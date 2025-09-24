# optimize/resampled_frontier.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Sequence, Tuple, Optional
from optimize.mvo import solve_mvo

def _sample_returns_normal(mu, Sigma, n_obs, rng):
    return rng.multivariate_normal(mu, Sigma, size=n_obs)

def _estimate_mu_cov(R):
    # R: (n, K)
    mu_hat = R.mean(axis=0)
    Sigma_hat = np.cov(R, rowvar=False, ddof=1)
    return mu_hat, Sigma_hat

def build_resampled_frontier(
    assets: Sequence[str],
    mu: np.ndarray,
    Sigma: np.ndarray,
    targets: Sequence[float],
    target_type: str = "vol",      # or "ret"
    K: int = 200,
    n_obs: int = 60,               # 재표본 가상의 ‘월’ 관측 수
    sampler: str = "normal",       # "normal" (향후 bootstrap 지원 가능)
    seed: int = 42,
    lb: Optional[Dict[str, float]] = None,
    ub: Optional[Dict[str, float]] = None,
    risky_cap: Optional[Tuple[Sequence[str], float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Resampled EF: target grid별로 K회 MVO → w 평균/표준편차/요약."""
    rng = np.random.default_rng(seed)
    assets = list(assets)
    K_assets = len(assets)
    mu = np.asarray(mu, float).reshape(-1)
    Sigma = np.asarray(Sigma, float)

    rows = []          # for ref_frontier long-form
    weights_all = {}   # target -> (K_runs, K_assets)

    for tgt in targets:
        w_runs = np.zeros((K, K_assets))
        stat_runs = []

        for k in range(K):
            if sampler == "normal":
                R = _sample_returns_normal(mu, Sigma, n_obs, rng)
            else:
                raise NotImplementedError("Only sampler='normal' supported in v1.0")

            mu_hat, Sigma_hat = _estimate_mu_cov(R)
            sol = solve_mvo(
                assets=assets, mu=mu_hat, Sigma=Sigma_hat,
                target=tgt, target_type=target_type,
                lb=lb, ub=ub, risky_cap=risky_cap,
            )
            w_runs[k, :] = sol["w"]
            stat_runs.append((sol["ret"], sol["vol"]))

        w_bar = w_runs.mean(axis=0)
        w_std = w_runs.std(axis=0, ddof=1)

        # store per-asset rows
        for a, wi, si in zip(assets, w_bar, w_std):
            rows.append({"target": tgt, "asset": a, "w_bar": wi, "w_std": si})

        # store aggregate for convenience
        weights_all[tgt] = w_runs

    ref_frontier = pd.DataFrame(rows)

    # 신뢰구간용 통계 (분위수)
    ci_rows = []
    for tgt, W in weights_all.items():
        for j, a in enumerate(assets):
            w_p05 = np.percentile(W[:, j], 5)
            w_p50 = np.percentile(W[:, j], 50)
            w_p95 = np.percentile(W[:, j], 95)
            ci_rows.append({"target": tgt, "asset": a, "w_p05": w_p05, "w_p50": w_p50, "w_p95": w_p95})
    weights_ci = pd.DataFrame(ci_rows)

    return ref_frontier, weights_ci
