# alm/sim_utils.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List

__all__ = [
    "generate_common_Z",
    "mvn_from_Z",
    "simulate_portfolio_paths",
    "min_contrib_schedule_paths",
    "cstar_for_path",
]

# -------- Common shock utilities --------
def generate_common_Z(assets: List[str], T: int, P: int, seed: int):
    """공통 표준정규 Z(T,P,K) 생성."""
    rng = np.random.default_rng(seed)
    K = len(assets)
    Z = rng.standard_normal((T, P, K))
    return {"assets": list(assets), "Z": Z}

def mvn_from_Z(mu: np.ndarray, cov: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """표준정규 Z(T,P,K)을 cov, mu로 선형변환 → X(T,P,K)."""
    K = len(mu)
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(K))
    T, P, _ = Z.shape
    X = Z.reshape(T * P, K) @ L.T
    X = X.reshape(T, P, K) + mu[None, None, :]
    return X

# -------- Path simulation --------
def simulate_portfolio_paths(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    n_years: int,
    n_paths: int,
    rng: np.random.Generator,
    common_Z: np.ndarray | None = None,
    common_assets: list[str] | None = None,
    policy_assets: list[str] | None = None,
) -> np.ndarray:
    """
    common_Z가 주어지면 공통 표준정규에서 policy 자산순서로 슬라이스해 사용.
    """
    if common_Z is None:
        X = rng.multivariate_normal(mean=mu, cov=cov, size=(n_years, n_paths))
    else:
        assert common_assets is not None and policy_assets is not None
        idx = [common_assets.index(a) for a in policy_assets]
        Zp = common_Z[:, :, idx]  # (T,P,K_policy)
        X = mvn_from_Z(mu, cov, Zp)  # (T,P,K_policy)
    Rp = (X @ weights).astype(float)  # (T, P)
    return Rp

# -------- Schedule mode: min required contribution per year --------
def min_contrib_schedule_paths(
    Rp: np.ndarray,
    A0: float,
    L0: float,
    liab_proj: pd.DataFrame,
    fr_targets: np.ndarray,
    pay_timing: str = "MID",
    liab_mode: str = "target",
) -> Dict[str, np.ndarray]:
    T, P = Rp.shape
    SC = liab_proj["service_cost"].values[:T]
    IC = liab_proj["interest_cost"].values[:T]
    B  = liab_proj["benefit_cf"].values[:T]
    L_input = liab_proj["closing_PBO"].values[:T]

    A = np.full(P, A0, dtype=float)
    L = float(L0)

    contribs = np.zeros((T, P), dtype=float)
    FRs      = np.zeros((T, P), dtype=float)
    F_open   = np.zeros((T, P), dtype=float)
    F_close  = np.zeros((T, P), dtype=float)
    ret_amt  = np.zeros((T, P), dtype=float)
    ret_rate = np.zeros((T, P), dtype=float)
    L_close  = np.zeros(T, dtype=float)

    eps = 1e-12
    for t in range(T):
        F_open[t, :] = A
        if liab_mode.lower() == "target":
            L = float(L_input[t])
        else:
            L = L + SC[t] + IC[t] - B[t]
        L_close[t] = L

        R = Rp[t, :]
        ret_rate[t, :] = R

        if pay_timing.upper() == "MID":
            denom = np.maximum(1.0 + R, eps)
            C_req = -A + B[t] + (fr_targets[t] * L) / denom
            C = np.maximum(C_req, 0.0)
            base = A + C - B[t]; ret = base * R; A = base * (1.0 + R)
        elif pay_timing.upper() == "BOP":
            denom = np.maximum(1.0 + R, eps)
            C_req = (fr_targets[t] * L + B[t]) / denom - A
            C = np.maximum(C_req, 0.0)
            base = A + C; ret = base * R; A = base * (1.0 + R) - B[t]
        else:  # EOP
            C_req = (fr_targets[t] * L) - A * (1.0 + R) + B[t]
            C = np.maximum(C_req, 0.0)
            base = A; ret = base * R; A = A * (1.0 + R) + C - B[t]

        contribs[t, :], ret_amt[t, :], F_close[t, :], FRs[t, :] = C, ret, A, A / max(L, 1e-9)

    return {
        "contribs": contribs,
        "FRs": FRs,
        "F_open": F_open,
        "F_close": F_close,
        "ret_amt": ret_amt,
        "ret_rate": ret_rate,
        "L_close": L_close,
    }

# -------- Fixed-C helper --------
def cstar_for_path(
    Rp_path: np.ndarray,
    A0: float,
    L0: float,
    liab_proj: pd.DataFrame,
    fr_targets: np.ndarray,
    pay_timing: str = "MID",
):
    T = len(Rp_path)
    SC = liab_proj["service_cost"].values[:T]
    IC = liab_proj["interest_cost"].values[:T]
    B  = liab_proj["benefit_cf"].values[:T]

    def ok(C):
        A, L = A0, L0
        for t in range(T):
            L = L + SC[t] + IC[t] - B[t]
            if pay_timing.upper() == "MID":
                A = (A + C - B[t]) * (1.0 + Rp_path[t])
            elif pay_timing.upper() == "BOP":
                A = (A + C) * (1.0 + Rp_path[t]) - B[t]
            else:
                A = A * (1.0 + Rp_path[t]) + C - B[t]
            if A / max(L, 1e-9) < fr_targets[t] - 1e-12:
                return False
        return True

    lo, hi = 0.0, 0.0
    if not ok(0.0):
        hi = max(1.0, B.max())
        while not ok(hi):
            hi *= 2.0
            if hi > 1e13:
                return np.nan
    return _bisect(lo, hi, ok)

def _bisect(lo, hi, pred, tol=1e-6, maxit=60):
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        if pred(mid):
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) <= tol * (1.0 + hi):
            break
    return hi
