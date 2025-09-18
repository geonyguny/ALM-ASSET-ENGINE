# sim_utils.py
from __future__ import annotations
import numpy as np
from typing import List

def simulate_portfolio_paths(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                             n_years: int, n_paths: int, rng: np.random.Generator):
    X = rng.multivariate_normal(mean=mu, cov=cov, size=(n_years, n_paths))
    return (X @ weights).astype(float)  # (T,P)

def generate_common_Z(global_assets: List[str], T: int, P: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(T, P, len(global_assets)))  # (T,P,K)

def simulate_paths_from_common_Z(Z_global: np.ndarray,
                                 global_assets: List[str],
                                 assets: List[str],
                                 mu: np.ndarray,
                                 cov: np.ndarray) -> np.ndarray:
    """
    공통 표준정규 Z를 정책 자산 순서에 맞춰 잘라 cov 촐레스키를 적용해 자산수익률 텐서 반환.
    반환: Y (T,P,k) ~ N(mu, cov)
    """
    T, P, _ = Z_global.shape
    idx = [global_assets.index(a) for a in map(str, assets)]
    Z_sub = Z_global[:, :, idx]                      # (T,P,k)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # 수치 안정(아주 작은 jitter)
        w, v = np.linalg.eigh(cov)
        w = np.clip(w, 1e-12, None)
        cov = (v @ np.diag(w) @ v.T)
        L = np.linalg.cholesky(cov)
    TP = T * P
    Z2 = Z_sub.reshape(TP, -1)
    Y  = Z2 @ L.T + mu[None, :]                      # (TP,k)
    return Y.reshape(T, P, -1)
