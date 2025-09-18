# corr_utils.py
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def shrink_to_psd(R: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    R = 0.5 * (R + R.T)
    try:
        m = np.linalg.eigvalsh(R).min()
        if m < eps:
            lam = min(0.1, (eps - m) / (1 - m + 1e-12))
            R = (1 - lam) * R + lam * np.eye(R.shape[0])
    except np.linalg.LinAlgError:
        R = R + eps * np.eye(R.shape[0])
    np.fill_diagonal(R, 1.0)
    return R

def fingerprint(arr) -> str:
    a = np.asarray(arr, float).ravel()
    return hashlib.sha1(a.tobytes()).hexdigest()[:12]

def build_cov(policy_assets: List[str],
              u_map: Dict[str, float],
              v_map: Dict[str, float],
              corr_long: pd.DataFrame,
              strict_corr: bool = True) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assets = list(policy_assets)
    k = len(assets)
    R = np.eye(k)
    seen = np.eye(k, dtype=bool)

    if corr_long is not None and len(corr_long):
        idx = {a: i for i, a in enumerate(assets)}
        for _, row in corr_long.iterrows():
            a, b, rho = str(row["a"]).strip(), str(row["b"]).strip(), float(row["rho"])
            if a in idx and b in idx and a != b:
                i, j = idx[a], idx[b]
                R[i, j] = R[j, i] = rho
                seen[i, j] = seen[j, i] = True

    if strict_corr:
        missing = []
        for i in range(k):
            for j in range(i+1, k):
                if not seen[i, j]:
                    missing.append(f"{assets[i]}-{assets[j]}")
        if missing:
            sample = ", ".join(missing[:12])
            raise ValueError(f"[corr] 누락된 자산쌍 {len(missing)}개 (예: {sample}). corr 시트를 보완하세요.")

    R = np.clip(R, -0.999, 0.999)
    np.fill_diagonal(R, 1.0)
    R = shrink_to_psd(R)

    sig = np.array([v_map[a] for a in assets], dtype=float)
    S = np.outer(sig, sig) * R
    mu = np.array([u_map[a] for a in assets], dtype=float)
    return assets, mu, sig, R, S