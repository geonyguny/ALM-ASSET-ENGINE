# stats_utils.py
from __future__ import annotations
import numpy as np
from typing import Dict

def _q_index(n: int, q: float) -> int:
    return int(np.clip(np.ceil(q * n) - 1, 0, n - 1))

def paired_quantiles(x: np.ndarray, y: np.ndarray,
                     qs=(0.05, 0.25, 0.50, 0.75, 0.95)) -> Dict[str, float]:
    out = {}
    n = len(x)
    order_x = np.argsort(x); y_by_x = y[order_x]
    for q in qs:
        out[f"FR_at_contrib_p{int(q*100)}"] = float(y_by_x[_q_index(n, q)])
    order_y = np.argsort(y); x_by_y = x[order_y]
    for q in qs:
        out[f"contrib_at_FR_p{int(q*100)}"] = float(x_by_y[_q_index(n, q)])
    return out

def dist_stats(prefix: str, arr: np.ndarray) -> Dict[str, float]:
    def q(p): return float(np.percentile(arr, p))
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_p5":   q(5),
        f"{prefix}_p25":  q(25),
        f"{prefix}_p50":  q(50),
        f"{prefix}_p75":  q(75),
        f"{prefix}_p95":  q(95),
    }
