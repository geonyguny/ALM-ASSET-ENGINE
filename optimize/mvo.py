# optimize/mvo.py
from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Sequence, Tuple
from scipy.optimize import minimize

def _as_bounds(assets: Sequence[str], lb: Optional[Dict[str, float]], ub: Optional[Dict[str, float]]):
    lbs, ubs = [], []
    for a in assets:
        lbs.append(0.0 if lb is None or a not in lb else lb[a])
        ubs.append(1.0 if ub is None or a not in ub else ub[a])
    return list(zip(lbs, ubs))

def _risk_cap_vector(assets: Sequence[str], risky_set: Optional[Sequence[str]]):
    if not risky_set:
        return None
    v = np.zeros(len(assets))
    idx = [assets.index(a) for a in risky_set if a in assets]
    v[idx] = 1.0
    return v

def solve_mvo(
    assets: Sequence[str],
    mu: np.ndarray,            # (K,)
    Sigma: np.ndarray,         # (K,K)
    target: float,
    target_type: str = "vol",  # "vol" or "ret"
    lb: Optional[Dict[str, float]] = None,
    ub: Optional[Dict[str, float]] = None,
    risky_cap: Optional[Tuple[Sequence[str], float]] = None,  # (risky_assets, cap)
    long_only: bool = True,
    tol: float = 1e-8,
    maxiter: int = 500,
) -> Dict[str, np.ndarray]:
    """Simple SLSQP-based constrained MVO."""
    K = len(assets)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    assert mu.shape[0] == K and Sigma.shape == (K, K)

    bounds = _as_bounds(assets, lb, ub)
    if not long_only:
        # allow shorting if no lb provided
        bounds = [(-1.0, ub if isinstance(ub, (int, float)) else 1.0) for _, ub in bounds]

    Aeq = np.ones((1, K))
    beq = np.array([1.0])

    risky_vec = None
    risky_cap_val = None
    if risky_cap:
        risky_vec = _risk_cap_vector(assets, risky_cap[0])
        risky_cap_val = float(risky_cap[1])

    def port_var(w): return float(w @ Sigma @ w)
    def port_vol(w): return np.sqrt(max(port_var(w), 0.0))
    def port_ret(w): return float(mu @ w)

    # objective + constraints
    if target_type.lower() == "vol":
        # maximize return s.t. vol <= target
        def obj(w): return -port_ret(w)
        cons = [
            {"type": "eq", "fun": lambda w, A=Aeq, b=beq: A @ w - b},
            {"type": "ineq", "fun": lambda w, t=target: t**2 - w @ Sigma @ w},
        ]
    elif target_type.lower() == "ret":
        # minimize variance s.t. return >= target
        def obj(w): return w @ Sigma @ w
        cons = [
            {"type": "eq", "fun": lambda w, A=Aeq, b=beq: A @ w - b},
            {"type": "ineq", "fun": lambda w, t=target: port_ret(w) - t},
        ]
    else:
        raise ValueError("target_type must be 'vol' or 'ret'.")

    if risky_vec is not None:
        cons.append({"type": "ineq", "fun": lambda w, v=risky_vec, cap=risky_cap_val: cap - float(v @ w)})

    w0 = np.full(K, 1.0 / K)
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons, tol=tol, options={"maxiter": maxiter})
    if not res.success:
        raise RuntimeError(f"MVO failed: {res.message}")

    w = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])
    w = w / w.sum()  # numerical cleanup
    return {
        "w": w,
        "ret": port_ret(w),
        "vol": port_vol(w),
        "var": port_var(w),
        "success": True,
        "message": res.message,
        "nit": res.nit,
    }
