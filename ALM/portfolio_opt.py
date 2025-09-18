# portfolio_opt.py
# 평균-분산 효율 포트폴리오 유틸 (u, v, R 입력 기반)
# - 공매도 허용 닫힌해(GMV/Target/MSR)
# - 롱온리/상하한/섹터 제약 SLSQP
# - 효율곡선 그리드 산출
# - MSR 견고화 유틸(solve_msr_robust)
# 사용 전: pip install numpy scipy

from __future__ import annotations
from typing import Iterable, List, Tuple, Optional
import numpy as np
from numpy.linalg import inv, eig
from scipy.optimize import minimize


# ----------------------------
# 기본 유틸
# ----------------------------
def cov_from_corr(v: np.ndarray, R: np.ndarray) -> np.ndarray:
    """표준편차 v와 상관행렬 R로 공분산행렬 Σ 생성."""
    v = np.asarray(v, dtype=float).ravel()
    R = np.asarray(R, dtype=float)
    D = np.diag(v)
    return D @ R @ D


def nearest_psd(S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """대칭 행렬을 간단히 PSD로 보정(고유값 하한 클리핑)."""
    S = 0.5 * (S + S.T)
    vals, vecs = eig(S)
    vals = np.real(vals)
    vecs = np.real(vecs)
    vals_clipped = np.clip(vals, eps, None)
    return (vecs @ np.diag(vals_clipped) @ vecs.T)


def validate_inputs(mu: np.ndarray, v: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """모양·대칭·대각을 표준화."""
    mu = np.asarray(mu, dtype=float).ravel()
    v  = np.asarray(v , dtype=float).ravel()
    R  = np.asarray(R , dtype=float)
    n = len(mu)
    if v.shape != (n,):
        raise ValueError(f"v length must match mu (got {v.shape} vs {n})")
    if R.shape != (n, n):
        raise ValueError(f"R must be {n}x{n} (got {R.shape})")
    # 대칭/대각 보정(수치 오차 허용)
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return mu, v, R


def portfolio_stats(w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, r_f: float = 0.0) -> tuple[float, float, float]:
    """포트폴리오 기대수익, 변동성, 샤프지수."""
    w = np.asarray(w, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()
    Sigma = np.asarray(Sigma, dtype=float)
    ret = float(w @ mu)
    var = float(w @ Sigma @ w)
    var = max(var, 0.0)
    vol = float(np.sqrt(var))
    sharpe = (ret - r_f) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


# ----------------------------
# 닫힌형 해(공매도 허용, 합=1)
# ----------------------------
def efficient_closed_form(mu: np.ndarray,
                          v: np.ndarray,
                          R: np.ndarray,
                          mode: str = "gmv",
                          mu_target: Optional[float] = None,
                          r_f: float = 0.0) -> np.ndarray:
    """
    mode:
      - "gmv": 글로벌 최소분산
      - "target": 목표수익 mu_target 효율 포트폴리오
      - "msr": 탄젠시(최대 샤프), 무위험수익률 r_f 사용
    공매도 허용, 합=1.
    """
    mu, v, R = validate_inputs(mu, v, R)
    Sigma = nearest_psd(cov_from_corr(v, R))
    iS = inv(Sigma)
    one = np.ones_like(mu)
    A = float(one @ iS @ one)
    B = float(one @ iS @ mu)
    C = float(mu  @ iS @ mu)
    Delta = A * C - B * B
    if Delta <= 0:
        # 수치적 불안정 시 가벼운 보정 재시도
        Sigma = nearest_psd(Sigma, eps=1e-10)
        iS = inv(Sigma)
        A = float(one @ iS @ one)
        B = float(one @ iS @ mu)
        C = float(mu  @ iS @ mu)
        Delta = A * C - B * B
        if Delta <= 0:
            raise RuntimeError("Singular covariance; cannot compute closed-form weights.")

    if mode == "gmv":
        w = (iS @ one) / A
    elif mode == "target":
        if mu_target is None:
            raise ValueError("mu_target required for mode='target'")
        w = iS @ (((C - B * mu_target) / Delta) * one + ((A * mu_target - B) / Delta) * mu)
    elif mode == "msr":
        z = iS @ (mu - r_f * one)
        denom = float(one @ z)
        if abs(denom) < 1e-16:
            raise RuntimeError("Degenerate tangent portfolio (denominator ~ 0).")
        w = z / denom  # 합=1 정규화
    else:
        raise ValueError("mode must be 'gmv', 'target', or 'msr'")
    return np.asarray(w, dtype=float)


# ----------------------------
# 제약 최적화(롱온리/상하한/섹터)
# ----------------------------
def _make_sector_constraints(sector_bounds: Optional[List[Tuple[Iterable[int], float, float]]]):
    """섹터 제약을 SLSQP ineq 제약으로 생성."""
    cons = []
    if not sector_bounds:
        return cons
    for idxs, lo, hi in sector_bounds:
        I = np.array(list(idxs), dtype=int)
        lo_val = float(lo)
        hi_val = float(hi)
        # 주의: 람다 캡처 시 기본값 인자로 박아야 함
        cons.append({'type': 'ineq', 'fun': (lambda w, I=I, lo_val=lo_val: float(np.sum(w[I]) - lo_val))})
        cons.append({'type': 'ineq', 'fun': (lambda w, I=I, hi_val=hi_val: float(hi_val - np.sum(w[I])))})
    return cons


def efficient_qp(mu: np.ndarray,
                 v: np.ndarray,
                 R: np.ndarray,
                 mode: str = "gmv",
                 mu_target: Optional[float] = None,
                 r_f: float = 0.0,
                 lb: float = 0.0,
                 ub: float = 1.0,
                 sector_bounds: Optional[List[Tuple[Iterable[int], float, float]]] = None,
                 w0: Optional[np.ndarray] = None,
                 maxiter: int = 20_000,
                 ftol: float = 1e-12) -> np.ndarray:
    """
    SLSQP 기반 제약 최적화.
    - 합=1, 경계 lb<=w_i<=ub
    - mode="gmv"/"target"/"msr"
    - sector_bounds: [(indices, lower, upper), ...] 형태로 섹터 합 제한
    """
    mu, v, R = validate_inputs(mu, v, R)
    Sigma = nearest_psd(cov_from_corr(v, R))

    n = len(mu)
    if w0 is None:
        w0 = np.full(n, 1.0 / n, dtype=float)
    else:
        w0 = np.asarray(w0, dtype=float).ravel()
        if w0.shape != (n,):
            raise ValueError(f"w0 must be shape ({n},), got {w0.shape}")

    def risk(w: np.ndarray) -> float:
        return float(w @ Sigma @ w)

    def neg_sharpe(w: np.ndarray) -> float:
        ret = float(w @ mu)
        vol = np.sqrt(max(risk(w), 1e-18))
        return -(ret - r_f) / vol

    # 제약: 합=1
    cons = [{'type': 'eq', 'fun': lambda w: float(np.sum(w) - 1.0)}]

    # 목표수익
    if mode == "target":
        if mu_target is None:
            raise ValueError("mu_target required for mode='target'")
        cons.append({'type': 'eq', 'fun': (lambda w, mu=mu, mt=float(mu_target): float(w @ mu - mt))})

    # 섹터합 제약
    cons.extend(_make_sector_constraints(sector_bounds))

    bounds = [(float(lb), float(ub))] * n

    if mode in ("gmv", "target"):
        obj = risk
    elif mode == "msr":
        obj = neg_sharpe
    else:
        raise ValueError("mode must be 'gmv', 'target', or 'msr'")

    res = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'ftol': ftol, 'maxiter': maxiter, 'disp': False})
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    w = np.asarray(res.x, dtype=float)
    # 수치 오차 보정(합=1, 경계 클리핑)
    w = np.clip(w, lb, ub)
    s = float(w.sum())
    if s <= 0:
        raise RuntimeError("Optimization produced non-positive sum of weights.")
    w = w / s
    return w


# ----------------------------
# 샤프 최대 견고화(그리드 스윕)
# ----------------------------
def solve_msr_robust(mu: np.ndarray,
                     v: np.ndarray,
                     R: np.ndarray,
                     rf: float = 0.0,
                     lb: float = 0.0,
                     ub: float = 1.0,
                     w0: Optional[np.ndarray] = None,
                     n_grid: int = 61,
                     eps: float = 1e-6) -> np.ndarray:
    """
    직접 샤프최대(msr) 대신:
    1) mu_target 그리드를 lo~hi로 생성
    2) 각 타깃에서 '분산 최소 target'을 풀고
    3) 샤프가 최대인 해를 선택 → 비선형 불능/플랫 샤프에 견고

    반환: 최적 w (롱온리/경계 준수)
    """
    mu, v, R = validate_inputs(mu, v, R)
    Sigma = nearest_psd(cov_from_corr(v, R))

    mu_min, mu_max = float(np.min(mu)), float(np.max(mu))
    span = (mu_max - mu_min)
    lo = mu_min + eps * (span + 1.0)
    hi = mu_max - eps * (span + 1.0)
    if lo >= hi:
        # 자산 1개/동일수익률 등 극단 케이스 → GMV로 대체
        return efficient_qp(mu, v, R, mode="gmv", lb=lb, ub=ub, w0=w0)

    targets = np.linspace(lo, hi, int(n_grid))
    best = None
    for mt in targets:
        w = efficient_qp(mu, v, R, mode="target", mu_target=float(mt), lb=lb, ub=ub, w0=w0)
        ret, vol, sh = portfolio_stats(w, mu, Sigma, r_f=rf)
        if (best is None) or (sh > best[0]):
            best = (sh, w)
    return best[1]


# ----------------------------
# 효율곡선(타깃 수익 그리드)
# ----------------------------
def efficient_frontier(mu: np.ndarray,
                       v: np.ndarray,
                       R: np.ndarray,
                       mu_min: Optional[float] = None,
                       mu_max: Optional[float] = None,
                       n: int = 50,
                       long_only: bool = False,
                       lb: float = 0.0,
                       ub: float = 1.0,
                       sector_bounds: Optional[List[Tuple[Iterable[int], float, float]]] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    목표수익을 mu_min~mu_max 사이 그리드로 훑어 효율 포트폴리오(리스크 최소) 계산.
    반환: (targets, vols, weights_matrix[n x k])
    """
    mu, v, R = validate_inputs(mu, v, R)
    Sigma = nearest_psd(cov_from_corr(v, R))

    if mu_min is None:
        mu_min = float(np.min(mu))
    if mu_max is None:
        mu_max = float(np.max(mu))
    targets = np.linspace(mu_min, mu_max, int(n))
    k = len(mu)
    W = np.zeros((len(targets), k), dtype=float)
    vols = np.zeros(len(targets), dtype=float)

    for i, mt in enumerate(targets):
        if long_only:
            w = efficient_qp(mu, v, R, mode="target", mu_target=float(mt),
                             lb=lb, ub=ub, sector_bounds=sector_bounds)
        else:
            w = efficient_closed_form(mu, v, R, mode="target", mu_target=float(mt))
        W[i] = w
        _, vol, _ = portfolio_stats(w, mu, Sigma)
        vols[i] = vol
    return targets, vols, W


# ----------------------------
# 편의 래퍼
# ----------------------------
def weights_gmv(mu: np.ndarray, v: np.ndarray, R: np.ndarray,
                long_only: bool = False, lb: float = 0.0, ub: float = 1.0) -> np.ndarray:
    return efficient_qp(mu, v, R, mode="gmv", lb=lb, ub=ub) if long_only \
        else efficient_closed_form(mu, v, R, mode="gmv")


def weights_msr(mu: np.ndarray, v: np.ndarray, R: np.ndarray, r_f: float = 0.0,
                long_only: bool = False, lb: float = 0.0, ub: float = 1.0) -> np.ndarray:
    return efficient_qp(mu, v, R, mode="msr", r_f=r_f, lb=lb, ub=ub) if long_only \
        else efficient_closed_form(mu, v, R, mode="msr", r_f=r_f)


def weights_target(mu: np.ndarray, v: np.ndarray, R: np.ndarray, mu_target: float,
                   long_only: bool = False, lb: float = 0.0, ub: float = 1.0) -> np.ndarray:
    return efficient_qp(mu, v, R, mode="target", mu_target=mu_target, lb=lb, ub=ub) if long_only \
        else efficient_closed_form(mu, v, R, mode="target", mu_target=mu_target)
