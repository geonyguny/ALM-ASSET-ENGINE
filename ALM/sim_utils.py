# alm/sim_utils.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

__all__ = [
    # 기존
    "generate_common_Z",
    "mvn_from_Z",
    "simulate_portfolio_paths",
    "min_contrib_schedule_paths",
    "cstar_for_path",
    # 신규
    "simulate_short_rate_paths",
    "infer_regime_codes",
    "build_timevarying_mu_cov",
    "mvn_from_Z_timevarying",
]

# ============================================================
# -------- Common shock utilities (기존) ----------------------
# ============================================================
def generate_common_Z(assets: List[str], T: int, P: int, seed: int):
    """공통 표준정규 Z(T,P,K) 생성."""
    rng = np.random.default_rng(seed)
    K = len(assets)
    Z = rng.standard_normal((T, P, K))
    return {"assets": list(assets), "Z": Z}

def mvn_from_Z(mu: np.ndarray, cov: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """표준정규 Z(T,P,K)을 cov, mu로 선형변환 → X(T,P,K) (정적 파라미터).
    - mu: (K,)
    - cov: (K,K)
    - Z:   (T,P,K) 표준정규
    """
    K = len(mu)
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(K))
    T, P, _ = Z.shape
    X = Z.reshape(T * P, K) @ L.T
    X = X.reshape(T, P, K) + mu[None, None, :]
    return X

# ============================================================
# -------- NEW: 금리 시나리오 & 연동 파트 ---------------------
# ============================================================
def simulate_short_rate_paths(
    model: str = "vasicek",
    r0: float = 0.03,
    kappa: float = 0.10,
    theta: float = 0.03,
    sigma_r: float = 0.01,
    n_years: int = 10,
    n_paths: int = 1000,
    dt: float = 1.0 / 12.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """단기금리 경로 시뮬레이션 (Vasicek/CIR).
    Returns: rates (T,P) with T = int(n_years/dt)
    """
    rng = np.random.default_rng(seed)
    steps = int(round(n_years / dt))
    rates = np.empty((steps, n_paths), dtype=float)
    rates[0, :] = r0
    sqrt_dt = np.sqrt(dt)

    if model.lower() == "vasicek":
        for t in range(1, steps):
            Z = rng.standard_normal(n_paths)
            rt = rates[t - 1, :]
            rates[t, :] = rt + kappa * (theta - rt) * dt + sigma_r * sqrt_dt * Z
    elif model.lower() == "cir":
        for t in range(1, steps):
            Z = rng.standard_normal(n_paths)
            rt = np.maximum(rates[t - 1, :], 0.0)
            # 표준 CIR: dr = kappa(theta-r)dt + sigma*sqrt(r)*dW
            rates[t, :] = np.abs(rt + kappa * (theta - rt) * dt + sigma_r * np.sqrt(rt) * sqrt_dt * Z)
    else:
        raise ValueError("model must be 'vasicek' or 'cir'")
    return rates


def infer_regime_codes(
    r_t: np.ndarray,
    infl_t: Optional[np.ndarray] = None,
    rate_thresh: Tuple[float, float] = (0.02, 0.035),
    infl_thresh: Tuple[float, float] = (0.02, 0.035),
) -> np.ndarray:
    """레짐코드 반환 (vectorized).
    0 = disinflation, 1 = reflation, 2 = stagflation
    - 금리와 인플레를 함께 쓰되, infl_t가 None이면 금리 기준만 사용.
    """
    low_r, high_r = rate_thresh
    if infl_t is None:
        # 금리만으로 단순 판정
        codes = np.where(r_t < low_r, 0, np.where(r_t < high_r, 1, 2))
        return codes.astype(int)

    low_i, high_i = infl_thresh
    # 인플레 기준 우선, 그 안에서 금리로 보조 판정
    codes = np.empty_like(r_t, dtype=int)
    # disinflation: 낮은 인플레
    mask_d = infl_t < low_i
    # reflation: 중간 인플레
    mask_r = (infl_t >= low_i) & (infl_t <= high_i)
    # stagflation: 높은 인플레
    mask_s = infl_t > high_i
    codes[mask_d] = 0
    codes[mask_r] = 1
    codes[mask_s] = 2

    # (선택) 매우 높은 금리구간은 스태그로 끌어올리는 보정
    codes = np.where(r_t > high_r, 2, codes)
    return codes


def build_timevarying_mu_cov(
    assets: List[str],
    short_rate_paths: np.ndarray,  # (T,P)
    *,
    # ----- Equity 설정 -----
    equity_assets: Optional[List[str]] = None,  # 지정 없으면 name에 "stock" 포함 탐지
    ERP_base: float = 0.035,            # 기본 ERP
    ERP_reflation_adj: float = -0.005,  # 리플레이션 ERP 보정
    ERP_stagflation_adj: float = 0.007, # 스태그플레이션 ERP 보정
    equity_sigma_base: float = 0.18,    # 기본 변동성
    equity_sigma_stag_add: float = 0.04, # 스태그플레이션 추가 변동성(+4%p 등)
    # ----- Bond 설정 -----
    bond_assets: Optional[List[str]] = None,   # 지정 없으면 name에 "bond" 포함 탐지
    bond_duration_map: Optional[Dict[str, float]] = None,  # 자산별 듀레이션
    sigma_r_param: float = 0.01,        # 금리 변동성 파라미터 (simulate_short_rate_paths와 일치 권장)
    bond_mu_mode: str = "rate_level",   # "rate_level" or "carry_roll"
    # ----- Cash 설정 -----
    cash_assets: Optional[List[str]] = None,   # 지정 없으면 name에 "cash" 포함 탐지
    cash_spread_adj: float = 0.0,       # 현금 수익률 보정(수수료 등)
    # ----- 상관(레짐별) -----
    regime_corr: Optional[Dict[int, np.ndarray]] = None,   # {0/1/2: (K,K) corr}
    # ----- 기타 -----
    dt: float = 1.0 / 12.0,
    inflation_paths: Optional[np.ndarray] = None,  # (T,P) or None
) -> Tuple[np.ndarray, np.ndarray]:
    """금리경로와 레짐을 바탕으로 per-step 자산 μ/Σ를 생성.
    Returns:
      mu_t:  (T,K)     각 시점 기대수익(연율)
      cov_t: (T,K,K)   각 시점 공분산(연율)

    규칙(간단화, 합리적 초기치):
    - Cash: μ = r_t + cash_spread_adj, σ ~ 0
    - Bond: μ ≈
         "rate_level" : μ = r_t (현 레벨 기반의 단순 기대)
         "carry_roll" : μ = r0 + (r_t - r0)*Duration (롤다운/레벨 민감 근사)
      σ = Duration * sigma_r_param
    - Equity: μ = r_t + ERP(state),  σ = base ± (레짐 추가),  corr = 레짐별 행렬
    """
    T, P = short_rate_paths.shape
    K = len(assets)
    mu_t = np.zeros((T, K), dtype=float)
    sig_t = np.zeros((T, K), dtype=float)

    # 자산군 자동 탐지(미지정 시)
    if equity_assets is None:
        equity_assets = [a for a in assets if "stock" in a.lower() or "equity" in a.lower()]
    if bond_assets is None:
        bond_assets = [a for a in assets if "bond" in a.lower()]
    if cash_assets is None:
        cash_assets = [a for a in assets if "cash" in a.lower()]

    # 인덱스 매핑
    idx_equity = [assets.index(a) for a in equity_assets if a in assets]
    idx_bond   = [assets.index(a) for a in bond_assets if a in assets]
    idx_cash   = [assets.index(a) for a in cash_assets if a in assets]

    # 듀레이션 맵 기본값
    if bond_duration_map is None:
        bond_duration_map = {a: 7.0 for a in bond_assets}

    # 레짐별 상관행렬 기본값 (없으면 합리적 초기값)
    if regime_corr is None:
        # 매우 단순한 기본: 모든 자산 간 0.2 상관, 주식-채권은 레짐에서 부호 전환
        base_corr = 0.2 * np.ones((K, K), dtype=float)
        np.fill_diagonal(base_corr, 1.0)
        corr_dis = base_corr.copy()
        corr_ref = base_corr.copy()
        corr_sta = base_corr.copy()
        # 주식-채권 쌍 조정
        for i in idx_equity:
            for j in idx_bond:
                corr_dis[i, j] = corr_dis[j, i] = -0.2
                corr_ref[i, j] = corr_ref[j, i] = +0.10
                corr_sta[i, j] = corr_sta[j, i] = +0.30
        regime_corr = {0: corr_dis, 1: corr_ref, 2: corr_sta}

    # 레짐 코드(T,P)
    infl = inflation_paths if inflation_paths is not None else None
    codes_tp = np.empty((T, P), dtype=int)
    for t in range(T):
        r_t = short_rate_paths[t, :]
        infl_t = infl[t, :] if infl is not None else None
        codes_tp[t, :] = infer_regime_codes(r_t, infl_t)

    # per-time 평균 레짐(경영 보고/간소화용): 각 t에서 다수표 결정을 사용
    # (엄밀히는 경로별 레짐으로 μ/Σ를 path-specific으로 잡을 수 있으나, 여긴 시간변동(공통) 파라미터로 둠)
    codes_t = np.zeros(T, dtype=int)
    for t in range(T):
        # 경로별 레짐의 최빈값
        vals, cnts = np.unique(codes_tp[t, :], return_counts=True)
        codes_t[t] = int(vals[np.argmax(cnts)])

    # 각 시점별 μ/σ 구성 (연율)
    r_bar_t = short_rate_paths.mean(axis=1)  # (T,) 평균 단기금리

    for t in range(T):
        code = codes_t[t]  # 0/1/2
        rbar = r_bar_t[t]

        # ----- Cash -----
        for i in idx_cash:
            mu_t[t, i]  = rbar + cash_spread_adj
            sig_t[t, i] = 0.0001  # 거의 0 (수치안정용 미소값)

        # ----- Bond -----
        for i in idx_bond:
            dur = float(bond_duration_map.get(assets[i], 7.0))
            if bond_mu_mode == "carry_roll":
                # 레벨 변화 민감 근사: r_t - r0 항을 사용하려면 r0 저장 필요.
                # 간략화: 현재 레벨(rbar)을 기대수익으로 사용
                mu_t[t, i] = rbar
            else:  # "rate_level"
                mu_t[t, i] = rbar
            sig_t[t, i] = max(dur * sigma_r_param, 1e-6)

        # ----- Equity -----
        if code == 0:
            erp = ERP_base
            sig_add = 0.0
        elif code == 1:
            erp = ERP_base + ERP_reflation_adj
            sig_add = 0.0
        else:
            erp = ERP_base + ERP_stagflation_adj
            sig_add = equity_sigma_stag_add

        for i in idx_equity:
            mu_t[t, i]  = rbar + erp
            sig_t[t, i] = max(equity_sigma_base + sig_add, 1e-6)

    # 각 시점 공분산 Σ_t = D σ_t * Corr(regime) * D σ_t
    cov_t = np.zeros((T, K, K), dtype=float)
    for t in range(T):
        Corr = regime_corr[int(codes_t[t])]
        D = np.diag(sig_t[t, :])
        cov_t[t, :, :] = D @ Corr @ D

    return mu_t, cov_t


def mvn_from_Z_timevarying(mu_t: np.ndarray, chol_t: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """시간변동 μ/Σ(Cholesky)를 사용해 Z(T,P,K) → X(T,P,K) 변환.
    - mu_t:   (T,K)
    - chol_t: (T,K,K)  (각 시점 공분산의 chol 분해)
    - Z:      (T,P,K)  표준정규
    """
    T, P, K = Z.shape
    X = np.empty((T, P, K), dtype=float)
    Zr = Z.reshape(T * P, K)
    # 블록별 적용
    offset = 0
    for t in range(T):
        Lt = chol_t[t]
        block = Zr[offset : offset + P, :] @ Lt.T   # (P,K)
        block += mu_t[t][None, :]
        X[t, :, :] = block
        offset += P
    return X

# ============================================================
# -------- Path simulation (리팩토링) -------------------------
# ============================================================
def simulate_portfolio_paths(
    weights: np.ndarray,
    mu: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    cov: Optional[np.ndarray],
    n_years: int,
    n_paths: int,
    rng: np.random.Generator,
    common_Z: Optional[np.ndarray] = None,
    common_assets: Optional[List[str]] = None,
    policy_assets: Optional[List[str]] = None,
) -> np.ndarray:
    """
    포트폴리오 경로 시뮬레이션 (연율 μ/Σ 가정).
    - 정적 모드(기존): mu=(K,), cov=(K,K)
    - 시간변동 모드(신규): mu=(T,K), cov=(T,K,K)
      * 이때 common_Z가 있으면 (T,P,K_policy) 형태로 동일 시계열 상관을 적용

    Returns:
      Rp: (T, P)  포트폴리오 연율 수익률(단일 기간 수익률; 후처리에서 dt 반영 가능)
    """
    # ---- 입력 파싱 ----
    if isinstance(mu, tuple):
        # (mu_t, cov_t)를 mu에 담아 온 경우 (호환 편의)
        mu_t, cov_t = mu
    else:
        mu_t, cov_t = mu, cov

    # 정적 vs 시간변동 분기
    timevarying = (mu_t is not None and mu_t.ndim == 2)

    if not timevarying:
        # ===== 정적 모드 (기존과 동일) =====
        assert cov is not None and mu is not None and mu.ndim == 1 and cov.ndim == 2
        if common_Z is None:
            X = rng.multivariate_normal(mean=mu, cov=cov, size=(n_years, n_paths))
        else:
            assert common_assets is not None and policy_assets is not None
            idx = [common_assets.index(a) for a in policy_assets]
            Zp = common_Z[:, :, idx]  # (T,P,K)
            X = mvn_from_Z(mu, cov, Zp)  # (T,P,K)
        Rp = (X @ weights).astype(float)  # (T,P)
        return Rp

    # ===== 시간변동 모드 =====
    T, K = mu_t.shape
    assert cov_t is not None and cov_t.shape == (T, K, K), "cov_t shape must be (T,K,K)"

    # chol 분해 (각 t)
    chol_t = np.empty_like(cov_t)
    eye = np.eye(K)
    for t in range(T):
        chol_t[t] = np.linalg.cholesky(cov_t[t] + 1e-12 * eye)

    # Z 생성/슬라이스
    if common_Z is None:
        Z = rng.standard_normal((T, n_paths, K))
    else:
        assert common_assets is not None and policy_assets is not None
        idx = [common_assets.index(a) for a in policy_assets]
        Z = common_Z[:, :, idx]  # (T,P,K) 재사용

    # X 생성
    X = mvn_from_Z_timevarying(mu_t, chol_t, Z)  # (T,P,K)
    Rp = (X @ weights).astype(float)  # (T,P)
    return Rp

# ============================================================
# -------- Schedule mode: min required contribution per year --
# ============================================================
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

# ============================================================
# -------- Fixed-C helper (기존) ------------------------------
# ============================================================
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
