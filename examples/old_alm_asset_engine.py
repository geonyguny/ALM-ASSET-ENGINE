# alm_asset_engine.py
# 목적: fr_target_t를 충당하기 위한 "연도별 최소 필요 기여금 C_t^* (>=0)" 산출
# - 기본 모드: schedule (연도별 스케줄 기여금)
# - 옵션 모드: fixed (연 고정 C)
# - targets 로더: 대/소문자, 공백, 언더스코어, 병합/퍼센트 표기 보정
# - liability 처리:
#     * liab_scenario 선택 (base/p50/기타) + 중복 연도 정리
#     * liab_mode: 'target' (엑셀 closing_PBO 고정) / 'roll' (SC/IC/B 롤포워드)
# - CSV: 정책별·연도별 집계, 전체 path 상세(자산/수익금/수익률/FR/PBO 입력/사용), 자산 base&mean 궤적
# - 통계: FR/기여금/자산/수익률 = mean, p5, p25, p50, p75, p95
# - NEW: 시뮬 직전 스냅샷 덤프(_inputs/), 상관행렬 엄격검사(strict_corr), R 해시 지문

import os
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union

# =========================
# 0) 유틸
# =========================
def _ensure_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] 시트에 필요한 컬럼이 없습니다: {missing}")

def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _last_hist_year(asset_df, liab_df):
    years_a = asset_df.loc[asset_df["closing_F"].notna(), "asset"]
    years_l = liab_df.loc[liab_df["closing_PBO"].notna(), "liability"]
    y = int(sorted(set(years_a).intersection(set(years_l)))[-1])
    return y

def _shrink_to_psd(R, eps=1e-10):
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

def _fingerprint(arr) -> str:
    """상관/공분산 등 배열의 지문(sha1 12자)."""
    a = np.asarray(arr, float).ravel()
    return hashlib.sha1(a.tobytes()).hexdigest()[:12]

# ---- paired quantile helpers ----
def _q_index(n: int, q: float) -> int:
    return int(np.clip(np.ceil(q * n) - 1, 0, n - 1))

def _paired_quantiles(x: np.ndarray, y: np.ndarray,
                      qs=(0.05, 0.25, 0.50, 0.75, 0.95)) -> dict:
    """
    x(예: 기여금)로 정렬했을 때 동일 경로의 y(예: FR)를 취한 '쌍 분위수'와,
    y로 정렬했을 때 동일 경로의 x 분위수도 함께 반환.
    (qs = p5/p25/p50/p75/p95)
    """
    out = {}
    n = len(x)
    order_x = np.argsort(x); y_by_x = y[order_x]
    for q in qs:
        out[f"FR_at_contrib_p{int(q*100)}"] = float(y_by_x[_q_index(n, q)])
    order_y = np.argsort(y); x_by_y = x[order_y]
    for q in qs:
        out[f"contrib_at_FR_p{int(q*100)}"] = float(x_by_y[_q_index(n, q)])
    return out

def _dist_stats(prefix: str, arr: np.ndarray) -> dict:
    """분위수 세트(mean, p5/p25/p50/p75/p95)를 prefix로 묶어 dict 반환."""
    def q(p): return float(np.percentile(arr, p))
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_p5":   q(5),
        f"{prefix}_p25":  q(25),
        f"{prefix}_p50":  q(50),
        f"{prefix}_p75":  q(75),
        f"{prefix}_p95":  q(95),
    }

# =========================
# 0-b) 스냅샷 유틸
# =========================
def _dump_snapshot(csvdir: Optional[str], *, policy: int,
                   seed: int, mode: str, liab_mode: str, liab_scenario: Optional[Union[str,int]],
                   assets: List[str], w: np.ndarray, u: np.ndarray, v: np.ndarray, R: np.ndarray,
                   A0: float, L0: float, fr_targets_head: float, strict_corr: bool):
    """
    시뮬 직전 입력을 csvdir/_inputs/에 저장(정책별 비교용).
    """
    if not csvdir:
        return
    dbg = Path(csvdir) / "_inputs"
    dbg.mkdir(parents=True, exist_ok=True)

    meta = {
        "policy": policy,
        "seed": seed,
        "mode": mode,
        "liab_mode": liab_mode,
        "liab_scenario": liab_scenario,
        "assets": list(map(str, assets)),
        "w_sum": float(np.sum(w)),
        "w_min": float(np.min(w)),
        "w_max": float(np.max(w)),
        "R_fingerprint": _fingerprint(R),
        "R_diag": np.diag(R).tolist(),
        "A0": float(A0),
        "L0": float(L0),
        "fr_target_first": float(fr_targets_head),
        "strict_corr": bool(strict_corr),
    }
    (dbg / f"inputs_policy{policy}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    pd.DataFrame({"asset": assets, "w": w, "u": u, "v": v}).to_csv(
        dbg / f"weights_policy{policy}.csv", index=False, encoding="utf-8-sig"
    )
    np.savetxt(dbg / f"corr_policy{policy}.csv", R, delimiter=",")

# =========================
# 1) 입력 로딩 (+ targets)
# =========================
def _normalize(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")

def _find_sheet(xls: pd.ExcelFile, target: str) -> Optional[str]:
    norm = _normalize(target)
    for name in xls.sheet_names:
        if _normalize(name) == norm:
            return name
    return None

def _coerce_num(x) -> Optional[float]:
    if pd.isna(x):
        return None
    s = str(x).strip().replace("%", "")
    if ("," in s) and ("." not in s):
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None

def _scan_header_row(raw: pd.DataFrame) -> Optional[Tuple[int, int, int]]:
    def norm_cell(v): return _normalize(v)
    max_scan = min(10, len(raw))
    for r in range(max_scan):
        row_vals = [norm_cell(v) for v in raw.iloc[r].tolist()]
        for yi, v in enumerate(row_vals):
            if v == "year":
                for fi, fv in enumerate(row_vals):
                    if fv in ("fr_target","frtarget","target_fr","fr","target"):
                        return r, yi, fi
    return None

def _load_targets_or_default(
    xls: pd.ExcelFile, start_year: int, horizon: int,
    fr_target_default: float, debug: bool = False
) -> np.ndarray:
    sheet = _find_sheet(xls, "targets")
    m: Dict[int, float] = {}
    if sheet is not None:
        try:
            df_try = pd.read_excel(xls, sheet_name=sheet)
            if debug:
                print(f"[DEBUG] targets sheet '{sheet}' columns:", list(df_try.columns))
            colmap = {_normalize(c): c for c in df_try.columns}
            year_key = next((k for k in colmap if k == "year"), None)
            fr_key   = next((k for k in colmap if k in ("fr_target","frtarget","target_fr","fr","target")), None)
            if year_key is None or fr_key is None:
                raw = pd.read_excel(xls, sheet_name=sheet, header=None)
                found = _scan_header_row(raw)
                if found is None:
                    raise ValueError(f"targets header not found (columns={list(df_try.columns)})")
                hdr, yi, fi = found
                if debug:
                    print(f"[DEBUG] detected header row at index {hdr}, year_col={yi}, fr_col={fi}")
                df = raw.iloc[hdr+1:, [yi, fi]].copy(); df.columns = ["year", "fr_target"]
            else:
                df = df_try[[colmap[year_key], colmap[fr_key]]].copy(); df.columns = ["year", "fr_target"]
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
            df["fr_target"] = df["fr_target"].map(_coerce_num)
            df = df.dropna(subset=["year"]).sort_values("year")
            m = {int(r.year): (float(r.fr_target) if r.fr_target is not None else fr_target_default)
                 for r in df.itertuples()}
        except Exception as e:
            if debug: print(f"[DEBUG] targets loading failed: {e}")
            m = {}
    else:
        if debug: print("[DEBUG] 'targets' sheet not found. Using default.]")

    out, last_val = [], fr_target_default
    for k in range(horizon):
        y = start_year + k
        val = m.get(y, last_val if k > 0 else fr_target_default)
        if (val is None) or (isinstance(val, float) and np.isnan(val)):
            val = last_val
        out.append(val); last_val = val
    v = np.array(out, dtype=float)
    if debug:
        print(f"[DEBUG] fr_targets: first 5 = {np.round(v[:5], 4)}, min={v.min():.4f}, max={v.max():.4f}")
    return v

def _pick_liability_scenario(liab_raw: pd.DataFrame,
                             liab_scenario: Optional[Union[str,int]],
                             debug: bool=False) -> pd.DataFrame:
    """liability 시트에서 scenario/sc 컬럼이 있으면 원하는 시나리오만 선택."""
    cols = {c.lower(): c for c in liab_raw.columns}
    scen_col = None
    for key in ("scenario", "sc"):
        if key in cols:
            scen_col = cols[key]
            break
    if scen_col is None:
        if debug: print("[DEBUG] liability: scenario column not found → single scenario assumed")
        return liab_raw.copy()

    df = liab_raw.copy()
    df["_scenario_norm"] = df[scen_col].astype(str).str.strip().str.lower()
    uniq = df["_scenario_norm"].unique().tolist()

    def _norm(x): return str(x).strip().lower()

    if liab_scenario is None:
        pick = "base" if "base" in uniq else ("p50" if "p50" in uniq else uniq[0])
    else:
        pick = _norm(liab_scenario)

    sel = df.loc[df["_scenario_norm"] == pick].drop(columns=["_scenario_norm"])
    if sel.empty:
        if debug:
            print(f"[DEBUG] liability scenario '{liab_scenario}' not found. Fallback to '{uniq[0]}'. (available={uniq})")
        sel = df.loc[df["_scenario_norm"] == uniq[0]].drop(columns=["_scenario_norm"])
    else:
        if debug:
            print(f"[DEBUG] liability scenario selected: '{pick}' (available={uniq})")
    return sel

def _dedup_by_year(liab_df: pd.DataFrame, debug: bool=False) -> pd.DataFrame:
    """연도 중복 시 첫 행 기준으로 정리."""
    if liab_df.duplicated(subset=["liability"]).any():
        if debug:
            dups = liab_df.loc[liab_df.duplicated(subset=["liability"], keep=False), "liability"].unique()
            print(f"[DEBUG] liability: duplicated years detected → taking first per year: {sorted(map(int, dups))}")
        liab_df = liab_df.drop_duplicates(subset=["liability"], keep="first")
    return liab_df

def load_inputs(xlsx_path: str,
                horizon_years: int = 10,
                n_paths: Optional[int] = None,
                seed: int = 2025,
                fr_target_default: float = 1.0,
                debug: bool = False,
                liab_scenario: Optional[Union[str,int]] = None):
    xls = pd.ExcelFile(xlsx_path)
    if debug:
        print("[DEBUG] sheets in workbook:", xls.sheet_names)

    # portfolio
    pf = pd.read_excel(xls, "portfolio"); pf = _strip_cols(pf)
    _ensure_cols(pf, ["policy", "asset_c2", "u", "v", "w"], "portfolio")
    pf["u"] = pf["u"].astype(float) / 100.0
    pf["v"] = pf["v"].astype(float) / 100.0
    chk = pf.groupby("policy")["w"].sum().round(9)
    if not np.allclose(chk.values, 1.0, atol=1e-6):
        raise ValueError(f"[portfolio] 정책별 비중합이 1이 아닙니다:\n{chk}")

    # corr
    try:
        corr = pd.read_excel(xls, "corr"); corr = _strip_cols(corr)
        _ensure_cols(corr, ["a", "b", "rho"], "corr")
        corr["a"] = corr["a"].astype(str).str.strip()
        corr["b"] = corr["b"].astype(str).str.strip()
        corr["rho"] = corr["rho"].astype(float)
    except Exception:
        corr = pd.DataFrame(columns=["a", "b", "rho"])

    # liability (시나리오 선택 + 중복 정리)
    liab_raw = pd.read_excel(xls, "liability"); liab_raw = _strip_cols(liab_raw)
    _ensure_cols(liab_raw, ["liability","opening_PBO","service_cost","interest_cost","benefit_cf","closing_PBO"], "liability")
    liab_sel = _pick_liability_scenario(liab_raw, liab_scenario, debug=debug)
    liab_sel["liability"] = pd.to_numeric(liab_sel["liability"], errors="coerce").astype("Int64")
    liab_sel = liab_sel.dropna(subset=["liability"]).astype({"liability":"int"})
    liab = _dedup_by_year(liab_sel, debug=debug)

    # asset (히스토리)
    asset = pd.read_excel(xls, "asset"); asset = _strip_cols(asset)
    _ensure_cols(asset, ["asset","opening_F","employer_contrib","actual_return","benefits_paid","closing_F"], "asset")

    # interest (미사용)
    try:
        rates = pd.read_excel(xls, "interest")
    except Exception:
        rates = None

    # 기준연도/초기치
    hist_y = _last_hist_year(asset, liab)
    A0 = float(asset.loc[asset["asset"]==hist_y, "closing_F"].iloc[0])
    L0 = float(liab.loc[liab["liability"]==hist_y, "closing_PBO"].iloc[0])

    # 투영 구간
    years = list(range(hist_y+1, hist_y+1+horizon_years))
    liab_idx = liab.set_index("liability")
    liab_take = liab_idx.loc[years, ["service_cost","interest_cost","benefit_cf","closing_PBO"]].astype(float)
    liab_proj = liab_take.copy()

    # 롤포워드 체크(정보성) – 가능 시에만
    try:
        prev = liab_idx.loc[[hist_y]+years[:-1], "closing_PBO"].values
        liab_proj["check_roll"] = (
            liab_idx.loc[years, "closing_PBO"].values
            - (prev + liab_proj["service_cost"].values
               + liab_proj["interest_cost"].values
               - liab_proj["benefit_cf"].values)
        )
    except Exception as e:
        if debug:
            print(f"[DEBUG] skip check_roll (non-unique index or shape mismatch): {e}")

    # 경로 수
    if n_paths is None:
        n_paths = 100
        if rates is not None and "path" in getattr(rates, "columns", []):
            n_paths = int(rates["path"].nunique())
    rng = np.random.default_rng(seed)

    # 연도별 FR 타깃
    fr_targets = _load_targets_or_default(
        xls, start_year=years[0], horizon=len(years),
        fr_target_default=fr_target_default, debug=debug
    )

    return dict(portfolio=pf, corr=corr, liab_proj=liab_proj,
                A0=A0, L0=L0, years=years, n_paths=n_paths, rng=rng,
                fr_targets=fr_targets)

# =========================
# 2) 상관/공분산 구성
# =========================
def build_cov(policy_assets: List[str], u_map: Dict[str,float], v_map: Dict[str,float],
              corr_long: pd.DataFrame, *, strict_corr: bool = True):
    """
    corr_long(a,b,rho) → 정책 자산 순서에 맞는 정방행렬 R 구성.
    strict_corr=True면 자산쌍(a,b) 중 누락된 pair 발견 시 에러.
    """
    assets = list(policy_assets)
    k = len(assets)
    R = np.eye(k)
    seen = np.eye(k, dtype=bool)
    pair_count = 0

    if corr_long is not None and len(corr_long):
        idx = {a: i for i, a in enumerate(assets)}
        for _, row in corr_long.iterrows():
            a, b, rho = str(row["a"]).strip(), str(row["b"]).strip(), float(row["rho"])
            if a in idx and b in idx and a != b:
                i, j = idx[a], idx[b]
                R[i, j] = R[j, i] = rho
                seen[i, j] = seen[j, i] = True
                pair_count += 1

    # 누락 쌍 검출
    if strict_corr:
        missing = []
        for i in range(k):
            for j in range(i+1, k):
                if not seen[i, j]:
                    missing.append(f"{assets[i]}-{assets[j]}")
        if missing:
            sample = ", ".join(missing[:12])
            raise ValueError(f"[corr] 누락된 자산쌍 {len(missing)}개 (예: {sample}). corr 시트를 보완하세요.")

    # 안전 보정
    R = np.clip(R, -0.999, 0.999); np.fill_diagonal(R, 1.0)
    R = _shrink_to_psd(R)

    sig = np.array([v_map[a] for a in assets], dtype=float)
    S = np.outer(sig, sig) * R
    mu = np.array([u_map[a] for a in assets], dtype=float)
    return assets, mu, sig, R, S

# =========================
# 3) 경로 수익률 생성
# =========================
def simulate_portfolio_paths(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                             n_years: int, n_paths: int, rng: np.random.Generator):
    X = rng.multivariate_normal(mean=mu, cov=cov, size=(n_years, n_paths))
    Rp = (X @ weights).astype(float)  # (T, P)
    return Rp

# =========================
# 4) 스케줄 모드: C_t^* 산출 + 상세 트래킹
# =========================
def min_contrib_schedule_paths(Rp: np.ndarray, A0: float, L0: float,
                               liab_proj: pd.DataFrame, fr_targets: np.ndarray,
                               pay_timing: str = "MID",
                               liab_mode: str = "target") -> Dict[str, np.ndarray]:
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

    return {"contribs": contribs, "FRs": FRs, "F_open": F_open, "F_close": F_close,
            "ret_amt": ret_amt, "ret_rate": ret_rate, "L_close": L_close}

# ----- (옵션) 고정 C 모드 유지 -----
def cstar_for_path(Rp_path: np.ndarray, A0: float, L0: float, liab_proj: pd.DataFrame,
                   fr_targets: np.ndarray, pay_timing: str = "MID"):
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
        mid = 0.5*(lo+hi)
        if pred(mid): hi = mid
        else: lo = mid
        if abs(hi-lo) <= tol*(1.0+hi): break
    return hi

# =========================
# 5) 메인 엔진
# =========================
def run_engine(xlsx_path: str,
               horizon_years: int = 10,
               success_q: float = 0.95,
               seed: int = 2025,
               fr_target_default: float = 1.0,
               debug: bool = False,
               csvdir: Optional[str] = None,
               mode: str = "schedule",
               liab_mode: str = "target",
               liab_scenario: Optional[Union[str,int]] = None,
               strict_corr: bool = True):
    data = load_inputs(
        xlsx_path, horizon_years=horizon_years, seed=seed,
        fr_target_default=fr_target_default, debug=debug,
        liab_scenario=liab_scenario
    )
    pf, corr_long, liab_proj = data["portfolio"], data["corr"], data["liab_proj"]
    A0, L0, years, n_paths, rng = data["A0"], data["L0"], data["years"], data["n_paths"], data["rng"]
    fr_targets = data["fr_targets"]

    summary = []
    fr_bands_all = []
    agg_rows = []; path_rows = []
    agg_contrib_rows = []; contrib_path_rows = []; asset_base_mean_rows = []

    for pol, g in pf.groupby("policy"):
        assets = g["asset_c2"].tolist()
        weights = g["w"].to_numpy(dtype=float)
        u_map = dict(zip(assets, g["u"].astype(float)))
        v_map = dict(zip(assets, g["v"].astype(float)))
        assets, mu, sig, R, S = build_cov(assets, u_map, v_map, corr_long, strict_corr=strict_corr)

        # ===== NEW: 시뮬 직전 스냅샷 저장 =====
        _dump_snapshot(csvdir,
                       policy=int(pol), seed=seed, mode=mode,
                       liab_mode=liab_mode, liab_scenario=liab_scenario,
                       assets=assets, w=weights, u=mu, v=sig, R=R,
                       A0=A0, L0=L0, fr_targets_head=float(fr_targets[0]), strict_corr=strict_corr)

        Rp = simulate_portfolio_paths(weights=weights, mu=mu, cov=S,
                                      n_years=len(years), n_paths=n_paths, rng=rng)

        if mode.lower() == "schedule":
            detail = min_contrib_schedule_paths(
                Rp, A0, L0, liab_proj, fr_targets, "MID", liab_mode=liab_mode
            )
            C_sched, FR_sched = detail["contribs"], detail["FRs"]
            F_open, F_close = detail["F_open"], detail["F_close"]
            ret_amt, ret_rate = detail["ret_amt"], detail["ret_rate"]
            L_used = detail["L_close"]
            T, P = C_sched.shape
            B_vec = liab_proj["benefit_cf"].values[:T]
            L_input = liab_proj["closing_PBO"].values[:T]

            # 연도별 집계 (mean, p5, p25, p50, p75, p95 + paired)
            rows = []
            for t in range(T):
                C_row, FR_row, R_row = C_sched[t, :], FR_sched[t, :], ret_rate[t, :]
                Aop, Acl = F_open[t, :], F_close[t, :]
                base = {
                    "policy": pol,
                    "year":   years[t],
                    **_dist_stats("contrib", C_row),
                    **_dist_stats("FR", FR_row),
                    **_dist_stats("return_rate", R_row),
                    **_dist_stats("F_open", Aop),
                    **_dist_stats("F_close", Acl),
                    "share_zero_contrib": float((C_row <= 1e-12).mean()),
                    "fr_target": float(fr_targets[t]),
                    "closing_PBO_input": float(L_input[t]),
                    "closing_PBO_used":  float(L_used[t]),
                }
                base.update(_paired_quantiles(C_row, FR_row,
                                              qs=(0.05, 0.25, 0.50, 0.75, 0.95)))
                rows.append(base)
            df_stats = pd.DataFrame(rows)
            agg_contrib_rows.append(df_stats)

            # 자산 base&mean 궤적 (base는 p50로 정의)
            asset_base_mean_rows.append(pd.DataFrame({
                "policy": pol, "year": years,
                "F_open_mean": df_stats["F_open_mean"].values,
                "F_open_base": df_stats["F_open_p50"].values,
                "F_close_mean": df_stats["F_close_mean"].values,
                "F_close_base": df_stats["F_close_p50"].values,
            }))

            # 전체 path × 연도 롱포맷
            df_long = pd.DataFrame({
                "policy":       pol,
                "year":         np.repeat(years, P),
                "path":         np.tile(np.arange(P, dtype=int), T),
                "closing_PBO_input": np.repeat(L_input, P),
                "closing_PBO_used":  np.repeat(L_used, P),
                "opening_F":    F_open.reshape(-1),
                "contribution": C_sched.reshape(-1),
                "actual_return":ret_amt.reshape(-1),
                "return_rate":  ret_rate.reshape(-1),
                "benefits_paid":np.repeat(B_vec, P),
                "closing_F":    F_close.reshape(-1),
                "fr":           FR_sched.reshape(-1),
                "fr_target":    np.repeat(fr_targets[:T], P),
            })
            contrib_path_rows.append(df_long)

            # 총 기여금(경로 합) 통계
            total_by_path = C_sched.sum(axis=0)
            summary.append({
                "policy": pol,
                **_dist_stats("total_contrib", total_by_path),
            })

            # FR 밴드 (표시용)
            fr_bands_all.append(pd.DataFrame({
                "policy": pol, "year": years,
                "FR_p5":  np.percentile(FR_sched, 5, axis=1),
                "FR_p25": np.percentile(FR_sched, 25, axis=1),
                "FR_p50": np.percentile(FR_sched, 50, axis=1),
                "FR_p75": np.percentile(FR_sched, 75, axis=1),
                "FR_p95": np.percentile(FR_sched, 95, axis=1),
                "fr_target": fr_targets.astype(float),
            }))

        else:  # fixed
            Cstars = np.array([
                cstar_for_path(Rp[:, p], A0, L0, liab_proj, fr_targets) for p in range(n_paths)
            ], dtype=float)
            Cstars = Cstars[np.isfinite(Cstars)]
            meanC = float(np.nanmean(Cstars)) if len(Cstars) else np.nan
            p5    = float(np.nanpercentile(Cstars, 5))  if len(Cstars) else np.nan
            p25   = float(np.nanpercentile(Cstars, 25)) if len(Cstars) else np.nan
            p50   = float(np.nanpercentile(Cstars, 50)) if len(Cstars) else np.nan
            p75   = float(np.nanpercentile(Cstars, 75)) if len(Cstars) else np.nan
            p95   = float(np.nanpercentile(Cstars, 95)) if len(Cstars) else np.nan

            def _fr_paths_at_C(C: float) -> np.ndarray:
                T, P = Rp.shape
                SC = liab_proj["service_cost"].values[:T]
                IC = liab_proj["interest_cost"].values[:T]
                B  = liab_proj["benefit_cf"].values[:T]
                FRs = np.empty((T, P), dtype=float)
                for p in range(P):
                    A, L = A0, L0
                    for t in range(T):
                        L = L + SC[t] + IC[t] - B[t]
                        A = (A + C - B[t]) * (1.0 + Rp[t, p])
                        FRs[t, p] = A / max(L, 1e-9)
                return FRs

            for label, Cval in {"mean": meanC, "p5": p5, "p25": p25, "p50": p50, "p75": p75, "p95": p95}.items():
                if not (np.isfinite(Cval) and Cval >= 0):
                    continue
                FRs = _fr_paths_at_C(Cval)
                agg_rows.append(pd.DataFrame({
                    "policy": pol, "C_label": label, "contribution": Cval,
                    "year": years,
                    "FR_mean": FRs.mean(axis=1),
                    "FR_p5":  np.percentile(FRs, 5, axis=1),
                    "FR_p25": np.percentile(FRs, 25, axis=1),
                    "FR_p50": np.percentile(FRs, 50, axis=1),
                    "FR_p75": np.percentile(FRs, 75, axis=1),
                    "FR_p95": np.percentile(FRs, 95, axis=1),
                    "fr_target": fr_targets.astype(float),
                }))
                T, P = FRs.shape
                df_long = pd.DataFrame({
                    "year": np.repeat(years, P),
                    "path": np.tile(np.arange(P, dtype=int), T),
                    "FR":   FRs.reshape(-1),
                })
                df_long.insert(0, "policy", pol)
                df_long.insert(1, "C_label", label)
                df_long["contribution"] = Cval
                path_rows.append(df_long)

            summary.append({
                "policy": pol,
                "mean_C": meanC, "p5_C": p5, "p25_C": p25, "p50_C": p50, "p75_C": p75, "p95_C": p95
            })

            fr_bands_all.append(pd.DataFrame({
                "policy": pol, "year": years,
                "FR_p5":  np.nan, "FR_p50": np.nan, "FR_p95": np.nan,
                "fr_target": fr_targets.astype(float),
            }))

    # DF 정리
    summary_df = pd.DataFrame(summary)
    fr_bands = pd.concat(fr_bands_all, ignore_index=True) if fr_bands_all else pd.DataFrame()

    # CSV 저장
    if csvdir:
        os.makedirs(csvdir, exist_ok=True)
        if mode.lower() == "schedule":
            policy_year_contrib_stats = pd.concat(agg_contrib_rows, ignore_index=True) if agg_contrib_rows else pd.DataFrame()
            policy_contrib_paths = pd.concat(contrib_path_rows, ignore_index=True) if contrib_path_rows else pd.DataFrame()
            policy_asset_base_mean = pd.concat(asset_base_mean_rows, ignore_index=True) if asset_base_mean_rows else pd.DataFrame()

            summary_df.to_csv(os.path.join(csvdir, "summary_schedule_total_contrib.csv"), index=False, encoding="utf-8-sig")
            policy_year_contrib_stats.to_csv(os.path.join(csvdir, "policy_year_contrib_stats.csv"), index=False, encoding="utf-8-sig")
            policy_contrib_paths.to_csv(os.path.join(csvdir, "policy_contrib_paths.csv"), index=False, encoding="utf-8-sig")
            policy_asset_base_mean.to_csv(os.path.join(csvdir, "policy_asset_base_mean.csv"), index=False, encoding="utf-8-sig")
            fr_bands.to_csv(os.path.join(csvdir, "fr_bands_schedule.csv"), index=False, encoding="utf-8-sig")

            print(f"\n[WRITE/CSV] '{csvdir}'에 저장:")
            print(" - summary_schedule_total_contrib.csv")
            print(" - policy_year_contrib_stats.csv  (mean/p5/p25/p50/p75/p95 + paired)")
            print(" - policy_contrib_paths.csv       (전체 path×연도 상세)")
            print(" - policy_asset_base_mean.csv     (자산 mean & base=p50 궤적)")
            print(" - fr_bands_schedule.csv")
        else:
            policy_year_stats = pd.concat(agg_rows, ignore_index=True) if agg_rows else pd.DataFrame()
            policy_fr_paths = pd.concat(path_rows, ignore_index=True) if path_rows else pd.DataFrame()
            summary_df.to_csv(os.path.join(csvdir, "summary_fixed.csv"), index=False, encoding="utf-8-sig")
            policy_year_stats.to_csv(os.path.join(csvdir, "policy_year_stats.csv"), index=False, encoding="utf-8-sig")
            policy_fr_paths.to_csv(os.path.join(csvdir, "policy_fr_paths.csv"), index=False, encoding="utf-8-sig")
            fr_bands.to_csv(os.path.join(csvdir, "fr_bands_fixed.csv"), index=False, encoding="utf-8-sig")
            print(f"\n[WRITE/CSV] '{csvdir}'에 저장 (fixed 모드):")
            print(" - summary_fixed.csv / policy_year_stats.csv / policy_fr_paths.csv / fr_bands_fixed.csv")

    return summary_df, fr_bands

# =========================
# 6) CLI
# =========================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="엑셀 파일 경로")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--q", type=float, default=0.95, help="(fixed 모드용) 성공확률 기준 — 단일 float만! (여러 개는 PowerShell 루프)")
    ap.add_argument("--fr_target_default", type=float, default=1.00, help="targets 시트 누락/결측 시 기본 FR 타깃")
    ap.add_argument("--debug", action="store_true", help="로딩/타깃 진단 메시지 출력")
    ap.add_argument("--csvdir", type=str, default="", help="CSV 저장 폴더 경로 (예: .\\out_csv)")
    ap.add_argument("--mode", type=str, default="schedule", choices=["schedule","fixed"], help="기여금 산출 모드")
    ap.add_argument("--liab_mode", type=str, default="target", choices=["target","roll"],
                    help="부채 처리 방식: target=엑셀 closing_PBO 고정, roll=SC/IC/B 롤포워드")
    ap.add_argument("--liab_scenario", type=str, default=None,
                    help="liability 시트의 scenario/sc 값 (예: base, p50, p5, 1, 2 등). 미지정 시 base→p50→첫 값 순.")
    ap.add_argument("--strict_corr", action="store_true", help="corr 누락쌍 발견 시 에러(권장). 기본 True.")
    ap.add_argument("--no_strict_corr", action="store_true", help="누락쌍을 0으로 두고 PSD 보정(디버그용).")
    args = ap.parse_args()

    strict_corr = True
    if args.no_strict_corr:
        strict_corr = False
    elif args.strict_corr:
        strict_corr = True

    summary, fr_bands = run_engine(
        args.excel,
        horizon_years=args.horizon,
        success_q=args.q,
        seed=args.seed,
        fr_target_default=args.fr_target_default,
        debug=args.debug,
        csvdir=(args.csvdir or None),
        mode=args.mode,
        liab_mode=args.liab_mode,
        liab_scenario=args.liab_scenario,
        strict_corr=strict_corr
    )

    if args.mode == "schedule":
        print("\n[SUMMARY - schedule] Total required contribution by policy (sum of C_t^*)")
        print(summary.to_string(index=False))
        print("\n[FR BANDS (schedule)] (top 5 rows by policy)")
        if not fr_bands.empty:
            print(fr_bands.groupby("policy").head(5).to_string(index=False))
        else:
            print("(empty)")
    else:
        print("\n[SUMMARY - fixed] Required constant C by policy (mean/p5/p25/p50/p75/p95)")
        print(summary.to_string(index=False))
        print("\n[FR BANDS (fixed)]")
        if not fr_bands.empty:
            print(fr_bands.groupby("policy").head(5).to_string(index=False))
        else:
            print("(empty)")
