# -*- coding: utf-8 -*-
"""
run_opt_from_excel.py
엑셀 → 최적 가중치 산출 → CSV 저장

입력 엑셀 스키마
- portfolio 시트: [policy, period, asset_c1, asset_c2, u, v, w]
  * u, v가 % 표기(예: 7.9)면 자동 감지해 100으로 나눔
- corr 시트: [a, b, rho] (a,b는 asset_c2 라벨)

제약
- 롱온리(0~1), 합=1 (efficient_qp에 lb/ub 전달)

출력
- <outdir>/recommended_weights.csv          (모든 policy/period 병합)
- <outdir>/recommended_summary.csv          (policy/period별 성과 요약)
- <outdir>/weights_policy{p}_period{t}.csv  (개별 파일, 선택)

개선점
- 상관행렬 엄격검증(strict_corr=True): 누락 pair 있으면 에러로 즉시 원인 파악
- MSR 견고화: 직접 샤프최대 실패 시, 목표수익 그리드 스윕으로 대체(solve_msr_robust)
- u,v 퍼센트 자동감지(스위치 없이도 안전)
- 여러 policy/period를 한 번에 실행(--policy all / --period all)
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from portfolio_opt import efficient_qp, cov_from_corr, portfolio_stats


# ============ 유틸 ============

def _ensure_cols(df: pd.DataFrame, need: set, name: str):
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] 필요한 컬럼이 없습니다: {missing} / 현재: {list(df.columns)}")

def _auto_detect_percent(u: np.ndarray, v: np.ndarray) -> bool:
    """
    u, v가 %단위(예: 7.9, 12.3)로 들어왔는지 자동감지.
    - 보수적으로: max(|u|,|v|) > 1이면 %로 간주.
    """
    mx = float(np.nanmax(np.abs(np.concatenate([u, v]))))
    return mx > 1.0

def _scale_uv_if_percent(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    is_percent = _auto_detect_percent(u, v)
    if is_percent:
        return u / 100.0, v / 100.0, True
    return u, v, False

def _strict_corr_from_pairs(assets: list[str], corr_df: pd.DataFrame, strict: bool = True) -> np.ndarray:
    """
    corr 시트(a,b,rho) → 정책 자산순서 기반 정방 상관행렬.
    strict=True면 누락 쌍 있으면 에러.
    """
    assets = list(map(str, assets))
    idx = {a: i for i, a in enumerate(assets)}
    n = len(assets)
    R = np.eye(n, dtype=float)
    seen = np.eye(n, dtype=bool)

    df = corr_df.dropna(subset=["a", "b", "rho"]).copy()
    df["a"] = df["a"].astype(str)
    df["b"] = df["b"].astype(str)

    for _, r in df.iterrows():
        a, b, rho = r["a"], r["b"], float(r["rho"])
        if a in idx and b in idx and a != b:
            i, j = idx[a], idx[b]
            R[i, j] = R[j, i] = rho
            seen[i, j] = seen[j, i] = True

    if strict:
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not seen[i, j]:
                    missing.append((assets[i], assets[j]))
        if missing:
            sample = ", ".join(f"{x}-{y}" for x, y in missing[:12])
            raise ValueError(f"[corr] 누락된 자산쌍 {len(missing)}개 (예: {sample}). corr 시트를 보완하세요.")
    # 안전 보정
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return R


# ============ MSR 견고화(그리드 스윕) ============

def solve_msr_robust(mu, v, R, rf, lb, ub, w0, n_grid=61, eps=1e-6):
    """
    샤프 최대를 직접 최적화하지 않고,
    '목표수익 고정-분산최소'를 여러 타깃(mu_target)으로 풀어 샤프가 최대인 해 선택.
    """
    Sigma = cov_from_corr(v, R)
    mu_min, mu_max = float(np.min(mu)), float(np.max(mu))
    span = (mu_max - mu_min)
    lo = mu_min + eps * (span + 1.0)
    hi = mu_max - eps * (span + 1.0)
    if lo >= hi:
        # 자산 1개/동일수익률 등 극단 케이스 → GMV로 대체
        return efficient_qp(mu, v, R, mode="gmv", lb=lb, ub=ub, w0=w0)

    targets = np.linspace(lo, hi, n_grid)
    best = None
    for mt in targets:
        w = efficient_qp(mu, v, R, mode="target", mu_target=float(mt),
                         lb=lb, ub=ub, w0=w0)
        ret, vol, sh = portfolio_stats(w, mu, Sigma, r_f=rf)
        if best is None or sh > best[0]:
            best = (sh, w)
    return best[1]


# ============ 로딩/실행 ============

def load_excel(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dfp = pd.read_excel(xlsx_path, sheet_name="portfolio")
    dfc = pd.read_excel(xlsx_path, sheet_name="corr")
    _ensure_cols(dfp, {"policy", "period", "asset_c1", "asset_c2", "u", "v", "w"}, "portfolio")
    _ensure_cols(dfc, {"a", "b", "rho"}, "corr")
    return dfp, dfc

def select_policies_periods(dfp: pd.DataFrame, policy_arg: str, period_arg: str):
    """
    policy_arg/period_arg: 'all' 또는 정수 문자열
    반환: [(policy, period), ...] (엑셀 존재 조합만)
    """
    dfp2 = dfp.copy()
    dfp2["policy"] = dfp2["policy"].astype(int)
    dfp2["period"] = dfp2["period"].astype(int)

    if policy_arg.lower() != "all":
        p = int(policy_arg)
        dfp2 = dfp2.loc[dfp2["policy"] == p]
    if period_arg.lower() != "all":
        t = int(period_arg)
        dfp2 = dfp2.loc[dfp2["period"] == t]

    pairs = sorted(dfp2[["policy", "period"]].drop_duplicates().itertuples(index=False, name=None))
    if not pairs:
        raise ValueError(f"선택된 policy/period 조합이 없습니다. (policy={policy_arg}, period={period_arg})")
    return pairs

def run_one(dfp: pd.DataFrame, dfc: pd.DataFrame, policy: int, period: int,
            mode: str, mu_target: float | None, rf: float,
            lb: float, ub: float, strict_corr: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    단일 policy/period 실행 → (weights_df, summary_df)
    """
    sel = dfp.loc[(dfp["policy"] == policy) & (dfp["period"] == period)].copy()
    if sel.empty:
        raise ValueError(f"portfolio 시트에 policy={policy}, period={period} 자료가 없습니다.")

    sel["asset_c2"] = sel["asset_c2"].astype(str)
    assets = sel["asset_c2"].tolist()
    u = sel["u"].astype(float).values
    v = sel["v"].astype(float).values
    w0 = sel["w"].astype(float).values

    u, v, was_percent = _scale_uv_if_percent(u, v)
    R = _strict_corr_from_pairs(assets, dfc, strict=strict_corr)

    if mode == "msr":
        try:
            w_rec = efficient_qp(u, v, R, mode="msr", r_f=float(rf), lb=lb, ub=ub, w0=w0)
        except Exception:
            w_rec = solve_msr_robust(u, v, R, rf=float(rf), lb=lb, ub=ub, w0=w0, n_grid=61)
    elif mode == "gmv":
        w_rec = efficient_qp(u, v, R, mode="gmv", lb=lb, ub=ub, w0=w0)
    elif mode == "target":
        if mu_target is None:
            raise ValueError("MODE='target'이면 --mu_target(소수)을 지정하세요.")
        w_rec = efficient_qp(u, v, R, mode="target", mu_target=float(mu_target), lb=lb, ub=ub, w0=w0)
    else:
        raise ValueError("MODE must be one of {'msr','gmv','target'}")

    Sigma = cov_from_corr(v, R)
    ret0, vol0, sh0 = portfolio_stats(w0, u, Sigma, r_f=float(rf))
    ret1, vol1, sh1 = portfolio_stats(w_rec, u, Sigma, r_f=float(rf))

    weights_df = pd.DataFrame({
        "policy": policy,
        "period": period,
        "asset_c2": assets,
        "w0": w0,
        "w_rec": w_rec,
        "delta": w_rec - w0,
        "u": u,
        "v": v,
        "was_percent_input": was_percent
    })

    summary_df = pd.DataFrame([{
        "policy": policy,
        "period": period,
        "mode": mode,
        "mu_target": mu_target,
        "rf": rf,
        "ret_w0": ret0, "vol_w0": vol0, "sharpe_w0": sh0,
        "ret_rec": ret1, "vol_rec": vol1, "sharpe_rec": sh1
    }])

    return weights_df, summary_df


# ============ 메인/CLI ============

def parse_args():
    ap = argparse.ArgumentParser(description="엑셀→최적 w 산출(롱온리, 합=1) + 요약 저장")
    ap.add_argument("--excel", type=str, required=True, help="입력 엑셀 경로 (예: C:\\00_ALM\\project\\test ALM IO_20250910.xlsx)")
    ap.add_argument("--policy", type=str, default="1", help="'all' 또는 정수 (예: 1)")
    ap.add_argument("--period", type=str, default="1", help="'all' 또는 정수 (예: 1)")
    ap.add_argument("--mode", type=str, default="msr", choices=["msr","gmv","target"])
    ap.add_argument("--mu_target", type=float, default=None, help="MODE='target'일 때 목표수익(소수)")
    ap.add_argument("--rf", type=float, default=0.0, help="무위험수익률(소수)")
    ap.add_argument("--lb", type=float, default=0.0, help="하한(기본 0)")
    ap.add_argument("--ub", type=float, default=1.0, help="상한(기본 1)")
    ap.add_argument("--outdir", type=str, default="./outputs/opt", help="CSV 저장 폴더")
    ap.add_argument("--strict_corr", action="store_true", help="corr 누락쌍 발견 시 에러(권장)")
    ap.add_argument("--allow_missing_corr", action="store_true", help="누락쌍을 0으로 두고 진행(디버깅용)")
    ap.add_argument("--emit_individual_files", action="store_true", help="policy/period별 개별 weights CSV도 저장")
    return ap.parse_args()

def main():
    args = parse_args()
    xlsx_path = Path(args.excel)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    strict_corr = True
    if args.allow_missing_corr:
        strict_corr = False
    elif args.strict_corr:
        strict_corr = True

    dfp, dfc = load_excel(xlsx_path)
    pairs = select_policies_periods(dfp, args.policy, args.period)

    all_weights = []
    all_summaries = []

    for (pol, per) in pairs:
        wdf, sdf = run_one(
            dfp, dfc, policy=pol, period=per,
            mode=args.mode, mu_target=args.mu_target, rf=args.rf,
            lb=args.lb, ub=args.ub, strict_corr=strict_corr
        )
        all_weights.append(wdf)
        all_summaries.append(sdf)

        if args.emit_individual_files:
            wfile = outdir / f"weights_policy{pol}_period{per}.csv"
            wdf.to_csv(wfile, index=False, encoding="utf-8-sig")
            print(f"[WRITE] {wfile}")

    weights_all = pd.concat(all_weights, ignore_index=True) if all_weights else pd.DataFrame()
    summary_all = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()

    weights_path = outdir / "recommended_weights.csv"
    summary_path = outdir / "recommended_summary.csv"

    weights_all.to_csv(weights_path, index=False, encoding="utf-8-sig")
    summary_all.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n=== SUMMARY ===")
    if not summary_all.empty:
        print(summary_all.to_string(index=False))
    else:
        print("(empty)")
    print(f"\nSaved:\n - {weights_path}\n - {summary_path}")

if __name__ == "__main__":
    main()
