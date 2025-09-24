# runner/cli.py
from __future__ import annotations
import argparse, os, re
import numpy as np
import pandas as pd

from optimize.resampled_frontier import build_resampled_frontier
from runner.adapter_ref_to_portfolio import ref_to_portfolio_excel

def _parse_float_expr(x: str | float | int) -> float:
    s = str(x).strip().replace("^", "**")
    if not re.fullmatch(r"[0-9eE\.\+\-\*\/\(\)\s]+", s):
        raise ValueError(f"Invalid numeric expression: {x!r}")
    return float(eval(s, {"__builtins__": None}, {}))

def _load_mu_cov(mu_csv: str, cov_csv: str):
    mu_df = pd.read_csv(mu_csv)
    assets = [str(a) for a in mu_df["asset"]]
    mu_map = dict(zip(assets, mu_df["mu"]))
    mu = np.array([float(mu_map[a]) for a in assets], dtype=float)

    cov_df = pd.read_csv(cov_csv)
    uniq = sorted(list(set(cov_df["asset_i"]).union(set(cov_df["asset_j"]))))
    # 자산 우주를 μ 파일 기준으로 우선 정렬, μ에 없는 자산은 뒤에 붙임
    ordered = [a for a in assets if a in uniq] + [a for a in uniq if a not in assets]
    idx = {a: i for i, a in enumerate(ordered)}
    K = len(ordered)
    Sigma = np.zeros((K, K), dtype=float)
    for _, row in cov_df.iterrows():
        i, j = idx[str(row["asset_i"])], idx[str(row["asset_j"])]
        Sigma[i, j] = _parse_float_expr(row["cov"])
    Sigma = np.maximum(Sigma, Sigma.T)
    # μ 재정렬
    if ordered != assets:
        mu = np.array([float(mu_map.get(a, 0.0)) for a in ordered], dtype=float)
    return ordered, mu, Sigma

def parse_targets(s: str):
    # "vol:0.06,0.08,0.10" or "ret:0.04,0.05"
    ttype, vals = s.split(":")[0], s.split(":")[1]
    targets = [float(x) for x in vals.split(",") if x.strip()!=""]
    return ttype, targets

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ref","alm","both"], required=True)
    p.add_argument("--mu_csv", required=False)
    p.add_argument("--cov_csv", required=False)
    p.add_argument("--targets", required=False)         # e.g., "vol:0.06,0.08,0.10"
    p.add_argument("--K", type=int, default=200)
    p.add_argument("--n_obs", type=int, default=60)
    p.add_argument("--sampler", default="normal")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", required=True)
    p.add_argument("--policy_yaml", required=False)     # (Phase2) 밴드 규칙
    p.add_argument("--ref_csv", required=False)         # alm 모드에서 사용
    p.add_argument("--profile", required=False)         # (Phase2)
    p.add_argument("--tag", default="baseline")
    p.add_argument("--base_io_xlsx", required=False)    # ✅ 기존 IO 엑셀(모든 시트) 경로

    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ---------- REF ----------
    if args.mode in ("ref", "both"):
        if not (args.mu_csv and args.cov_csv and args.targets):
            raise SystemExit("--mode ref requires --mu_csv --cov_csv --targets")
        assets, mu, Sigma = _load_mu_cov(args.mu_csv, args.cov_csv)
        ttype, targets = parse_targets(args.targets)

        # 예시: 위험자산 캡(70%) - 필요시 조정
        risky_assets = [a for a in assets if any(k in a.lower() for k in ["eq","gold","hy","em"])]
        ref_frontier, weights_ci = build_resampled_frontier(
            assets=assets, mu=mu, Sigma=Sigma,
            targets=targets, target_type=ttype,
            K=args.K, n_obs=args.n_obs, sampler=args.sampler, seed=args.seed,
            lb=None, ub=None, risky_cap=(risky_assets, 0.70),
        )

        ref_dir = os.path.join(args.outdir, f"ref_{args.tag}")
        os.makedirs(ref_dir, exist_ok=True)
        ref_csv = os.path.join(ref_dir, "ref_frontier.csv")
        ci_csv  = os.path.join(ref_dir, "weights_ci.csv")
        ref_frontier.to_csv(ref_csv, index=False, encoding="utf-8-sig")
        weights_ci.to_csv(ci_csv, index=False, encoding="utf-8-sig")
        print("REF saved:", ref_csv, ci_csv)

        # ✅ 어댑터: base IO가 주어지면 FULL 엑셀 생성
        out_xlsx = os.path.join(ref_dir, "sample_ALM_IO_REF.xlsx")
        ref_to_portfolio_excel(
            ref_frontier_csv=ref_csv,
            mu_csv=args.mu_csv,
            cov_csv=args.cov_csv,
            out_xlsx=out_xlsx,
            policy_start=101,
            base_io_xlsx=args.base_io_xlsx,  # <-- 여기!
        )
        print("ALM input xlsx saved:", out_xlsx)

        # alm-only 실행에서 재사용하도록
        args.ref_csv = ref_csv

    # ---------- ALM ----------
    if args.mode in ("alm", "both"):
        if not args.ref_csv:
            raise SystemExit("--mode alm requires --ref_csv (or run --mode ref first)")
        # REF 디렉토리 내 FULL 엑셀 경로(위에서 생성된 파일)
        ref_dir = os.path.dirname(args.ref_csv)
        alm_xlsx = os.path.join(ref_dir, "sample_ALM_IO_REF.xlsx")

        # 만약 아직 FULL이 없다면, base_io_xlsx 없이는 실행할 수 없으므로 안내
        if not os.path.exists(alm_xlsx):
            if not (args.mu_csv and args.cov_csv):
                raise SystemExit("Missing FULL xlsx. Re-run --mode ref with --base_io_xlsx to build FULL input.")
            # 마지막 시도: base_io_xlsx가 주어졌다면 지금 생성
            if args.base_io_xlsx:
                ref_to_portfolio_excel(
                    ref_frontier_csv=args.ref_csv,
                    mu_csv=args.mu_csv,
                    cov_csv=args.cov_csv,
                    out_xlsx=alm_xlsx,
                    policy_start=101,
                    base_io_xlsx=args.base_io_xlsx,
                )
                print("ALM input xlsx saved:", alm_xlsx)
            else:
                raise SystemExit("No FULL xlsx found. Provide --base_io_xlsx in REF step.")

        from alm.engine import run_engine
        alm_dir = os.path.join(args.outdir, f"alm_{args.tag}")
        os.makedirs(alm_dir, exist_ok=True)

        summary, bands = run_engine(
            xlsx_path=alm_xlsx,
            horizon_years=30,
            mode="schedule",
            liab_mode="target",
            csvdir=alm_dir,
            common_shock=True,
            strict_corr=True,
            rate_linked=False,
        )
        print("ALM saved to:", alm_dir)

if __name__ == "__main__":
    main()
