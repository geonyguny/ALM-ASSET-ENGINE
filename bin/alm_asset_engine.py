# alm_asset_engine.py
from __future__ import annotations
import argparse
from engine import run_engine

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--q", type=float, default=0.95, help="(fixed 모드용) 성공확률 기준 — 현재 로직엔 미연결")
    ap.add_argument("--fr_target_default", type=float, default=1.00)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--csvdir", type=str, default="")
    ap.add_argument("--mode", type=str, default="schedule", choices=["schedule","fixed"])
    ap.add_argument("--liab_mode", type=str, default="target", choices=["target","roll"])
    ap.add_argument("--liab_scenario", type=str, default=None)
    ap.add_argument("--strict_corr", action="store_true", help="누락쌍 발견 시 에러(권장)")
    ap.add_argument("--no_strict_corr", action="store_true", help="누락쌍을 0으로 두고 진행(디버그)")
    ap.add_argument("--no_common_shock", action="store_true", help="공통충격 사용하지 않음(기존 방식)")
    args = ap.parse_args()

    strict_corr = not args.no_strict_corr if args.strict_corr or args.no_strict_corr else True
    common_shock = not args.no_common_shock

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
        strict_corr=strict_corr,
        common_shock=common_shock,
    )

    if args.mode == "schedule":
        print("\n[SUMMARY - schedule] Total required contribution by policy (sum of C_t^*)")
        print(summary.to_string(index=False))
    else:
        print("\n[SUMMARY - fixed] Required constant C by policy (mean/p5/p25/p50/p75/p95)")
        print(summary.to_string(index=False))
