# engine.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List

from .alm_io import load_inputs
from .corr_utils import build_cov
from .sim_utils import (
    simulate_portfolio_paths,
    min_contrib_schedule_paths,
    cstar_for_path,
    generate_common_Z,
)
from .stats_utils import dist_stats, paired_quantiles
from .snapshot import dump_snapshot


def run_engine(
    xlsx_path: str,
    horizon_years: int = 10,
    success_q: float = 0.95,   # 현재 로직에는 미사용(향후 확률제약 연결용)
    seed: int = 2025,
    fr_target_default: float = 1.0,
    debug: bool = False,
    csvdir: Optional[str] = None,
    mode: str = "schedule",
    liab_mode: str = "target",
    liab_scenario: Optional[Union[str, int]] = None,
    strict_corr: bool = True,
    common_shock: bool = True,
):
    # 1) 입력 로드
    data = load_inputs(
        xlsx_path,
        horizon_years=horizon_years,
        seed=seed,
        fr_target_default=fr_target_default,
        debug=debug,
        liab_scenario=liab_scenario,
    )
    pf, corr_long, liab_proj = data["portfolio"], data["corr"], data["liab_proj"]
    A0, L0, years, n_paths, rng = (
        data["A0"],
        data["L0"],
        data["years"],
        data["n_paths"],
        data["rng"],
    )
    fr_targets = data["fr_targets"]

    # 2) 공통충격(옵션)
    if common_shock:
        global_assets = sorted(map(str, pf["asset_c2"].astype(str).unique()))
        Z_global = generate_common_Z(global_assets, T=len(years), P=n_paths, seed=seed)

    summary = []
    fr_bands_all = []
    agg_rows = []
    path_rows = []
    agg_contrib_rows = []
    contrib_path_rows = []
    asset_base_mean_rows = []

    # 3) 정책 루프
    for pol, g in pf.groupby("policy"):
        assets = g["asset_c2"].astype(str).tolist()
        weights = g["w"].to_numpy(dtype=float)
        u_map = dict(zip(assets, g["u"].astype(float)))
        v_map = dict(zip(assets, g["v"].astype(float)))

        # 상관/공분산 구성(엄격 검사 옵션 포함)
        assets, mu, sig, R, S = build_cov(
            assets, u_map, v_map, corr_long, strict_corr=strict_corr
        )

        # 스냅샷
        dump_snapshot(
            csvdir,
            policy=int(pol),
            seed=seed,
            mode=mode,
            liab_mode=liab_mode,
            liab_scenario=liab_scenario,
            assets=assets,
            w=weights,
            u=mu,
            v=sig,
            R=R,
            A0=A0,
            L0=L0,
            fr_targets_head=float(fr_targets[0]),
            strict_corr=strict_corr,
        )

        # 4) 경로 수익률 생성 (공통충격 여부)
        if common_shock:
            Rp = simulate_portfolio_paths(
                weights=weights,
                mu=mu,
                cov=S,
                n_years=len(years),
                n_paths=n_paths,
                rng=rng,
                common_Z=Z_global["Z"],
                common_assets=Z_global["assets"],
                policy_assets=assets,
            )
        else:
            Rp = simulate_portfolio_paths(
                weights=weights,
                mu=mu,
                cov=S,
                n_years=len(years),
                n_paths=n_paths,
                rng=rng,
            )

        # 5) 모드별 결과 집계
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

            # 연도별 통계
            rows = []
            for t in range(T):
                C_row, FR_row, R_row = C_sched[t, :], FR_sched[t, :], ret_rate[t, :]
                Aop, Acl = F_open[t, :], F_close[t, :]
                base = {
                    "policy": pol,
                    "year": years[t],
                    **dist_stats("contrib", C_row),
                    **dist_stats("FR", FR_row),
                    **dist_stats("return_rate", R_row),
                    **dist_stats("F_open", Aop),
                    **dist_stats("F_close", Acl),
                    "share_zero_contrib": float((C_row <= 1e-12).mean()),
                    "fr_target": float(fr_targets[t]),
                    "closing_PBO_input": float(L_input[t]),
                    "closing_PBO_used": float(L_used[t]),
                }
                base.update(
                    paired_quantiles(
                        C_row, FR_row, qs=(0.05, 0.25, 0.50, 0.75, 0.95)
                    )
                )
                rows.append(base)
            df_stats = pd.DataFrame(rows)
            agg_contrib_rows.append(df_stats)

            # 자산 mean/base 궤적
            asset_base_mean_rows.append(
                pd.DataFrame(
                    {
                        "policy": pol,
                        "year": years,
                        "F_open_mean": df_stats["F_open_mean"].values,
                        "F_open_base": df_stats["F_open_p50"].values,
                        "F_close_mean": df_stats["F_close_mean"].values,
                        "F_close_base": df_stats["F_close_p50"].values,
                    }
                )
            )

            # 전체 path × 연도 롱포맷
            df_long = pd.DataFrame(
                {
                    "policy": pol,
                    "year": np.repeat(years, P),
                    "path": np.tile(np.arange(P, dtype=int), T),
                    "closing_PBO_input": np.repeat(L_input, P),
                    "closing_PBO_used": np.repeat(L_used, P),
                    "opening_F": F_open.reshape(-1),
                    "contribution": C_sched.reshape(-1),
                    "actual_return": ret_amt.reshape(-1),
                    "return_rate": ret_rate.reshape(-1),
                    "benefits_paid": np.repeat(B_vec, P),
                    "closing_F": F_close.reshape(-1),
                    "fr": FR_sched.reshape(-1),
                    "fr_target": np.repeat(fr_targets[:T], P),
                }
            )
            contrib_path_rows.append(df_long)

            # 총 기여금(경로 합) 통계
            total_by_path = C_sched.sum(axis=0)
            summary.append({"policy": pol, **dist_stats("total_contrib", total_by_path)})

            # FR 밴드
            fr_bands_all.append(
                pd.DataFrame(
                    {
                        "policy": pol,
                        "year": years,
                        "FR_p5": np.percentile(FR_sched, 5, axis=1),
                        "FR_p25": np.percentile(FR_sched, 25, axis=1),
                        "FR_p50": np.percentile(FR_sched, 50, axis=1),
                        "FR_p75": np.percentile(FR_sched, 75, axis=1),
                        "FR_p95": np.percentile(FR_sched, 95, axis=1),
                        "fr_target": fr_targets.astype(float),
                    }
                )
            )

        else:  # fixed
            Cstars = np.array(
                [cstar_for_path(Rp[:, p], A0, L0, liab_proj, fr_targets) for p in range(n_paths)],
                dtype=float,
            )
            Cstars = Cstars[np.isfinite(Cstars)]
            meanC = float(np.nanmean(Cstars)) if len(Cstars) else np.nan
            p5 = float(np.nanpercentile(Cstars, 5)) if len(Cstars) else np.nan
            p25 = float(np.nanpercentile(Cstars, 25)) if len(Cstars) else np.nan
            p50 = float(np.nanpercentile(Cstars, 50)) if len(Cstars) else np.nan
            p75 = float(np.nanpercentile(Cstars, 75)) if len(Cstars) else np.nan
            p95 = float(np.nanpercentile(Cstars, 95)) if len(Cstars) else np.nan

            def _fr_paths_at_C(C: float) -> np.ndarray:
                T, P = Rp.shape
                SC = liab_proj["service_cost"].values[:T]
                IC = liab_proj["interest_cost"].values[:T]
                B = liab_proj["benefit_cf"].values[:T]
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
                agg_rows.append(
                    pd.DataFrame(
                        {
                            "policy": pol,
                            "C_label": label,
                            "contribution": Cval,
                            "year": years,
                            "FR_mean": FRs.mean(axis=1),
                            "FR_p5": np.percentile(FRs, 5, axis=1),
                            "FR_p25": np.percentile(FRs, 25, axis=1),
                            "FR_p50": np.percentile(FRs, 50, axis=1),
                            "FR_p75": np.percentile(FRs, 75, axis=1),
                            "FR_p95": np.percentile(FRs, 95, axis=1),
                            "fr_target": fr_targets.astype(float),
                        }
                    )
                )
                T, P = FRs.shape
                df_long = pd.DataFrame(
                    {
                        "year": np.repeat(years, P),
                        "path": np.tile(np.arange(P, dtype=int), T),
                        "FR": FRs.reshape(-1),
                    }
                )
                df_long.insert(0, "policy", pol)
                df_long.insert(1, "C_label", label)
                df_long["contribution"] = Cval
                path_rows.append(df_long)

            summary.append(
                {
                    "policy": pol,
                    "mean_C": meanC,
                    "p5_C": p5,
                    "p25_C": p25,
                    "p50_C": p50,
                    "p75_C": p75,
                    "p95_C": p95,
                }
            )
            fr_bands_all.append(
                pd.DataFrame(
                    {
                        "policy": pol,
                        "year": years,
                        "FR_p5": np.nan,
                        "FR_p50": np.nan,
                        "FR_p95": np.nan,
                        "fr_target": fr_targets.astype(float),
                    }
                )
            )

    # 6) 결과 및 CSV
    summary_df = pd.DataFrame(summary)
    fr_bands = pd.concat(fr_bands_all, ignore_index=True) if fr_bands_all else pd.DataFrame()

    if csvdir:
        os.makedirs(csvdir, exist_ok=True)
        if mode.lower() == "schedule":
            policy_year_contrib_stats = (
                pd.concat(agg_contrib_rows, ignore_index=True) if agg_contrib_rows else pd.DataFrame()
            )
            policy_contrib_paths = (
                pd.concat(contrib_path_rows, ignore_index=True) if contrib_path_rows else pd.DataFrame()
            )
            policy_asset_base_mean = (
                pd.concat(asset_base_mean_rows, ignore_index=True) if asset_base_mean_rows else pd.DataFrame()
            )

            summary_df.to_csv(os.path.join(csvdir, "summary_schedule_total_contrib.csv"), index=False, encoding="utf-8-sig")
            policy_year_contrib_stats.to_csv(os.path.join(csvdir, "policy_year_contrib_stats.csv"), index=False, encoding="utf-8-sig")
            policy_contrib_paths.to_csv(os.path.join(csvdir, "policy_contrib_paths.csv"), index=False, encoding="utf-8-sig")
            policy_asset_base_mean.to_csv(os.path.join(csvdir, "policy_asset_base_mean.csv"), index=False, encoding="utf-8-sig")
            fr_bands.to_csv(os.path.join(csvdir, "fr_bands_schedule.csv"), index=False, encoding="utf-8-sig")
        else:
            policy_year_stats = pd.concat(agg_rows, ignore_index=True) if agg_rows else pd.DataFrame()
            policy_fr_paths = pd.concat(path_rows, ignore_index=True) if path_rows else pd.DataFrame()

            summary_df.to_csv(os.path.join(csvdir, "summary_fixed.csv"), index=False, encoding="utf-8-sig")
            policy_year_stats.to_csv(os.path.join(csvdir, "policy_year_stats.csv"), index=False, encoding="utf-8-sig")
            policy_fr_paths.to_csv(os.path.join(csvdir, "policy_fr_paths.csv"), index=False, encoding="utf-8-sig")
            fr_bands.to_csv(os.path.join(csvdir, "fr_bands_fixed.csv"), index=False, encoding="utf-8-sig")

    return summary_df, fr_bands
