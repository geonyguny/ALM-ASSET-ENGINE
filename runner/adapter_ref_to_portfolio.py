# runner/adapter_ref_to_portfolio.py
from __future__ import annotations
import math
import re
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd


def _parse_float_expr(x: str | float | int) -> float:
    """
    공분산 long CSV의 'cov' 필드에 '0.18^2', '0.18*0.16*0.35' 등 간단 수식이 오는 경우를 안전하게 평가.
    허용: 숫자/공백, + - * / ( ), e/E, ^(제곱 → **로 치환)
    """
    if isinstance(x, (float, int)):
        return float(x)
    s = str(x).strip().replace("^", "**")
    if not re.fullmatch(r"[0-9eE\.\+\-\*\/\(\)\s]+", s):
        raise ValueError(f"Invalid cov expression: {x!r}")
    # 안전한 eval: 내장 비활성화
    return float(eval(s, {"__builtins__": None}, {}))


def _cov_from_long(df_cov: pd.DataFrame, assets: List[str]) -> np.ndarray:
    """
    long 포맷 공분산 -> 대칭 (K,K) 행렬.
    - df_cov: columns ['asset_i','asset_j','cov']
    - assets: 최종 자산 순서(행/열 순서로 사용)
    """
    K = len(assets)
    idx = {a: i for i, a in enumerate(assets)}
    Sigma = np.zeros((K, K), dtype=float)

    # 값 채우기
    for _, row in df_cov.iterrows():
        ai = str(row["asset_i"])
        aj = str(row["asset_j"])
        if ai not in idx or aj not in idx:
            # 정의되지 않은 자산 레코드는 무시(또는 raise 하고 싶으면 바꾸세요)
            continue
        i, j = idx[ai], idx[aj]
        Sigma[i, j] = _parse_float_expr(row["cov"])

    # 대칭화(상삼각/하삼각 중 큰 값 사용)
    Sigma = np.maximum(Sigma, Sigma.T)

    # 대각선 보정(>=0)
    for i in range(K):
        if Sigma[i, i] <= 0:
            # 아주 작은 양수로 보정 (수치안정)
            Sigma[i, i] = 1e-8
    return Sigma


def _ensure_asset_alignment(mu_df: pd.DataFrame, cov_df: pd.DataFrame) -> List[str]:
    """
    mu_df, cov_df의 자산 우주를 합집합으로 모아 정렬된 자산 리스트를 반환.
    - mu_df: columns ['asset','mu']
    - cov_df: columns ['asset_i','asset_j','cov']
    """
    mu_assets = set(map(str, mu_df["asset"].unique()))
    cov_assets = set(map(str, pd.concat([cov_df["asset_i"], cov_df["asset_j"]]).unique()))
    all_assets = sorted(mu_assets.union(cov_assets))

    missing_in_mu = [a for a in all_assets if a not in mu_assets]
    missing_in_cov = [a for a in all_assets if a not in cov_assets]
    if missing_in_mu:
        print(f"[adapter] Warning: assets not in mu file: {missing_in_mu}")
    if missing_in_cov:
        print(f"[adapter] Warning: assets not in cov file: {missing_in_cov}")
    return all_assets


def _build_corr_from_cov(Sigma: np.ndarray) -> np.ndarray:
    sig = np.sqrt(np.clip(np.diag(Sigma), 1e-16, None))
    denom = np.outer(sig, sig)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = Sigma / denom
        R[~np.isfinite(R)] = 0.0
        np.fill_diagonal(R, 1.0)
    return R


def ref_to_portfolio_excel(
    ref_frontier_csv: str,
    mu_csv: str,
    cov_csv: str,
    out_xlsx: str,
    policy_start: int = 101,
    base_io_xlsx: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    REF 결과(ref_frontier.csv) + μ/Σ 입력을 기존 ALM 엔진 엑셀 스키마로 변환 저장.
    - 기본 저장: portfolio, corr 두 시트만(out_xlsx).
    - base_io_xlsx 지정 시: base IO의 모든 시트를 로드해 portfolio/corr만 교체 후 out_xlsx에 FULL로 저장.
    반환: (portfolio_df, corr_df)
    """
    # 입력 로드
    ref = pd.read_csv(ref_frontier_csv)
    mu_df = pd.read_csv(mu_csv)
    cov_df = pd.read_csv(cov_csv)

    # 자산 우주 정렬(μ/Σ 간 충돌 대비)
    assets = _ensure_asset_alignment(mu_df, cov_df)

    # μ 매핑 (누락 자산은 0으로 보정 가능—여기서는 존재 필수로 가정)
    mu_map: Dict[str, float] = {}
    mu_src = dict(zip(map(str, mu_df["asset"]), mu_df["mu"]))
    for a in assets:
        if a not in mu_src:
            raise ValueError(f"Missing mu for asset {a!r} in {mu_csv}")
        mu_map[a] = float(mu_src[a])

    # Σ -> corr
    Sigma = _cov_from_long(cov_df, assets)
    sig_vec = np.sqrt(np.diag(Sigma))
    corr = _build_corr_from_cov(Sigma)

    # portfolio 시트(타깃별 정책 번호)
    policies_rows = []
    targets_sorted = sorted(ref["target"].unique())
    for pid, tgt in enumerate(targets_sorted, start=policy_start):
        sub = ref[ref["target"] == tgt].copy()
        # 자산별 w_bar 매핑 (누락 자산은 0)
        w_map = dict(zip(map(str, sub["asset"]), sub["w_bar"]))
        for a_idx, a in enumerate(assets):
            policies_rows.append(
                {
                    "policy": int(pid),
                    "asset_c2": a,
                    "w": float(w_map.get(a, 0.0)),
                    "u": float(mu_map[a]),
                    "v": float(sig_vec[a_idx]),
                }
            )
    portfolio_df = pd.DataFrame(policies_rows)

    # corr 시트
    corr_rows = []
    for i, a in enumerate(assets):
        for j, b in enumerate(assets):
            corr_rows.append({"asset_i": a, "asset_j": b, "corr": float(corr[i, j])})
    corr_df = pd.DataFrame(corr_rows)

    # 저장
    if base_io_xlsx is None:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            portfolio_df.to_excel(w, sheet_name="portfolio", index=False)
            corr_df.to_excel(w, sheet_name="corr", index=False)
    else:
        # 기존 IO의 모든 시트를 불러와 portfolio/corr만 교체하여 FULL 저장
        base_sheets = pd.read_excel(base_io_xlsx, sheet_name=None)
        base_sheets["portfolio"] = portfolio_df
        base_sheets["corr"] = corr_df
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            for name, df in base_sheets.items():
                # 엑셀의 빈 시트 방지: 최소 한 행이 필요한 경우가 있어 형식 유지
                df_out = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
                df_out.to_excel(w, sheet_name=name, index=False)

    return portfolio_df, corr_df
