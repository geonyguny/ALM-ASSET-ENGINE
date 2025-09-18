# alm_io.py
from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

def _ensure_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] 시트에 필요한 컬럼이 없습니다: {missing}")

def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _normalize(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")

def _find_sheet(xls: pd.ExcelFile, target: str) -> Optional[str]:
    norm = _normalize(target)
    for name in xls.sheet_names:
        if _normalize(name) == norm:
            return name
    return None

def _coerce_num(x) -> Optional[float]:
    if pd.isna(x): return None
    s = str(x).strip().replace("%", "")
    if ("," in s) and ("." not in s): s = s.replace(",", ".")
    try: return float(s)
    except ValueError: return None

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

def _last_hist_year(asset_df, liab_df):
    years_a = asset_df.loc[asset_df["closing_F"].notna(), "asset"]
    years_l = liab_df.loc[liab_df["closing_PBO"].notna(), "liability"]
    return int(sorted(set(years_a).intersection(set(years_l)))[-1])

def _auto_scale_uv(df: pd.DataFrame) -> pd.DataFrame:
    u = df["u"].astype(float).values
    v = df["v"].astype(float).values
    is_percent = (np.nanmax(np.abs(np.concatenate([u, v]))) > 1.0)
    if is_percent:
        df["u"] = df["u"].astype(float) / 100.0
        df["v"] = df["v"].astype(float) / 100.0
        df["_was_percent"] = True
    else:
        df["_was_percent"] = False
    return df

def _pick_liability_scenario(liab_raw: pd.DataFrame,
                             liab_scenario: Optional[Union[str,int]],
                             debug: bool=False) -> pd.DataFrame:
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
    pick = "base" if liab_scenario is None else _norm(liab_scenario)
    if (liab_scenario is None) and ("base" not in uniq): pick = ("p50" if "p50" in uniq else uniq[0])
    sel = df.loc[df["_scenario_norm"] == pick].drop(columns=["_scenario_norm"])
    if sel.empty:
        if debug: print(f"[DEBUG] scenario '{liab_scenario}' not found. Fallback to '{uniq[0]}'")
        sel = df.loc[df["_scenario_norm"] == uniq[0]].drop(columns=["_scenario_norm"])
    else:
        if debug: print(f"[DEBUG] liability scenario selected: '{pick}' (available={uniq})")
    return sel

def _dedup_by_year(liab_df: pd.DataFrame, debug: bool=False) -> pd.DataFrame:
    if liab_df.duplicated(subset=["liability"]).any():
        if debug:
            dups = liab_df.loc[liab_df.duplicated(subset=["liability"], keep=False), "liability"].unique()
            print(f"[DEBUG] liability: duplicated years → first per year: {sorted(map(int, dups))}")
        liab_df = liab_df.drop_duplicates(subset=["liability"], keep="first")
    return liab_df

def _load_targets_or_default(
    xls: pd.ExcelFile, start_year: int, horizon: int,
    fr_target_default: float, debug: bool = False
) -> np.ndarray:
    sheet = _find_sheet(xls, "targets")
    m: Dict[int, float] = {}
    if sheet is not None:
        try:
            df_try = pd.read_excel(xls, sheet_name=sheet)
            if debug: print(f"[DEBUG] targets sheet '{sheet}' columns:", list(df_try.columns))
            colmap = {_normalize(c): c for c in df_try.columns}
            year_key = next((k for k in colmap if k == "year"), None)
            fr_key   = next((k for k in colmap if k in ("fr_target","frtarget","target_fr","fr","target")), None)
            if year_key is None or fr_key is None:
                raw = pd.read_excel(xls, sheet_name=sheet, header=None)
                hdr = _scan_header_row(raw)
                if hdr is None:
                    raise ValueError("targets header not found")
                r, yi, fi = hdr
                df = raw.iloc[r+1:, [yi, fi]].copy(); df.columns = ["year", "fr_target"]
            else:
                df = df_try[[colmap[year_key], colmap[fr_key]]].copy(); df.columns = ["year", "fr_target"]
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
            df["fr_target"] = df["fr_target"].map(_coerce_num)
            df = df.dropna(subset=["year"]).sort_values("year")
            m = {int(r.year): (float(r.fr_target) if r.fr_target is not None else fr_target_default) for r in df.itertuples()}
        except Exception as e:
            if debug: print(f"[DEBUG] targets loading failed: {e}")
            m = {}
    else:
        if debug: print("[DEBUG] 'targets' sheet not found. Using default.]")

    out, last_val = [], fr_target_default
    for k in range(horizon):
        y = start_year + k
        val = m.get(y, last_val if k > 0 else fr_target_default)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = last_val
        out.append(val); last_val = val
    v = np.array(out, dtype=float)
    if debug:
        print(f"[DEBUG] fr_targets: first 5 = {np.round(v[:5], 4)}, min={v.min():.4f}, max={v.max():.4f}")
    return v

def load_inputs(xlsx_path: str,
                horizon_years: int = 10,
                n_paths: int | None = None,
                seed: int = 2025,
                fr_target_default: float = 1.0,
                debug: bool = False,
                liab_scenario: Optional[Union[str,int]] = None):
    xls = pd.ExcelFile(xlsx_path)
    if debug: print("[DEBUG] sheets in workbook:", xls.sheet_names)

    pf = pd.read_excel(xls, "portfolio"); pf = _strip_cols(pf)
    _ensure_cols(pf, ["policy", "asset_c2", "u", "v", "w"], "portfolio")
    pf = _auto_scale_uv(pf)
    chk = pf.groupby("policy")["w"].sum().round(9)
    if not np.allclose(chk.values, 1.0, atol=1e-6):
        raise ValueError(f"[portfolio] 정책별 비중합이 1이 아닙니다:\n{chk}")

    try:
        corr = pd.read_excel(xls, "corr"); corr = _strip_cols(corr)
        _ensure_cols(corr, ["a", "b", "rho"], "corr")
        corr["a"] = corr["a"].astype(str).str.strip()
        corr["b"] = corr["b"].astype(str).str.strip()
        corr["rho"] = corr["rho"].astype(float)
    except Exception:
        corr = pd.DataFrame(columns=["a", "b", "rho"])

    liab_raw = pd.read_excel(xls, "liability"); liab_raw = _strip_cols(liab_raw)
    _ensure_cols(liab_raw, ["liability","opening_PBO","service_cost","interest_cost","benefit_cf","closing_PBO"], "liability")
    liab_sel = _pick_liability_scenario(liab_raw, liab_scenario, debug=debug)
    liab_sel["liability"] = pd.to_numeric(liab_sel["liability"], errors="coerce").astype("Int64")
    liab_sel = liab_sel.dropna(subset=["liability"]).astype({"liability":"int"})
    liab = _dedup_by_year(liab_sel, debug=debug)

    asset = pd.read_excel(xls, "asset"); asset = _strip_cols(asset)
    _ensure_cols(asset, ["asset","opening_F","employer_contrib","actual_return","benefits_paid","closing_F"], "asset")

    try:
        rates = pd.read_excel(xls, "interest")
    except Exception:
        rates = None

    hist_y = _last_hist_year(asset, liab)
    A0 = float(asset.loc[asset["asset"]==hist_y, "closing_F"].iloc[0])
    L0 = float(liab.loc[liab["liability"]==hist_y, "closing_PBO"].iloc[0])

    years = list(range(hist_y+1, hist_y+1+horizon_years))
    liab_idx = liab.set_index("liability")
    liab_take = liab_idx.loc[years, ["service_cost","interest_cost","benefit_cf","closing_PBO"]].astype(float)
    liab_proj = liab_take.copy()
    try:
        prev = liab_idx.loc[[hist_y]+years[:-1], "closing_PBO"].values
        liab_proj["check_roll"] = (
            liab_idx.loc[years, "closing_PBO"].values
            - (prev + liab_proj["service_cost"].values + liab_proj["interest_cost"].values - liab_proj["benefit_cf"].values)
        )
    except Exception:
        pass

    if n_paths is None:
        n_paths = 100
        if rates is not None and "path" in getattr(rates, "columns", []):
            n_paths = int(rates["path"].nunique())

    rng = np.random.default_rng(seed)
    fr_targets = _load_targets_or_default(xls, start_year=years[0], horizon=len(years),
                                          fr_target_default=fr_target_default, debug=debug)

    return dict(portfolio=pf, corr=corr, liab_proj=liab_proj,
                A0=A0, L0=L0, years=years, n_paths=n_paths, rng=rng,
                fr_targets=fr_targets)
