# snapshot.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import List, Optional, Union
from corr_utils import fingerprint

def dump_snapshot(csvdir: Optional[str], *,
                  policy: int, seed: int, mode: str,
                  liab_mode: str, liab_scenario: Optional[Union[str,int]],
                  assets: List[str], w: np.ndarray, u: np.ndarray, v: np.ndarray, R: np.ndarray,
                  A0: float, L0: float, fr_targets_head: float, strict_corr: bool):
    if not csvdir:
        return
    dbg = Path(csvdir) / "_inputs"
    dbg.mkdir(parents=True, exist_ok=True)
    meta = {
        "policy": policy, "seed": seed, "mode": mode,
        "liab_mode": liab_mode, "liab_scenario": liab_scenario,
        "assets": list(map(str, assets)),
        "w_sum": float(np.sum(w)), "w_min": float(np.min(w)), "w_max": float(np.max(w)),
        "R_fingerprint": fingerprint(R), "R_diag": np.diag(R).tolist(),
        "A0": float(A0), "L0": float(L0),
        "fr_target_first": float(fr_targets_head),
        "strict_corr": bool(strict_corr),
    }
    (dbg / f"inputs_policy{policy}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    pd.DataFrame({"asset": assets, "w": w, "u": u, "v": v}).to_csv(
        dbg / f"weights_policy{policy}.csv", index=False, encoding="utf-8-sig"
    )
    np.savetxt(dbg / f"corr_policy{policy}.csv", R, delimiter=",")
