import pandas as pd
from alm.engine import run_engine

def test_smoke_run(tmp_path):
    # 예제 파일로 짧게 돌려보기
    csvdir = tmp_path / "out"
    summary, bands = run_engine(
        xlsx_path="examples/sample_ALM_IO.xlsx",
        horizon_years=3, seed=42, csvdir=str(csvdir),
        mode="schedule", liab_mode="target",
        liab_scenario="base", strict_corr=True, common_shock=True
    )
    assert not summary.empty
