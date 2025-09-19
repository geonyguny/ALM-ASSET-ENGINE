# ALM Asset Engine

퇴직연금 **자산–부채(ALM)** 시뮬레이션 엔진.  
엑셀 입력을 기반으로 Monte-Carlo 자산경로를 생성하고, 스케줄/고정 기여금 정책에서 **최소 필요 기여금**과 **FR(Funded Ratio)** 분포를 계산합니다.

## 🔧 Quick Start (PowerShell)

```powershell
git clone <repo-url> C:\00_ALM\alm-asset-engine
cd C:\00_ALM\alm-asset-engine
python -m venv .venv
.\.venv\Scripts\pip install -U pip
.\.venv\Scripts\pip install -r requirements.txt

# 실행 예시 (스케줄 모드)
.\.venv\Scripts\python.exe .\bin\alm_asset_engine.py `
  --excel "C:\00_ALM\project\sample_ALM_IO.xlsx" `
  --mode schedule --liab_mode target --liab_scenario base `
  --horizon 10 --seed 42 `
  --csvdir ".\outputs\alm_independent_seed42" `
  --strict_corr --no_common_shock

# 프로젝트 구조
alm-asset-engine/
├─ README.md
├─ requirements.txt
├─ bin/
│  └─ alm_asset_engine.py     # 실행 진입점 (CLI)
├─ alm/
│  ├─ __init__.py
│  ├─ engine.py               # 메인 워크플로우 (입력→시뮬→통계→CSV)
│  ├─ alm_io.py               # Excel 입력 로더 (자산/부채/정책/상관/옵션)
│  ├─ corr_utils.py           # 공분산·상관 행렬 구성/검증
│  ├─ sim_utils.py            # 경로 생성/공통충격/스케줄·고정C 계산
│  ├─ stats_utils.py          # 통계 요약(mean, p5, p25, p50, p75, p95)
│  ├─ snapshot.py             # 실행 스냅샷 저장(메타/파라미터)
│  └─ types.py                # (선택) 데이터클래스/타이핑 모음
└─ outputs/
   └─ ...                     # 실행 결과 CSV/로그

# 파일별 역할

bin/alm_asset_engine.py
CLI 파서(옵션 해석) → alm.engine.run_engine() 호출.

alm/engine.py
엔진 오케스트레이션: load_inputs → 경로 시뮬 → 기여금 계산 → 통계 요약 → CSV 저장.

alm/alm_io.py
Excel 다중 시트 로딩, 스키마 검증, 정책/시나리오 파싱, 네이밍 정규화(대/소문자·공백·%).

alm/corr_utils.py
상관→공분산 변환, PSD(양의 준정부호) 검증, --strict_corr 처리(수정/실패).

alm/sim_utils.py

generate_common_Z(...): 전역 난수 Z_global 생성(공통충격용)

simulate_portfolio_paths(..., common_Z, common_assets, policy_assets)

min_contrib_schedule_paths(...): 연도별 C_t^* 경로/FR 계산

cstar_for_path(...): 경로별 최소 고정 C 바이섹션 탐색

alm/stats_utils.py
분포요약(평균/분위수), 페어드 시드 통계 등.

alm/snapshot.py
실행 시점의 입력/파라미터/행렬/기본 통계 저장(재현성).


# 실행 옵션(요약)

--mode: schedule(연도별 C_t^*) | fixed(고정 C 탐색)
--liab_mode: target(Closing PBO 기준) | roll(SC/IC/B 롤포워드)
--liab_scenario: base/p50/기타 시나리오 키
--horizon: 연 단위 시뮬 기간(예: 10)
--seed: 난수 시드
--csvdir: 결과 저장 폴더
--common_shock / --no_common_shock:
전 자산군이 같은 난수 Z 충격을 받도록(상관 보존 & 정책 비교의 일관성↑)
--strict_corr: 공분산 정합성(대칭/PSD) 강제 체크(위반 시 보정/에러)


# 출력물(모드별 CSV)

모든 CSV는 UTF-8 with BOM (utf-8-sig) 로 저장합니다(Excel 호환).
mode = schedule

summary_schedule_total_contrib.csv
policy_year_contrib_stats.csv
policy_contrib_paths.csv
policy_asset_base_mean.csv
fr_bands_schedule.csv
mode = fixed
summary_fixed.csv
policy_year_stats.csv
policy_fr_paths.csv
fr_bands_fixed.csv

# 통계 요약 규칙

기본 분위수: mean, p5, p25, p50, p75, p95
스케줄/고정 모드 모두 동일한 규칙으로 FR/자산/기여금 통계를 산출

# 스냅샷

alm/snapshot.py 가 각 실행의 입력 요약, 상관/공분산, 파라미터를 JSON/CSV로 남겨 재현성과 디버그를 지원합니다.

# 개발 가이드

코딩 규칙: 모듈 단일 책임, Numpy 연산 벡터화, I/O와 계산 로직 분리
예외 처리: 명확한 오류 메시지 (입력파일, 상관행렬 등)
성능 팁: 공통충격 + 경로 캐싱 → 정책간 비교 반복 시 속도 개선
테스트: 샘플 엑셀 기반 회귀 테스트 권장

# 트러블슈팅

FileNotFoundError: sample_ALM_IO.xlsx
경로/확장자/드라이브 확인. PowerShell:

Test-Path "C:\00_ALM\project\sample_ALM_IO.xlsx"
Get-ChildItem C:\00_ALM -Recurse -Filter "sample_ALM_IO*.xls*"

openpyxl 오류 → pip install -U openpyxl pandas
--strict_corr 실패 → 상관 행렬 대칭/대각=1 확인