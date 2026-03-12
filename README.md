# STATIZ KBO 2026 Prediction

STATIZ AI 승부예측대회용 KBO 경기 승부예측 프로젝트 레포입니다.

이 레포의 목표는 세 가지입니다.
- 팀이 같은 기준으로 데이터를 수집하고 가공할 수 있게 한다.
- 대회 규정에 맞는 피처와 백테스트를 재현 가능하게 관리한다.
- EC2에서 실행한 실험 결과를 GitHub 문서로 팀이 함께 이해할 수 있게 정리한다.

## 지금 가장 중요한 원칙

- 데이터 사용 범위는 `2023+` 입니다.
- 실제 학습/평가의 중심은 `2024~2025` 입니다.
- 경기일 `D` 예측에는 반드시 `D-1`까지의 정보만 사용합니다.
- API Key, Secret, pem, raw data는 GitHub에 커밋하지 않습니다.

## 작업 공간을 아주 쉽게 이해하기

이 프로젝트는 작업 공간이 세 군데로 나뉩니다.

1. 로컬 GitHub clone
- 코드 읽기, 수정, 커밋, 문서 작성의 기준 위치
- 이 레포가 팀 협업의 공식 원본입니다.

2. GitHub
- 팀원이 함께 보는 공식 저장소입니다.
- 코드, 문서, 작은 요약 결과만 올립니다.

3. EC2 `~/statiz`
- 실제 데이터 수집과 실험을 돌리는 실행 환경입니다.
- 원천 데이터와 대용량 결과는 여기 남깁니다.
- 현재 EC2에서는 파이썬 스크립트가 repo root에 직접 놓여 있습니다.

정리하면 다음 한 문장으로 이해하면 됩니다.
- 코드는 GitHub repo의 `scripts/` 에서 관리하고, 실행은 EC2 `~/statiz` 에서 한다.

## 먼저 읽을 문서

1. [docs/README.md](docs/README.md)
2. [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)
3. [docs/PIPELINE.md](docs/PIPELINE.md)
4. [docs/EXPERIMENT_STATUS.md](docs/EXPERIMENT_STATUS.md)
5. [docs/COLLABORATION.md](docs/COLLABORATION.md)

## 디렉터리 구조

```text
kbo-2026-prediction/
|-- README.md
|-- docs/
|   |-- README.md
|   |-- PROJECT_OVERVIEW.md
|   |-- PIPELINE.md
|   |-- EXPERIMENT_STATUS.md
|   |-- COLLABORATION.md
|-- scripts/
|   |-- download_*.py
|   |-- build_*.py
|   |-- backtest_*.py
|   |-- inspect_*.py
|-- data/                # 로컬/EC2 생성 산출물, 기본적으로 커밋하지 않음
|-- logs/                # 실행 로그, 기본적으로 커밋하지 않음
```

## 현재 주요 스크립트

데이터 수집/가공
- `scripts/download_schedule.py`
- `scripts/build_game_index.py`
- `scripts/download_game_details.py`
- `scripts/build_lineup_table.py`
- `scripts/build_player_year_index.py`
- `scripts/download_playerday.py`
- `scripts/build_playerday_tables_v2.py`

기준선 실험
- `scripts/build_features_v0.py`
- `scripts/backtest_v0_online_lr.py`

현재 핵심 실험
- `scripts/build_features_v1_paper.py`
- `scripts/backtest_v1_online_lr.py`
- `scripts/inspect_lr_coef_v1.py`

후보 피처 탐색
- `scripts/build_features_v2_candidates.py`
- `scripts/feature_subset_search_v1.py`
- `scripts/backtest_top10_block_report_v1.py`

## 현재 진행 상황

끝난 것
- API 수집 파이프라인 구축
- `lineup_long`, `playerday_*_long` 생성
- v0 baseline 백테스트 실행
- v1 4피처 실험 실행
- v2 후보 피처 생성 및 subset 탐색 실행

지금 하고 있는 것
- 좋은 피처 조합을 찾고 해석하는 단계
- 문서와 GitHub 협업 구조를 정리하는 단계

## 빠른 시작

실행 순서가 필요하면 [docs/PIPELINE.md](docs/PIPELINE.md)를 보면 됩니다.
현재 실험 상태와 추천 다음 단계는 [docs/EXPERIMENT_STATUS.md](docs/EXPERIMENT_STATUS.md)에 정리되어 있습니다.
