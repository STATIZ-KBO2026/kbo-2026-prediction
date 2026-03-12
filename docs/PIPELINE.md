# Pipeline

이 문서는 실제로 어떤 순서로 스크립트를 돌리는지 설명합니다.

## 제일 먼저 알아둘 점

현재 실행 환경은 EC2 `~/statiz` 입니다.
GitHub repo에서는 스크립트가 `scripts/` 아래에 정리되어 있지만, EC2에서는 현재 스크립트가 repo root에 직접 있습니다.

예시
- GitHub: `scripts/build_features_v1_paper.py`
- EC2: `~/statiz/build_features_v1_paper.py`

즉, 스크립트 이름은 같고 위치만 다를 수 있습니다.

## 전체 흐름

1. schedule raw 저장
2. 경기 인덱스 생성
3. lineup / boxscore raw 저장
4. lineup long 생성
5. player-year 인덱스 생성
6. playerDay raw 저장
7. batter / pitcher long 생성
8. feature table 생성
9. expanding window 백테스트

## 단계별 스크립트와 산출물

### 1) 일정 수집
- 스크립트: `download_schedule.py`
- 목적: `gameSchedule` raw 저장
- 산출물: `data/raw_schedule/*.json`

### 2) 경기 인덱스 생성
- 스크립트: `build_game_index.py`
- 목적: 경기번호 `s_no` 목록 생성 및 실제 치러진 경기만 필터링
- 산출물:
  - `data/game_index.csv`
  - `data/game_index_played.csv`

### 3) 경기 상세 수집
- 스크립트: `download_game_details.py`
- 목적: 각 경기의 `gameLineup`, `gameBoxscore` 저장
- 산출물:
  - `data/raw_lineup/*.json`
  - `data/raw_boxscore/*.json`

### 4) 라인업 테이블 생성
- 스크립트: `build_lineup_table.py`
- 산출물: `data/lineup_long.csv`

### 5) 선수-연도 인덱스 생성
- 스크립트: `build_player_year_index.py`
- 산출물: `data/player_year_index.csv`

### 6) playerDay 수집
- 스크립트: `download_playerday.py`
- 산출물: `data/raw_playerday/*.json`

### 7) playerDay 테이블 생성
- 스크립트:
  - `build_playerday_tables.py`
  - `build_playerday_tables_v2.py`
- 주요 산출물:
  - `data/playerday_batter_long.csv`
  - `data/playerday_pitcher_long.csv`

### 8) 피처 생성

baseline
- 스크립트: `build_features_v0.py`
- 산출물: `data/features_v0.csv`

현재 핵심 v1
- 스크립트: `build_features_v1_paper.py`
- 산출물: `data/features_v1_paper.csv`
- 역할: 기존 4개 핵심 피처 생성

후보 확장 v2
- 스크립트: `build_features_v2_candidates.py`
- 산출물: `data/features_v2_candidates.csv`
- 역할: 기존 4개 + 새 후보 피처를 함께 생성

### 9) 백테스트

baseline
- 스크립트: `backtest_v0_online_lr.py`
- 산출물: `data/backtest_pred_v0.csv`

v1 기준선
- 스크립트: `backtest_v1_online_lr.py`
- 역할: expanding window 기준 평가

subset 탐색
- 스크립트: `feature_subset_search_v1.py`
- 산출물:
  - `data/feature_subset_search_v1.csv`
  - `data/feature_subset_top20_v1.csv`

상위 조합 상세 리포트
- 스크립트: `backtest_top10_block_report_v1.py`
- 산출물:
  - `data/top10_subset_summary_v1.csv`
  - `data/top10_subset_block_metrics_v1.csv`
  - `data/top10_subset_pred_v1.csv`

## 실제 컬럼 확인 결과

핵심 CSV 컬럼은 실제 데이터를 기준으로 아래처럼 확인했습니다.

`lineup_long.csv`
- `date, s_no, t_code, side, battingOrder, position, starting, lineupState, p_no, p_name, p_bat, p_throw, p_backNumber`

`playerday_batter_long.csv`
- 타자 누적 계산에 필요한 `PA, AB, H, 1B, 2B, 3B, HR, TB, BB, HP, SF, OPS, NP` 포함

`playerday_pitcher_long.csv`
- 투수 누적 계산에 필요한 `IP, BB, TBF, H, TB, NP, S, HD, OPS` 포함

## 운영상 주의

- 데이터 규정상 `2023+` 만 사용한다.
- output과 학습/평가 중심은 `2024~2025` 로 맞춘다.
- 모든 피처는 경기일 `D` 예측 시 `D-1` 까지 정보만 사용해야 한다.
- raw data와 대용량 결과는 GitHub에 올리지 않는다.
