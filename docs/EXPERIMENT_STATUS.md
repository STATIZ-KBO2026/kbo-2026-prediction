# Experiment Status

## 지금까지의 흐름

### 1) baseline v0
- 목적: 파이프라인이 끝까지 도는지 확인
- 구성: 기본 피처 + 로지스틱 회귀 백테스트
- 의미: 수집부터 백테스트까지 재현 가능한 기준선 확보

### 2) v1 핵심 피처 실험
현재 v1의 핵심 4피처는 아래와 같습니다.
- `diff_sum_ops_smooth`
- `diff_sum_ops_recent5`
- `diff_sp_oops`
- `diff_bullpen_fatigue`

기존 실험에서 확인된 방향
- `diff_sum_ops_smooth`, `diff_sp_oops` 는 상대적으로 유효했다.
- `diff_sum_ops_recent5`, `diff_bullpen_fatigue` 는 약한 편이었다.
- 다만 불펜 정의를 개선한 최신 버전에서는 다시 볼 필요가 있다.

v1 최신 결과
- ACC `0.5470`
- LOGLOSS `0.6865`
- BRIER `0.2467`
- AUC `0.5676`

### 3) v2 후보 피처 확장
v2 후보 생성 스크립트에는 아래 피처들이 포함됩니다.

기존 4개
- `diff_sum_ops_smooth`
- `diff_sum_ops_recent5`
- `diff_sp_oops`
- `diff_bullpen_fatigue`

새 후보
- `diff_opp_sp_platoon_cnt`
- `diff_sp_bbip`
- `diff_pythag_winpct`
- `diff_recent10_winpct`
- `diff_top5_ops_smooth`
- `diff_team_stadium_winpct`
- `park_factor_stadium`
- `diff_team_stadium_winpct_pfadj`

## subset 탐색 결과 요약

평가 설정
- 모델: `StandardScaler + LogisticRegression`
- 방식: expanding window
- seed train: `2024`
- test: `2025`
- 재학습 block: `7일`
- 우선 비교 지표: `LOGLOSS`, `BRIER`, `AUC`, `ACC`

전체 subset 탐색에서 가장 좋게 나온 조합
- `diff_bullpen_fatigue + diff_sp_bbip + diff_pythag_winpct`

성능
- ACC `0.5820`
- LOGLOSS `0.6746`
- BRIER `0.2408`
- AUC `0.6153`

이 결과의 의미
- 현재는 피처를 많이 넣는 것보다, 정보량이 좋은 일부 조합이 더 잘 나왔다.
- 특히 `선발 볼넷 관련 지표` 와 `팀 누적 전력 수준` 계열이 의미가 있었을 가능성이 있다.

## 현재 남은 핵심 과제

1. cold start 검증
- 시즌 초반에 prior와 smoothing이 실제로 안정적으로 작동하는지 확인

2. 상위 조합 해석
- 왜 이 조합이 잘 나왔는지 야구적으로 설명 가능해야 함

3. 재현성 강화
- GitHub 문서와 브랜치 기준으로 실험 과정을 남겨야 함

4. 다음 후보 확장 검토
- 손대응, 구장, 최근 폼, 역할 정보 피처를 더 체계적으로 늘릴지 검토

## 현재 결과 파일 위치

이 결과들은 EC2 `~/statiz/data` 에 생성됩니다.

- `features_v2_candidates.csv`
- `feature_subset_search_v1.csv`
- `feature_subset_top20_v1.csv`
- `top10_subset_summary_v1.csv`
- `top10_subset_block_metrics_v1.csv`
- `top10_subset_pred_v1.csv`

GitHub에는 원칙적으로 raw data나 대용량 결과 전체를 올리지 않고, 문서와 요약만 남깁니다.
