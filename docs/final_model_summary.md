# KBO 2026 예측 최종 모델 요약

작성일: 2026-03-28

## 1) 최종 모델 선정 기준
- 기준 리포트: `statiz/data/backtest_v5_model_report_2023_2024_to_2025.csv`
- 선택 룰: 테스트 구간(`split=test`)에서 `logloss`가 가장 낮은 모델을 최종 모델로 정의
- 결과: `sk_random_forest`가 최저 `logloss`로 1위

## 2) 최종 모델 정보
- 모델명: `sk_random_forest`
- 하이퍼파라미터: `n_estimators=1200`, `max_depth=10`, `min_samples_leaf=5`, `max_features=sqrt`
- 학습/평가 구간: train 2023~2024, test 2025
- 모델 비중(앙상블 관점): **100%** (단일 모델)

## 3) 성능 (기준 리포트)
| 모델 | Test LogLoss | Test Accuracy | Test AUC | 비고 |
|---|---:|---:|---:|---|
| sk_random_forest | **0.667621** | 0.602828 | **0.632910** | 최종 모델 |
| blend_topk_composite | 0.668248 | 0.597686 | 0.632851 | blend |
| blend_topk_logloss | 0.668825 | 0.592545 | 0.632177 | blend |
| sk_extra_trees | 0.670848 | **0.604113** | 0.629718 | 단일 트리계열 후보 |

## 4) 피처 구성
기준 데이터: `statiz/data/features_v0.csv`

- 전체 피처: 405개
- 사용 피처: 356개
- 제외 피처: 49개 (`strict` 필터)

### 4-1) 사용 피처 카테고리(개수 기준)
| 카테고리 | 개수 | 비율 |
|---|---:|---:|
| lineup | 110 | 30.90% |
| starter_pitcher | 106 | 29.78% |
| team_strength | 62 | 17.42% |
| bullpen | 49 | 13.76% |
| environment_context | 20 | 5.62% |
| other_context | 9 | 2.53% |

### 4-2) 제외 피처 특징
제외 사유 태그 비중(`statiz/data/backtest_v5_dropped_features_2023_2024_to_2025.csv`):
- `manual_drop`: 25개 (51.02%)
- `constant`: 18개 (36.73%)
- `near_all_zero`: 18개 (36.73%)
- `countlike_high_shift`: 4개 (8.16%)

대표 제외 예시:
- 상수/희소: `home_sp_sched_missing`, `away_sp_replaced`, `weather_rain_probability`
- 분포 이동 과다: `home_team_G`, `away_team_G`, `home_team_home_G`, `away_team_away_G`
- 수동 제외(노이즈/중복성): `home_sp_nohist`, `away_sp_nohist`, `home_lineup_avg_obp`, `away_lineup_blend_ops`

## 5) 피처 비중(중요도)
아래 중요도는 위 최종 모델 설정으로 재학습해 산출한 `feature_importances_` 기준.

### 5-1) 카테고리별 중요도 비중
| 카테고리 | 중요도 비중 |
|---|---:|
| lineup | **35.63%** |
| starter_pitcher | **29.33%** |
| team_strength | 17.66% |
| bullpen | 10.58% |
| other_context | 3.96% |
| environment_context | 2.84% |

### 5-2) 방향성(접두어) 기준 중요도 비중
| 그룹 | 중요도 비중 |
|---|---:|
| `diff_*` | **32.95%** |
| `home_*` | 32.11% |
| `away_*` | 31.88% |
| 기타 | 3.05% |

### 5-3) 상위 핵심 피처 TOP 15
| 순위 | 피처 | 중요도 비중 |
|---:|---|---:|
| 1 | `diff_count_matchup_edge` | 0.772% |
| 2 | `diff_sp_prior_K9` | 0.710% |
| 3 | `diff_sp_blend_K9` | 0.705% |
| 4 | `away_count_matchup_edge` | 0.675% |
| 5 | `away_lineup_state_y_rate` | 0.657% |
| 6 | `away_lineup_state_y_cnt` | 0.636% |
| 7 | `diff_lineup_time_split_ops` | 0.631% |
| 8 | `away_sp_blend_K9` | 0.629% |
| 9 | `diff_lineup_bb_per_pa` | 0.618% |
| 10 | `diff_lineup_prior_so_per_pa` | 0.597% |
| 11 | `diff_lineup_blend_ops` | 0.584% |
| 12 | `diff_lineup_vs_throw_split_so_per_pa` | 0.566% |
| 13 | `diff_lineup_blend_so_per_pa` | 0.536% |
| 14 | `home_lineup_bot6_blend_ops` | 0.533% |
| 15 | `away_sp_prior_K9` | 0.527% |

### 5-4) TOP 15 피처 상세 설명
해석 전제:
- `home_*`: 홈팀 값
- `away_*`: 원정팀 값
- `diff_*`: `home - away`
- `prior_*`: 누적 기록을 리그 평균으로 수축(shrink)한 사전 추정치
- `blend_*`: 당해/전년/커리어 + 콜드스타트 보정을 섞은 혼합 추정치

| 피처 | 계산/정의 | 해석 포인트 |
|---|---|---|
| `diff_count_matchup_edge` | `home_count_matchup_edge - away_count_matchup_edge` | 양수면 홈 타선의 볼넷-삼진 제구 우위가 더 큼 |
| `diff_sp_prior_K9` | `home_sp_prior_K9 - away_sp_prior_K9` | 양수면 홈 선발의 사전 추정 탈삼진 능력(K/9) 우위 |
| `diff_sp_blend_K9` | `home_sp_blend_K9 - away_sp_blend_K9` | 양수면 최근/이전시즌/커리어를 반영한 홈 선발 K/9 우위 |
| `away_count_matchup_edge` | `(away_lineup_bb_per_pa - away_lineup_so_per_pa) - ((home_sp_blend_BB9/9) - (home_sp_blend_K9/9))` | 원정 타선의 선구안/삼진 억제 vs 홈 선발 제구·탈삼진 상성 |
| `away_lineup_state_y_rate` | `away_lineup_state_y_cnt / away_lineup_known_cnt` | 원정 라인업에서 `lineupState == "Y"` 비율 |
| `away_lineup_state_y_cnt` | 원정 라인업 타자 중 `lineupState == "Y"` 개수 | 라인업 상태 플래그의 절대량 |
| `diff_lineup_time_split_ops` | `home_lineup_time_split_ops - away_lineup_time_split_ops` | 경기 시간대 분할 OPS 기준 홈-원정 타선 우위 |
| `away_sp_blend_K9` | 원정 선발의 `blend K/9` | 원정 선발 탈삼진 기대치(혼합 추정) |
| `diff_lineup_bb_per_pa` | `home_lineup_bb_per_pa - away_lineup_bb_per_pa` | 양수면 홈 타선 볼넷 생산률 우위 |
| `diff_lineup_prior_so_per_pa` | `home_lineup_prior_so_per_pa - away_lineup_prior_so_per_pa` | 양수면 홈 타선의 사전 추정 삼진률이 더 높음 |
| `diff_lineup_blend_ops` | `home_lineup_blend_ops - away_lineup_blend_ops` | 양수면 혼합 추정 OPS 기준 홈 타선 우위 |
| `diff_lineup_vs_throw_split_so_per_pa` | `home_lineup_vs_throw_split_so_per_pa - away_lineup_vs_throw_split_so_per_pa` | 상대 선발 투구손 유형 상대로의 삼진률 격차 |
| `diff_lineup_blend_so_per_pa` | `home_lineup_blend_so_per_pa - away_lineup_blend_so_per_pa` | 양수면 홈 타선 혼합 추정 삼진률이 더 높음 |
| `home_lineup_bot6_blend_ops` | 홈 라인업 4~9번 `blend OPS` 평균 | 하위 타선 화력(깊이) 지표 |
| `away_sp_prior_K9` | 원정 선발의 `prior K/9` | 원정 선발 탈삼진 능력의 안정화 추정치 |

참고:
- `count_matchup_edge`는 타자측 `(BB/PA - SO/PA)`와 투수측 `(BB9/9 - K9/9)`를 결합한 상성 지표입니다.
- `lineup_time_split_ops`는 타자의 경기 시간대(`day/night`) 분할 성적을 축소 추정해 평균한 값입니다.
- `lineup_vs_throw_split_so_per_pa`는 상대 선발의 좌/우 투구손 분할 히스토리를 반영한 삼진률입니다.

## 6) 해석 요약
- 모델의 중심 신호는 **타선(lineup) + 선발투수(starter_pitcher)** 조합이다.
- 중요도 상위에 `diff_*` 피처가 다수 올라, 절대값보다 **홈-원정 격차 신호**를 강하게 활용한다.
- 특히 `K9`, `lineup SO/BB`, `matchup edge` 계열이 예측력의 핵심 축이다.

## 7) 메모
- 리포트의 공식 성능 수치는 3번 표를 기준으로 사용.
- 중요도 표는 동일 설정 재학습 결과이며, 코드/데이터 미세 변경으로 소수점 단위 차이가 발생할 수 있다.
