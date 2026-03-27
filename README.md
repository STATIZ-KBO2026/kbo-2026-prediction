# 🧢 STATIZ KBO 2026 승부예측 (팀 프로젝트)

**STATIZ Prediction API → 데이터 파이프라인 → 백테스트 → 제출**까지 팀이 같은 기준으로 재현 가능하게 만들기 위한 레포입니다.

> ⚠️ 공지 기준: **대회 데이터는 2023년부터 사용 가능**  
> → 파이프라인의 기본 수집 기간도 2023~로 맞춰 실행하는 것을 권장합니다.

---

## 🔎 한눈에 보기 (TL;DR)

이 레포에서 할 수 있는 것

- ✅ STATIZ Prediction API 데이터 **자동 수집**
- ✅ raw JSON → 분석 가능한 **테이블(CSV) 변환**
- ✅ **경기 1개 = 1행** 피처 테이블 생성 (룩어헤드 방지)
- ✅ **expanding/online** 방식 백테스트로 재현 가능한 실험

---

## 🗺️ Project Map (파이프라인 맵)

~~~mermaid
flowchart LR
  A["STATIZ Prediction API"] --> B["raw JSON 저장\n(raw_schedule / raw_lineup / raw_boxscore / raw_playerday)"]
  B --> C["테이블화\n(game_index / lineup_long / playerday_long)"]
  C --> D["피처 생성\n(features_v0.csv)"]
  D --> E["백테스트\n(expanding/online)"]
  E --> F["제출 자동화\n(POST prediction)"]
~~~

---

## 📦 레포 구성 (Repository Layout)

- `scripts/` : 수집/가공/백테스트 코드
- `docs/` : 실행 가이드(PIPELINE), 운영 규칙
- `data/`, `logs/` : **로컬(EC2)에서 생성되는 산출물 위치** (기본적으로 깃에 커밋 X)

---

## ✅ 현재 진행상태 (체크리스트)

- [x] 스케줄 기반 경기 인덱스 생성 (`gameSchedule → game_index.csv`)
- [x] 경기 상세 수집 (`gameLineup`, `gameBoxscore`)
- [x] 라인업 테이블 생성 (`lineup_long.csv`)
- [x] playerDay 수집(라인업 등장 선수만) + 테이블화
- [x] 룩어헤드 없는 피처 테이블 생성 (`features_v0.csv`)
- [x] 베이스라인 백테스트 실행(파이프라인 검증용)
- [x] 노이즈 피처 자동 분석/드롭 규칙 적용
- [x] 모델 Zoo 비교(선형/부스팅/포레스트/NB)
- [x] 챔피언 RF 모델 백테스트 스크립트 고정
- [x] 모델 예측 결과 제출 스크립트 추가 (`savePrediction` POST)

- [ ] 피처 개선(전년도/커리어 prior, 최근 N경기 폼 등)
- [ ] 외부 라이브러리 기반 모델(LightGBM/CatBoost) 추가 검증
- [ ] 제출 자동화(당일 스케줄 → 라인업 → 예측 → POST)

---

## 🚀 Quick Start (팀원이 바로 실행)

> 전제: STATIZ API는 **허용 IP(화이트리스트)** 기반일 수 있어 **EC2(등록된 고정 IP)에서 실행** 권장  
> 개인 PC(Postman)로는 403이 날 수 있습니다.

### 1) Key/Secret 환경변수 설정 (필수)
~~~bash
export STATIZ_API_KEY="..."
export STATIZ_SECRET="..."
python3 -c "import os; print('KEY OK' if os.getenv('STATIZ_API_KEY') else 'KEY NO'); print('SECRET OK' if os.getenv('STATIZ_SECRET') else 'SECRET NO')"
~~~

### 2) 실행 순서
👉 `docs/PIPELINE.md`를 그대로 따라가면 됩니다.

---

## 🧰 스크립트 목록 (팀원이 실제로 쓰는 것)

| 목적 | 스크립트 | 주요 산출물 |
|---|---|---|
| 날짜별 스케줄 raw 저장 | `scripts/download_schedule.py` | `data/raw_schedule/YYYYMMDD.json` |
| 경기 인덱스 생성 | `scripts/build_game_index.py` | `data/game_index.csv`, `data/game_index_played.csv` |
| 라인업/박스스코어 raw 저장 | `scripts/download_game_details.py` | `data/raw_lineup/*.json`, `data/raw_boxscore/*.json` |
| 라인업 테이블 생성 | `scripts/build_lineup_table.py` | `data/lineup_long.csv` |
| (p_no,year) 인덱스 생성 | `scripts/build_player_year_index.py` | `data/player_year_index.csv` |
| playerDay raw 저장 | `scripts/download_playerday.py` | `data/raw_playerday/*.json` |
| playerDay 테이블 생성 | `scripts/build_playerday_tables_v2.py` | `data/playerday_*_long.csv` |
| 피처 생성(베이스라인) | `scripts/build_features_v0.py` | `data/features_v0.csv` |
| 베이스라인 백테스트 | `scripts/backtest_v0_online_lr.py` | `data/backtest_pred_v0.csv` |
| 노이즈 피처 분석 | `scripts/analyze_feature_noise.py` | `data/feature_noise_report.csv` |
| 모델 비교(Model Zoo) | `scripts/backtest_v3_model_zoo.py` | `data/backtest_v3_model_report.csv` |
| 챔피언 RF 백테스트 | `scripts/backtest_v4_champion_rf.py` | `data/backtest_pred_v4_champion_rf.csv` |
| 모델 예측 제출(POST) | `scripts/submit_predictions.py` | `~/statiz/data/save_prediction_result.csv` |
| 제출 파이프라인(생성+제출) | `scripts/run_submit_pipeline_v5.py` | `~/statiz/data/backtest_pred_v5_best.csv`, `~/statiz/data/save_prediction_result.csv` |

---

## 🔒 보안 / 협업 규칙 (중요)

**절대 커밋 금지**
- API Key/Secret, `.env`, `.pem`
- `data/raw_*` (원본 JSON)
- 대용량 산출물 CSV(원칙적으로 로컬 생성)

데이터 공유가 급하면:
- `features_v0.csv` 같은 “가공된 소형 결과물”만 별도로 공유(드라이브/카톡 등) 권장

---

## 🧭 다음 실험 방향 (Roadmap)

성능을 올리려면 보통 아래 순서가 효율적입니다.

1) 시즌 초반/신인/복귀 선수 처리  
   - 전년도/커리어 prior로 “기록 없음” 구간 보완
2) 최근 N경기 폼(rolling), 홈/원정, 구장(s_code), 날씨 등 컨텍스트
3) 모델: LightGBM/CatBoost + 확률 캘리브레이션
4) expanding window 평가를 시즌 단위로 더 정교하게

---

## 🙋‍♀️ 도움이 필요하면
막히는 지점이 생기면 `docs/PIPELINE.md`의 트러블슈팅을 먼저 확인하고, 그래도 안 되면 에러 로그/상황을 팀 채팅에 공유해주세요.
