# PIPELINE — 데이터 수집/가공/백테스트 실행 순서

이 문서는 팀원이 **“어떤 순서로 무엇을 실행하면 어떤 파일이 생기는지”** 그대로 따라가도록 만든 실행 가이드입니다.

> ⚠️ 공지 기준: 데이터는 **2023년부터 사용 가능**  
> → 수집 기간을 2023~로 맞춰 실행하는 것을 권장합니다.

---

## 0) 한 장 요약 (Pipeline at a glance)

~~~mermaid
flowchart TD
  S["A. gameSchedule 날짜별 수집"] --> I["B. game_index.csv 생성"]
  I --> D["C. gameLineup / gameBoxscore 수집"]
  D --> L["D. lineup_long.csv 생성"]
  L --> P["E. player_year_index.csv 생성"]
  P --> R["F. playerDay (p_no, year) 수집"]
  R --> T["G. playerday_long CSV 생성"]
  T --> F["H. features_v0.csv 생성"]
  F --> B["I. expanding/online 백테스트"]
~~~

---

## 1) 사전 준비(필수)

### 1-1) 실행 환경
- Ubuntu(EC2) + Python3
- 인터넷 연결(EC2에서 API 호출)

### 1-2) IP 화이트리스트 주의
STATIZ API는 허용 IP에서만 접근 가능할 수 있습니다.  
개인 PC에서 Postman으로 호출하면 `403 허용되지 않은 IP`가 날 수 있으니 **등록된 EC2에서 실행**을 권장합니다.

### 1-3) API Key/Secret 설정(필수)
절대 코드에 하드코딩하지 말고 환경변수로만 설정합니다.

~~~bash
export STATIZ_API_KEY="..."
export STATIZ_SECRET="..."
python3 -c "import os; print('KEY OK' if os.getenv('STATIZ_API_KEY') else 'KEY NO'); print('SECRET OK' if os.getenv('STATIZ_SECRET') else 'SECRET NO')"
~~~

---

## 2) 데이터 저장 위치(중요)

- 원본 JSON: `data/raw_*`
- 가공 CSV: `data/*.csv`
- 로그: `logs/`

원본 데이터는 용량/규정 이슈가 있을 수 있어 **깃에 올리지 않는 것을 권장**합니다.

---

## 3) 실행 단계 (Step-by-step)

아래는 “한 단계 실행 → 결과 확인” 방식입니다.

---

### STEP A) 스케줄 수집 (기본: 2023-01-01 ~ 오늘)
목표: 날짜별 `gameSchedule` 원본 JSON 저장

~~~bash
python3 scripts/download_schedule.py
~~~

결과:
- `data/raw_schedule/YYYYMMDD.json`

확인:
~~~bash
ls data/raw_schedule | wc -l
~~~

> 기본값은 `2023-01-01`부터 오늘까지입니다. 필요하면 `scripts/download_schedule.py`의 `start/end`를 수정하세요.

---

### STEP B) 경기 인덱스 생성 (s_no 목록)
목표: 경기번호 `s_no` 목록 생성 + played 경기 인덱스 분리

~~~bash
python3 scripts/build_game_index.py
~~~

결과:
- `data/game_index.csv`
- `data/game_index_played.csv`

설명:
- `game_index.csv`: 스케줄에서 수집된 전체 경기 인덱스
- `game_index_played.csv`: 점수가 존재하는(종료된) 경기 인덱스

확인:
~~~bash
head -n 3 data/game_index_played.csv
~~~

---

### STEP C) 경기 상세 수집 (라인업/박스스코어)
목표: 각 s_no로 라인업/박스스코어 원본 저장

기본(전체):
~~~bash
python3 scripts/download_game_details.py
~~~

테스트(20경기만):
~~~bash
python3 scripts/download_game_details.py --limit 20
~~~

전체 백그라운드 실행:
~~~bash
nohup python3 scripts/download_game_details.py --limit 0 > logs/game_details.log 2>&1 &
~~~

진행 확인:
~~~bash
ls data/raw_lineup | wc -l
ls data/raw_boxscore | wc -l
ps -ef | grep download_game_details.py | grep -v grep
~~~

---

### STEP D) 라인업 테이블 생성
목표: 라인업 JSON → long CSV 변환

~~~bash
python3 scripts/build_lineup_table.py
~~~

결과:
- `data/lineup_long.csv`

확인:
~~~bash
head -n 3 data/lineup_long.csv
~~~

---

### STEP E) (p_no, year) 인덱스 생성
목표: 호출량을 줄이기 위해 “실제로 라인업에 등장한 선수만” 대상으로 playerDay 수집 준비

~~~bash
python3 scripts/build_player_year_index.py
~~~

결과:
- `data/player_year_index.csv`

확인:
~~~bash
head -n 3 data/player_year_index.csv
~~~

---

### STEP F) playerDay 수집
목표: `player_year_index.csv` 기반으로 playerDay raw 저장

기본(전체):
~~~bash
python3 scripts/download_playerday.py
~~~

테스트(20개만):
~~~bash
python3 scripts/download_playerday.py --limit 20
~~~

전체 백그라운드 실행:
~~~bash
nohup python3 scripts/download_playerday.py --limit 0 > logs/playerday.log 2>&1 &
~~~

진행 확인:
~~~bash
ls data/raw_playerday | wc -l
ps -ef | grep download_playerday.py | grep -v grep
~~~

---

### STEP G) playerDay 테이블 생성(타자/투수)
목표: raw_playerday JSON → long CSV 변환

~~~bash
python3 scripts/build_playerday_tables_v2.py
~~~

결과:
- `data/playerday_batter_long.csv`
- `data/playerday_pitcher_long.csv`

확인:
~~~bash
wc -l data/playerday_batter_long.csv
wc -l data/playerday_pitcher_long.csv
~~~

---

### STEP H) 피처 생성 (경기 1개 = 1행, 룩어헤드 방지)
목표: 해당 경기 “전날까지” 누적 기록만 사용해 피처 생성

~~~bash
python3 scripts/build_features_v0.py
~~~

결과:
- `data/features_v0.csv`
- 미종료 경기(예: 당일 경기)는 `y_home_win`이 빈 값으로 저장되며, 추론/제출용으로 사용

확인:
~~~bash
head -n 3 data/features_v0.csv
wc -l data/features_v0.csv
~~~

---

### STEP I) 베이스라인 백테스트
목표: 파이프라인이 끝까지 잘 도는지 확인 + baseline metric 확보

~~~bash
python3 scripts/backtest_v0_online_lr.py
~~~

결과:
- `data/backtest_pred_v0.csv`

---

### STEP J) 고급 백테스트(노이즈 분석 + 모델 비교 + 챔피언 모델)

노이즈 피처 리포트:
~~~bash
python3 scripts/analyze_feature_noise.py --train-start 20240101 --train-end 20241231 --test-start 20250101 --test-end 20251231
~~~

모델 Zoo 비교:
~~~bash
python3 scripts/backtest_v3_model_zoo.py --train-start 20240101 --train-end 20241231 --test-start 20250101 --test-end 20251231
~~~

챔피언 RF 백테스트:
~~~bash
python3 scripts/backtest_v4_champion_rf.py --train-start 20240101 --train-end 20241231 --test-start 20250101 --test-end 20251231
~~~

산출물:
- `data/feature_noise_report.csv`
- `data/backtest_v3_model_report.csv`
- `data/backtest_pred_v3_best.csv`
- `data/backtest_pred_v4_champion_rf.csv`

---

### STEP K) 모델 결과 제출 (`savePrediction` POST)
목표: 모델 예측 CSV를 `/prediction/savePrediction`에 전송

`statiz_prediction_v4.xlsx`의 **승부예측 결과 입력** 시트 기준:
- 요청 필드: `s_no`, `percent`
- 응답 필드: `cdoe`, `result_msg`

사전 점검(dry-run):
~~~bash
python3 scripts/submit_predictions.py --in-csv ~/statiz/data/backtest_pred_v5_best.csv --dry-run --limit 5
~~~

실제 제출:
~~~bash
python3 scripts/submit_predictions.py --in-csv ~/statiz/data/backtest_pred_v5_best.csv
~~~

재실행 시 기존 성공건 스킵:
~~~bash
python3 scripts/submit_predictions.py --in-csv ~/statiz/data/backtest_pred_v5_best.csv --resume
~~~

특정 날짜만 제출(예: 2026-03-18 경기):
~~~bash
python3 scripts/submit_predictions.py --in-csv ~/statiz/data/backtest_pred_v5_best.csv --date 20260318
~~~

결과:
- `~/statiz/data/save_prediction_result.csv`
- 컬럼: `date`, `s_no`, `percent`, `cdoe`, `result_msg`, `http_status`, `ok`, `error`, `raw_body`

원커맨드 파이프라인(백테스트 생성 + 제출):
~~~bash
python3 scripts/run_submit_pipeline_v5.py --dry-run --date 20260318
~~~

시범경기 기반 단순 파이프라인 테스트(제출 전 검증용):
~~~bash
python3 scripts/run_submit_pipeline_v5.py --season-mode exhibition --pipeline-test --dry-run --date 20260318
~~~

---

## 4) 트러블슈팅

### 4-1) 403 허용되지 않은 IP
- PC에서 실행하면 403이 날 수 있음
- **화이트리스트 등록된 EC2에서 실행**

### 4-2) SSH 끊김
- SSH 끊겨도 EC2 내부 파일은 유지됨
- 긴 작업은 `nohup ... &`로 실행

### 4-3) 환경변수 사라짐
- SSH 재접속 후 `STATIZ_API_KEY/SECRET`가 사라질 수 있음  
  → 다시 export 후 진행

---

## 5) 다음 단계(성능 개선)
- 전년도/커리어 prior로 시즌 초반 결측 보완
- 최근 N경기 rolling 피처
- 구장(s_code)/날씨/홈원정 스플릿
- 모델: LightGBM/CatBoost + 확률 캘리브레이션
