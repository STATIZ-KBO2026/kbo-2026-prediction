# PIPELINE — 데이터 수집/가공/백테스트 실행 순서

이 문서는 팀원이 **“어떤 순서로 무엇을 실행하면 어떤 파일이 생기는지”** 그대로 따라가도록 만든 실행 가이드입니다.

> ⚠️ 공지 기준: 데이터는 **2023년부터 사용 가능**  
> → 수집 기간을 2023~로 맞춰 실행하는 것을 권장합니다.

---

## 0) 한 장 요약 (Pipeline at a glance)

flowchart TD
  S["A. gameSchedule 날짜별 수집"] --> I["B. game_index_played.csv 생성"]
  I --> D["C. gameLineup / gameBoxscore 수집"]
  D --> L["D. lineup_long.csv 생성"]
  L --> P["E. player_year_index.csv 생성"]
  P --> R["F. playerDay (p_no, year) 수집"]
  R --> T["G. playerday_long CSV 생성"]
  T --> F["H. features_v0.csv 생성"]
  F --> B["I. expanding/online 백테스트"]

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

### STEP A) 스케줄 수집 (2023~ 권장)
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

> 수집 기간(2023~)은 `scripts/download_schedule.py` 내부의 `start/end`를 수정해 맞춥니다.

---

### STEP B) 경기 인덱스 생성 (s_no 목록)
목표: 경기번호 `s_no` 목록 생성 + 정규시즌/정상 경기 필터링

~~~bash
python3 scripts/build_game_index.py
~~~

결과:
- `data/game_index.csv`
- `data/game_index_played.csv`

확인:
~~~bash
head -n 3 data/game_index_played.csv
~~~

---

### STEP C) 경기 상세 수집 (라인업/박스스코어)
목표: 각 s_no로 라인업/박스스코어 원본 저장

테스트(20경기):
~~~bash
python3 scripts/download_game_details.py --limit 20
~~~

전체(백그라운드 권장):
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

테스트(20개):
~~~bash
python3 scripts/download_playerday.py --limit 20
~~~

전체(백그라운드 권장):
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