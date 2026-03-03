\# PIPELINE.md — 데이터 수집/가공/백테스트 실행 순서



이 문서는 팀원이 \*\*“어떤 순서로 무엇을 실행하면 어떤 파일이 생기는지”\*\*를 한 번에 따라올 수 있도록 만든 실행 가이드입니다.  

가능하면 \*\*화이트리스트 IP가 등록된 EC2\*\*에서 실행하는 것을 전제로 합니다.



---



\## 0. 큰 그림(한 문장)

1\) 날짜별 스케줄로 전체 경기번호(s\_no) 목록을 만든 뒤  

2\) s\_no로 라인업/박스스코어를 수집하고  

3\) 라인업에 나온 선수만 골라 playerDay를 수집해서  

4\) “경기 1개 = 1행” 피처 테이블을 만든 뒤  

5\) expanding/online 방식으로 백테스트를 돌립니다.



---



\## 1. 사전 준비(필수)



\### 1) 실행 환경

\- Ubuntu(EC2) + Python3

\- 인터넷 연결(EC2에서 API 호출)



\### 2) IP 화이트리스트

STATIZ API는 허용된 IP에서만 접근 가능합니다.  

개인 PC에서 Postman을 쓰면 `403 허용되지 않은 IP`가 뜰 수 있으니, \*\*허용 IP(EC2)에서 실행\*\*하세요.



\### 3) API Key/Secret 설정(필수)

절대 코드에 하드코딩하지 말고 환경변수로만 설정합니다.



예시(터미널에서):

```bash

export STATIZ\_API\_KEY="..."

export STATIZ\_SECRET="..."

```



확인(값은 출력하지 않음):

```bash

python3 -c "import os; print('KEY OK' if os.getenv('STATIZ\_API\_KEY') else 'KEY NO'); print('SECRET OK' if os.getenv('STATIZ\_SECRET') else 'SECRET NO')"

```



---



\## 2. 데이터 저장 위치(중요)

이 파이프라인은 로컬(EC2) 파일 시스템에 데이터를 저장합니다.



\- 원본 JSON: `data/raw\_\*`

\- 가공 CSV: `data/\*.csv`

\- 로그: `logs/`



원본 데이터는 용량이 크고 규정 이슈가 있을 수 있어 \*\*깃에 올리지 않는 것을 권장\*\*합니다.



---



\## 3. 실행 단계(순서대로)



아래는 “한 단계씩 실행 → 결과 파일 확인” 흐름입니다.  

(스크립트 파일명은 현재 개발/실행에 사용한 이름 기준입니다.)



---



\### STEP A) 2022~2025 스케줄 수집 (날짜별)

목표: 날짜별 경기일정 원본 JSON 저장



\- 실행:

```bash

python3 download\_schedule.py

```



\- 결과:

&nbsp; - `data/raw\_schedule/YYYYMMDD.json` 가 2022~2025 전체 날짜로 저장됨

&nbsp; - 정상 저장 확인(개수 예시):

```bash

ls data/raw\_schedule | wc -l

```



---



\### STEP B) 경기 인덱스 생성 (s\_no 목록)

목표: 날짜별 JSON에서 경기번호를 뽑아 `game\_index.csv` 생성



\- 실행:

```bash

python3 build\_game\_index.py

```



\- 결과:

&nbsp; - `data/game\_index.csv` (전체 경기 목록)

&nbsp; - 여기에는 취소/기록없음/시범경기 등이 섞일 수 있음



---



\### STEP C) 실제 치러진 정규시즌만 필터링

목표: 모델링 가능한 “실제 경기”만 남긴 인덱스 생성



\- 필터 규칙(현재 적용):

&nbsp; - `state == 3` (정상 진행된 경기로 간주)

&nbsp; - score 존재

&nbsp; - `leagueType == 10100` (정규시즌)



\- 결과:

&nbsp; - `data/game\_index\_played.csv` (2870경기)



---



\### STEP D) 경기 상세 수집 (라인업 / 박스스코어)

목표: 각 경기번호(s\_no)로 라인업/박스스코어 원본 JSON 저장



\- 실행(테스트 20개):

```bash

python3 download\_game\_details.py --limit 20

```



\- 전체 실행(백그라운드 권장):

```bash

nohup python3 download\_game\_details.py --limit 0 > logs/game\_details.log 2>\&1 \&

```



\- 진행 확인:

```bash

ls data/raw\_lineup | wc -l

ls data/raw\_boxscore | wc -l

ps -ef | grep download\_game\_details.py | grep -v grep

```



\- 결과:

&nbsp; - `data/raw\_lineup/{s\_no}.json`

&nbsp; - `data/raw\_boxscore/{s\_no}.json`

&nbsp; - 최종적으로 둘 다 2870개여야 정상



---



\### STEP E) 라인업 테이블 생성

목표: 라인업 JSON을 “long 포맷” CSV로 변환



\- 실행:

```bash

python3 build\_lineup\_table.py

```



\- 결과:

&nbsp; - `data/lineup\_long.csv`

&nbsp; - 한 행은 대략 “한 경기의 한 선수(타순/포지션 포함)”을 의미

&nbsp; - battingOrder는 1~9 또는 P(투수)



---



\### STEP F) 필요한 선수만 골라 (p\_no, year) 목록 만들기

목표: 호출량을 줄이기 위해 “실제로 등장한 선수만” 수집 대상으로 잡음



\- 실행:

```bash

python3 build\_player\_year\_index.py

```



\- 결과:

&nbsp; - `data/player\_year\_index.csv` (p\_no-year 조합 1531개)



---



\### STEP G) playerDay 수집 (선수/연도별)

목표: `player\_year\_index.csv` 기준으로 `GET /prediction/playerDay` 수집



\- 테스트 20개:

```bash

python3 download\_playerday.py --limit 20

```



\- 전체 실행(백그라운드 권장):

```bash

nohup python3 download\_playerday.py --limit 0 > logs/playerday.log 2>\&1 \&

```



\- 진행 확인:

```bash

ls data/raw\_playerday | wc -l

ps -ef | grep download\_playerday.py | grep -v grep

```



\- 결과:

&nbsp; - `data/raw\_playerday/{p\_no}\_{year}.json`



> 참고: 일부 선수는 해당 연도 기록이 없어도 `result\_cd=100`만 내려오는 “빈 성공” JSON이 있을 수 있습니다. 이는 정상 케이스입니다.



---



\### STEP H) playerDay 테이블 생성(타자/투수)

목표: raw\_playerday JSON을 long CSV로 변환



\- 실행(구조 반영 버전 사용):

```bash

python3 build\_playerday\_tables\_v2.py

```



\- 결과:

&nbsp; - `data/playerday\_batter\_long.csv`

&nbsp; - `data/playerday\_pitcher\_long.csv`



---



\### STEP I) 피처 생성 (경기 1개 = 1행)

목표: 룩어헤드 없이 “전날까지” 누적 지표로 `features\_v0.csv` 생성



\- 실행:

```bash

python3 build\_features\_v0.py

```



\- 결과:

&nbsp; - `data/features\_v0.csv` (2870행)

&nbsp; - 주요 피처: 팀 전적 요약, 선발투수 누적, 라인업(1~9) 누적 평균/비율, 각종 diff



---



\### STEP J) 베이스라인 백테스트(online logistic regression)

목표: 파이프라인이 제대로 동작하는지 “끝까지” 확인하는 최소 기준



\- 실행:

```bash

python3 backtest\_v0\_online\_lr.py

```



\- 결과:

&nbsp; - `data/backtest\_pred\_v0.csv` (각 경기의 예측 확률)

&nbsp; - 현재는 성능이 낮을 수 있음(피처/모델이 매우 단순)



---



\## 4. 트러블슈팅(자주 막히는 것)



\### 1) 403 허용되지 않은 IP

\- 개인 PC에서 실행하면 403이 날 수 있음

\- 반드시 \*\*화이트리스트 등록된 EC2\*\*에서 실행



\### 2) SSH 끊김

\- SSH는 끊겨도 EC2 내부 파일은 남아있음

\- 긴 작업은 `nohup ... \&`로 실행하면 안전



\### 3) Key/Secret이 갑자기 없음

\- SSH 재접속하면 환경변수 세션이 초기화될 수 있음  

&nbsp; → `export STATIZ\_API\_KEY=...` 다시 설정



---



\## 5. 데이터 공유(팀 협업 팁)

\- raw JSON은 용량이 크고 규정 이슈가 있을 수 있어 깃에는 올리지 않는 것을 권장

\- 급하게 공유가 필요하면:

&nbsp; - `features\_v0.csv` 같은 “가공된 결과물”만 드라이브/카톡으로 전달

&nbsp; - 또는 레포를 Private로 바꾸고 접근 제한



---



\## 6. 다음 단계(실험 로드맵)

\- 피처 개선(전년도/커리어 prior, 최근 N경기 폼, 구장(s\_code), 날씨 등)

\- 모델 변경(LightGBM/CatBoost)

\- expanding window 평가를 더 정교하게(시즌별, 리그 변화 반영)

\- 제출 자동화(당일 스케줄→라인업→예측→POST 제출)

