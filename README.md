# \# STATIZ KBO 2026 승부예측 (팀 프로젝트)

# 

# 이 레포는 \*\*STATIZ Prediction API\*\*로부터 데이터를 받아 \*\*백테스트 가능한 형태(경기 1개 = 1행)\*\*로 정리하고, 이후 모델 실험/제출까지 이어지는 파이프라인을 팀이 함께 관리하기 위한 공간입니다.

# 

# ---

# 

# \## 지금까지 어디까지 했나? (현재 진행상태)

# 

# 현재(2026-03-01 기준)까지 아래 단계까지 “재현 가능”하게 구현되어 있습니다.

# 

# 1\) \*\*(경기 인덱스) 2022~2025 정규시즌 경기 목록 구축\*\*

# \- `GET /prediction/gameSchedule` 를 날짜별로 호출해 원본 JSON 저장

# \- 여기서 경기번호 `s\_no`를 뽑아 `game\_index.csv` 생성

# \- “취소/기록 없음/시범경기” 등을 제외하고, 실제 치러진 정규시즌만 필터링하여  

# &nbsp; `game\_index\_played.csv` (총 \*\*2870경기\*\*) 생성

# 

# 2\) \*\*(경기 상세) 2870경기 전체 라인업/박스스코어 수집\*\*

# \- `GET /prediction/gameLineup` → `raw\_lineup/{s\_no}.json`

# \- `GET /prediction/gameBoxscore` → `raw\_boxscore/{s\_no}.json`

# \- 누락/에러 파일 없이 수집 완료

# 

# 3\) \*\*(테이블화) 라인업을 long 포맷 CSV로 변환\*\*

# \- `lineup\_long.csv` 생성 (경기/팀/타순/포지션/선수번호 등)

# 

# 4\) \*\*(선수 기록) 라인업 기반으로 필요한 선수만 골라 playerDay 수집\*\*

# \- 라인업에 실제로 등장한 선수들 기준으로 `(p\_no, year)` 목록 생성  

# &nbsp; `player\_year\_index.csv` (총 \*\*1531 조합\*\*, 선수 \*\*681명\*\*)

# \- `GET /prediction/playerDay` 를 해당 조합만 호출해 원본 JSON 저장

# \- 이를 테이블로 펼쳐:

# &nbsp; - `playerday\_batter\_long.csv` (약 7만 행)

# &nbsp; - `playerday\_pitcher\_long.csv` (약 1만 행)

# 

# 5\) \*\*(피처 생성) 룩어헤드 없는 베이스라인 피처 생성\*\*

# \- `features\_v0.csv` 생성 (\*\*2870행\*\*, 경기 1개 = 1행)

# \- 핵심 원칙: \*\*해당 경기 “전날까지”의 누적 기록만 사용\*\*(expanding/online 방식)

# 

# 6\) \*\*(베이스라인 백테스트)\*\*

# \- 온라인 로지스틱 회귀(매일 예측 → 그날 결과로 학습 업데이트) 베이스라인 실행 가능  

# &nbsp; 단, 현재 성능은 거의 랜덤(피처/모델 모두 매우 단순한 상태)

# 

# ---

# 

# \## 왜 Postman이 아니라 EC2에서 돌리나?

# 

# STATIZ API는 \*\*IP 화이트리스트(허용 IP)\*\* 기반으로 접근이 제한됩니다.  

# 즉, 팀에서 등록한 고정 IP(예: AWS EC2 Elastic IP)에서만 호출이 가능하고, 개인 PC에서 Postman으로 호출하면 `403 허용되지 않은 IP`가 뜰 수 있습니다.

# 

# 그래서 이 파이프라인은 기본적으로 \*\*화이트리스트에 등록된 서버(EC2)에서 실행\*\*하는 것을 전제로 작성되어 있습니다.

# 

# ---

# 

# \## 레포 구조(권장)

# 

# 이 레포는 “코드”만 버전관리하고, “데이터(원본/대용량)”는 로컬에서 생성하는 방식이 안전합니다.

# 

# ```

# scripts/     # 수집/가공/백테스트 스크립트

# docs/        # 실행 순서, 규칙, 운영 메모

# data/        # (로컬 생성물) raw\_\*/csv가 쌓이는 위치 (git에는 올리지 않음)

# logs/        # (로컬 생성물) nohup 실행 로그

# ```

# 

# > 주의: API Key/Secret, pem 키 파일, raw JSON 전체는 절대 커밋 금지

# 

# ---

# 

# \## 빠른 시작(팀원이 “바로 따라할 수 있게”)

# 

# 실행 순서는 `docs/PIPELINE.md`에 단계별로 자세히 적어두었습니다.  

# 핵심 흐름은 아래처럼 이해하면 됩니다.

# 

# 1\) 환경변수 설정  

# \- `STATIZ\_API\_KEY`, `STATIZ\_SECRET` 를 \*\*환경변수로만\*\* 설정 (코드에 하드코딩 금지)

# 

# 2\) 경기 인덱스 생성  

# \- `download\_schedule.py` → `build\_game\_index.py` → `game\_index\_played.csv`

# 

# 3\) 경기 상세 수집(라인업/박스스코어)  

# \- `download\_game\_details.py`

# 

# 4\) 라인업/선수기록 테이블 생성  

# \- `build\_lineup\_table.py`  

# \- `build\_player\_year\_index.py`  

# \- `download\_playerday.py`  

# \- `build\_playerday\_tables\_v2.py`

# 

# 5\) 피처 생성 \& 백테스트  

# \- `build\_features\_v0.py`  

# \- `backtest\_v0\_online\_lr.py`

# 

# ---

# 

# \## 보안/협업 규칙(사고 방지)

# 

# \- \*\*절대 커밋 금지\*\*

# &nbsp; - API Key/Secret

# &nbsp; - `.pem` 키 파일

# &nbsp; - `data/raw\_\*` (원본 JSON)

# &nbsp; - 대용량 csv(필요 시 따로 공유)

# \- 데이터 공유가 급하면:

# &nbsp; - `features\_v0.csv` 같은 “가공된 결과물”만 카톡/드라이브로 공유하거나

# &nbsp; - 레포를 Private로 전환 후 제한적으로 공유 권장

# 

# ---

# 

# \## 다음 할 일(TODO)

# 

# 현재 파이프라인은 “수집/정리/룩어헤드 없는 피처 생성”까지 완료된 상태입니다.  

# 다음은 모델 성능을 올리는 실험 단계로 넘어갑니다.

# 

# \- 피처 개선

# &nbsp; - 시즌 초반/신인/복귀선수 문제를 위한 \*\*전년도 prior / 커리어 prior\*\*

# &nbsp; - 최근 N경기 폼(rolling), 홈/원정 스플릿, 구장 효과(s\_code), 날씨 등

# &nbsp; - 불펜/수비/팀 컨텍스트 추가(가능 범위에서)

# \- 모델

# &nbsp; - LightGBM / CatBoost 등 트리 기반 모델

# &nbsp; - 캘리브레이션(확률 예측 안정화)

# \- 평가

# &nbsp; - expanding window / 시계열 분할을 더 정교하게 설계

# &nbsp; - 최종 제출 API(POST) 자동화

# 

# ---

# 

# \## 문의/연락

# 파이프라인 실행이 막히면(특히 IP/키/권한 문제) `docs/PIPELINE.md`의 “트러블슈팅”을 먼저 확인해주세요.

