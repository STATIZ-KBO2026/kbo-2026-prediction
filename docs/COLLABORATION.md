# Collaboration Guide

## 이 프로젝트에서 역할을 나누는 법

헷갈리지 않게 아래처럼 역할을 고정합니다.

### 1) 로컬 GitHub clone
- 코드 읽기
- 코드 수정
- 문서 작성
- git add / commit / push

즉, 사람이 주로 보는 공간입니다.

### 2) GitHub
- 팀 협업의 공식 원본
- 브랜치, PR, 문서 리뷰 기준점

### 3) EC2 `~/statiz`
- 실제 실행 환경
- API 호출
- feature 생성
- 백테스트 실행
- data, logs 저장

즉, 계산과 실행의 공간입니다.

## 앞으로의 기본 원칙

- 코드는 GitHub repo `scripts/` 기준으로 관리한다.
- EC2에서 새로 생긴 코드는 가능한 빨리 GitHub repo `scripts/` 로 옮긴다.
- raw data, 대용량 csv, 비밀정보는 커밋하지 않는다.
- 문서 없이 코드만 올리지 않는다. 중요한 실험은 반드시 문서도 같이 남긴다.

## 브랜치 규칙

- `main`: 안정 버전만 둔다.
- 작업은 항상 `codex/...` 또는 팀 합의 prefix 브랜치에서 한다.
- 브랜치 하나에는 가능하면 작업 주제 하나만 담는다.

예시
- `codex/ec2-sync-v1-v2`
- `codex/coldstart-check`
- `codex/docs-rewrite`

## 커밋 규칙

좋은 커밋은 한 문장만 읽어도 무슨 변화인지 알 수 있어야 합니다.

예시
- `docs: rewrite README and project docs for team onboarding`
- `feat: add v2 candidate feature builder and subset search scripts`
- `chore: sync EC2 experiment scripts into repo scripts directory`

## PR에 꼭 적을 내용

1. 왜 이 작업을 했는지
2. 어떤 파일이 바뀌었는지
3. 실행 위치가 어디인지
- 로컬인지, EC2인지
4. 데이터 범위와 누수 방지 기준
- `2023+`, `D-1`
5. 결과 요약
- 주요 지표, 해석, 다음 액션

## 커밋하지 말아야 하는 것

- API Key, Secret
- `.env`
- `*.pem`
- `data/raw_*`
- 대용량 중간 산출물
- 개인 로컬 설정 파일

## 가장 쉬운 협업 흐름

1. 로컬 repo에서 브랜치 생성
2. 코드 수정
3. 문서 업데이트
4. git status 로 변경 파일 확인
5. 커밋
6. GitHub push
7. PR 생성
8. EC2 실행 결과를 문서에 반영

## 읽고 수정하기 쉬운 작업 방식

터미널 복붙만으로 계속 작업하면 나중에 코드 위치를 놓치기 쉽습니다.
앞으로는 아래 조합을 추천합니다.

- 코드 읽기/수정: VS Code
- git 관리: VS Code Source Control 또는 GitHub Desktop
- EC2 접속/실행: 터미널 또는 VS Code Remote-SSH
- CSV 확인: Excel 또는 pandas 출력

핵심은 이 한 줄입니다.
- 수정은 로컬 repo에서 하고, 실행은 EC2에서 하고, 공유는 GitHub에서 한다.
