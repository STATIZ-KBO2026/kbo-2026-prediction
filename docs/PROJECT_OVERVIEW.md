# Project Overview

## 프로젝트 한 줄 설명

STATIZ AI 승부예측대회용 KBO 경기 승부예측 모델을 만들고, 이를 재현 가능한 데이터 파이프라인과 백테스트 체계로 관리하는 프로젝트입니다.

## 현재 목표

현재 목표는 이미 돌아가는 기본 파이프라인 위에서 `좋은 피처 조합`을 찾는 것입니다.
즉, 지금은 데이터를 처음부터 다시 모으는 단계가 아니라, 다음 단계에 있습니다.

- 후보 피처를 여러 개 만든다.
- expanding window 백테스트로 공정하게 비교한다.
- LOGLOSS, BRIER, AUC 기준으로 좋은 조합을 고른다.
- 팀이 결과를 이해할 수 있게 문서화한다.

## 데이터 사용 규정

반드시 지켜야 하는 기준은 아래와 같습니다.

- `2022` 데이터는 사용하지 않는다.
- `2023` 데이터는 prior, smoothing, cold start 용으로만 사용 가능하다.
- 실제 feature output 및 학습/평가의 중심은 `2024~2025` 이다.
- 경기일 `D` 예측에는 반드시 `D-1`까지의 정보만 사용한다.

## 현재 데이터 흐름

1. `gameSchedule` 수집
2. `game_index.csv`, `game_index_played.csv` 생성
3. `gameLineup`, `gameBoxscore` 수집
4. `lineup_long.csv` 생성
5. `player_year_index.csv` 생성
6. `playerDay` 수집
7. `playerday_batter_long.csv`, `playerday_pitcher_long.csv` 생성
8. feature table 생성
9. expanding window 백테스트 실행

## 현재 기준 파일 위치

로컬 GitHub clone
- 코드와 문서의 공식 관리 위치
- `scripts/`, `docs/` 를 여기서 수정한다.

EC2 `~/statiz`
- 실제 수집/실험 실행 위치
- 원천 데이터와 대용량 결과는 여기 저장된다.
- 현재는 파이썬 스크립트가 root에 직접 존재한다.

이 차이 때문에 헷갈리기 쉬우므로, 앞으로는 아래 원칙으로 관리한다.
- GitHub repo `scripts/` 를 공식 코드 원본으로 본다.
- EC2 실행 전에는 GitHub 기준 코드와 맞춰 sync 한다.

## 현재 단계

이미 끝난 단계
- API 수집 파이프라인 구축
- baseline 모델 구축
- v1 피처 실험
- v2 후보 피처 탐색

지금 해야 하는 단계
- cold start 검증
- 상위 조합 해석
- 실험 재현 문서 정리
- 브랜치 기반 협업 정착
