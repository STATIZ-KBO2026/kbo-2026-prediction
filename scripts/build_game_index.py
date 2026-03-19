"""
경기 일정 raw JSON → 경기 인덱스 CSV 변환기

download_schedule.py가 저장한 날짜별 JSON 파일들을 읽어서,
모든 경기를 한 줄씩 정리한 game_index.csv를 생성합니다.

이 CSV는 이후 스크립트들(라인업 수집, 피처 생성 등)에서
"어떤 경기가 있었는지"를 조회할 때 기준 테이블로 사용됩니다.

[파이프라인 위치]
  2단계 — download_schedule.py 다음에 실행합니다.

[입력]
  - ~/statiz/data/raw_schedule/*.json (날짜별 경기 일정 raw JSON)

[출력]
  - ~/statiz/data/game_index.csv
    컬럼: date, s_no, state, leagueType, s_code, awayTeam, homeTeam,
          awaySP, homeSP, awaySPName, homeSPName, awayScore, homeScore, hm, gameDate
"""

import os, json, glob, csv

# 입력/출력 경로
RAW_DIR = os.path.expanduser("~/statiz/data/raw_schedule")
OUT_CSV = os.path.expanduser("~/statiz/data/game_index.csv")


def main():
    """raw_schedule 폴더의 JSON들을 읽어 경기 인덱스 CSV를 만듭니다."""
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    rows = []
    for fp in files:
        # 파일명에서 날짜 추출 (예: 20240503.json → "20240503")
        ymd = os.path.basename(fp).replace(".json", "")

        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        # result_cd가 100이 아니면 비정상 응답이므로 스킵
        if data.get("result_cd") != 100:
            continue

        # 날짜 키 찾기: API 응답에서 "0503" 같은 4자리 숫자 키가 경기 목록을 담고 있음
        date_key = None
        for k in data.keys():
            if k.isdigit() and len(k) == 4:  # MMDD 형식
                date_key = k
                break
        if not date_key:
            continue

        # 해당 날짜의 경기 목록을 순회하며 필요한 필드만 추출
        games = data.get(date_key, [])
        for g in games:
            rows.append({
                "date": ymd,                           # 날짜 (YYYYMMDD)
                "s_no": g.get("s_no"),                 # 경기 고유번호
                "state": g.get("state"),               # 경기 상태 (예: 종료, 진행중)
                "leagueType": g.get("leagueType"),     # 리그 종류 (정규, 포스트 등)
                "s_code": g.get("s_code"),             # 구장 코드
                "awayTeam": g.get("awayTeam"),         # 원정팀 코드
                "homeTeam": g.get("homeTeam"),         # 홈팀 코드
                "awaySP": g.get("awaySP"),             # 원정 선발투수 번호
                "homeSP": g.get("homeSP"),             # 홈 선발투수 번호
                "awaySPName": g.get("awaySPName"),     # 원정 선발투수 이름
                "homeSPName": g.get("homeSPName"),     # 홈 선발투수 이름
                "awayScore": g.get("awayScore"),       # 원정팀 점수
                "homeScore": g.get("homeScore"),       # 홈팀 점수
                "hm": g.get("hm"),                     # 경기 시간
                "gameDate": g.get("gameDate"),         # API 기준 경기 날짜
            })

    # s_no(경기번호)가 없는 행은 제거하고, 날짜+경기번호 순으로 정렬
    rows = [r for r in rows if r["s_no"] is not None]
    rows.sort(key=lambda r: (r["date"], r["s_no"]))

    # CSV 헤더 설정: 데이터가 있으면 첫 번째 행의 키를 사용
    fieldnames = list(rows[0].keys()) if rows else [
        "date","s_no","state","leagueType","s_code","awayTeam","homeTeam",
        "awaySP","homeSP","awaySPName","homeSPName","awayScore","homeScore","hm","gameDate"
    ]

    # CSV 파일로 저장
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("DONE", "games=", len(rows), "out=", OUT_CSV)

if __name__ == "__main__":
    main()
