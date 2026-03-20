"""
라인업 raw JSON → 라인업 long 테이블 CSV 변환기

download_game_details.py가 저장한 라인업 raw JSON들을 읽어서,
경기-팀-선수 단위로 풀어낸 long 형태의 CSV를 생성합니다.

이 CSV(lineup_long.csv)에는 각 경기의:
  - 타순 1~9번 타자 정보
  - 선발투수 정보
  - 홈/원정 구분

등이 들어 있어서, 이후 피처 생성 시 "어떤 선수가 몇 번 타순으로 출장했는지"를
조회하는 데 사용됩니다.

[파이프라인 위치]
  4단계 — download_game_details.py 이후에 실행합니다.

[입력]
  - ~/statiz/data/raw_lineup/*.json  (경기별 라인업 raw JSON)
  - ~/statiz/data/game_index_played.csv  (경기 메타정보: 날짜, 홈/원정팀)

[출력]
  - ~/statiz/data/lineup_long.csv
"""

import os, csv, json, glob

# 입력/출력 경로
RAW_LINEUP_DIR = os.path.expanduser("~/statiz/data/raw_lineup")
INDEX_CSV      = os.path.expanduser("~/statiz/data/game_index_played.csv")
OUT_CSV        = os.path.expanduser("~/statiz/data/lineup_long.csv")


def load_index():
    """
    game_index_played.csv에서 경기별 메타정보를 딕셔너리로 읽어옵니다.

    Returns:
        {s_no: {"date": "YYYYMMDD", "homeTeam": int, "awayTeam": int}} 형태의 dict
    """
    idx = {}
    with open(INDEX_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            s_no = int(row["s_no"])
            idx[s_no] = {
                "date": row["date"],                      # YYYYMMDD
                "homeTeam": int(row["homeTeam"]),
                "awayTeam": int(row["awayTeam"]),
            }
    return idx


def main():
    """라인업 raw JSON들을 읽어 long 형태 CSV로 저장합니다."""
    idx = load_index()
    files = sorted(glob.glob(os.path.join(RAW_LINEUP_DIR, "*.json")))

    rows = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        # result_cd가 100이 아니면 비정상 응답이므로 스킵
        if isinstance(data, dict) and data.get("result_cd") != 100:
            continue

        # 파일명에서 s_no 추출 (예: 20240001.json → 20240001)
        s_no = int(os.path.basename(fp).replace(".json", ""))
        meta = idx.get(s_no)
        if not meta:
            # game_index_played에 없는 경기면 스킵
            continue

        homeTeam = meta["homeTeam"]
        awayTeam = meta["awayTeam"]
        date = meta["date"]

        # API 응답에서 팀 코드(숫자 문자열)를 키로 쓰는 항목만 선수 목록임
        for team_key, players in data.items():
            if not (isinstance(team_key, str) and team_key.isdigit() and isinstance(players, list)):
                continue

            # 팀 코드로 홈/원정 구분
            t_code = int(team_key)
            if t_code == homeTeam:
                side = "home"
            elif t_code == awayTeam:
                side = "away"
            else:
                side = "unknown"

            # 선수 한 명씩 행으로 추가
            for p in players:
                rows.append({
                    "date": date,
                    "s_no": s_no,
                    "t_code": t_code,                           # 팀 코드
                    "side": side,                               # 홈/원정 구분
                    "battingOrder": p.get("battingOrder"),      # 타순 ("1"~"9" 또는 "P"=투수)
                    "position": p.get("position"),              # 포지션
                    "starting": p.get("starting"),              # 선발 여부
                    "lineupState": p.get("lineupState"),        # 라인업 상태
                    "p_no": p.get("p_no"),                      # 선수 고유번호
                    "p_name": p.get("p_name"),                  # 선수 이름
                    "p_bat": p.get("p_bat"),                    # 타석 (1=우타, 2=좌타, 3=스위치)
                    "p_throw": p.get("p_throw"),                # 투구 손 (1,2=우투, 3,4=좌투)
                    "p_backNumber": p.get("p_backNumber"),      # 등번호
                })

    # CSV 파일로 저장
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = [
        "date","s_no","t_code","side",
        "battingOrder","position","starting","lineupState",
        "p_no","p_name","p_bat","p_throw","p_backNumber"
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("DONE", "rows=", len(rows), "out=", OUT_CSV)

if __name__ == "__main__":
    main()
