"""
라인업 기반 선수-연도 인덱스 생성기

lineup_long.csv에서 각 선수(p_no)가 연도별로 몇 경기에 출장했는지를 집계하여
player_year_index.csv를 생성합니다.

이 인덱스는 다음 단계(download_playerday.py)에서 어떤 선수-연도 조합의
개인 기록(playerDay)을 다운로드할지 결정하는 기준 목록으로 사용됩니다.

[파이프라인 위치]
  5단계 — build_lineup_table.py 이후에 실행합니다.

[입력]
  - ~/statiz/data/lineup_long.csv  (경기-팀-선수 long 테이블)

[출력]
  - ~/statiz/data/player_year_index.csv
    컬럼: p_no, year, games_in_lineup, first_date, last_date, has_pitcher, has_batter
"""

import os, csv
from collections import defaultdict

# 입력/출력 경로
IN_CSV  = os.path.expanduser("~/statiz/data/lineup_long.csv")
OUT_CSV = os.path.expanduser("~/statiz/data/player_year_index.csv")


def main():
    """라인업 데이터를 집계하여 선수-연도 인덱스를 만듭니다."""

    # (p_no, year) 조합별로 출장 기록을 집계할 딕셔너리
    agg = {}  # key: (p_no, year) → value: 집계 결과 dict

    with open(IN_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            p_no = row.get("p_no")
            date = row.get("date")  # YYYYMMDD
            bo   = row.get("battingOrder")  # "1"~"9" 또는 "P" (투수)

            if not p_no or not date or len(date) < 4:
                continue

            year = int(date[:4])
            key = (int(p_no), year)

            # 처음 나타난 선수-연도 조합이면 초기화
            if key not in agg:
                agg[key] = {
                    "p_no": int(p_no),
                    "year": year,
                    "games_in_lineup": 0,       # 라인업에 등장한 경기 수
                    "first_date": date,          # 시즌 첫 출장 날짜
                    "last_date": date,           # 시즌 마지막 출장 날짜
                    "has_pitcher": 0,            # 투수로 출장한 기록이 있는지 (0 또는 1)
                    "has_batter": 0,             # 타자로 출장한 기록이 있는지 (0 또는 1)
                }

            s = agg[key]
            s["games_in_lineup"] += 1

            # 첫/마지막 출장 날짜 갱신
            if date < s["first_date"]:
                s["first_date"] = date
            if date > s["last_date"]:
                s["last_date"] = date

            # 투수/타자 구분 기록
            if bo == "P":
                s["has_pitcher"] = 1
            else:
                s["has_batter"] = 1

    # 연도 → 선수번호 순으로 정렬
    rows = list(agg.values())
    rows.sort(key=lambda x: (x["year"], x["p_no"]))

    # CSV 저장
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = ["p_no","year","games_in_lineup","first_date","last_date","has_pitcher","has_batter"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    unique_players = len(set(r["p_no"] for r in rows))
    print("DONE", "player_year_rows=", len(rows), "unique_players=", unique_players, "out=", OUT_CSV)

if __name__ == "__main__":
    main()
