"""
raw_lineup JSON? ??? ??-?-?? ??? long ???? ??? ??????.

???? `lineup_long.csv` ?
- ?? ??? 1~9? ??
- ????
- ??/?? ? ??
? ?? ?? ?? feature ??? ?? ??? ??.
"""

import os, csv, json, glob

RAW_LINEUP_DIR = os.path.expanduser("~/statiz/data/raw_lineup")
INDEX_CSV      = os.path.expanduser("~/statiz/data/game_index_played.csv")
OUT_CSV        = os.path.expanduser("~/statiz/data/lineup_long.csv")

def load_index():
    """????? ??? ?/?? ? ??? ??? ?? ?? ???? ???."""
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
    """??? raw? long ?? CSV? ????."""
    idx = load_index()
    files = sorted(glob.glob(os.path.join(RAW_LINEUP_DIR, "*.json")))

    rows = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 정상 응답만
        if isinstance(data, dict) and data.get("result_cd") != 100:
            continue

        # s_no는 파일명에서 추출 (예: 20220001.json)
        s_no = int(os.path.basename(fp).replace(".json", ""))
        meta = idx.get(s_no)
        if not meta:
            # played index에 없는 경기면 스킵
            continue

        homeTeam = meta["homeTeam"]
        awayTeam = meta["awayTeam"]
        date = meta["date"]

        # 팀 키(숫자 문자열)만 처리
        for team_key, players in data.items():
            if not (isinstance(team_key, str) and team_key.isdigit() and isinstance(players, list)):
                continue

            t_code = int(team_key)
            if t_code == homeTeam:
                side = "home"
            elif t_code == awayTeam:
                side = "away"
            else:
                side = "unknown"

            for p in players:
                rows.append({
                    "date": date,
                    "s_no": s_no,
                    "t_code": t_code,
                    "side": side,
                    "battingOrder": p.get("battingOrder"),   # "1"~"9" or "P"
                    "position": p.get("position"),
                    "starting": p.get("starting"),
                    "lineupState": p.get("lineupState"),
                    "p_no": p.get("p_no"),
                    "p_name": p.get("p_name"),
                    "p_bat": p.get("p_bat"),
                    "p_throw": p.get("p_throw"),
                    "p_backNumber": p.get("p_backNumber"),
                })

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
