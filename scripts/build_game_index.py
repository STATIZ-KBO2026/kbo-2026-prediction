import os, json, glob, csv

RAW_DIR = os.path.expanduser("~/statiz/data/raw_schedule")
OUT_CSV = os.path.expanduser("~/statiz/data/game_index.csv")

def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    rows = []
    for fp in files:
        ymd = os.path.basename(fp).replace(".json", "")  # YYYYMMDD
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 정상 응답만
        if data.get("result_cd") != 100:
            continue

        # 날짜 키는 "0504" 같은 형태로 들어있음
        date_key = None
        for k in data.keys():
            if k.isdigit() and len(k) == 4:  # MMDD
                date_key = k
                break
        if not date_key:
            continue

        games = data.get(date_key, [])
        for g in games:
            # 필요한 것만 최소로 뽑음 (나중에 늘리면 됨)
            rows.append({
                "date": ymd,
                "s_no": g.get("s_no"),
                "state": g.get("state"),
                "leagueType": g.get("leagueType"),
                "s_code": g.get("s_code"),
                "awayTeam": g.get("awayTeam"),
                "homeTeam": g.get("homeTeam"),
                "awaySP": g.get("awaySP"),
                "homeSP": g.get("homeSP"),
                "awaySPName": g.get("awaySPName"),
                "homeSPName": g.get("homeSPName"),
                "awayScore": g.get("awayScore"),
                "homeScore": g.get("homeScore"),
                "hm": g.get("hm"),
                "gameDate": g.get("gameDate"),
            })

    # s_no 없는 행 제거 + date/s_no 기준 정렬
    rows = [r for r in rows if r["s_no"] is not None]
    rows.sort(key=lambda r: (r["date"], r["s_no"]))

    fieldnames = list(rows[0].keys()) if rows else [
        "date","s_no","state","leagueType","s_code","awayTeam","homeTeam",
        "awaySP","homeSP","awaySPName","homeSPName","awayScore","homeScore","hm","gameDate"
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("DONE", "games=", len(rows), "out=", OUT_CSV)

if __name__ == "__main__":
    main()
