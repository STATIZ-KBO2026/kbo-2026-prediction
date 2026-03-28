import os, json, glob, csv

RAW_DIR = os.path.expanduser("~/statiz/data/raw_schedule")
OUT_ALL_CSV = os.path.expanduser("~/statiz/data/game_index.csv")
OUT_PLAYED_CSV = os.path.expanduser("~/statiz/data/game_index_played.csv")
MIN_DATE = "20230101"  # 대회 규칙: 2023년 데이터부터 사용

def safe_int(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "none":
            return None
        return int(float(s))
    except Exception:
        return None

def is_played_row(row):
    # 결과 점수가 확정된 경기만 played로 간주
    hs = safe_int(row.get("homeScore"))
    aw = safe_int(row.get("awayScore"))
    return hs is not None and aw is not None

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def extract_schedule_games(data):
    # v4 spec shape: {"date": [ ...games... ], ...}
    date_block = data.get("date")
    if isinstance(date_block, list):
        return date_block
    if isinstance(date_block, dict):
        out = []
        for v in date_block.values():
            if isinstance(v, list):
                out.extend(v)
        if out:
            return out

    # legacy shape: {"MMDD": [ ...games... ], ...}
    for k, v in data.items():
        if isinstance(k, str) and k.isdigit() and len(k) == 4 and isinstance(v, list):
            return v
    return []

def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    os.makedirs(os.path.dirname(OUT_ALL_CSV), exist_ok=True)

    rows = []
    for fp in files:
        ymd = os.path.basename(fp).replace(".json", "")  # YYYYMMDD
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 정상 응답만
        if data.get("result_cd") != 100:
            continue

        games = extract_schedule_games(data)
        for g in games:
            # 예측에 유용한 경기 컨텍스트(날씨/환경 포함)를 함께 저장
            rows.append({
                "date": ymd,
                "s_no": g.get("s_no"),
                "state": g.get("s_state", g.get("state")),
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
                "weather": g.get("weather"),
                "temperature": g.get("temperature"),
                "humidity": g.get("humidity"),
                "windDirection": g.get("windDirection"),
                "windSpeed": g.get("windSpeed"),
                "rainprobability": g.get("rainprobability"),
            })

    # s_no 없는 행 제거 + date/s_no 기준 정렬
    rows = [r for r in rows if r["s_no"] is not None and str(r["date"]) >= MIN_DATE]
    rows.sort(key=lambda r: (r["date"], r["s_no"]))

    fieldnames = list(rows[0].keys()) if rows else [
        "date","s_no","state","leagueType","s_code","awayTeam","homeTeam",
        "awaySP","homeSP","awaySPName","homeSPName","awayScore","homeScore","hm","gameDate",
        "weather","temperature","humidity","windDirection","windSpeed","rainprobability"
    ]

    played_rows = [r for r in rows if is_played_row(r)]

    write_csv(OUT_ALL_CSV, rows, fieldnames)
    write_csv(OUT_PLAYED_CSV, played_rows, fieldnames)

    print("DONE")
    print("all_games=", len(rows), "out=", OUT_ALL_CSV)
    print("played_games=", len(played_rows), "out=", OUT_PLAYED_CSV)

if __name__ == "__main__":
    main()
