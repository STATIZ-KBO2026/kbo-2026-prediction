import os, csv, json, glob

RAW_DIR   = os.path.expanduser("~/statiz/data/raw_playerday")
INDEX_CSV = os.path.expanduser("~/statiz/data/game_index_played.csv")

OUT_BAT   = os.path.expanduser("~/statiz/data/playerday_batter_long.csv")
OUT_PIT   = os.path.expanduser("~/statiz/data/playerday_pitcher_long.csv")

PITCHER_HINT_KEYS = {"IP","TBF","WHIP","ERA","W","L","S","HD","BS","SHO","CG"}

BATTER_COLS = [
    "date","s_no","p_no","year","t_code","vs_tCode","side","homeTeam","awayTeam",
    "awayTeam_rec","awayScore","homeScore","gameDate",
    "G","GS","PA","ePA","AB","H","2B","3B","HR","TB","RBI","SB","CS","BB","HP","IB","SO","GDP","SH","SF",
    "AVG","OBP","SLG","OPS","NP","situation","battingOrder","position"
]

PITCHER_COLS = [
    "date","s_no","p_no","year","t_code","vs_tCode","side","homeTeam","awayTeam",
    "awayTeam_rec","awayScore","homeScore","gameDate",
    "G","GS","IP","R","rRA","ER","TBF","AB","H","2B","3B","HR","BB","IB","HP","SO","NP","TB","SF",
    "W","L","CG","SHO","S","HD","BS","BH",
    "WHIP","AVG","OBP","SLG","OPS","ERA","situation"
]

def load_game_index():
    idx = {}
    with open(INDEX_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            s_no = int(row["s_no"])
            idx[s_no] = {
                "date": row["date"],
                "homeTeam": int(row["homeTeam"]),
                "awayTeam": int(row["awayTeam"]),
            }
    return idx

def get_records(obj: dict):
    # 스펙상: {"s_no":[{...}, {...}], "result_cd":100, ...}
    arr = obj.get("s_no")
    if isinstance(arr, list):
        return arr
    # 혹시 구조가 다를 때 대비: list value 중 s_no 필드가 있는 것 찾기
    for v in obj.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and ("s_no" in v[0]):
            return v
    return None  # 데이터 없음

def is_pitcher(rec: dict):
    return any(k in rec and rec.get(k) not in (None, "") for k in PITCHER_HINT_KEYS)

def pick(rec: dict, cols: list):
    return {c: rec.get(c, None) for c in cols}

def main():
    idx = load_game_index()
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))

    bat_rows = []
    pit_rows = []

    for fp in files:
        base = os.path.basename(fp).replace(".json","")  # pno_year
        try:
            p_no_str, year_str = base.split("_")
            p_no_file = int(p_no_str)
            year_file = int(year_str)
        except Exception:
            continue

        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict) or data.get("result_cd") != 100:
            continue

        recs = get_records(data)
        if not recs:
            continue  # ✅ 이게 바로 "성공인데 데이터 없는" 케이스(작은 파일)

        for r in recs:
            if not isinstance(r, dict):
                continue

            s_no = r.get("s_no")
            if s_no is None:
                continue
            try:
                s_no = int(s_no)
            except Exception:
                continue

            meta = idx.get(s_no)
            if not meta:
                continue

            homeTeam = meta["homeTeam"]
            awayTeam = meta["awayTeam"]
            t_code = r.get("t_code")
            side = "unknown"
            try:
                if int(t_code) == homeTeam:
                    side = "home"
                elif int(t_code) == awayTeam:
                    side = "away"
            except Exception:
                pass

            common = {
                "date": meta["date"],
                "s_no": s_no,
                "p_no": int(r.get("p_no") or p_no_file),
                "year": year_file,
                "t_code": r.get("t_code"),
                "vs_tCode": r.get("vs_tCode"),
                "side": side,
                "homeTeam": homeTeam,
                "awayTeam": awayTeam,
                "awayTeam_rec": r.get("awayTeam"),
                "awayScore": r.get("awayScore"),
                "homeScore": r.get("homeScore"),
                "gameDate": r.get("gameDate"),
            }

            # 나머지 필드들 합치기
            merged = {**r, **common}

            if is_pitcher(r):
                pit_rows.append(pick(merged, PITCHER_COLS))
            else:
                bat_rows.append(pick(merged, BATTER_COLS))

    os.makedirs(os.path.dirname(OUT_BAT), exist_ok=True)

    with open(OUT_BAT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BATTER_COLS)
        w.writeheader()
        w.writerows(bat_rows)

    with open(OUT_PIT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PITCHER_COLS)
        w.writeheader()
        w.writerows(pit_rows)

    print("DONE", "batter_rows=", len(bat_rows), "pitcher_rows=", len(pit_rows))
    print("OUT", OUT_BAT)
    print("OUT", OUT_PIT)

if __name__ == "__main__":
    main()
