"""
raw_playerday JSON? ?? ?? ??? ?? ????? ??? v2 ??????.

?? long ??? ??? ?? ??? ? ????.
??? ?? key ??? ???? ?????, ?? ?? ??? ?? ??/?? CSV? ??? ???.
"""

import os, csv, json, glob

RAW_DIR   = os.path.expanduser("~/statiz/data/raw_playerday")
INDEX_CSV = os.path.expanduser("~/statiz/data/game_index_played.csv")

OUT_BAT   = os.path.expanduser("~/statiz/data/playerday_batter_long.csv")
OUT_PIT   = os.path.expanduser("~/statiz/data/playerday_pitcher_long.csv")

# 엑셀(날짜별 선수 기록) + 실제 응답 키 기준
BATTER_COLS = [
    "date","s_no","p_no","year","homeTeam","awayTeam","side",
    "t_code","vs_tCode","awayTeam_rec","awayScore","homeScore","gameDate",
    "G","GS","PA","ePA","AB","R","H","1B","2B","3B","HR","TB","RBI","SB","CS","BB","HP","IB","SO","GDP","SH","SF",
    "AVG","OBP","SLG","OPS","NP","situation","battingOrder","position"
]

PITCHER_COLS = [
    "date","s_no","p_no","year","homeTeam","awayTeam","side",
    "t_code","vs_tCode","awayTeam_rec","awayScore","homeScore","gameDate",
    "G","GS","IP","R","rRA","ER","TBF","AB","H","2B","3B","HR","BB","IB","HP","SO","NP","TB","SF",
    "W","L","CG","SHO","S","HD","BS","BH",
    "WHIP","AVG","OBP","SLG","OPS","ERA","situation"
]

PITCHER_HINT_KEYS = {"IP","TBF","WHIP","ERA","W","L","S","HD","BS","SHO","CG"}

def load_game_index():
    """????? ??? ?/?? ? ??? ?? ?? ??."""
    idx = {}
    with open(INDEX_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            s_no = int(row["s_no"])
            idx[s_no] = {
                "date": row["date"],  # YYYYMMDD
                "homeTeam": int(row["homeTeam"]),
                "awayTeam": int(row["awayTeam"]),
            }
    return idx

def is_pitcher(rec: dict):
    """?? ?? ??? ??? ?? ???? ????."""
    return any(k in rec and rec.get(k) not in (None, "") for k in PITCHER_HINT_KEYS)

def pick(merged: dict, cols: list):
    """?? CSV? ?? ??? ?? dict? ????."""
    return {c: merged.get(c, None) for c in cols}

def iter_records(data: dict):
    """playerDay ???? ????(s_no)? ?? dict? ????."""
    """
    실제 API 응답 구조:
      {
        "20250003": { ... 기록 ... },
        "20250008": { ... },
        ...
        "result_cd":100, ...
      }
    """
    for k, v in data.items():
        if k in ("result_cd","result_msg","update_time"):
            continue
        if not (isinstance(k, str) and k.isdigit() and len(k) == 8):
            continue

        # 보통 v는 dict(한 경기 기록)인데 혹시 list면 확장 처리
        if isinstance(v, dict):
            yield int(k), v
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    yield int(k), item

def main():
    """v2 ???? playerDay raw? ??/?? long CSV? ????."""
    idx = load_game_index()
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))

    bat_rows, pit_rows = [], []

    for fp in files:
        base = os.path.basename(fp).replace(".json","")  # pno_year
        try:
            p_no_file, year_file = base.split("_")
            p_no_file = int(p_no_file)
            year_file = int(year_file)
        except Exception:
            continue

        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict) or data.get("result_cd") != 100:
            continue

        for s_no_key, rec in iter_records(data):
            # s_no는 rec에도 있지만, key가 더 믿을만함
            s_no = s_no_key

            meta = idx.get(s_no)
            if not meta:
                # (우리 데이터는 정규시즌만이라 거의 없겠지만) 그래도 안전하게 스킵
                continue

            homeTeam = meta["homeTeam"]
            awayTeam = meta["awayTeam"]
            date = meta["date"]

            # side 판정
            side = "unknown"
            try:
                t_code_int = int(rec.get("t_code"))
                if t_code_int == homeTeam:
                    side = "home"
                elif t_code_int == awayTeam:
                    side = "away"
            except Exception:
                pass

            merged = {
                **rec,
                "date": date,
                "s_no": s_no,
                "p_no": int(rec.get("p_no") or p_no_file),
                "year": year_file,
                "homeTeam": homeTeam,
                "awayTeam": awayTeam,
                "side": side,
                "awayTeam_rec": rec.get("awayTeam"),
            }

            if is_pitcher(rec):
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
