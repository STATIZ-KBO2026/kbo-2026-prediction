import os, csv
from collections import defaultdict

IN_CSV  = os.path.expanduser("~/statiz/data/lineup_long.csv")
OUT_CSV = os.path.expanduser("~/statiz/data/player_year_index.csv")

def safe_int(x, default=0):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "" or s.lower() == "none":
            return default
        return int(float(s))
    except Exception:
        return default

def is_pitcher_row(batting_order, position):
    bo = str(batting_order or "").strip().upper()
    pos = safe_int(position, 0)
    return bo == "P" or pos == 1

def main():
    agg = {}  # (p_no, year) -> stats

    with open(IN_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            p_no = row.get("p_no")
            date = row.get("date")  # YYYYMMDD
            bo   = row.get("battingOrder")  # "1"~"9" or "P"
            pos  = row.get("position")

            if not p_no or not date or len(date) < 4:
                continue

            year = int(date[:4])
            key = (int(p_no), year)

            if key not in agg:
                agg[key] = {
                    "p_no": int(p_no),
                    "year": year,
                    "games_in_lineup": 0,
                    "first_date": date,
                    "last_date": date,
                    "has_pitcher": 0,
                    "has_batter": 0,
                }

            s = agg[key]
            s["games_in_lineup"] += 1
            if date < s["first_date"]:
                s["first_date"] = date
            if date > s["last_date"]:
                s["last_date"] = date

            if is_pitcher_row(bo, pos):
                s["has_pitcher"] = 1
            else:
                s["has_batter"] = 1

    rows = list(agg.values())
    rows.sort(key=lambda x: (x["year"], x["p_no"]))

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
