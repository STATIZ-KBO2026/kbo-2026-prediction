import csv
import os
from collections import defaultdict

LINEUP_CSV = os.path.expanduser("~/statiz/data/lineup_long.csv")
OUT_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "manual_player_tags_candidates.csv"))


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
    players = defaultdict(lambda: {"name": "", "first_year": 9999, "pitcher_games": 0, "batter_games": 0})

    with open(LINEUP_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            p_no = safe_int(row.get("p_no"))
            if not p_no:
                continue
            p = players[p_no]
            nm = (row.get("p_name") or "").strip()
            if nm:
                p["name"] = nm
            y = safe_int(str(row.get("date"))[:4], 9999)
            if y and y < p["first_year"]:
                p["first_year"] = y
            bo = (row.get("battingOrder") or "").strip()
            pos = row.get("position")
            if is_pitcher_row(bo, pos):
                p["pitcher_games"] += 1
            else:
                p["batter_games"] += 1

    rows = []
    for p_no, v in players.items():
        if v["first_year"] >= 2024:
            rows.append(
                {
                    "p_no": p_no,
                    "p_name": v["name"],
                    "first_year": v["first_year"],
                    "pitcher_games": v["pitcher_games"],
                    "batter_games": v["batter_games"],
                    "is_foreign": "",
                    "is_rookie": "",
                    "note": "",
                }
            )

    rows.sort(key=lambda x: (x["first_year"], -(x["pitcher_games"] + x["batter_games"]), x["p_no"]))
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "p_no",
                "p_name",
                "first_year",
                "pitcher_games",
                "batter_games",
                "is_foreign",
                "is_rookie",
                "note",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print("DONE", "rows=", len(rows), "out=", OUT_CSV)


if __name__ == "__main__":
    main()
