import os, csv, json, glob, argparse

RAW_LINEUP_DIR = os.path.expanduser("~/statiz/data/raw_lineup")
INDEX_CSV      = os.path.expanduser("~/statiz/data/game_index.csv")
OUT_CSV        = os.path.expanduser("~/statiz/data/lineup_long.csv")

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

def iter_team_players(data: dict):
    # legacy shape: {"5002": [players...], "7002":[players...], ...}
    team_rows = []
    for team_key, players in data.items():
        if team_key in ("result_cd", "result_msg", "update_time"):
            continue
        if isinstance(team_key, str) and team_key.isdigit() and isinstance(players, list):
            team_rows.append((int(team_key), players))
    if team_rows:
        for item in team_rows:
            yield item
        return

    # possible v4 shapes: {"t_code":[{...}, ...]} or {"t_code":{"5002":[...], ...}}
    t_code_block = data.get("t_code", data.get("t_cdoe"))
    if isinstance(t_code_block, dict):
        for team_key, players in t_code_block.items():
            if isinstance(team_key, str) and team_key.isdigit() and isinstance(players, list):
                yield int(team_key), players
        return

    if isinstance(t_code_block, list):
        grouped = {}
        for p in t_code_block:
            if not isinstance(p, dict):
                continue
            tc = safe_int(p.get("t_code"))
            if not tc:
                continue
            grouped.setdefault(tc, []).append(p)
        for tc, players in grouped.items():
            yield tc, players

def load_index(index_csv):
    idx = {}
    with open(index_csv, "r", encoding="utf-8") as f:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", default=INDEX_CSV, help="game index csv path")
    args = ap.parse_args()

    idx = load_index(args.index_csv)
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
            # index에 없는 경기면 스킵
            continue

        homeTeam = meta["homeTeam"]
        awayTeam = meta["awayTeam"]
        date = meta["date"]

        for t_code, players in iter_team_players(data):
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

    print("DONE", "rows=", len(rows), "out=", OUT_CSV, "index_csv=", args.index_csv)

if __name__ == "__main__":
    main()
