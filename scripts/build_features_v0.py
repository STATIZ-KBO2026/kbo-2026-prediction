import os, csv, math
from collections import defaultdict

GAMES_CSV  = os.path.expanduser("~/statiz/data/game_index_played.csv")
LINEUP_CSV = os.path.expanduser("~/statiz/data/lineup_long.csv")
BAT_CSV    = os.path.expanduser("~/statiz/data/playerday_batter_long.csv")
PIT_CSV    = os.path.expanduser("~/statiz/data/playerday_pitcher_long.csv")
OUT_CSV    = os.path.expanduser("~/statiz/data/features_v0.csv")

def safe_int(x, default=0):
    try:
        if x is None: return default
        s = str(x).strip()
        if s == "" or s.lower() == "none": return default
        return int(float(s))
    except:
        return default

def safe_float(x, default=0.0):
    try:
        if x is None: return default
        s = str(x).strip()
        if s == "" or s.lower() == "none": return default
        return float(s)
    except:
        return default

def div(a, b):
    return (a / b) if b else 0.0

def parse_ip(x):
    """Baseball IP: 5.1 means 5 + 1/3, 5.2 means 5 + 2/3."""
    if x is None: return 0.0
    s = str(x).strip()
    if s == "" or s.lower() == "none": return 0.0
    # if looks like "5.1"
    if "." in s:
        a, b = s.split(".", 1)
        a = safe_int(a, 0)
        b = safe_int(b, 0)
        if b in (0, 1, 2):
            return a + (b / 3.0)
    # fallback
    return safe_float(s, 0.0)

def load_games():
    games = []
    with open(GAMES_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            games.append(row)
    games.sort(key=lambda x: (x["date"], safe_int(x["s_no"])))
    return games

def load_lineup_map():
    # lineup_map[s_no][side]['P'] = p_no
    # lineup_map[s_no][side]['batters'][order] = p_no
    lineup_map = defaultdict(lambda: {
        "home": {"P": None, "batters": {}},
        "away": {"P": None, "batters": {}},
    })
    with open(LINEUP_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            side = row.get("side")
            if side not in ("home", "away"):
                continue
            s_no = safe_int(row.get("s_no"))
            p_no = safe_int(row.get("p_no"))
            bo = str(row.get("battingOrder", "")).strip()

            if not s_no or not p_no:
                continue

            if bo == "P":
                lineup_map[s_no][side]["P"] = p_no
            else:
                order = safe_int(bo, 0)
                if 1 <= order <= 9:
                    lineup_map[s_no][side]["batters"][order] = p_no

    return lineup_map

def group_playerday_by_date(csv_path):
    by_date = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            d = row.get("date")
            if d:
                by_date[d].append(row)
    return by_date

def batter_metrics(cum):
    # cum: AB,H,BB,HP,SF,TB,HR,SO,PA
    ab = cum["AB"]; h = cum["H"]; bb = cum["BB"]; hp = cum["HP"]; sf = cum["SF"]
    tb = cum["TB"]; hr = cum["HR"]; so = cum["SO"]; pa = cum["PA"]
    avg = div(h, ab)
    obp = div(h + bb + hp, ab + bb + hp + sf)
    slg = div(tb, ab)
    ops = obp + slg
    hr_rate = div(hr, pa)
    so_rate = div(so, pa)
    return avg, obp, slg, ops, hr_rate, so_rate, pa

def pitcher_metrics(cum):
    ip = cum["IP"]; er = cum["ER"]; h = cum["H"]; bb = cum["BB"]; so = cum["SO"]
    era = div(er * 9.0, ip)
    whip = div(bb + h, ip)
    k9 = div(so * 9.0, ip)
    bb9 = div(bb * 9.0, ip)
    ip_per_start = div(ip, cum["GS"])
    return era, whip, k9, bb9, ip, cum["G"], cum["GS"], ip_per_start

def team_metrics(cum):
    g = cum["G"]; rs = cum["RS"]; ra = cum["RA"]; w = cum["W"]
    return g, div(rs, g), div(ra, g), div(w, g)

def main():
    games = load_games()
    lineup_map = load_lineup_map()

    bat_by_date = group_playerday_by_date(BAT_CSV)
    pit_by_date = group_playerday_by_date(PIT_CSV)

    # cumulative stats (previous dates only)
    bat_cum = defaultdict(lambda: {"AB":0,"H":0,"BB":0,"HP":0,"SF":0,"TB":0,"HR":0,"SO":0,"PA":0})
    pit_cum = defaultdict(lambda: {"IP":0.0,"ER":0,"H":0,"BB":0,"SO":0,"HR":0,"G":0,"GS":0})
    team_cum = defaultdict(lambda: {"G":0,"RS":0,"RA":0,"W":0})

    # group games by date
    games_by_date = defaultdict(list)
    for g in games:
        games_by_date[g["date"]].append(g)

    dates = sorted(games_by_date.keys())

    fieldnames = [
        "date","s_no","s_code","homeTeam","awayTeam",
        "y_home_win","homeScore","awayScore",

        "home_team_G","home_team_RS_perG","home_team_RA_perG","home_team_winpct",
        "away_team_G","away_team_RS_perG","away_team_RA_perG","away_team_winpct",
        "diff_team_RS_perG","diff_team_RA_perG","diff_team_winpct",

        "home_sp_p_no","away_sp_p_no",
        "home_sp_G","home_sp_GS","home_sp_IP","home_sp_ERA","home_sp_WHIP","home_sp_K9","home_sp_BB9","home_sp_IP_per_start","home_sp_nohist",
        "away_sp_G","away_sp_GS","away_sp_IP","away_sp_ERA","away_sp_WHIP","away_sp_K9","away_sp_BB9","away_sp_IP_per_start","away_sp_nohist",
        "diff_sp_ERA","diff_sp_WHIP","diff_sp_K9","diff_sp_BB9",

        "home_lineup_avg_avg","home_lineup_avg_obp","home_lineup_avg_slg","home_lineup_avg_ops",
        "home_lineup_hr_per_pa","home_lineup_so_per_pa","home_lineup_avg_pa","home_lineup_nohist_cnt",
        "away_lineup_avg_avg","away_lineup_avg_obp","away_lineup_avg_slg","away_lineup_avg_ops",
        "away_lineup_hr_per_pa","away_lineup_so_per_pa","away_lineup_avg_pa","away_lineup_nohist_cnt",
        "diff_lineup_avg_ops","diff_lineup_hr_per_pa","diff_lineup_so_per_pa","diff_lineup_nohist_cnt",
    ]

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out_rows = []

    for d in dates:
        todays_games = games_by_date[d]

        # 1) features computed using ONLY cumulative stats (<= previous day)
        for g in todays_games:
            s_no = safe_int(g["s_no"])
            home = safe_int(g["homeTeam"])
            away = safe_int(g["awayTeam"])
            s_code = safe_int(g.get("s_code"))

            hs = safe_int(g.get("homeScore"))
            as_ = safe_int(g.get("awayScore"))
            y = 1 if hs > as_ else 0

            # team prior
            hg, hrs, hra, hwp = team_metrics(team_cum[home])
            ag, ars, ara, awp = team_metrics(team_cum[away])

            # starters from lineup
            home_sp = lineup_map[s_no]["home"]["P"]
            away_sp = lineup_map[s_no]["away"]["P"]

            # pitcher prior
            h_era,h_whip,h_k9,h_bb9,h_ip,h_pg,h_pgs,h_ipps = pitcher_metrics(pit_cum[home_sp]) if home_sp else (0,0,0,0,0,0,0,0)
            a_era,a_whip,a_k9,a_bb9,a_ip,a_pg,a_pgs,a_ipps = pitcher_metrics(pit_cum[away_sp]) if away_sp else (0,0,0,0,0,0,0,0)

            # lineup prior (batters 1~9)
            def lineup_agg(side):
                batters = lineup_map[s_no][side]["batters"]
                p_list = [batters.get(i) for i in range(1,10)]
                p_list = [p for p in p_list if p]

                avgs=[]; obps=[]; slgs=[]; opss=[]; pas=[]
                hr_sum=0; so_sum=0; pa_sum=0
                nohist=0

                for p in p_list:
                    cum = bat_cum[p]
                    avg, obp, slg, ops, hr_rate, so_rate, pa = batter_metrics(cum)
                    if pa == 0:
                        nohist += 1
                    avgs.append(avg); obps.append(obp); slgs.append(slg); opss.append(ops); pas.append(pa)
                    # rates는 합산해서 다시 계산(팀 기준)
                    hr_sum += cum["HR"]
                    so_sum += cum["SO"]
                    pa_sum += cum["PA"]

                n = len(p_list) if p_list else 0
                avg_avg = sum(avgs)/n if n else 0.0
                avg_obp = sum(obps)/n if n else 0.0
                avg_slg = sum(slgs)/n if n else 0.0
                avg_ops = sum(opss)/n if n else 0.0
                hr_per_pa = div(hr_sum, pa_sum)
                so_per_pa = div(so_sum, pa_sum)
                avg_pa = sum(pas)/n if n else 0.0
                return avg_avg, avg_obp, avg_slg, avg_ops, hr_per_pa, so_per_pa, avg_pa, nohist

            h_la, h_lo, h_ls, h_lops, h_hrpa, h_sopa, h_avgpa, h_nohist = lineup_agg("home")
            a_la, a_lo, a_ls, a_lops, a_hrpa, a_sopa, a_avgpa, a_nohist = lineup_agg("away")

            out_rows.append({
                "date": d,
                "s_no": s_no,
                "s_code": s_code,
                "homeTeam": home,
                "awayTeam": away,
                "y_home_win": y,
                "homeScore": hs,
                "awayScore": as_,

                "home_team_G": hg,
                "home_team_RS_perG": round(hrs,6),
                "home_team_RA_perG": round(hra,6),
                "home_team_winpct": round(hwp,6),

                "away_team_G": ag,
                "away_team_RS_perG": round(ars,6),
                "away_team_RA_perG": round(ara,6),
                "away_team_winpct": round(awp,6),

                "diff_team_RS_perG": round(hrs-ars,6),
                "diff_team_RA_perG": round(hra-ara,6),
                "diff_team_winpct": round(hwp-awp,6),

                "home_sp_p_no": home_sp or 0,
                "away_sp_p_no": away_sp or 0,

                "home_sp_G": h_pg,
                "home_sp_GS": h_pgs,
                "home_sp_IP": round(h_ip,6),
                "home_sp_ERA": round(h_era,6),
                "home_sp_WHIP": round(h_whip,6),
                "home_sp_K9": round(h_k9,6),
                "home_sp_BB9": round(h_bb9,6),
                "home_sp_IP_per_start": round(h_ipps,6),
                "home_sp_nohist": 1 if h_ip == 0 else 0,

                "away_sp_G": a_pg,
                "away_sp_GS": a_pgs,
                "away_sp_IP": round(a_ip,6),
                "away_sp_ERA": round(a_era,6),
                "away_sp_WHIP": round(a_whip,6),
                "away_sp_K9": round(a_k9,6),
                "away_sp_BB9": round(a_bb9,6),
                "away_sp_IP_per_start": round(a_ipps,6),
                "away_sp_nohist": 1 if a_ip == 0 else 0,

                "diff_sp_ERA": round(h_era-a_era,6),
                "diff_sp_WHIP": round(h_whip-a_whip,6),
                "diff_sp_K9": round(h_k9-a_k9,6),
                "diff_sp_BB9": round(h_bb9-a_bb9,6),

                "home_lineup_avg_avg": round(h_la,6),
                "home_lineup_avg_obp": round(h_lo,6),
                "home_lineup_avg_slg": round(h_ls,6),
                "home_lineup_avg_ops": round(h_lops,6),
                "home_lineup_hr_per_pa": round(h_hrpa,6),
                "home_lineup_so_per_pa": round(h_sopa,6),
                "home_lineup_avg_pa": round(h_avgpa,6),
                "home_lineup_nohist_cnt": h_nohist,

                "away_lineup_avg_avg": round(a_la,6),
                "away_lineup_avg_obp": round(a_lo,6),
                "away_lineup_avg_slg": round(a_ls,6),
                "away_lineup_avg_ops": round(a_lops,6),
                "away_lineup_hr_per_pa": round(a_hrpa,6),
                "away_lineup_so_per_pa": round(a_sopa,6),
                "away_lineup_avg_pa": round(a_avgpa,6),
                "away_lineup_nohist_cnt": a_nohist,

                "diff_lineup_avg_ops": round(h_lops-a_lops,6),
                "diff_lineup_hr_per_pa": round(h_hrpa-a_hrpa,6),
                "diff_lineup_so_per_pa": round(h_sopa-a_sopa,6),
                "diff_lineup_nohist_cnt": h_nohist-a_nohist,
            })

        # 2) after all games of the day: update cumulative stats with TODAY results (so next day can use)
        for g in todays_games:
            home = safe_int(g["homeTeam"])
            away = safe_int(g["awayTeam"])
            hs = safe_int(g.get("homeScore"))
            as_ = safe_int(g.get("awayScore"))

            # team update
            team_cum[home]["G"] += 1
            team_cum[home]["RS"] += hs
            team_cum[home]["RA"] += as_
            team_cum[home]["W"]  += (1 if hs > as_ else 0)

            team_cum[away]["G"] += 1
            team_cum[away]["RS"] += as_
            team_cum[away]["RA"] += hs
            team_cum[away]["W"]  += (1 if as_ > hs else 0)

        # batter update
        for row in bat_by_date.get(d, []):
            p = safe_int(row.get("p_no"))
            if not p: continue
            c = bat_cum[p]
            c["AB"] += safe_int(row.get("AB"))
            c["H"]  += safe_int(row.get("H"))
            c["BB"] += safe_int(row.get("BB"))
            c["HP"] += safe_int(row.get("HP"))
            c["SF"] += safe_int(row.get("SF"))
            c["TB"] += safe_int(row.get("TB"))
            c["HR"] += safe_int(row.get("HR"))
            c["SO"] += safe_int(row.get("SO"))
            # PA가 없으면 AB+BB+HP+SF로 근사
            pa = row.get("PA")
            c["PA"] += safe_int(pa, safe_int(row.get("AB")) + safe_int(row.get("BB")) + safe_int(row.get("HP")) + safe_int(row.get("SF")))

        # pitcher update
        for row in pit_by_date.get(d, []):
            p = safe_int(row.get("p_no"))
            if not p: continue
            c = pit_cum[p]
            c["IP"] += parse_ip(row.get("IP"))
            c["ER"] += safe_int(row.get("ER"))
            c["H"]  += safe_int(row.get("H"))
            c["BB"] += safe_int(row.get("BB"))
            c["SO"] += safe_int(row.get("SO"))
            c["HR"] += safe_int(row.get("HR"))
            c["G"]  += 1
            c["GS"] += safe_int(row.get("GS"))

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print("DONE", "rows=", len(out_rows), "out=", OUT_CSV)

if __name__ == "__main__":
    main()
