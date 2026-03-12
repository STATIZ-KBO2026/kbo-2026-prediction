# build_features_v1_paper.py
# Paper-style features (v1):
# - sum OPS_smooth of lineup 1~9
# - sum recent7 OPS of lineup 1~9 (PA<10 -> OPS_smooth)
# - starter pitcher allowed OPS (sp_oops_smooth)
# - bullpen fatigue: weighted pitches in last 5 DAYS (5..1)
#
# Notes:
# - Uses ONLY 2023+ data (no 2022).
# - Writes features for 2024+ games, but processes 2023 internally for priors.
# - IMPORTANT: output column names keep "recent5" for compatibility with existing backtest code,
#   but the actual calculation is based on RECENT 7 GAMES.

import os
import csv
from collections import defaultdict, deque
from datetime import datetime, timedelta

DATA_DIR = os.path.expanduser("~/statiz/data")

GAMES_CSV  = os.path.join(DATA_DIR, "game_index_played.csv")
LINEUP_CSV = os.path.join(DATA_DIR, "lineup_long.csv")
BAT_CSV    = os.path.join(DATA_DIR, "playerday_batter_long.csv")
PIT_CSV    = os.path.join(DATA_DIR, "playerday_pitcher_long.csv")

OUT_CSV    = os.path.join(DATA_DIR, "features_v1_paper.csv")

# ---- Hyperparams (fixed by discussion) ----
K_SMOOTH = 20
MIN_PA_LASTSEASON = 60
MIN_PA_RECENT = 10
RECENT_GAMES = 7
EARLY_TEAM_GAMES = 10  # n=10
MIN_FEATURE_YEAR = 2024

FALLBACK_PRIOR_OPS = 0.700  # used only if prev-year league OPS is unavailable (shouldn't matter for 2024+)

# ---- utils ----
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

def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "" or s.lower() == "none":
            return default
        return float(s)
    except Exception:
        return default

def yyyymmdd_to_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d")

def dt_to_yyyymmdd(d: datetime) -> str:
    return d.strftime("%Y%m%d")

def calc_ops_from_counts(H, BB, HP, AB, SF, TB):
    # OBP = (H+BB+HP)/(AB+BB+HP+SF)
    # SLG = TB/AB
    denom_obp = AB + BB + HP + SF
    obp = (H + BB + HP) / denom_obp if denom_obp else 0.0
    slg = TB / AB if AB else 0.0
    ops = obp + slg
    pa = denom_obp
    return ops, pa

def smooth_value(curr_val, curr_w, prior_val, K):
    # (w/(w+K))*curr + (K/(w+K))*prior
    if curr_w < 0:
        curr_w = 0
    return (curr_w / (curr_w + K)) * curr_val + (K / (curr_w + K)) * prior_val

# ---- loaders ----
def load_games():
    games = []
    with open(GAMES_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # expects: date, s_no, s_code, homeTeam, awayTeam, homeScore, awayScore
            if not row.get("date") or not row.get("s_no"):
                continue
            games.append(row)
    games.sort(key=lambda x: (x["date"], safe_int(x["s_no"])))
    return games

def load_lineup_map():
    # lineup_map[s_no][side]["P"] = p_no
    # lineup_map[s_no][side]["batters"][1..9] = p_no
    lineup_map = defaultdict(lambda: {"home": {"P": None, "batters": {}},
                                      "away": {"P": None, "batters": {}}})
    with open(LINEUP_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            side = (row.get("side") or "").strip()
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

def group_by_date(csv_path):
    by_date = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            d = row.get("date")
            if d:
                by_date[d].append(row)
    return by_date

def group_pitcher_by_date_sno(pit_by_date):
    # {date: {s_no: [rows...]}}
    out = {}
    for d, rows in pit_by_date.items():
        m = defaultdict(list)
        for row in rows:
            s_no = safe_int(row.get("s_no"))
            if s_no:
                m[s_no].append(row)
        for s_no in m:
            m[s_no].sort(key=lambda x: (safe_int(x.get("t_code")), safe_int(x.get("p_no"))))
        out[d] = m
    return out

# ---- main ----
def main():
    games = load_games()
    lineup_map = load_lineup_map()

    bat_by_date = group_by_date(BAT_CSV)
    pit_by_date = group_by_date(PIT_CSV)
    pit_by_date_sno = group_pitcher_by_date_sno(pit_by_date)

    # group games by date
    games_by_date = defaultdict(list)
    for g in games:
        games_by_date[g["date"]].append(g)
    dates = sorted(games_by_date.keys())

    # cumulative season counts (keyed by (p_no, year))
    bat_cum = defaultdict(lambda: {"AB":0,"H":0,"BB":0,"HP":0,"SF":0,"TB":0})
    pit_cum = defaultdict(lambda: {"AB":0,"H":0,"BB":0,"HP":0,"SF":0,"TB":0,"BF":0})

    # league batting totals per year (for league OPS of that year)
    league_tot = defaultdict(lambda: {"AB":0,"H":0,"BB":0,"HP":0,"SF":0,"TB":0})

    # recent windows (per season)
    bat_recent = defaultdict(lambda: deque(maxlen=RECENT_GAMES))   # stores dict counts for last 7 games
    pit_recent_gs5 = defaultdict(lambda: deque(maxlen=5))          # stores last 5 GS flags (0/1), per (p_no, year)

    # pitcher GS totals per season (for early-season role)
    pit_season_gs = defaultdict(int)  # (p_no, year) -> GS count

    # team game count per season (for early-season window n=10)
    team_game_cnt = defaultdict(int)  # (t_code, year) -> games played so far (up to previous date)

    # bullpen pitches per (team, date)
    bullpen_pitches = defaultdict(int)

    def league_ops(year: int) -> float:
        tot = league_tot.get(year)
        if not tot:
            return FALLBACK_PRIOR_OPS
        ops, _ = calc_ops_from_counts(tot["H"], tot["BB"], tot["HP"], tot["AB"], tot["SF"], tot["TB"])
        return ops if ops > 0 else FALLBACK_PRIOR_OPS

    def batter_ops_smooth(p_no: int, year: int):
        cur = bat_cum[(p_no, year)]
        cur_ops, cur_pa = calc_ops_from_counts(cur["H"], cur["BB"], cur["HP"], cur["AB"], cur["SF"], cur["TB"])

        prev_year = year - 1
        prev = bat_cum.get((p_no, prev_year))
        if prev:
            prev_ops, prev_pa = calc_ops_from_counts(prev["H"], prev["BB"], prev["HP"], prev["AB"], prev["SF"], prev["TB"])
        else:
            prev_ops, prev_pa = 0.0, 0

        prior = prev_ops if prev_pa >= MIN_PA_LASTSEASON else league_ops(prev_year)
        smooth = smooth_value(cur_ops, cur_pa, prior, K_SMOOTH)
        return smooth, cur_pa

    def batter_ops_recent_or_smooth(p_no: int, year: int, ops_smooth_val: float):
        dq = bat_recent[(p_no, year)]
        if not dq:
            return ops_smooth_val, 0

        AB=H=BB=HP=SF=TB=0
        for it in dq:
            AB += it["AB"]; H += it["H"]; BB += it["BB"]; HP += it["HP"]; SF += it["SF"]; TB += it["TB"]
        ops_recent, pa_recent = calc_ops_from_counts(H, BB, HP, AB, SF, TB)
        if pa_recent < MIN_PA_RECENT:
            return ops_smooth_val, pa_recent
        return ops_recent, pa_recent

    def pitcher_allowed_ops_smooth(p_no: int, year: int):
        cur = pit_cum[(p_no, year)]
        cur_ops, _ = calc_ops_from_counts(cur["H"], cur["BB"], cur["HP"], cur["AB"], cur["SF"], cur["TB"])
        cur_bf = cur["BF"]

        prev_year = year - 1
        prev = pit_cum.get((p_no, prev_year))
        if prev:
            prev_ops, _ = calc_ops_from_counts(prev["H"], prev["BB"], prev["HP"], prev["AB"], prev["SF"], prev["TB"])
            prev_bf = prev["BF"]
        else:
            prev_ops, prev_bf = 0.0, 0

        prior = prev_ops if prev_bf >= MIN_PA_LASTSEASON else league_ops(prev_year)
        smooth = smooth_value(cur_ops, cur_bf, prior, K_SMOOTH)
        return smooth, cur_bf

    def bullpen_fatigue(team: int, date_str: str) -> int:
        d0 = yyyymmdd_to_dt(date_str)
        f = 0
        for lag in range(1, 6):
            w = 6 - lag  # lag=1 -> 5, lag=5 -> 1
            dp = dt_to_yyyymmdd(d0 - timedelta(days=lag))
            f += w * bullpen_pitches.get((team, dp), 0)
        return f

    def is_starter_like(p_no: int, year: int, gs_today: int, team: int, team_game_no: int) -> bool:
        # 시즌 초반(팀 게임 1~10): prev season GS>=5면 선발 취급 (단, 오늘 GS=1이면 무조건 선발)
        if gs_today == 1:
            return True
        if team_game_no <= EARLY_TEAM_GAMES:
            return pit_season_gs.get((p_no, year-1), 0) >= 5

        # 시즌 중반: "최근 6경기 내 GS=1" => (오늘 GS=1) or (이전 5번 등판 중 GS=1)
        dq = pit_recent_gs5[(p_no, year)]
        return any(x == 1 for x in dq)

    fieldnames = [
        "date","s_no","s_code","homeTeam","awayTeam",
        "y_home_win","homeScore","awayScore",
        "home_sum_ops_smooth","away_sum_ops_smooth","diff_sum_ops_smooth",
        "home_sum_ops_recent5","away_sum_ops_recent5","diff_sum_ops_recent5",
        "home_sp_oops","away_sp_oops","diff_sp_oops",
        "home_bullpen_fatigue","away_bullpen_fatigue","diff_bullpen_fatigue",
        "home_sp_p_no","away_sp_p_no",
    ]

    rows_out = []

    for date in dates:
        todays_games = games_by_date[date]
        todays_games.sort(key=lambda x: safe_int(x.get("s_no")))

        year = safe_int(date[:4])

        # ---- (1) FEATURES: use stats up to PREVIOUS day only ----
        for g in todays_games:
            s_no = safe_int(g.get("s_no"))
            if not s_no:
                continue

            home = safe_int(g.get("homeTeam"))
            away = safe_int(g.get("awayTeam"))
            s_code = safe_int(g.get("s_code"))
            hs = safe_int(g.get("homeScore"))
            aw = safe_int(g.get("awayScore"))
            y = 1 if hs > aw else 0

            def lineup_sums(side: str):
                batters = lineup_map[s_no][side]["batters"]
                sum_smooth = 0.0
                sum_recent = 0.0
                for order in range(1, 10):
                    p_no = safe_int(batters.get(order))
                    if not p_no:
                        sum_smooth += league_ops(year-1)
                        sum_recent += league_ops(year-1)
                        continue
                    ops_s, _ = batter_ops_smooth(p_no, year)
                    ops_r, _ = batter_ops_recent_or_smooth(p_no, year, ops_s)
                    sum_smooth += ops_s
                    sum_recent += ops_r
                return sum_smooth, sum_recent

            home_sum_s, home_sum_r = lineup_sums("home")
            away_sum_s, away_sum_r = lineup_sums("away")

            # starter pitchers from lineup
            home_sp = lineup_map[s_no]["home"]["P"]
            away_sp = lineup_map[s_no]["away"]["P"]

            if home_sp:
                home_sp_oops, _ = pitcher_allowed_ops_smooth(home_sp, year)
            else:
                home_sp_oops = league_ops(year-1)

            if away_sp:
                away_sp_oops, _ = pitcher_allowed_ops_smooth(away_sp, year)
            else:
                away_sp_oops = league_ops(year-1)

            home_fat = bullpen_fatigue(home, date) if home else 0
            away_fat = bullpen_fatigue(away, date) if away else 0

            if year >= MIN_FEATURE_YEAR:
                rows_out.append({
                    "date": date,
                    "s_no": s_no,
                    "s_code": s_code,
                    "homeTeam": home,
                    "awayTeam": away,
                    "y_home_win": y,
                    "homeScore": hs,
                    "awayScore": aw,
                    "home_sum_ops_smooth": round(home_sum_s, 6),
                    "away_sum_ops_smooth": round(away_sum_s, 6),
                    "diff_sum_ops_smooth": round(home_sum_s - away_sum_s, 6),
                    # 컬럼명은 recent5 유지(호환성용), 실제 계산은 recent7
                    "home_sum_ops_recent5": round(home_sum_r, 6),
                    "away_sum_ops_recent5": round(away_sum_r, 6),
                    "diff_sum_ops_recent5": round(home_sum_r - away_sum_r, 6),
                    "home_sp_oops": round(home_sp_oops, 6),
                    "away_sp_oops": round(away_sp_oops, 6),
                    "diff_sp_oops": round(home_sp_oops - away_sp_oops, 6),
                    "home_bullpen_fatigue": home_fat,
                    "away_bullpen_fatigue": away_fat,
                    "diff_bullpen_fatigue": home_fat - away_fat,
                    "home_sp_p_no": safe_int(home_sp),
                    "away_sp_p_no": safe_int(away_sp),
                })

        # ---- (2) TODAY bullpen pitches: use today's pitcher rows, process games in order ----
        pit_map = pit_by_date_sno.get(date, {})
        for g in todays_games:
            s_no = safe_int(g.get("s_no"))
            if not s_no:
                continue
            home = safe_int(g.get("homeTeam"))
            away = safe_int(g.get("awayTeam"))

            home_game_no = team_game_cnt[(home, year)] + 1 if home else 9999
            away_game_no = team_game_cnt[(away, year)] + 1 if away else 9999

            for row in pit_map.get(s_no, []):
                p_no = safe_int(row.get("p_no"))
                t_code = safe_int(row.get("t_code"))
                gs_today = safe_int(row.get("GS"))
                np_today = safe_int(row.get("NP"))

                if not p_no or not t_code:
                    continue

                team_game_no = home_game_no if t_code == home else (away_game_no if t_code == away else 9999)
                starter_like = is_starter_like(p_no, year, gs_today, t_code, team_game_no)

                if not starter_like:
                    bullpen_pitches[(t_code, date)] += np_today

                pit_recent_gs5[(p_no, year)].append(gs_today)

            if home:
                team_game_cnt[(home, year)] += 1
            if away:
                team_game_cnt[(away, year)] += 1

        # ---- (3) After day: update cumulative stats with TODAY results ----
        bat_rows = bat_by_date.get(date, [])
        bat_rows.sort(key=lambda x: safe_int(x.get("s_no")))
        for row in bat_rows:
            p_no = safe_int(row.get("p_no"))
            if not p_no:
                continue
            y = safe_int(row.get("year")) or year
            AB = safe_int(row.get("AB"))
            H  = safe_int(row.get("H"))
            BB = safe_int(row.get("BB"))
            HP = safe_int(row.get("HP"))
            SF = safe_int(row.get("SF"))
            TB = safe_int(row.get("TB"))

            bat_cum[(p_no, y)]["AB"] += AB
            bat_cum[(p_no, y)]["H"]  += H
            bat_cum[(p_no, y)]["BB"] += BB
            bat_cum[(p_no, y)]["HP"] += HP
            bat_cum[(p_no, y)]["SF"] += SF
            bat_cum[(p_no, y)]["TB"] += TB

            league_tot[y]["AB"] += AB
            league_tot[y]["H"]  += H
            league_tot[y]["BB"] += BB
            league_tot[y]["HP"] += HP
            league_tot[y]["SF"] += SF
            league_tot[y]["TB"] += TB

            bat_recent[(p_no, y)].append({"AB":AB,"H":H,"BB":BB,"HP":HP,"SF":SF,"TB":TB})

        pit_rows = pit_by_date.get(date, [])
        pit_rows.sort(key=lambda x: safe_int(x.get("s_no")))
        for row in pit_rows:
            p_no = safe_int(row.get("p_no"))
            if not p_no:
                continue
            y = safe_int(row.get("year")) or year

            AB = safe_int(row.get("AB"))
            H  = safe_int(row.get("H"))
            BB = safe_int(row.get("BB"))
            HP = safe_int(row.get("HP"))
            SF = safe_int(row.get("SF"))
            TB = safe_int(row.get("TB"))
            TBF = safe_int(row.get("TBF"))
            gs  = safe_int(row.get("GS"))

            BF = TBF if TBF > 0 else (AB + BB + HP + SF)

            pit_cum[(p_no, y)]["AB"] += AB
            pit_cum[(p_no, y)]["H"]  += H
            pit_cum[(p_no, y)]["BB"] += BB
            pit_cum[(p_no, y)]["HP"] += HP
            pit_cum[(p_no, y)]["SF"] += SF
            pit_cum[(p_no, y)]["TB"] += TB
            pit_cum[(p_no, y)]["BF"] += BF

            pit_season_gs[(p_no, y)] += gs

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print(f"[OK] wrote: {OUT_CSV} rows={len(rows_out)} (year>={MIN_FEATURE_YEAR})")


if __name__ == "__main__":
    main()
