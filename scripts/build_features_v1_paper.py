# build_features_v1_paper.py
# Paper-style features (v1):
# - sum OPS_smooth of lineup 1~9
# - sum recent5 OPS of lineup 1~9 (PA<10 -> OPS_smooth)
# - starter pitcher allowed OPS (sp_oops_smooth)
# - bullpen fatigue: sum of fatigue scores of CORE 4 bullpen arms
#
# Notes:
# - Uses ONLY 2023+ data (no 2022).
# - Writes features for 2024+ games, but processes 2023 internally for priors.
# - IMPORTANT: output column names keep "recent5" for compatibility with existing backtest code.

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

# ---- Hyperparams ----
K_SMOOTH = 20
MIN_PA_LASTSEASON = 60
MIN_PA_RECENT = 10
RECENT_GAMES = 5

EARLY_BULLPEN_TEAM_GAMES = 20         # 시즌 초반: 팀 경기 1~20
RECENT_TEAM_GAMES_FOR_SP = 7          # 이후: 최근 7팀경기 선발 이력 있으면 선발군
PREV_SEASON_GS_THRESHOLD = 5          # 시즌 초반 선발군 기준: 직전 시즌 GS >= 5

MIN_FEATURE_YEAR = 2024
FALLBACK_PRIOR_OPS = 0.700

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
    if curr_w < 0:
        curr_w = 0
    return (curr_w / (curr_w + K)) * curr_val + (K / (curr_w + K)) * prior_val

def get_save_from_row(row):
    for key in ("SV", "sv", "Save", "save"):
        if key in row:
            return safe_int(row.get(key))
    return 0

def get_hold_from_row(row):
    for key in ("HLD", "HD", "hld", "hd", "Hold", "hold"):
        if key in row:
            return safe_int(row.get(key))
    return 0

def get_svhld_from_row(row):
    return get_save_from_row(row) + get_hold_from_row(row)

# ---- loaders ----
def load_games():
    games = []
    with open(GAMES_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
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

# ---- main ----
def main():
    games = load_games()
    lineup_map = load_lineup_map()

    bat_by_date = group_by_date(BAT_CSV)
    pit_by_date = group_by_date(PIT_CSV)

    games_by_date = defaultdict(list)
    for g in games:
        games_by_date[g["date"]].append(g)
    dates = sorted(games_by_date.keys())

    # cumulative season counts
    bat_cum = defaultdict(lambda: {"AB":0, "H":0, "BB":0, "HP":0, "SF":0, "TB":0})
    pit_cum = defaultdict(lambda: {"AB":0, "H":0, "BB":0, "HP":0, "SF":0, "TB":0, "BF":0})

    # league totals per year
    league_tot = defaultdict(lambda: {"AB":0, "H":0, "BB":0, "HP":0, "SF":0, "TB":0})

    # recent batting window (last 5 games)
    bat_recent = defaultdict(lambda: deque(maxlen=RECENT_GAMES))

    # season-level pitcher info
    pit_season_gs = defaultdict(int)        # (p_no, year) -> GS
    pitcher_svhld_season = defaultdict(int) # (p_no, year) -> SV+HLD

    # player pitch count history by date
    pitcher_np_by_date = defaultdict(int)   # (p_no, date) -> NP

    # team-level context
    team_game_cnt = defaultdict(int)                                # (team, year) -> games played up to previous date
    team_recent_starters = defaultdict(lambda: deque(maxlen=RECENT_TEAM_GAMES_FOR_SP))  # (team, year) -> starter p_no deque
    team_pitchers_by_year = defaultdict(set)                        # (team, year) -> set(p_no)

    def league_ops(year: int) -> float:
        tot = league_tot.get(year)
        if not tot:
            return FALLBACK_PRIOR_OPS
        ops, _ = calc_ops_from_counts(tot["H"], tot["BB"], tot["HP"], tot["AB"], tot["SF"], tot["TB"])
        return ops if ops > 0 else FALLBACK_PRIOR_OPS

    def batter_ops_smooth(p_no: int, year: int):
        cur = bat_cum[(p_no, year)]
        cur_ops, cur_pa = calc_ops_from_counts(cur["H"], cur["BB"], cur["HP"], cur["AB"], cur["SF"], cur["TB"])

        prev = bat_cum.get((p_no, year - 1))
        if prev:
            prev_ops, prev_pa = calc_ops_from_counts(prev["H"], prev["BB"], prev["HP"], prev["AB"], prev["SF"], prev["TB"])
        else:
            prev_ops, prev_pa = 0.0, 0

        prior = prev_ops if prev_pa >= MIN_PA_LASTSEASON else league_ops(year - 1)
        smooth = smooth_value(cur_ops, cur_pa, prior, K_SMOOTH)
        return smooth, cur_pa

    def batter_ops_recent_or_smooth(p_no: int, year: int, ops_smooth_val: float):
        dq = bat_recent[(p_no, year)]
        if not dq:
            return ops_smooth_val, 0

        AB = H = BB = HP = SF = TB = 0
        for it in dq:
            AB += it["AB"]
            H  += it["H"]
            BB += it["BB"]
            HP += it["HP"]
            SF += it["SF"]
            TB += it["TB"]

        ops_recent, pa_recent = calc_ops_from_counts(H, BB, HP, AB, SF, TB)
        if pa_recent < MIN_PA_RECENT:
            return ops_smooth_val, pa_recent
        return ops_recent, pa_recent

    def pitcher_allowed_ops_smooth(p_no: int, year: int):
        cur = pit_cum[(p_no, year)]
        cur_ops, _ = calc_ops_from_counts(cur["H"], cur["BB"], cur["HP"], cur["AB"], cur["SF"], cur["TB"])
        cur_bf = cur["BF"]

        prev = pit_cum.get((p_no, year - 1))
        if prev:
            prev_ops, _ = calc_ops_from_counts(prev["H"], prev["BB"], prev["HP"], prev["AB"], prev["SF"], prev["TB"])
            prev_bf = prev["BF"]
        else:
            prev_ops, prev_bf = 0.0, 0

        prior = prev_ops if prev_bf >= MIN_PA_LASTSEASON else league_ops(year - 1)
        smooth = smooth_value(cur_ops, cur_bf, prior, K_SMOOTH)
        return smooth, cur_bf

    def is_starter_group(team: int, p_no: int, year: int, team_game_no: int) -> bool:
        # 시즌 초반: 직전 시즌 GS >= 5 이면 선발군
        if team_game_no <= EARLY_BULLPEN_TEAM_GAMES:
            return pit_season_gs.get((p_no, year - 1), 0) >= PREV_SEASON_GS_THRESHOLD

        # 이후: 최근 7팀경기 내 선발 이력 있으면 선발군
        recent_sp_set = set(team_recent_starters[(team, year)])
        return p_no in recent_sp_set

    def pitcher_fatigue_score(p_no: int, date_str: str) -> int:
        d0 = yyyymmdd_to_dt(date_str)
        score = 0
        for lag in range(1, 6):
            w = 6 - lag  # D-1:5, D-2:4, ..., D-5:1
            dp = dt_to_yyyymmdd(d0 - timedelta(days=lag))
            score += w * pitcher_np_by_date.get((p_no, dp), 0)
        return score

    def select_core_bullpen(team: int, year: int, team_game_no: int, today_starter_p_no: int):
        candidates = set()
        candidates.update(team_pitchers_by_year[(team, year - 1)])
        candidates.update(team_pitchers_by_year[(team, year)])

        if today_starter_p_no:
            candidates.discard(today_starter_p_no)

        if not candidates:
            return []

        bullpen_pool = [p for p in candidates if not is_starter_group(team, p, year, team_game_no)]

        if team_game_no <= EARLY_BULLPEN_TEAM_GAMES:
            ranked = sorted(
                bullpen_pool,
                key=lambda p: (
                    pitcher_svhld_season.get((p, year - 1), 0),
                    pitcher_svhld_season.get((p, year), 0),
                    p
                ),
                reverse=True
            )
        else:
            ranked = sorted(
                bullpen_pool,
                key=lambda p: (
                    pitcher_svhld_season.get((p, year), 0),
                    pitcher_svhld_season.get((p, year - 1), 0),
                    p
                ),
                reverse=True
            )

        return ranked[:4]

    def team_core_bullpen_fatigue(team: int, year: int, team_game_no: int, date_str: str, today_starter_p_no: int) -> int:
        core4 = select_core_bullpen(team, year, team_game_no, today_starter_p_no)
        return sum(pitcher_fatigue_score(p_no, date_str) for p_no in core4)

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

        # --------------------------------------------------
        # (1) build today's features using stats up to D-1
        # --------------------------------------------------
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

            home_game_no = team_game_cnt[(home, year)] + 1 if home else 9999
            away_game_no = team_game_cnt[(away, year)] + 1 if away else 9999

            def lineup_sums(side: str):
                batters = lineup_map[s_no][side]["batters"]
                sum_smooth = 0.0
                sum_recent = 0.0
                for order in range(1, 10):
                    p_no = safe_int(batters.get(order))
                    if not p_no:
                        fallback = league_ops(year - 1)
                        sum_smooth += fallback
                        sum_recent += fallback
                        continue
                    ops_s, _ = batter_ops_smooth(p_no, year)
                    ops_r, _ = batter_ops_recent_or_smooth(p_no, year, ops_s)
                    sum_smooth += ops_s
                    sum_recent += ops_r
                return sum_smooth, sum_recent

            home_sum_s, home_sum_r = lineup_sums("home")
            away_sum_s, away_sum_r = lineup_sums("away")

            home_sp = safe_int(lineup_map[s_no]["home"]["P"])
            away_sp = safe_int(lineup_map[s_no]["away"]["P"])

            if home_sp:
                home_sp_oops, _ = pitcher_allowed_ops_smooth(home_sp, year)
            else:
                home_sp_oops = league_ops(year - 1)

            if away_sp:
                away_sp_oops, _ = pitcher_allowed_ops_smooth(away_sp, year)
            else:
                away_sp_oops = league_ops(year - 1)

            home_fat = team_core_bullpen_fatigue(home, year, home_game_no, date, home_sp) if home else 0
            away_fat = team_core_bullpen_fatigue(away, year, away_game_no, date, away_sp) if away else 0

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
                    "home_sum_ops_recent5": round(home_sum_r, 6),
                    "away_sum_ops_recent5": round(away_sum_r, 6),
                    "diff_sum_ops_recent5": round(home_sum_r - away_sum_r, 6),
                    "home_sp_oops": round(home_sp_oops, 6),
                    "away_sp_oops": round(away_sp_oops, 6),
                    "diff_sp_oops": round(home_sp_oops - away_sp_oops, 6),
                    "home_bullpen_fatigue": home_fat,
                    "away_bullpen_fatigue": away_fat,
                    "diff_bullpen_fatigue": home_fat - away_fat,
                    "home_sp_p_no": home_sp,
                    "away_sp_p_no": away_sp,
                })

        # --------------------------------------------------
        # (2) after today's games: update team-level game context
        # --------------------------------------------------
        for g in todays_games:
            s_no = safe_int(g.get("s_no"))
            if not s_no:
                continue

            home = safe_int(g.get("homeTeam"))
            away = safe_int(g.get("awayTeam"))

            home_sp = safe_int(lineup_map[s_no]["home"]["P"])
            away_sp = safe_int(lineup_map[s_no]["away"]["P"])

            if home:
                team_game_cnt[(home, year)] += 1
                if home_sp:
                    team_recent_starters[(home, year)].append(home_sp)

            if away:
                team_game_cnt[(away, year)] += 1
                if away_sp:
                    team_recent_starters[(away, year)].append(away_sp)

        # --------------------------------------------------
        # (3) after today's games: update batter cumulative stats
        # --------------------------------------------------
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

            bat_recent[(p_no, y)].append({
                "AB": AB, "H": H, "BB": BB, "HP": HP, "SF": SF, "TB": TB
            })

        # --------------------------------------------------
        # (4) after today's games: update pitcher cumulative stats
        # --------------------------------------------------
        pit_rows = pit_by_date.get(date, [])
        pit_rows.sort(key=lambda x: safe_int(x.get("s_no")))

        for row in pit_rows:
            p_no = safe_int(row.get("p_no"))
            if not p_no:
                continue

            y = safe_int(row.get("year")) or year
            t_code = safe_int(row.get("t_code"))

            AB = safe_int(row.get("AB"))
            H  = safe_int(row.get("H"))
            BB = safe_int(row.get("BB"))
            HP = safe_int(row.get("HP"))
            SF = safe_int(row.get("SF"))
            TB = safe_int(row.get("TB"))
            TBF = safe_int(row.get("TBF"))
            GS  = safe_int(row.get("GS"))
            NP  = safe_int(row.get("NP"))
            SVHLD = get_svhld_from_row(row)

            BF = TBF if TBF > 0 else (AB + BB + HP + SF)

            pit_cum[(p_no, y)]["AB"] += AB
            pit_cum[(p_no, y)]["H"]  += H
            pit_cum[(p_no, y)]["BB"] += BB
            pit_cum[(p_no, y)]["HP"] += HP
            pit_cum[(p_no, y)]["SF"] += SF
            pit_cum[(p_no, y)]["TB"] += TB
            pit_cum[(p_no, y)]["BF"] += BF

            pit_season_gs[(p_no, y)] += GS
            pitcher_svhld_season[(p_no, y)] += SVHLD
            pitcher_np_by_date[(p_no, date)] += NP

            if t_code:
                team_pitchers_by_year[(t_code, y)].add(p_no)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print(f"[OK] wrote: {OUT_CSV} rows={len(rows_out)} (year>={MIN_FEATURE_YEAR})")

if __name__ == "__main__":
    main()
