"""
?? v1 ?? 4??? ?? ?? ??? ???, ?? subset ??? ?? ???? ??? ??????.

?? ??
- ?? ??? 2023+ ? ????.
- ?? ?? ?? 2024+ ? ???.
- ??? D? ??? ??? D-1?? ??? ????.

? ????? `?? ?? ??? ?? ?? ?, ?? ???? ????` ? ?? ???? ??? ?? ????.
"""
import os
import csv
from collections import defaultdict, deque
from datetime import datetime, timedelta


DATA_DIR = os.path.expanduser("~/statiz/data")

GAMES_CSV = os.path.join(DATA_DIR, "game_index_played.csv")
LINEUP_CSV = os.path.join(DATA_DIR, "lineup_long.csv")
BAT_CSV = os.path.join(DATA_DIR, "playerday_batter_long.csv")
PIT_CSV = os.path.join(DATA_DIR, "playerday_pitcher_long.csv")

OUT_CSV = os.path.join(DATA_DIR, "features_v2_candidates.csv")

# ---- year scope ----
MIN_INTERNAL_YEAR = 2023
MIN_FEATURE_YEAR = 2024

# ---- v1-compatible OPS / starter / bullpen params ----
K_SMOOTH = 20.0
MIN_PA_LASTSEASON = 60
MIN_PA_RECENT = 10
RECENT_GAMES = 5

EARLY_BULLPEN_TEAM_GAMES = 20
RECENT_TEAM_GAMES_FOR_SP = 7
PREV_SEASON_GS_THRESHOLD = 5

# ---- candidate feature params ----
TOP_ORDER_N = 5

K_BBIP_IP = 30.0
MIN_IP_LASTSEASON = 30.0
FALLBACK_PRIOR_BBIP = 0.35

PYTHAG_EXP = 1.83
K_PYTHAG_GAMES = 20.0
MIN_PYTHAG_PRIOR_GAMES = 30

K_RECENT10_GAMES = 10.0

K_STADIUM_WIN_GAMES = 10.0
MIN_STADIUM_PRIOR_GAMES = 5

K_PF_GAMES = 30.0
MIN_PF_PRIOR_GAMES = 30

FALLBACK_PRIOR_OPS = 0.700


def safe_int(x, default=0):
    """?? ??? ??? ?? ???? ??? ???."""
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
    """?? ??? ??? ?? ???? ??? ???."""
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "" or s.lower() == "none":
            return default
        return float(s)
    except Exception:
        return default


def yyyymmdd_to_dt(s):
    """YYYYMMDD ???? datetime?? ???."""
    return datetime.strptime(s, "%Y%m%d")


def dt_to_yyyymmdd(d):
    """datetime? YYYYMMDD ???? ???."""
    return d.strftime("%Y%m%d")


def calc_ops_from_counts(H, BB, HP, AB, SF, TB):
    """??? ???? OPS? PA? ????."""
    denom_obp = AB + BB + HP + SF
    obp = (H + BB + HP) / denom_obp if denom_obp else 0.0
    slg = TB / AB if AB else 0.0
    return obp + slg, denom_obp


def smooth_value(curr_val, curr_w, prior_val, k):
    """?? ??? prior? ????? ???? ?? ???."""
    if curr_w < 0:
        curr_w = 0
    return (curr_w / (curr_w + k)) * curr_val + (k / (curr_w + k)) * prior_val


def ip_to_outs(ip_val):
    """KBO? ?? ??(?: 6.1, 6.2)? ?????? ???."""
    s = str(ip_val).strip()
    if s == "" or s.lower() == "none":
        return 0
    if "." not in s:
        return max(0, safe_int(s)) * 3
    whole, frac = s.split(".", 1)
    w = max(0, safe_int(whole))
    # KBO-style IP notation:
    # .1 -> 1 out, .2 -> 2 outs
    f = frac[:1]
    add = 1 if f == "1" else (2 if f == "2" else 0)
    return w * 3 + add


def pythag_winpct(rs, ra, exp=PYTHAG_EXP):
    """??/???? ????? ????? ????."""
    rs = max(0.0, float(rs))
    ra = max(0.0, float(ra))
    if rs == 0 and ra == 0:
        return 0.5
    rs_e = rs ** exp
    ra_e = ra ** exp
    denom = rs_e + ra_e
    return (rs_e / denom) if denom > 0 else 0.5


def result_point(score_for, score_against):
    """?=1, ?=0, ?=0.5 ???? ?? ??? ???? ???."""
    if score_for > score_against:
        return 1.0
    if score_for < score_against:
        return 0.0
    return 0.5


def batter_hand_from_p_bat(p_bat):
    """API? ?? ? ???? R/L/S ??? ???."""
    # API docs (teamRecord pt): 1=R, 2=L, 3=S
    if p_bat == 1:
        return "R"
    if p_bat == 2:
        return "L"
    if p_bat == 3:
        return "S"
    return None


def pitcher_hand_from_p_throw(p_throw):
    """API? ?? ? ???? R/L ??? ???."""
    # API docs (pitcher type): 1/2 right variants, 3/4 left variants
    if p_throw in (1, 2):
        return "R"
    if p_throw in (3, 4):
        return "L"
    return None


def get_save_from_row(row):
    """?? ??? ??? ??? ??? ?? ???."""
    for key in ("SV", "sv", "Save", "save", "S"):
        if key in row:
            return safe_int(row.get(key))
    return 0


def get_hold_from_row(row):
    """?? ??? ??? ??? ?? ?? ???."""
    for key in ("HLD", "HD", "hld", "hd", "Hold", "hold"):
        if key in row:
            return safe_int(row.get(key))
    return 0


def get_svhld_from_row(row):
    """???+??? ?? ?? ??? ??? ???."""
    return get_save_from_row(row) + get_hold_from_row(row)


def load_games():
    """2023+ ?? ?? ??? ????? ???."""
    games = []
    with open(GAMES_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            date = (row.get("date") or "").strip()
            s_no = safe_int(row.get("s_no"))
            if not date or not s_no:
                continue
            year = safe_int(date[:4])
            if year < MIN_INTERNAL_YEAR:
                continue
            games.append(row)
    games.sort(key=lambda x: (x["date"], safe_int(x["s_no"])))
    return games


def load_lineup_map():
    """??? ????? 1~9? ??, ? ??? ??? ?? ?? ???."""
    # lineup_map[s_no][side]:
    #   - P: {"p_no","p_throw","p_bat"}
    #   - batters[1..9]: {"p_no","p_bat","p_throw"}
    lineup_map = defaultdict(
        lambda: {
            "home": {"P": None, "batters": {}},
            "away": {"P": None, "batters": {}},
        }
    )
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

            info = {
                "p_no": p_no,
                "p_bat": safe_int(row.get("p_bat")),
                "p_throw": safe_int(row.get("p_throw")),
            }

            if bo == "P":
                lineup_map[s_no][side]["P"] = info
            else:
                order = safe_int(bo, 0)
                if 1 <= order <= 9:
                    lineup_map[s_no][side]["batters"][order] = info
    return lineup_map


def group_by_date(csv_path):
    """???? ?? ?? D-1 ?? ???? ??? ???? ?? ???."""
    by_date = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            d = row.get("date")
            if d:
                by_date[d].append(row)
    return by_date


def main():
    games = load_games()
    lineup_map = load_lineup_map()
    bat_by_date = group_by_date(BAT_CSV)
    pit_by_date = group_by_date(PIT_CSV)

    games_by_date = defaultdict(list)
    for g in games:
        games_by_date[g["date"]].append(g)
    dates = sorted(games_by_date.keys())

    # cumulative player stats
    bat_cum = defaultdict(lambda: {"AB": 0, "H": 0, "BB": 0, "HP": 0, "SF": 0, "TB": 0})
    pit_cum = defaultdict(
        lambda: {
            "AB": 0,
            "H": 0,
            "BB": 0,
            "HP": 0,
            "SF": 0,
            "TB": 0,
            "BF": 0,
            "OUTS": 0,
        }
    )

    # league totals
    league_bat_tot = defaultdict(lambda: {"AB": 0, "H": 0, "BB": 0, "HP": 0, "SF": 0, "TB": 0})
    league_pit_tot = defaultdict(lambda: {"BB": 0, "OUTS": 0})
    league_game_runs = defaultdict(lambda: {"R": 0, "G": 0})

    # recent player/team windows
    bat_recent = defaultdict(lambda: deque(maxlen=RECENT_GAMES))
    team_recent_results = defaultdict(lambda: deque(maxlen=10))

    # bullpen context
    pit_season_gs = defaultdict(int)  # (p_no, year) -> GS
    pitcher_svhld_season = defaultdict(int)  # (p_no, year) -> SV+HLD
    pitcher_np_by_date = defaultdict(int)  # (p_no, date) -> NP
    team_game_cnt = defaultdict(int)  # (team, year)
    team_recent_starters = defaultdict(lambda: deque(maxlen=RECENT_TEAM_GAMES_FOR_SP))
    team_pitchers_by_year = defaultdict(set)  # (team, year) -> set(p_no)

    # team-level score/win context
    team_runs = defaultdict(lambda: {"RS": 0, "RA": 0, "G": 0})  # (team, year)
    team_wins = defaultdict(lambda: {"WPTS": 0.0, "G": 0})  # tie=0.5
    team_stadium_wins = defaultdict(lambda: {"WPTS": 0.0, "G": 0})  # (team, year, stadium)
    stadium_runs = defaultdict(lambda: {"R": 0, "G": 0})  # (stadium, year)

    def league_ops(year):
        """?? ?? ?? prior? ??? ? ? ?? ?? OPS? ????."""
        tot = league_bat_tot.get(year)
        if not tot:
            return FALLBACK_PRIOR_OPS
        val, _ = calc_ops_from_counts(
            tot["H"], tot["BB"], tot["HP"], tot["AB"], tot["SF"], tot["TB"]
        )
        return val if val > 0 else FALLBACK_PRIOR_OPS

    def league_bbip(year):
        """?? ?? ?? prior? ??? ? ? ?? ?? BB/IP? ????."""
        tot = league_pit_tot.get(year)
        if not tot:
            return FALLBACK_PRIOR_BBIP
        outs = tot["OUTS"]
        if outs <= 0:
            return FALLBACK_PRIOR_BBIP
        ip = outs / 3.0
        return (tot["BB"] / ip) if ip > 0 else FALLBACK_PRIOR_BBIP

    def team_prev_winpct(team, year):
        """?? ?? ? ??? ?? ? prior? ???? ?? ????."""
        prev = team_wins.get((team, year - 1))
        if not prev or prev["G"] <= 0:
            return 0.5
        return prev["WPTS"] / prev["G"]

    def league_rpg(year):
        """???? ??? ??? ?? ?? ??? ??? ????."""
        rec = league_game_runs.get(year)
        if not rec or rec["G"] <= 0:
            return None
        return rec["R"] / rec["G"]

    def batter_ops_smooth(p_no, year):
        """??? ?? OPS? ?? ?? ?? ?? ??? ?? ?????."""
        cur = bat_cum[(p_no, year)]
        cur_ops, cur_pa = calc_ops_from_counts(
            cur["H"], cur["BB"], cur["HP"], cur["AB"], cur["SF"], cur["TB"]
        )

        prev = bat_cum.get((p_no, year - 1))
        if prev:
            prev_ops, prev_pa = calc_ops_from_counts(
                prev["H"], prev["BB"], prev["HP"], prev["AB"], prev["SF"], prev["TB"]
            )
        else:
            prev_ops, prev_pa = 0.0, 0

        prior = prev_ops if prev_pa >= MIN_PA_LASTSEASON else league_ops(year - 1)
        smooth = smooth_value(cur_ops, cur_pa, prior, K_SMOOTH)
        return smooth, cur_pa

    def batter_ops_recent_or_smooth(p_no, year, ops_smooth_val):
        """?? 5?? OPS? ??? ??? ??? OPS_smooth? ????."""
        dq = bat_recent[(p_no, year)]
        if not dq:
            return ops_smooth_val, 0
        ab = h = bb = hp = sf = tb = 0
        for it in dq:
            ab += it["AB"]
            h += it["H"]
            bb += it["BB"]
            hp += it["HP"]
            sf += it["SF"]
            tb += it["TB"]
        ops_recent, pa_recent = calc_ops_from_counts(h, bb, hp, ab, sf, tb)
        if pa_recent < MIN_PA_RECENT:
            return ops_smooth_val, pa_recent
        return ops_recent, pa_recent

    def pitcher_allowed_ops_smooth(p_no, year):
        """???? ?OPS? ???? ?? ???? ????."""
        cur = pit_cum[(p_no, year)]
        cur_ops, _ = calc_ops_from_counts(
            cur["H"], cur["BB"], cur["HP"], cur["AB"], cur["SF"], cur["TB"]
        )
        cur_bf = cur["BF"]

        prev = pit_cum.get((p_no, year - 1))
        if prev:
            prev_ops, _ = calc_ops_from_counts(
                prev["H"], prev["BB"], prev["HP"], prev["AB"], prev["SF"], prev["TB"]
            )
            prev_bf = prev["BF"]
        else:
            prev_ops, prev_bf = 0.0, 0

        prior = prev_ops if prev_bf >= MIN_PA_LASTSEASON else league_ops(year - 1)
        smooth = smooth_value(cur_ops, cur_bf, prior, K_SMOOTH)
        return smooth, cur_bf

    def pitcher_bbip_smooth(p_no, year):
        """???? BB/IP? prior? ?? ?? ? ???? ???."""
        cur = pit_cum[(p_no, year)]
        cur_ip = cur["OUTS"] / 3.0 if cur["OUTS"] > 0 else 0.0
        cur_val = (cur["BB"] / cur_ip) if cur_ip > 0 else 0.0

        prev = pit_cum.get((p_no, year - 1))
        if prev:
            prev_ip = prev["OUTS"] / 3.0 if prev["OUTS"] > 0 else 0.0
            prev_val = (prev["BB"] / prev_ip) if prev_ip > 0 else 0.0
        else:
            prev_ip, prev_val = 0.0, 0.0

        prior = prev_val if prev_ip >= MIN_IP_LASTSEASON else league_bbip(year - 1)
        smooth = smooth_value(cur_val, cur_ip, prior, K_BBIP_IP)
        return smooth, cur_ip

    def team_pythag_smooth(team, year):
        """? ?? ??? ?? ????? ??? prior? ?? ????."""
        cur = team_runs[(team, year)]
        cur_val = pythag_winpct(cur["RS"], cur["RA"], PYTHAG_EXP)
        cur_w = cur["G"]

        prev = team_runs.get((team, year - 1))
        if prev and prev["G"] >= MIN_PYTHAG_PRIOR_GAMES:
            prior = pythag_winpct(prev["RS"], prev["RA"], PYTHAG_EXP)
        else:
            prior = 0.5

        return smooth_value(cur_val, cur_w, prior, K_PYTHAG_GAMES)

    def team_recent10_winpct_smooth(team, year):
        """?? 10?? ??? ???? ?? ??? ?? ?? ??? ???."""
        dq = team_recent_results[(team, year)]
        if dq:
            cur_val = sum(dq) / len(dq)
            cur_w = len(dq)
        else:
            cur_val = 0.5
            cur_w = 0
        prior = team_prev_winpct(team, year)
        return smooth_value(cur_val, cur_w, prior, K_RECENT10_GAMES)

    def team_stadium_winpct_smooth(team, year, stadium):
        """?? ????? ? ??? ?? ? prior? ?? ?????."""
        cur = team_stadium_wins[(team, year, stadium)]
        if cur["G"] > 0:
            cur_val = cur["WPTS"] / cur["G"]
            cur_w = cur["G"]
        else:
            cur_val, cur_w = 0.5, 0

        prev = team_stadium_wins.get((team, year - 1, stadium))
        if prev and prev["G"] >= MIN_STADIUM_PRIOR_GAMES:
            prior = prev["WPTS"] / prev["G"]
        else:
            prior = team_prev_winpct(team, year)

        return smooth_value(cur_val, cur_w, prior, K_STADIUM_WIN_GAMES)

    def stadium_park_factor_smooth(stadium, year):
        """?? ??? ?? ??? ?? ?? ?? ??? ??? ????? ????."""
        cur = stadium_runs[(stadium, year)]
        cur_g = cur["G"]
        lg_rpg = league_rpg(year)
        if cur_g > 0 and lg_rpg and lg_rpg > 0:
            cur_val = (cur["R"] / cur_g) / lg_rpg
        else:
            cur_val = 1.0

        prev = stadium_runs.get((stadium, year - 1))
        prev_lg_rpg = league_rpg(year - 1)
        if prev and prev["G"] >= MIN_PF_PRIOR_GAMES and prev_lg_rpg and prev_lg_rpg > 0:
            prior = (prev["R"] / prev["G"]) / prev_lg_rpg
        else:
            prior = 1.0

        return smooth_value(cur_val, cur_g, prior, K_PF_GAMES)

    def is_starter_group(team, p_no, year, team_game_no):
        """?? ??? ??? ?? ???/???? ????."""
        if team_game_no <= EARLY_BULLPEN_TEAM_GAMES:
            return pit_season_gs.get((p_no, year - 1), 0) >= PREV_SEASON_GS_THRESHOLD
        recent_sp_set = set(team_recent_starters[(team, year)])
        return p_no in recent_sp_set

    def pitcher_fatigue_score(p_no, date_str):
        """D-1~D-5 ???? ????? ?? ?? ???? ????."""
        d0 = yyyymmdd_to_dt(date_str)
        score = 0
        for lag in range(1, 6):
            w = 6 - lag
            dp = dt_to_yyyymmdd(d0 - timedelta(days=lag))
            score += w * pitcher_np_by_date.get((p_no, dp), 0)
        return score

    def select_core_bullpen(team, year, team_game_no, today_starter_p_no):
        """?? ?? ??? ??? ?? ?? 4?? ???."""
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
                    p,
                ),
                reverse=True,
            )
        else:
            ranked = sorted(
                bullpen_pool,
                key=lambda p: (
                    pitcher_svhld_season.get((p, year), 0),
                    pitcher_svhld_season.get((p, year - 1), 0),
                    p,
                ),
                reverse=True,
            )
        return ranked[:4]

    def team_core_bullpen_fatigue(team, year, team_game_no, date_str, today_starter_p_no):
        """?? ?? 4?? ???? ?? ? ?? ???? ????."""
        core4 = select_core_bullpen(team, year, team_game_no, today_starter_p_no)
        return sum(pitcher_fatigue_score(p_no, date_str) for p_no in core4)

    def count_platoon_advantage(batters, opp_sp_throw_code):
        """?? ?????? ??? ??? ?? ?? ?? ??."""
        p_hand = pitcher_hand_from_p_throw(opp_sp_throw_code)
        cnt = 0
        for order in range(1, 10):
            info = batters.get(order)
            if not info:
                continue
            b_hand = batter_hand_from_p_bat(safe_int(info.get("p_bat")))
            if b_hand == "S":
                cnt += 1
            elif p_hand == "L" and b_hand == "R":
                cnt += 1
            elif p_hand == "R" and b_hand == "L":
                cnt += 1
        return cnt

    fieldnames = [
        "date",
        "s_no",
        "s_code",
        "homeTeam",
        "awayTeam",
        "y_home_win",
        "homeScore",
        "awayScore",
        "home_sum_ops_smooth",
        "away_sum_ops_smooth",
        "diff_sum_ops_smooth",
        "home_sum_ops_recent5",
        "away_sum_ops_recent5",
        "diff_sum_ops_recent5",
        "home_top5_ops_smooth",
        "away_top5_ops_smooth",
        "diff_top5_ops_smooth",
        "home_sp_oops",
        "away_sp_oops",
        "diff_sp_oops",
        "home_sp_bbip",
        "away_sp_bbip",
        "diff_sp_bbip",
        "home_platoon_cnt_vs_opp_sp",
        "away_platoon_cnt_vs_opp_sp",
        "diff_opp_sp_platoon_cnt",
        "home_bullpen_fatigue",
        "away_bullpen_fatigue",
        "diff_bullpen_fatigue",
        "home_pythag_winpct",
        "away_pythag_winpct",
        "diff_pythag_winpct",
        "home_recent10_winpct",
        "away_recent10_winpct",
        "diff_recent10_winpct",
        "home_stadium_winpct",
        "away_stadium_winpct",
        "diff_team_stadium_winpct",
        "park_factor_stadium",
        "diff_team_stadium_winpct_pfadj",
        "home_sp_p_no",
        "away_sp_p_no",
    ]

    rows_out = []

    for date in dates:
        todays_games = games_by_date[date]
        todays_games.sort(key=lambda x: safe_int(x.get("s_no")))
        year = safe_int(date[:4])

        # (1) ?? ?? ??? ?? ???.
        #     ?? ?? ?? ??? ???? ??? ??? D-1??? ????.
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

            def lineup_sums(side):
                side_info = lineup_map[s_no][side]
                batters = side_info["batters"]
                sum_smooth = 0.0
                sum_recent = 0.0
                sum_top_smooth = 0.0

                for order in range(1, 10):
                    info = batters.get(order)
                    p_no = safe_int(info.get("p_no")) if info else 0
                    if not p_no:
                        fallback = league_ops(year - 1)
                        sum_smooth += fallback
                        sum_recent += fallback
                        if order <= TOP_ORDER_N:
                            sum_top_smooth += fallback
                        continue

                    ops_s, _ = batter_ops_smooth(p_no, year)
                    ops_r, _ = batter_ops_recent_or_smooth(p_no, year, ops_s)

                    sum_smooth += ops_s
                    sum_recent += ops_r
                    if order <= TOP_ORDER_N:
                        sum_top_smooth += ops_s

                return sum_smooth, sum_recent, sum_top_smooth

            home_sum_s, home_sum_r, home_top_s = lineup_sums("home")
            away_sum_s, away_sum_r, away_top_s = lineup_sums("away")

            home_sp_info = lineup_map[s_no]["home"]["P"] or {}
            away_sp_info = lineup_map[s_no]["away"]["P"] or {}

            home_sp = safe_int(home_sp_info.get("p_no"))
            away_sp = safe_int(away_sp_info.get("p_no"))
            home_sp_throw = safe_int(home_sp_info.get("p_throw"))
            away_sp_throw = safe_int(away_sp_info.get("p_throw"))

            if home_sp:
                home_sp_oops, _ = pitcher_allowed_ops_smooth(home_sp, year)
                home_sp_bbip, _ = pitcher_bbip_smooth(home_sp, year)
            else:
                home_sp_oops = league_ops(year - 1)
                home_sp_bbip = league_bbip(year - 1)

            if away_sp:
                away_sp_oops, _ = pitcher_allowed_ops_smooth(away_sp, year)
                away_sp_bbip, _ = pitcher_bbip_smooth(away_sp, year)
            else:
                away_sp_oops = league_ops(year - 1)
                away_sp_bbip = league_bbip(year - 1)

            home_batters = lineup_map[s_no]["home"]["batters"]
            away_batters = lineup_map[s_no]["away"]["batters"]
            home_platoon = count_platoon_advantage(home_batters, away_sp_throw)
            away_platoon = count_platoon_advantage(away_batters, home_sp_throw)

            home_fat = team_core_bullpen_fatigue(home, year, home_game_no, date, home_sp) if home else 0
            away_fat = team_core_bullpen_fatigue(away, year, away_game_no, date, away_sp) if away else 0

            home_pyth = team_pythag_smooth(home, year) if home else 0.5
            away_pyth = team_pythag_smooth(away, year) if away else 0.5

            home_recent10 = team_recent10_winpct_smooth(home, year) if home else 0.5
            away_recent10 = team_recent10_winpct_smooth(away, year) if away else 0.5

            home_stadium_wr = team_stadium_winpct_smooth(home, year, s_code) if home and s_code else 0.5
            away_stadium_wr = team_stadium_winpct_smooth(away, year, s_code) if away and s_code else 0.5
            diff_stadium_wr = home_stadium_wr - away_stadium_wr

            park_factor = stadium_park_factor_smooth(s_code, year) if s_code else 1.0
            diff_stadium_wr_pfadj = diff_stadium_wr * park_factor

            if year >= MIN_FEATURE_YEAR:
                rows_out.append(
                    {
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
                        "home_top5_ops_smooth": round(home_top_s, 6),
                        "away_top5_ops_smooth": round(away_top_s, 6),
                        "diff_top5_ops_smooth": round(home_top_s - away_top_s, 6),
                        "home_sp_oops": round(home_sp_oops, 6),
                        "away_sp_oops": round(away_sp_oops, 6),
                        "diff_sp_oops": round(home_sp_oops - away_sp_oops, 6),
                        "home_sp_bbip": round(home_sp_bbip, 6),
                        "away_sp_bbip": round(away_sp_bbip, 6),
                        "diff_sp_bbip": round(home_sp_bbip - away_sp_bbip, 6),
                        "home_platoon_cnt_vs_opp_sp": home_platoon,
                        "away_platoon_cnt_vs_opp_sp": away_platoon,
                        "diff_opp_sp_platoon_cnt": home_platoon - away_platoon,
                        "home_bullpen_fatigue": home_fat,
                        "away_bullpen_fatigue": away_fat,
                        "diff_bullpen_fatigue": home_fat - away_fat,
                        "home_pythag_winpct": round(home_pyth, 6),
                        "away_pythag_winpct": round(away_pyth, 6),
                        "diff_pythag_winpct": round(home_pyth - away_pyth, 6),
                        "home_recent10_winpct": round(home_recent10, 6),
                        "away_recent10_winpct": round(away_recent10, 6),
                        "diff_recent10_winpct": round(home_recent10 - away_recent10, 6),
                        "home_stadium_winpct": round(home_stadium_wr, 6),
                        "away_stadium_winpct": round(away_stadium_wr, 6),
                        "diff_team_stadium_winpct": round(diff_stadium_wr, 6),
                        "park_factor_stadium": round(park_factor, 6),
                        "diff_team_stadium_winpct_pfadj": round(diff_stadium_wr_pfadj, 6),
                        "home_sp_p_no": home_sp,
                        "away_sp_p_no": away_sp,
                    }
                )

        # (2) ?? ?? ??? ? ?? ?? ?? ?? ??? ????.
        for g in todays_games:
            s_no = safe_int(g.get("s_no"))
            if not s_no:
                continue

            home = safe_int(g.get("homeTeam"))
            away = safe_int(g.get("awayTeam"))
            home_sp = safe_int((lineup_map[s_no]["home"]["P"] or {}).get("p_no"))
            away_sp = safe_int((lineup_map[s_no]["away"]["P"] or {}).get("p_no"))

            if home:
                team_game_cnt[(home, year)] += 1
                if home_sp:
                    team_recent_starters[(home, year)].append(home_sp)
            if away:
                team_game_cnt[(away, year)] += 1
                if away_sp:
                    team_recent_starters[(away, year)].append(away_sp)

        # (2b) ?? ?? ??? ??? ? ??, ?????, ?? ??, ???? ??? ????.
        for g in todays_games:
            home = safe_int(g.get("homeTeam"))
            away = safe_int(g.get("awayTeam"))
            s_code = safe_int(g.get("s_code"))
            hs = safe_int(g.get("homeScore"))
            aw = safe_int(g.get("awayScore"))

            if home:
                team_runs[(home, year)]["RS"] += hs
                team_runs[(home, year)]["RA"] += aw
                team_runs[(home, year)]["G"] += 1
                hp = result_point(hs, aw)
                team_wins[(home, year)]["WPTS"] += hp
                team_wins[(home, year)]["G"] += 1
                team_recent_results[(home, year)].append(hp)
                if s_code:
                    team_stadium_wins[(home, year, s_code)]["WPTS"] += hp
                    team_stadium_wins[(home, year, s_code)]["G"] += 1

            if away:
                team_runs[(away, year)]["RS"] += aw
                team_runs[(away, year)]["RA"] += hs
                team_runs[(away, year)]["G"] += 1
                ap = result_point(aw, hs)
                team_wins[(away, year)]["WPTS"] += ap
                team_wins[(away, year)]["G"] += 1
                team_recent_results[(away, year)].append(ap)
                if s_code:
                    team_stadium_wins[(away, year, s_code)]["WPTS"] += ap
                    team_stadium_wins[(away, year, s_code)]["G"] += 1

            if s_code:
                stadium_runs[(s_code, year)]["R"] += (hs + aw)
                stadium_runs[(s_code, year)]["G"] += 1
            league_game_runs[year]["R"] += (hs + aw)
            league_game_runs[year]["G"] += 1

        # (3) ?? ?? ??? ??? ?? ????? ?? ??? ?? ?? ????.
        bat_rows = bat_by_date.get(date, [])
        bat_rows.sort(key=lambda x: safe_int(x.get("s_no")))
        for row in bat_rows:
            p_no = safe_int(row.get("p_no"))
            if not p_no:
                continue
            y = safe_int(row.get("year")) or year
            if y < MIN_INTERNAL_YEAR:
                continue

            ab = safe_int(row.get("AB"))
            h = safe_int(row.get("H"))
            bb = safe_int(row.get("BB"))
            hp = safe_int(row.get("HP"))
            sf = safe_int(row.get("SF"))
            tb = safe_int(row.get("TB"))

            bat_cum[(p_no, y)]["AB"] += ab
            bat_cum[(p_no, y)]["H"] += h
            bat_cum[(p_no, y)]["BB"] += bb
            bat_cum[(p_no, y)]["HP"] += hp
            bat_cum[(p_no, y)]["SF"] += sf
            bat_cum[(p_no, y)]["TB"] += tb

            league_bat_tot[y]["AB"] += ab
            league_bat_tot[y]["H"] += h
            league_bat_tot[y]["BB"] += bb
            league_bat_tot[y]["HP"] += hp
            league_bat_tot[y]["SF"] += sf
            league_bat_tot[y]["TB"] += tb

            bat_recent[(p_no, y)].append({"AB": ab, "H": h, "BB": bb, "HP": hp, "SF": sf, "TB": tb})

        # (4) ?? ?? ??? ??? ?? ????? ?OPS, BB/IP, ?? ??? ??? ????.
        pit_rows = pit_by_date.get(date, [])
        pit_rows.sort(key=lambda x: safe_int(x.get("s_no")))
        for row in pit_rows:
            p_no = safe_int(row.get("p_no"))
            if not p_no:
                continue
            y = safe_int(row.get("year")) or year
            if y < MIN_INTERNAL_YEAR:
                continue
            t_code = safe_int(row.get("t_code"))

            ab = safe_int(row.get("AB"))
            h = safe_int(row.get("H"))
            bb = safe_int(row.get("BB"))
            hp = safe_int(row.get("HP"))
            sf = safe_int(row.get("SF"))
            tb = safe_int(row.get("TB"))
            tbf = safe_int(row.get("TBF"))
            gs = safe_int(row.get("GS"))
            np = safe_int(row.get("NP"))
            svhld = get_svhld_from_row(row)
            outs = ip_to_outs(row.get("IP"))

            bf = tbf if tbf > 0 else (ab + bb + hp + sf)

            pit_cum[(p_no, y)]["AB"] += ab
            pit_cum[(p_no, y)]["H"] += h
            pit_cum[(p_no, y)]["BB"] += bb
            pit_cum[(p_no, y)]["HP"] += hp
            pit_cum[(p_no, y)]["SF"] += sf
            pit_cum[(p_no, y)]["TB"] += tb
            pit_cum[(p_no, y)]["BF"] += bf
            pit_cum[(p_no, y)]["OUTS"] += outs

            league_pit_tot[y]["BB"] += bb
            league_pit_tot[y]["OUTS"] += outs

            pit_season_gs[(p_no, y)] += gs
            pitcher_svhld_season[(p_no, y)] += svhld
            pitcher_np_by_date[(p_no, date)] += np

            if t_code:
                team_pitchers_by_year[(t_code, y)].add(p_no)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print(
        f"[OK] wrote: {OUT_CSV} rows={len(rows_out)} "
        f"(internal_year>={MIN_INTERNAL_YEAR}, output_year>={MIN_FEATURE_YEAR})"
    )


if __name__ == "__main__":
    main()
