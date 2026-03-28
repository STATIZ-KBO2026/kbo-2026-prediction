import os, csv, math
from collections import defaultdict, deque
from datetime import datetime

GAMES_CSV  = os.path.expanduser("~/statiz/data/game_index.csv")
LINEUP_CSV = os.path.expanduser("~/statiz/data/lineup_long.csv")
BAT_CSV    = os.path.expanduser("~/statiz/data/playerday_batter_long.csv")
PIT_CSV    = os.path.expanduser("~/statiz/data/playerday_pitcher_long.csv")
OUT_CSV    = os.path.expanduser("~/statiz/data/features_v1.csv")
MANUAL_TAG_CSV = os.getenv("MANUAL_TAG_CSV", "").strip()

# prior/rolling hyper-parameters (feature engineering baseline)
BAT_PRIOR_PA = float(os.getenv("BAT_PRIOR_PA", "40.0"))
PIT_PRIOR_IP = float(os.getenv("PIT_PRIOR_IP", "15.0"))
TEAM_PRIOR_G = 12.0
PREV_YEAR_W = float(os.getenv("PREV_YEAR_W", "0.8"))
CAREER_W = float(os.getenv("CAREER_W", "0.8"))
ROLL_N = 5
SCHEDULE_WIN_DAYS = 7

# v1: cold-start thresholds
COLD_TEAM_G = 10       # 팀 10경기 미만 → 작년 데이터 보강
COLD_BAT_G  = 5        # 타자 5경기(≈20PA) 미만
COLD_PIT_G  = 3        # 투수 3경기 미만
# v1: key bullpen fatigue weights (전날*5, 2일전*3, 3일전*1)
BP_FATIGUE_W = [5.0, 3.0, 1.0]
KEY_BP_N = 4            # 핵심불펜 = S+HD 상위 4명

def safe_int(x, default=0):
    try:
        if x is None: return default
        s = str(x).strip()
        if s == "" or s.lower() == "none": return default
        return int(float(s))
    except:
        return default

def safe_int_or_none(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "none":
            return None
        return int(float(s))
    except:
        return None

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

def shrink(raw_value, sample_size, prior_value, prior_size):
    return div(raw_value * sample_size + prior_value * prior_size, sample_size + prior_size)

def parse_ymd(s):
    return datetime.strptime(str(s), "%Y%m%d").date()

def rest_days(prev_ymd, cur_ymd):
    if not prev_ymd:
        return -1
    return max((parse_ymd(cur_ymd) - parse_ymd(prev_ymd)).days - 1, 0)

def day_diff(a_ymd, b_ymd):
    return (parse_ymd(a_ymd) - parse_ymd(b_ymd)).days

def parse_game_hour(hm):
    s = str(hm or "").strip()
    if not s:
        return -1
    try:
        return int(s.split(":")[0])
    except Exception:
        return safe_int(s, -1)

def game_time_bucket(game_hour):
    if game_hour >= 18:
        return "night"
    if 0 <= game_hour < 18:
        return "day"
    return "day"

def throw_side(code):
    # API_CODE.handPit: 1=R over, 2=R under/sidearm, 3=L over, 4=L under/sidearm
    s = str(code or "").strip()
    if s in ("1", "2"):
        return "R"
    if s in ("3", "4"):
        return "L"
    return None

def throw_is_under(code):
    s = str(code or "").strip()
    return 1 if s in ("2", "4") else 0

def league_type_flags(code):
    c = safe_int(code)
    is_regular = 1 if c == 10100 else 0
    is_post = 1 if (10200 <= c < 10300) else 0
    is_exhi = 1 if c == 10400 else 0
    is_other = 1 if (c and not (is_regular or is_post or is_exhi)) else 0
    return is_regular, is_post, is_exhi, is_other

def bat_side_vs_throw(bat_code, opp_throw_side):
    # 1=R, 2=L, 3=Switch (bats opposite to pitcher side)
    b = str(bat_code or "").strip()
    if b == "1":
        return "R"
    if b == "2":
        return "L"
    if b == "3":
        if opp_throw_side == "R":
            return "L"
        if opp_throw_side == "L":
            return "R"
    return None

def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

def better_low(value, ref, mult):
    return min(value, ref * mult) if ref > 0 else value

def worse_high(value, ref, mult):
    return max(value, ref * mult) if ref > 0 else value

def normalize3(a, b, c):
    s = a + b + c
    if s <= 0:
        return (1.0/3.0, 1.0/3.0, 1.0/3.0)
    return (a / s, b / s, c / s)

def has_latin(name):
    if not name:
        return False
    for ch in str(name):
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            return True
    return False

def korean_len(name):
    if not name:
        return 0
    n = 0
    for ch in str(name):
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3:
            n += 1
    return n

def foreign_name_hint(name):
    # 약한 힌트만 사용: 과신을 피하기 위해 작은 가중만 부여한다.
    if has_latin(name):
        return 0.5
    klen = korean_len(name)
    if klen and klen != 3:
        return 0.08
    return 0.0

def order_bucket(order):
    if 1 <= order <= 3:
        return "top"
    if 4 <= order <= 5:
        return "core"
    return "bottom"

def load_manual_tags():
    tags = {}
    candidates = []
    if MANUAL_TAG_CSV:
        candidates.append(os.path.expanduser(MANUAL_TAG_CSV))
    candidates.append(os.path.expanduser("~/statiz/data/manual_player_tags.csv"))
    repo_default = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "manual_player_tags.csv"))
    candidates.append(repo_default)

    tag_path = None
    for p in candidates:
        if p and os.path.exists(p):
            tag_path = p
            break
    if not tag_path:
        return tags
    with open(tag_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            p = safe_int(row.get("p_no"))
            if not p:
                continue
            tags[p] = {
                "is_foreign": 1 if safe_int(row.get("is_foreign"), 0) == 1 else 0,
                "is_rookie": 1 if safe_int(row.get("is_rookie"), 0) == 1 else 0,
            }
    return tags

def estimate_batter_cold_probs(name, order, state, cur_pa, prev_pa, car_pa, month, tag, first_year, year, min_data_year):
    if tag and tag.get("is_foreign") == 1:
        return (1.0, 0.0, 0.0)
    if tag and tag.get("is_rookie") == 1:
        return (0.0, 1.0, 0.0)

    is_new_to_dataset_year = (first_year == year and year > min_data_year)
    first_kbo = is_new_to_dataset_year and (prev_pa <= 0 and car_pa <= 0)
    returnee_like = (year > min_data_year) and (prev_pa <= 0 and car_pa > 0)
    name_hint = foreign_name_hint(name)

    if first_kbo:
        f = 0.24 + (0.23 if order <= 5 else 0.0) + (0.10 if order == 4 else 0.0) + name_hint
        r = 0.56 + (0.16 if order >= 6 else 0.0) + (0.06 if state != "Y" else 0.0)
        rt = 0.20
        return normalize3(max(f, 0.05), max(r, 0.05), max(rt, 0.05))

    if returnee_like:
        f = 0.18 + (0.06 if order <= 5 else 0.0) + name_hint
        r = 0.17 + (0.08 if order >= 6 else 0.0)
        rt = 0.65
        return normalize3(f, r, rt)

    # 기존 KBO 히스토리 보유 선수
    f = 0.10 + name_hint
    r = 0.07 if cur_pa < 40 else 0.03
    rt = 0.83
    return normalize3(f, r, rt)

def estimate_pitcher_cold_probs(name, cur_ip, prev_ip, car_ip, month, tag, first_year, year, min_data_year):
    if tag and tag.get("is_foreign") == 1:
        return (1.0, 0.0, 0.0)
    if tag and tag.get("is_rookie") == 1:
        return (0.0, 1.0, 0.0)

    is_new_to_dataset_year = (first_year == year and year > min_data_year)
    first_kbo = is_new_to_dataset_year and (prev_ip <= 0 and car_ip <= 0)
    returnee_like = (year > min_data_year) and (prev_ip <= 0 and car_ip > 0)
    name_hint = foreign_name_hint(name)

    if first_kbo:
        f = 0.52 + (0.08 if month <= 5 else 0.0) + name_hint
        r = 0.34 + (0.06 if month >= 7 else 0.0)
        rt = 0.14
        return normalize3(max(f, 0.05), max(r, 0.05), max(rt, 0.05))

    if returnee_like:
        f = 0.24 + name_hint
        r = 0.12
        rt = 0.64
        return normalize3(f, r, rt)

    f = 0.11 + name_hint
    r = 0.06 if cur_ip < 20 else 0.03
    rt = 0.83
    return normalize3(f, r, rt)

def mean_or_zero(vals):
    return (sum(vals) / len(vals)) if vals else 0.0

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

def lineup_row_priority(starting, lineup_state):
    score = 0
    if str(starting or "").strip().upper() == "Y":
        score += 2
    if str(lineup_state or "").strip().upper() == "Y":
        score += 1
    return score

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
    # lineup_map[s_no][side]['P_throw'] = p_throw
    # lineup_map[s_no][side]['P_name'] = p_name
    # lineup_map[s_no][side]['batters'][order] = p_no
    # lineup_map[s_no][side]['bat_hands'][order] = p_bat
    # lineup_map[s_no][side]['bat_names'][order] = p_name
    # lineup_map[s_no][side]['bat_states'][order] = lineupState
    lineup_map = defaultdict(lambda: {
        "home": {"P": None, "P_throw": None, "P_name": "", "P_state": "", "batters": {}, "bat_hands": {}, "bat_names": {}, "bat_states": {}},
        "away": {"P": None, "P_throw": None, "P_name": "", "P_state": "", "batters": {}, "bat_hands": {}, "bat_names": {}, "bat_states": {}},
    })
    pick_score = defaultdict(lambda: {
        "home": {"P": -999, "bat": defaultdict(lambda: -999)},
        "away": {"P": -999, "bat": defaultdict(lambda: -999)},
    })
    player_meta = defaultdict(lambda: {"name": "", "bat": "", "throw": "", "first_year": 9999})
    with open(LINEUP_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            side = row.get("side")
            if side not in ("home", "away"):
                continue
            s_no = safe_int(row.get("s_no"))
            p_no = safe_int(row.get("p_no"))
            bo = str(row.get("battingOrder", "")).strip()
            bo_u = bo.upper()
            ls = str(row.get("lineupState") or "").strip()
            st = str(row.get("starting") or "").strip()
            name = str(row.get("p_name") or "").strip()
            bat_code = str(row.get("p_bat") or "").strip()
            throw_code = str(row.get("p_throw") or "").strip()
            pos_code = safe_int(row.get("position"))
            order = safe_int(bo, 0)
            is_pitcher_slot = (bo_u == "P") or (pos_code == 1)
            pri = lineup_row_priority(st, ls)

            if not s_no or not p_no:
                continue

            pm = player_meta[p_no]
            if name:
                pm["name"] = name
            if bat_code:
                pm["bat"] = bat_code
            if throw_code:
                pm["throw"] = throw_code
            py = safe_int(str(row.get("date"))[:4], 9999)
            if py and py < pm["first_year"]:
                pm["first_year"] = py

            if is_pitcher_slot and pri >= pick_score[s_no][side]["P"]:
                lineup_map[s_no][side]["P"] = p_no
                lineup_map[s_no][side]["P_throw"] = throw_code
                lineup_map[s_no][side]["P_name"] = name
                lineup_map[s_no][side]["P_state"] = ls
                pick_score[s_no][side]["P"] = pri
            else:
                if 1 <= order <= 9 and pri >= pick_score[s_no][side]["bat"][order]:
                    lineup_map[s_no][side]["batters"][order] = p_no
                    lineup_map[s_no][side]["bat_hands"][order] = bat_code
                    lineup_map[s_no][side]["bat_names"][order] = name
                    lineup_map[s_no][side]["bat_states"][order] = ls
                    pick_score[s_no][side]["bat"][order] = pri

    return lineup_map, player_meta

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

def pythag_winpct(rs, ra, exponent=1.83):
    rs_e = rs ** exponent if rs > 0 else 0.0
    ra_e = ra ** exponent if ra > 0 else 0.0
    den = rs_e + ra_e
    if den == 0:
        return 0.5
    return rs_e / den

def team_recent_metrics(hist):
    # hist: deque[(rs, ra, win01)]
    n = len(hist)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    rs = sum(x[0] for x in hist)
    ra = sum(x[1] for x in hist)
    w = sum(x[2] for x in hist)
    return n, div(rs, n), div(ra, n), div(w, n)

def recent_count_by_days(date_hist, cur_ymd, days):
    c = 0
    for old_d in date_hist:
        dd = day_diff(cur_ymd, old_d)
        if 1 <= dd <= days:
            c += 1
    return c

def recent_pitch_usage_by_days(hist, cur_ymd, days):
    # hist: deque[(date, np, ip, appeared01)]
    np_sum = 0.0
    ip_sum = 0.0
    app = 0
    for old_d, np_v, ip_v, appeared in hist:
        dd = day_diff(cur_ymd, old_d)
        if 1 <= dd <= days:
            np_sum += np_v
            ip_sum += ip_v
            app += appeared
    return np_sum, ip_sum, app

def recent_pitch_usage_by_games(hist, n_games):
    # hist: deque[(np, ip, app)] or deque[(date, np, ip, app)] in chronological game order
    items = list(hist)[-n_games:]
    if not items:
        return 0.0, 0.0, 0
    def to_trip(x):
        if len(x) == 4:
            return x[1], x[2], x[3]
        return x[0], x[1], x[2]
    trip = [to_trip(x) for x in items]
    np_sum = sum(x[0] for x in trip)
    ip_sum = sum(x[1] for x in trip)
    app_sum = sum(x[2] for x in trip)
    return np_sum, ip_sum, app_sum

def recent_bp_quality_by_games(hist, n_games):
    # hist: deque[(date, ip, h, bb, er, so)] in chronological game order
    items = list(hist)[-n_games:]
    if not items:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    ip = sum(x[1] for x in items)
    h = sum(x[2] for x in items)
    bb = sum(x[3] for x in items)
    er = sum(x[4] for x in items)
    so = sum(x[5] for x in items)
    whip = div(h + bb, ip)
    era = div(er * 9.0, ip)
    k9 = div(so * 9.0, ip)
    bb9 = div(bb * 9.0, ip)
    return ip, whip, era, k9, bb9

def recent_bp_quality_by_days(hist, cur_ymd, days):
    # hist: deque[(date, ip, h, bb, er, so)]
    ip = 0.0
    h = 0.0
    bb = 0.0
    er = 0.0
    so = 0.0
    for old_d, ip_v, h_v, bb_v, er_v, so_v in hist:
        dd = day_diff(cur_ymd, old_d)
        if 1 <= dd <= days:
            ip += ip_v
            h += h_v
            bb += bb_v
            er += er_v
            so += so_v
    whip = div(h + bb, ip)
    era = div(er * 9.0, ip)
    k9 = div(so * 9.0, ip)
    bb9 = div(bb * 9.0, ip)
    return ip, whip, era, k9, bb9

def build_bp_usage_by_game(pit_by_date):
    # from relievers only (GS==0)
    # usage[(s_no, team_code)] = (np, ip, app)
    # quality[(s_no, team_code)] = (ip, h, bb, er, so)
    usage = defaultdict(lambda: [0.0, 0.0, 0])
    quality = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    # v1: per-pitcher reliever data per game
    # bp_pitcher[(date, team, p_no)] = [np, s, hd, ip, h, bb, er, so, hr, ab, sf, hp]
    bp_pitcher = defaultdict(lambda: [0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _, rows in pit_by_date.items():
        for row in rows:
            s_no = safe_int(row.get("s_no"))
            team_code = safe_int(row.get("t_code"))
            gs = safe_int(row.get("GS"))
            if not s_no or not team_code or gs != 0:
                continue
            key = (s_no, team_code)
            np_v = safe_float(row.get("NP"))
            ip_v = parse_ip(row.get("IP"))
            h_v = safe_float(row.get("H"))
            bb_v = safe_float(row.get("BB"))
            er_v = safe_float(row.get("ER"))
            so_v = safe_float(row.get("SO"))
            usage[key][0] += np_v
            usage[key][1] += ip_v
            usage[key][2] += 1
            quality[key][0] += ip_v
            quality[key][1] += h_v
            quality[key][2] += bb_v
            quality[key][3] += er_v
            quality[key][4] += so_v
            # v1: per-pitcher tracking
            p_no = safe_int(row.get("p_no"))
            date_v = row.get("date")
            s_v = safe_int(row.get("S"))
            hd_v = safe_int(row.get("HD"))
            hr_v = safe_int(row.get("HR"))
            ab_v = safe_int(row.get("AB"))
            sf_v = safe_int(row.get("SF"))
            hp_v = safe_int(row.get("HP"))
            pk = (date_v, team_code, p_no)
            bp_pitcher[pk][0] += np_v
            bp_pitcher[pk][1] += s_v
            bp_pitcher[pk][2] += hd_v
            bp_pitcher[pk][3] += ip_v
            bp_pitcher[pk][4] += h_v
            bp_pitcher[pk][5] += bb_v
            bp_pitcher[pk][6] += er_v
            bp_pitcher[pk][7] += so_v
            bp_pitcher[pk][8] += hr_v
            bp_pitcher[pk][9] += ab_v
            bp_pitcher[pk][10] += sf_v
            bp_pitcher[pk][11] += hp_v
    usage_map = {k: (v[0], v[1], v[2]) for k, v in usage.items()}
    quality_map = {k: (v[0], v[1], v[2], v[3], v[4]) for k, v in quality.items()}
    bp_pitcher_map = {k: tuple(v) for k, v in bp_pitcher.items()}
    return usage_map, quality_map, bp_pitcher_map

def bat_sub(a, b):
    return {
        "AB": max(a["AB"] - b["AB"], 0),
        "H": max(a["H"] - b["H"], 0),
        "BB": max(a["BB"] - b["BB"], 0),
        "HP": max(a["HP"] - b["HP"], 0),
        "SF": max(a["SF"] - b["SF"], 0),
        "TB": max(a["TB"] - b["TB"], 0),
        "HR": max(a["HR"] - b["HR"], 0),
        "SO": max(a["SO"] - b["SO"], 0),
        "PA": max(a["PA"] - b["PA"], 0),
    }

def pit_sub(a, b):
    return {
        "IP": max(a["IP"] - b["IP"], 0.0),
        "ER": max(a["ER"] - b["ER"], 0),
        "H": max(a["H"] - b["H"], 0),
        "BB": max(a["BB"] - b["BB"], 0),
        "SO": max(a["SO"] - b["SO"], 0),
        "HR": max(a["HR"] - b["HR"], 0),
        "G": max(a["G"] - b["G"], 0),
        "GS": max(a["GS"] - b["GS"], 0),
    }

def blend_four(cur_v, cur_n, prev_v, prev_n, car_v, car_n, lg_v, lg_n):
    n = cur_n + PREV_YEAR_W * prev_n + CAREER_W * car_n + lg_n
    if n == 0:
        return 0.0
    s = (cur_v * cur_n) + (prev_v * PREV_YEAR_W * prev_n) + (car_v * CAREER_W * car_n) + (lg_v * lg_n)
    return s / n

def batter_recent_metrics_by_games(hist, n):
    # hist: deque[(ab,h,bb,hp,sf,tb,hr,so,pa)]
    items = list(hist)[-n:]
    if not items:
        return 0, 0.0, 0.0, 0.0, 0
    cum = {"AB":0, "H":0, "BB":0, "HP":0, "SF":0, "TB":0, "HR":0, "SO":0, "PA":0}
    for ab,h,bb,hp,sf,tb,hr,so,pa in items:
        cum["AB"] += ab
        cum["H"] += h
        cum["BB"] += bb
        cum["HP"] += hp
        cum["SF"] += sf
        cum["TB"] += tb
        cum["HR"] += hr
        cum["SO"] += so
        cum["PA"] += pa
    _, _, _, ops, hr_rate, so_rate, pa = batter_metrics(cum)
    return len(items), ops, hr_rate, so_rate, pa

def pitcher_recent_metrics(hist):
    # hist: deque[(ip, er, h, bb, so)]
    n = len(hist)
    if n == 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    ip = sum(x[0] for x in hist)
    er = sum(x[1] for x in hist)
    h = sum(x[2] for x in hist)
    bb = sum(x[3] for x in hist)
    so = sum(x[4] for x in hist)
    era = div(er * 9.0, ip)
    whip = div(h + bb, ip)
    k9 = div(so * 9.0, ip)
    bb9 = div(bb * 9.0, ip)
    return n, era, whip, k9, bb9

def main():
    games = load_games()
    lineup_map, player_meta = load_lineup_map()
    manual_tags = load_manual_tags()

    bat_by_date = group_playerday_by_date(BAT_CSV)
    pit_by_date = group_playerday_by_date(PIT_CSV)
    bp_usage_by_game, bp_quality_by_game, bp_pitcher_map = build_bp_usage_by_game(pit_by_date)

    # cumulative stats (previous dates only)
    bat_cum = defaultdict(lambda: {"AB":0,"H":0,"BB":0,"HP":0,"SF":0,"TB":0,"HR":0,"SO":0,"PA":0})
    bat_year_cum = defaultdict(lambda: {"AB":0,"H":0,"BB":0,"HP":0,"SF":0,"TB":0,"HR":0,"SO":0,"PA":0})
    pit_cum = defaultdict(lambda: {"IP":0.0,"ER":0,"H":0,"BB":0,"SO":0,"HR":0,"G":0,"GS":0})
    pit_year_cum = defaultdict(lambda: {"IP":0.0,"ER":0,"H":0,"BB":0,"SO":0,"HR":0,"G":0,"GS":0})
    team_cum = defaultdict(lambda: {"G":0,"RS":0,"RA":0,"W":0})
    team_year_g = defaultdict(int)  # v1: (team, year) -> games played this season
    team_home_cum = defaultdict(lambda: {"G":0,"RS":0,"RA":0,"W":0})
    team_away_cum = defaultdict(lambda: {"G":0,"RS":0,"RA":0,"W":0})
    team_lg_cum = {"G":0,"RS":0,"RA":0,"W":0}
    team_lg_home_cum = {"G":0,"RS":0,"RA":0,"W":0}
    team_lg_away_cum = {"G":0,"RS":0,"RA":0,"W":0}
    bat_lg_cum = {"AB":0,"H":0,"BB":0,"HP":0,"SF":0,"TB":0,"HR":0,"SO":0,"PA":0}
    bat_recent_games = defaultdict(lambda: deque(maxlen=30))  # (ab,h,bb,hp,sf,tb,hr,so,pa)
    bat_order_cum = defaultdict(lambda: {"AB":0,"H":0,"BB":0,"HP":0,"SF":0,"TB":0,"HR":0,"SO":0,"PA":0})
    bat_hand_cum = defaultdict(lambda: {"AB":0,"H":0,"BB":0,"HP":0,"SF":0,"TB":0,"HR":0,"SO":0,"PA":0})
    bat_time_cum = defaultdict(lambda: {"AB":0,"H":0,"BB":0,"HP":0,"SF":0,"TB":0,"HR":0,"SO":0,"PA":0})
    bat_vs_throw_cum = defaultdict(lambda: {"AB":0,"H":0,"BB":0,"HP":0,"SF":0,"TB":0,"HR":0,"SO":0,"PA":0})
    pit_lg_cum = {"IP":0.0,"ER":0,"H":0,"BB":0,"SO":0,"HR":0,"G":0,"GS":0}
    pit_sp_all_cum = {"IP":0.0,"ER":0,"H":0,"BB":0,"SO":0,"HR":0,"G":0,"GS":0}
    pit_sp_side_cum = defaultdict(lambda: {"IP":0.0,"ER":0,"H":0,"BB":0,"SO":0,"HR":0,"G":0,"GS":0})
    pit_time_cum = defaultdict(lambda: {"IP":0.0,"ER":0,"H":0,"BB":0,"SO":0,"HR":0,"G":0,"GS":0})
    team_recent = defaultdict(lambda: deque(maxlen=ROLL_N))
    pit_recent = defaultdict(lambda: deque(maxlen=ROLL_N))
    team_last_game_date = {}
    team_game_dates = defaultdict(lambda: deque(maxlen=32))
    team_last_side = {}
    team_away_streak = defaultdict(int)
    team_consec_days = defaultdict(int)
    team_bp_usage = defaultdict(lambda: deque(maxlen=60))  # (date, np, ip, app) by game order
    team_bp_quality = defaultdict(lambda: deque(maxlen=60))  # (date, ip, h, bb, er, so) by game order
    team_opp_wp_hist = defaultdict(lambda: deque(maxlen=20))
    park_cum = defaultdict(lambda: {"G":0,"RUNS":0})
    lg_game_cum = {"G":0,"RUNS":0}
    pit_recent_usage = defaultdict(lambda: deque(maxlen=20))  # (date, np, ip, app)
    pit_last_game_date = {}

    # --- v1 NEW cumulative structures ---
    h2h_cum = defaultdict(lambda: {"G":0, "W":0})  # (teamA, teamB) -> record of teamA vs teamB
    team_year_end = {}  # (team, year) -> {"G","RS","RA","W"} snapshot at season end
    # key bullpen tracking
    bp_career_sh = defaultdict(int)          # (team, pitcher) -> cumulative S+HD this season
    bp_pitcher_daily_np = defaultdict(float)  # (team, pitcher, date) -> NP that day
    # foreign player average (for cold start)
    foreign_bat_year = defaultdict(lambda: {"AB":0,"H":0,"BB":0,"HP":0,"SF":0,"TB":0,"HR":0,"SO":0,"PA":0})
    foreign_pit_year = defaultdict(lambda: {"IP":0.0,"ER":0,"H":0,"BB":0,"SO":0,"HR":0,"G":0,"GS":0})
    # park HR factor
    park_hr_cum = defaultdict(lambda: {"G":0,"HR":0})
    lg_hr_cum = {"G":0,"HR":0}
    # pitcher SLG-against (need AB, TB per pitcher)
    pit_slg_cum = defaultdict(lambda: {"AB":0,"TB":0})
    pit_slg_year_cum = defaultdict(lambda: {"AB":0,"TB":0})
    pit_slg_lg_cum = {"AB":0,"TB":0}
    # pitcher OPS-against (need AB, H, BB, HP, SF, TB per reliever for bullpen OPS-against)
    bp_ops_cum = defaultdict(lambda: {"AB":0,"H":0,"BB":0,"HP":0,"SF":0,"TB":0})  # (team, pitcher)

    # group games by date
    games_by_date = defaultdict(list)
    for g in games:
        games_by_date[g["date"]].append(g)

    dates = sorted(games_by_date.keys())
    min_data_year = safe_int(str(dates[0])[:4], 2023) if dates else 2023

    fieldnames = [
        "date","s_no","s_code","homeTeam","awayTeam",
        "y_home_win","homeScore","awayScore",

        "home_team_G","home_team_RS_perG","home_team_RA_perG","home_team_winpct",
        "away_team_G","away_team_RS_perG","away_team_RA_perG","away_team_winpct",
        "diff_team_RS_perG","diff_team_RA_perG","diff_team_winpct",
        "home_team_pyth_winpct","away_team_pyth_winpct","diff_team_pyth_winpct",
        "home_team_prior_RS_perG","home_team_prior_RA_perG","home_team_prior_winpct",
        "away_team_prior_RS_perG","away_team_prior_RA_perG","away_team_prior_winpct",
        "diff_team_prior_RS_perG","diff_team_prior_RA_perG","diff_team_prior_winpct",
        "home_team_home_G","home_team_home_RS_perG","home_team_home_RA_perG","home_team_home_winpct",
        "away_team_away_G","away_team_away_RS_perG","away_team_away_RA_perG","away_team_away_winpct",
        "diff_team_homeaway_RS_perG","diff_team_homeaway_RA_perG","diff_team_homeaway_winpct",
        "home_team_home_prior_winpct","away_team_away_prior_winpct","diff_team_homeaway_prior_winpct",
        "league_home_winpct","home_field_adv_to_date",
        "league_type_regular","league_type_postseason","league_type_exhibition","league_type_other",
        "home_context_boost","expected_run_edge_homeaway",
        "home_team_r5_G","home_team_r5_RS_perG","home_team_r5_RA_perG","home_team_r5_winpct",
        "away_team_r5_G","away_team_r5_RS_perG","away_team_r5_RA_perG","away_team_r5_winpct",
        "diff_team_r5_RS_perG","diff_team_r5_RA_perG","diff_team_r5_winpct",
        "home_team_rest_days","away_team_rest_days","diff_team_rest_days",
        "home_team_consec_days","away_team_consec_days","diff_team_consec_days",
        "home_team_games_last7","away_team_games_last7","diff_team_games_last7",
        "home_team_away_streak","away_team_away_streak","diff_team_away_streak",
        "home_team_oppwp_avg","away_team_oppwp_avg","diff_team_oppwp_avg",
        "home_team_oppwp_r5","away_team_oppwp_r5","diff_team_oppwp_r5",
        # v1: H2H head-to-head
        "home_h2h_G","home_h2h_winpct","away_h2h_winpct","diff_h2h_winpct",
        "home_bp_np_l1","home_bp_np_l3","home_bp_ip_l3","home_bp_app_l3",
        "away_bp_np_l1","away_bp_np_l3","away_bp_ip_l3","away_bp_app_l3",
        "diff_bp_np_l3","diff_bp_ip_l3","diff_bp_app_l3",
        "home_bp_np_per_ip_l3","away_bp_np_per_ip_l3","diff_bp_np_per_ip_l3",
        "home_bp_np_per_app_l3","away_bp_np_per_app_l3","diff_bp_np_per_app_l3",
        "home_bp_whip_l3","away_bp_whip_l3","diff_bp_whip_l3",
        "home_bp_era_l3","away_bp_era_l3","diff_bp_era_l3",
        "home_bp_stress_index","away_bp_stress_index","diff_bp_stress_index",
        "home_bp_day_np_l1","home_bp_day_np_l3","home_bp_day_ip_l3","home_bp_day_app_l3",
        "away_bp_day_np_l1","away_bp_day_np_l3","away_bp_day_ip_l3","away_bp_day_app_l3",
        "diff_bp_day_np_l3","diff_bp_day_ip_l3","diff_bp_day_app_l3",
        "home_bp_day_np_per_ip_l3","away_bp_day_np_per_ip_l3","diff_bp_day_np_per_ip_l3",
        "home_bp_day_np_per_app_l3","away_bp_day_np_per_app_l3","diff_bp_day_np_per_app_l3",
        "home_bp_day_whip_l3","away_bp_day_whip_l3","diff_bp_day_whip_l3",
        "home_bp_day_era_l3","away_bp_day_era_l3","diff_bp_day_era_l3",
        # v1: key bullpen
        "home_key_bp_fatigue","away_key_bp_fatigue","diff_key_bp_fatigue",
        "home_key_bp_ops_against","away_key_bp_ops_against","diff_key_bp_ops_against",
        "weather_code","weather_temperature","weather_humidity","weather_wind_direction","weather_wind_speed","weather_rain_probability",
        "weather_unknown","weather_hot","weather_cold","weather_windy","weather_rainy",
        "game_month","game_weekday","game_is_weekend","game_hour","game_is_night","game_is_day",
        "park_run_factor",
        "park_hr_factor",

        "home_sp_sched_p_no","away_sp_sched_p_no",
        "home_sp_sched_missing","away_sp_sched_missing","diff_sp_sched_missing",
        "home_sp_replaced","away_sp_replaced","diff_sp_replaced",
        "home_sp_p_no","away_sp_p_no",
        "home_sp_G","home_sp_GS","home_sp_IP","home_sp_ERA","home_sp_WHIP","home_sp_K9","home_sp_BB9","home_sp_IP_per_start","home_sp_nohist",
        "away_sp_G","away_sp_GS","away_sp_IP","away_sp_ERA","away_sp_WHIP","away_sp_K9","away_sp_BB9","away_sp_IP_per_start","away_sp_nohist",
        "diff_sp_ERA","diff_sp_WHIP","diff_sp_K9","diff_sp_BB9",
        # v1: SP SLG-against / BB per IP / SO per IP
        "home_sp_SLG_against","away_sp_SLG_against","diff_sp_SLG_against",
        "home_sp_BB_per_IP","away_sp_BB_per_IP","diff_sp_BB_per_IP",
        "home_sp_SO_per_IP","away_sp_SO_per_IP","diff_sp_SO_per_IP",
        "home_sp_blend_ERA","home_sp_blend_WHIP","home_sp_blend_K9","home_sp_blend_BB9",
        "away_sp_blend_ERA","away_sp_blend_WHIP","away_sp_blend_K9","away_sp_blend_BB9",
        "diff_sp_blend_ERA","diff_sp_blend_WHIP","diff_sp_blend_K9","diff_sp_blend_BB9",
        "home_sp_prior_ERA","home_sp_prior_WHIP","home_sp_prior_K9","home_sp_prior_BB9",
        "away_sp_prior_ERA","away_sp_prior_WHIP","away_sp_prior_K9","away_sp_prior_BB9",
        "diff_sp_prior_ERA","diff_sp_prior_WHIP","diff_sp_prior_K9","diff_sp_prior_BB9",
        "home_sp_r5_G","home_sp_r5_ERA","home_sp_r5_WHIP","home_sp_r5_K9","home_sp_r5_BB9",
        "away_sp_r5_G","away_sp_r5_ERA","away_sp_r5_WHIP","away_sp_r5_K9","away_sp_r5_BB9",
        "diff_sp_r5_ERA","diff_sp_r5_WHIP","diff_sp_r5_K9","diff_sp_r5_BB9",
        "home_sp_form_conf","away_sp_form_conf","diff_sp_form_conf",
        "home_sp_form_ERA_adj","away_sp_form_ERA_adj","diff_sp_form_ERA_adj",
        "home_sp_form_WHIP_adj","away_sp_form_WHIP_adj","diff_sp_form_WHIP_adj",
        "home_sp_form_K9_adj","away_sp_form_K9_adj","diff_sp_form_K9_adj",
        "home_sp_time_split_ERA","away_sp_time_split_ERA","diff_sp_time_split_ERA",
        "home_sp_time_split_WHIP","away_sp_time_split_WHIP","diff_sp_time_split_WHIP",
        "home_sp_np_l1d","home_sp_np_l3d","home_sp_ip_l3d","home_sp_app_l3d","home_sp_rest_days",
        "away_sp_np_l1d","away_sp_np_l3d","away_sp_ip_l3d","away_sp_app_l3d","away_sp_rest_days",
        "diff_sp_np_l3d","diff_sp_ip_l3d","diff_sp_rest_days",
        "home_sp_fatigue_index","away_sp_fatigue_index","diff_sp_fatigue_index",
        "home_sp_short_rest","away_sp_short_rest","diff_sp_short_rest",
        "home_sp_expected_bp_ip","away_sp_expected_bp_ip","diff_sp_expected_bp_ip",
        "home_pitch_handoff_risk","away_pitch_handoff_risk","diff_pitch_handoff_risk",
        "home_sp_throw_unknown","away_sp_throw_unknown",
        "home_sp_throw_R","home_sp_throw_L","away_sp_throw_R","away_sp_throw_L",
        "home_sp_throw_under","away_sp_throw_under","diff_sp_throw_under",
        "home_sp_cold_foreign_prob","home_sp_cold_rookie_prob","home_sp_cold_returnee_prob","home_sp_hist_reliability",
        "away_sp_cold_foreign_prob","away_sp_cold_rookie_prob","away_sp_cold_returnee_prob","away_sp_hist_reliability",
        "diff_sp_cold_foreign_prob","diff_sp_hist_reliability",

        "home_lineup_avg_avg","home_lineup_avg_obp","home_lineup_avg_slg","home_lineup_avg_ops",
        "home_lineup_hr_per_pa","home_lineup_so_per_pa","home_lineup_bb_per_pa","home_lineup_avg_pa","home_lineup_nohist_cnt",
        "away_lineup_avg_avg","away_lineup_avg_obp","away_lineup_avg_slg","away_lineup_avg_ops",
        "away_lineup_hr_per_pa","away_lineup_so_per_pa","away_lineup_bb_per_pa","away_lineup_avg_pa","away_lineup_nohist_cnt",
        "diff_lineup_avg_ops","diff_lineup_hr_per_pa","diff_lineup_so_per_pa","diff_lineup_bb_per_pa","diff_lineup_nohist_cnt",
        "home_lineup_prior_avg_ops","home_lineup_prior_hr_per_pa","home_lineup_prior_so_per_pa",
        "away_lineup_prior_avg_ops","away_lineup_prior_hr_per_pa","away_lineup_prior_so_per_pa",
        "diff_lineup_prior_avg_ops","diff_lineup_prior_hr_per_pa","diff_lineup_prior_so_per_pa",
        "home_lineup_blend_ops","home_lineup_blend_hr_per_pa","home_lineup_blend_so_per_pa",
        "away_lineup_blend_ops","away_lineup_blend_hr_per_pa","away_lineup_blend_so_per_pa",
        "diff_lineup_blend_ops","diff_lineup_blend_hr_per_pa","diff_lineup_blend_so_per_pa",
        "home_lineup_r3_ops","home_lineup_r3_hr_per_pa","home_lineup_r3_so_per_pa",
        "away_lineup_r3_ops","away_lineup_r3_hr_per_pa","away_lineup_r3_so_per_pa",
        "diff_lineup_r3_ops","diff_lineup_r3_hr_per_pa","diff_lineup_r3_so_per_pa",
        "home_lineup_r7_ops","home_lineup_r7_hr_per_pa","home_lineup_r7_so_per_pa",
        "away_lineup_r7_ops","away_lineup_r7_hr_per_pa","away_lineup_r7_so_per_pa",
        "diff_lineup_r7_ops","diff_lineup_r7_hr_per_pa","diff_lineup_r7_so_per_pa",
        "home_lineup_time_split_ops","away_lineup_time_split_ops","diff_lineup_time_split_ops",
        "home_lineup_vs_throw_split_ops","away_lineup_vs_throw_split_ops","diff_lineup_vs_throw_split_ops",
        "home_lineup_vs_throw_split_hr_per_pa","away_lineup_vs_throw_split_hr_per_pa","diff_lineup_vs_throw_split_hr_per_pa",
        "home_lineup_vs_throw_split_so_per_pa","away_lineup_vs_throw_split_so_per_pa","diff_lineup_vs_throw_split_so_per_pa",
        "home_count_matchup_edge","away_count_matchup_edge","diff_count_matchup_edge",
        "home_lineup_momentum_ops","away_lineup_momentum_ops","diff_lineup_momentum_ops",
        "home_lineup_r14_ops","home_lineup_r14_hr_per_pa","home_lineup_r14_so_per_pa",
        "away_lineup_r14_ops","away_lineup_r14_hr_per_pa","away_lineup_r14_so_per_pa",
        "diff_lineup_r14_ops","diff_lineup_r14_hr_per_pa","diff_lineup_r14_so_per_pa",
        "home_lineup_known_cnt","away_lineup_known_cnt","diff_lineup_known_cnt",
        "home_lineup_missing_cnt","away_lineup_missing_cnt","diff_lineup_missing_cnt",
        "home_lineup_nohist_ratio","away_lineup_nohist_ratio","diff_lineup_nohist_ratio",
        "home_lineup_state_y_cnt","away_lineup_state_y_cnt","diff_lineup_state_y_cnt",
        "home_lineup_state_y_rate","away_lineup_state_y_rate","diff_lineup_state_y_rate",
        "home_lineup_hand1_cnt","home_lineup_hand2_cnt","home_lineup_hand3_cnt",
        "away_lineup_hand1_cnt","away_lineup_hand2_cnt","away_lineup_hand3_cnt",
        "home_lineup_vs_opp_sp_samehand_rate","away_lineup_vs_opp_sp_samehand_rate","diff_lineup_vs_opp_sp_samehand_rate",
        "home_lineup_vs_opp_sp_platoon_rate","away_lineup_vs_opp_sp_platoon_rate","diff_lineup_vs_opp_sp_platoon_rate",
        "home_lineup_platoon_edge","away_lineup_platoon_edge","diff_lineup_platoon_edge",
        "home_lineup_cold_foreign_prob_avg","home_lineup_cold_rookie_prob_avg","home_lineup_cold_returnee_prob_avg","home_lineup_hist_reliability_avg",
        "away_lineup_cold_foreign_prob_avg","away_lineup_cold_rookie_prob_avg","away_lineup_cold_returnee_prob_avg","away_lineup_hist_reliability_avg",
        "diff_lineup_cold_foreign_prob_avg","diff_lineup_hist_reliability_avg",
        "home_lineup_top3_blend_ops","home_lineup_bot6_blend_ops","home_lineup_top3_minus_bot6_blend_ops",
        "away_lineup_top3_blend_ops","away_lineup_bot6_blend_ops","away_lineup_top3_minus_bot6_blend_ops",
        "diff_lineup_top3_minus_bot6_blend_ops",
        # v1: lineup sum features (1-9 합산)
        "home_lineup_sum_ops","away_lineup_sum_ops","diff_lineup_sum_ops",
        "home_lineup_sum_slg","away_lineup_sum_slg","diff_lineup_sum_slg",
        "home_lineup_sum_hr_per_ab","away_lineup_sum_hr_per_ab","diff_lineup_sum_hr_per_ab",
        "home_lineup_sum_so_per_ab","away_lineup_sum_so_per_ab","diff_lineup_sum_so_per_ab",
        # v1: core5 (1-5번) vs bottom4 (6-9번)
        "home_lineup_core5_ops","away_lineup_core5_ops","diff_lineup_core5_ops",
        "home_lineup_core5_slg","away_lineup_core5_slg","diff_lineup_core5_slg",
        "home_lineup_core5_hr_per_ab","away_lineup_core5_hr_per_ab","diff_lineup_core5_hr_per_ab",
        "home_lineup_bottom4_ops","away_lineup_bottom4_ops","diff_lineup_bottom4_ops",
        # v1: cold start confidence
        "home_team_cold_start","away_team_cold_start","diff_team_cold_start",
    ]

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out_rows = []

    prev_year = None
    for d in dates:
        todays_games = games_by_date[d]
        year = safe_int(str(d)[:4])
        pregame_wp = {}

        # v1: snapshot team stats at year boundary for cold start prior
        if prev_year is not None and year != prev_year:
            for team_key, team_stats in team_cum.items():
                if isinstance(team_key, int) and team_stats["G"] > 0:
                    team_year_end[(team_key, prev_year)] = dict(team_stats)
        prev_year = year

        # 1) features computed using ONLY cumulative stats (<= previous day)
        for g in todays_games:
            s_no = safe_int(g["s_no"])
            home = safe_int(g["homeTeam"])
            away = safe_int(g["awayTeam"])
            s_code = safe_int(g.get("s_code"))

            hs_raw = safe_int_or_none(g.get("homeScore"))
            as_raw = safe_int_or_none(g.get("awayScore"))
            state_code = safe_int_or_none(g.get("state"))
            is_final_state = True if state_code is None else (state_code in (3, 5))
            has_result = (hs_raw is not None and as_raw is not None and is_final_state)
            hs = hs_raw if hs_raw is not None else 0
            as_ = as_raw if as_raw is not None else 0
            y = (1 if hs > as_ else 0) if has_result else ""
            weather_raw = g.get("weather")
            temp_raw = g.get("temperature")
            hum_raw = g.get("humidity")
            wdir_raw = g.get("windDirection")
            wsp_raw = g.get("windSpeed")
            rain_raw = g.get("rainprobability")
            weather_code = safe_int(weather_raw)
            weather_temp = safe_float(temp_raw)
            weather_humidity = safe_float(hum_raw)
            weather_wdir = safe_int(wdir_raw)
            weather_wspd = safe_float(wsp_raw)
            weather_rain = safe_float(rain_raw)
            weather_unknown = 1 if (
                weather_raw in (None, "", "None")
                and temp_raw in (None, "", "None")
                and hum_raw in (None, "", "None")
                and wdir_raw in (None, "", "None")
                and wsp_raw in (None, "", "None")
                and rain_raw in (None, "", "None")
            ) else 0
            weather_hot = 1 if weather_temp >= 28.0 else 0
            weather_cold = 1 if (weather_temp > 0 and weather_temp <= 10.0) else 0
            weather_windy = 1 if weather_wspd >= 4.0 else 0
            weather_rainy = 1 if weather_rain >= 50.0 else 0
            league_type_regular, league_type_post, league_type_exhi, league_type_other = league_type_flags(g.get("leagueType"))
            game_month = safe_int(str(d)[4:6], 0)
            game_weekday = parse_ymd(d).weekday()
            game_is_weekend = 1 if game_weekday >= 5 else 0
            game_hour = parse_game_hour(g.get("hm"))
            game_time_key = game_time_bucket(game_hour)
            game_is_night = 1 if game_hour >= 18 else 0
            game_is_day = 1 if (0 <= game_hour < 18) else 0

            # team prior (v1: cold start - use prev year team data as prior when <COLD_TEAM_G games)
            hg, hrs, hra, hwp = team_metrics(team_cum[home])
            ag, ars, ara, awp = team_metrics(team_cum[away])
            h_pyth = pythag_winpct(team_cum[home]["RS"], team_cum[home]["RA"])
            a_pyth = pythag_winpct(team_cum[away]["RS"], team_cum[away]["RA"])
            _, lrs, lra, lwp = team_metrics(team_lg_cum)

            def team_cold_prior(team, metric_fn):
                """Use prev year team data as prior if available, else league avg."""
                prev = team_year_end.get((team, year - 1))
                if prev and prev["G"] >= 50:
                    _, p_rs, p_ra, p_wp = team_metrics(prev)
                    return p_rs, p_ra, p_wp
                return lrs, lra, lwp

            h_pr_rs, h_pr_ra, h_pr_wp = team_cold_prior(home, team_metrics)
            a_pr_rs, a_pr_ra, a_pr_wp = team_cold_prior(away, team_metrics)
            h_prs = shrink(hrs, hg, h_pr_rs, TEAM_PRIOR_G)
            h_pra = shrink(hra, hg, h_pr_ra, TEAM_PRIOR_G)
            h_pwp = shrink(hwp, hg, h_pr_wp, TEAM_PRIOR_G)
            a_prs = shrink(ars, ag, a_pr_rs, TEAM_PRIOR_G)
            a_pra = shrink(ara, ag, a_pr_ra, TEAM_PRIOR_G)
            a_pwp = shrink(awp, ag, a_pr_wp, TEAM_PRIOR_G)
            hhg, hhrs, hhra, hhwp = team_metrics(team_home_cum[home])
            aag, aars, aara, aawp = team_metrics(team_away_cum[away])
            _, _, _, l_home_wp = team_metrics(team_lg_home_cum)
            _, _, _, l_away_wp = team_metrics(team_lg_away_cum)
            h_home_pwp = shrink(hhwp, hhg, l_home_wp, TEAM_PRIOR_G)
            a_away_pwp = shrink(aawp, aag, l_away_wp, TEAM_PRIOR_G)
            home_field_adv_to_date = (l_home_wp - 0.5) if team_lg_home_cum["G"] > 0 else 0.0
            home_context_boost = (hhwp - hwp) - (aawp - awp) + home_field_adv_to_date
            expected_run_edge_homeaway = (hhrs - aara) - (aars - hhra)
            pregame_wp[s_no] = (hwp, awp)

            h_r5g, h_r5rs, h_r5ra, h_r5wp = team_recent_metrics(team_recent[home])
            a_r5g, a_r5rs, a_r5ra, a_r5wp = team_recent_metrics(team_recent[away])
            h_rest = rest_days(team_last_game_date.get(home), d)
            a_rest = rest_days(team_last_game_date.get(away), d)
            h_consec = team_consec_days.get(home, 0)
            a_consec = team_consec_days.get(away, 0)
            h_g7 = recent_count_by_days(team_game_dates[home], d, SCHEDULE_WIN_DAYS)
            a_g7 = recent_count_by_days(team_game_dates[away], d, SCHEDULE_WIN_DAYS)
            h_away_streak = team_away_streak.get(home, 0)
            a_away_streak = team_away_streak.get(away, 0)

            h_opp_hist = list(team_opp_wp_hist[home])
            a_opp_hist = list(team_opp_wp_hist[away])
            h_opp_avg = mean_or_zero(h_opp_hist)
            a_opp_avg = mean_or_zero(a_opp_hist)
            h_opp_r5 = mean_or_zero(h_opp_hist[-5:])
            a_opp_r5 = mean_or_zero(a_opp_hist[-5:])

            # v1: H2H head-to-head
            h2h_hg = h2h_cum[(home, away)]["G"]
            h2h_hw = h2h_cum[(home, away)]["W"]
            h2h_aw = h2h_cum[(away, home)]["W"]
            h2h_h_wp = div(h2h_hw, h2h_hg) if h2h_hg > 0 else 0.5
            h2h_a_wp = div(h2h_aw, h2h_hg) if h2h_hg > 0 else 0.5

            # bullpen windows use recent games (not recent days)
            h_bp_np_l1, _, _ = recent_pitch_usage_by_games(team_bp_usage[home], 1)
            h_bp_np_l3, h_bp_ip_l3, h_bp_app_l3 = recent_pitch_usage_by_games(team_bp_usage[home], 3)
            a_bp_np_l1, _, _ = recent_pitch_usage_by_games(team_bp_usage[away], 1)
            a_bp_np_l3, a_bp_ip_l3, a_bp_app_l3 = recent_pitch_usage_by_games(team_bp_usage[away], 3)
            h_bp_np_per_ip_l3 = div(h_bp_np_l3, h_bp_ip_l3)
            a_bp_np_per_ip_l3 = div(a_bp_np_l3, a_bp_ip_l3)
            h_bp_np_per_app_l3 = div(h_bp_np_l3, h_bp_app_l3)
            a_bp_np_per_app_l3 = div(a_bp_np_l3, a_bp_app_l3)
            _, h_bp_whip_l3, h_bp_era_l3, _, _ = recent_bp_quality_by_games(team_bp_quality[home], 3)
            _, a_bp_whip_l3, a_bp_era_l3, _, _ = recent_bp_quality_by_games(team_bp_quality[away], 3)
            h_bp_stress_idx = h_bp_np_per_ip_l3 * (1.0 + div(h_bp_era_l3, 9.0))
            a_bp_stress_idx = a_bp_np_per_ip_l3 * (1.0 + div(a_bp_era_l3, 9.0))
            # hybrid: keep day-window bullpen usage as separate features
            h_bp_day_np_l1, _, _ = recent_pitch_usage_by_days(team_bp_usage[home], d, 1)
            h_bp_day_np_l3, h_bp_day_ip_l3, h_bp_day_app_l3 = recent_pitch_usage_by_days(team_bp_usage[home], d, 3)
            a_bp_day_np_l1, _, _ = recent_pitch_usage_by_days(team_bp_usage[away], d, 1)
            a_bp_day_np_l3, a_bp_day_ip_l3, a_bp_day_app_l3 = recent_pitch_usage_by_days(team_bp_usage[away], d, 3)
            h_bp_day_np_per_ip_l3 = div(h_bp_day_np_l3, h_bp_day_ip_l3)
            a_bp_day_np_per_ip_l3 = div(a_bp_day_np_l3, a_bp_day_ip_l3)
            h_bp_day_np_per_app_l3 = div(h_bp_day_np_l3, h_bp_day_app_l3)
            a_bp_day_np_per_app_l3 = div(a_bp_day_np_l3, a_bp_day_app_l3)
            _, h_bp_day_whip_l3, h_bp_day_era_l3, _, _ = recent_bp_quality_by_days(team_bp_quality[home], d, 3)
            _, a_bp_day_whip_l3, a_bp_day_era_l3, _, _ = recent_bp_quality_by_days(team_bp_quality[away], d, 3)

            # v1: key bullpen fatigue & OPS-against
            def key_bp_features(team, cur_date):
                # identify key relievers: top KEY_BP_N by cumulative S+HD
                all_bp = [(p, sh) for (t, p), sh in bp_career_sh.items() if t == team]
                all_bp.sort(key=lambda x: -x[1])
                key_pitchers = [p for p, _ in all_bp[:KEY_BP_N]]
                if not key_pitchers:
                    return 0.0, 0.0
                # fatigue = sum over key pitchers of (yesterday_NP*5 + 2days_ago_NP*3 + 3days_ago_NP*1)
                fatigue = 0.0
                for p in key_pitchers:
                    for days_back, w in enumerate(BP_FATIGUE_W, 1):
                        try:
                            past_d = int((parse_ymd(cur_date) - __import__('datetime').timedelta(days=days_back)).strftime("%Y%m%d"))
                        except:
                            continue
                        fatigue += bp_pitcher_daily_np.get((team, p, str(past_d)), 0.0) * w
                # OPS-against = sum of individual OPS-against for key pitchers
                ops_sum = 0.0
                for p in key_pitchers:
                    c = bp_ops_cum[(team, p)]
                    ab = c["AB"]; h = c["H"]; bb = c["BB"]; hp = c["HP"]; sf = c["SF"]; tb = c["TB"]
                    obp = div(h + bb + hp, ab + bb + hp + sf)
                    slg = div(tb, ab)
                    ops_sum += obp + slg
                ops_avg = div(ops_sum, len(key_pitchers))
                return fatigue, ops_avg

            h_key_bp_fatigue, h_key_bp_ops = key_bp_features(home, d)
            a_key_bp_fatigue, a_key_bp_ops = key_bp_features(away, d)

            park_g = park_cum[s_code]["G"] if s_code else 0
            park_rpg = div(park_cum[s_code]["RUNS"], park_g) if park_g else 0.0
            lg_rpg = div(lg_game_cum["RUNS"], lg_game_cum["G"])
            park_factor = div(park_rpg, lg_rpg) if (park_g >= 20 and lg_game_cum["G"] >= 50 and lg_rpg > 0) else 1.0
            # v1: park HR factor
            park_hr_g = park_hr_cum[s_code]["G"] if s_code else 0
            park_hrpg = div(park_hr_cum[s_code]["HR"], park_hr_g) if park_hr_g else 0.0
            lg_hrpg = div(lg_hr_cum["HR"], lg_hr_cum["G"]) if lg_hr_cum["G"] > 0 else 0.0
            park_hr_factor = div(park_hrpg, lg_hrpg) if (park_hr_g >= 20 and lg_hr_cum["G"] >= 50 and lg_hrpg > 0) else 1.0

            # starters from lineup
            home_sp = lineup_map[s_no]["home"]["P"]
            away_sp = lineup_map[s_no]["away"]["P"]
            home_sp_throw = lineup_map[s_no]["home"]["P_throw"]
            away_sp_throw = lineup_map[s_no]["away"]["P_throw"]
            home_sp_name = lineup_map[s_no]["home"]["P_name"] or player_meta[home_sp]["name"] if home_sp else ""
            away_sp_name = lineup_map[s_no]["away"]["P_name"] or player_meta[away_sp]["name"] if away_sp else ""
            month = safe_int(str(d)[4:6], 0)
            home_sp_sched = safe_int(g.get("homeSP"))
            away_sp_sched = safe_int(g.get("awaySP"))
            home_sp_sched_missing = 1 if not home_sp_sched else 0
            away_sp_sched_missing = 1 if not away_sp_sched else 0
            home_sp_replaced = 1 if (home_sp and home_sp_sched and home_sp != home_sp_sched) else 0
            away_sp_replaced = 1 if (away_sp and away_sp_sched and away_sp != away_sp_sched) else 0

            # pitcher prior
            h_era,h_whip,h_k9,h_bb9,h_ip,h_pg,h_pgs,h_ipps = pitcher_metrics(pit_cum[home_sp]) if home_sp else (0,0,0,0,0,0,0,0)
            a_era,a_whip,a_k9,a_bb9,a_ip,a_pg,a_pgs,a_ipps = pitcher_metrics(pit_cum[away_sp]) if away_sp else (0,0,0,0,0,0,0,0)
            l_era,l_whip,l_k9,l_bb9,_,_,_,_ = pitcher_metrics(pit_lg_cum)
            h_pera = shrink(h_era, h_ip, l_era, PIT_PRIOR_IP) if home_sp else 0.0
            h_pwhip = shrink(h_whip, h_ip, l_whip, PIT_PRIOR_IP) if home_sp else 0.0
            h_pk9 = shrink(h_k9, h_ip, l_k9, PIT_PRIOR_IP) if home_sp else 0.0
            h_pbb9 = shrink(h_bb9, h_ip, l_bb9, PIT_PRIOR_IP) if home_sp else 0.0
            a_pera = shrink(a_era, a_ip, l_era, PIT_PRIOR_IP) if away_sp else 0.0
            a_pwhip = shrink(a_whip, a_ip, l_whip, PIT_PRIOR_IP) if away_sp else 0.0
            a_pk9 = shrink(a_k9, a_ip, l_k9, PIT_PRIOR_IP) if away_sp else 0.0
            a_pbb9 = shrink(a_bb9, a_ip, l_bb9, PIT_PRIOR_IP) if away_sp else 0.0

            # v1: SP SLG-against, BB/IP, SO/IP
            h_sp_slg_a = div(pit_slg_cum[home_sp]["TB"], pit_slg_cum[home_sp]["AB"]) if home_sp else 0.0
            a_sp_slg_a = div(pit_slg_cum[away_sp]["TB"], pit_slg_cum[away_sp]["AB"]) if away_sp else 0.0
            h_sp_bb_per_ip = div(pit_cum[home_sp]["BB"], h_ip) if home_sp else 0.0
            a_sp_bb_per_ip = div(pit_cum[away_sp]["BB"], a_ip) if away_sp else 0.0
            h_sp_so_per_ip = div(pit_cum[home_sp]["SO"], h_ip) if home_sp else 0.0
            a_sp_so_per_ip = div(pit_cum[away_sp]["SO"], a_ip) if away_sp else 0.0

            def sp_side_prior(throw_code):
                side = throw_side(throw_code)
                base = pit_sp_side_cum[side] if (side and pit_sp_side_cum[side]["IP"] >= PIT_PRIOR_IP) else pit_sp_all_cum
                if base["IP"] <= 0:
                    base = pit_lg_cum
                era, whip, k9, bb9, _, _, _, _ = pitcher_metrics(base)
                return era, whip, k9, bb9

            if home_sp:
                h_cur = pit_year_cum[(home_sp, year)]
                h_prev = pit_year_cum[(home_sp, year - 1)]
                h_car = pit_sub(pit_cum[home_sp], h_cur)
                h_cur_era,h_cur_whip,h_cur_k9,h_cur_bb9,h_cur_ip,_,_,_ = pitcher_metrics(h_cur)
                h_prev_era,h_prev_whip,h_prev_k9,h_prev_bb9,h_prev_ip,_,_,_ = pitcher_metrics(h_prev)
                h_car_era,h_car_whip,h_car_k9,h_car_bb9,h_car_ip,_,_,_ = pitcher_metrics(h_car)
                h_bera = blend_four(h_cur_era, h_cur_ip, h_prev_era, h_prev_ip, h_car_era, h_car_ip, l_era, PIT_PRIOR_IP)
                h_bwhip = blend_four(h_cur_whip, h_cur_ip, h_prev_whip, h_prev_ip, h_car_whip, h_car_ip, l_whip, PIT_PRIOR_IP)
                h_bk9 = blend_four(h_cur_k9, h_cur_ip, h_prev_k9, h_prev_ip, h_car_k9, h_car_ip, l_k9, PIT_PRIOR_IP)
                h_bbb9 = blend_four(h_cur_bb9, h_cur_ip, h_prev_bb9, h_prev_ip, h_car_bb9, h_car_ip, l_bb9, PIT_PRIOR_IP)
                h_hist_n = h_cur_ip + PREV_YEAR_W * h_prev_ip + CAREER_W * h_car_ip
                h_hist_rel = div(h_hist_n, h_hist_n + PIT_PRIOR_IP)
                h_pf, h_pr, h_prt = estimate_pitcher_cold_probs(
                    home_sp_name, h_cur_ip, h_prev_ip, h_car_ip, month, manual_tags.get(home_sp),
                    player_meta[home_sp]["first_year"], year, min_data_year
                )
                h_role_era, h_role_whip, h_role_k9, h_role_bb9 = sp_side_prior(home_sp_throw)
                h_fore_era = better_low(h_role_era, l_era, 0.96)
                h_fore_whip = better_low(h_role_whip, l_whip, 0.97)
                h_fore_k9 = worse_high(h_role_k9, l_k9, 1.05)
                h_fore_bb9 = better_low(h_role_bb9, l_bb9, 0.98)
                h_rook_era = worse_high(h_role_era, l_era, 1.08)
                h_rook_whip = worse_high(h_role_whip, l_whip, 1.08)
                h_rook_k9 = better_low(h_role_k9, l_k9, 0.95)
                h_rook_bb9 = worse_high(h_role_bb9, l_bb9, 1.08)
                h_ret_era, h_ret_whip, h_ret_k9, h_ret_bb9 = h_role_era, h_role_whip, h_role_k9, h_role_bb9
                h_cold_era = (h_pf * h_fore_era) + (h_pr * h_rook_era) + (h_prt * h_ret_era)
                h_cold_whip = (h_pf * h_fore_whip) + (h_pr * h_rook_whip) + (h_prt * h_ret_whip)
                h_cold_k9 = (h_pf * h_fore_k9) + (h_pr * h_rook_k9) + (h_prt * h_ret_k9)
                h_cold_bb9 = (h_pf * h_fore_bb9) + (h_pr * h_rook_bb9) + (h_prt * h_ret_bb9)
                h_bera = (h_hist_rel * h_bera) + ((1.0 - h_hist_rel) * h_cold_era)
                h_bwhip = (h_hist_rel * h_bwhip) + ((1.0 - h_hist_rel) * h_cold_whip)
                h_bk9 = (h_hist_rel * h_bk9) + ((1.0 - h_hist_rel) * h_cold_k9)
                h_bbb9 = (h_hist_rel * h_bbb9) + ((1.0 - h_hist_rel) * h_cold_bb9)
            else:
                h_bera,h_bwhip,h_bk9,h_bbb9 = 0.0,0.0,0.0,0.0
                h_pf, h_pr, h_prt, h_hist_rel = 0.0, 0.0, 0.0, 0.0

            if away_sp:
                a_cur = pit_year_cum[(away_sp, year)]
                a_prev = pit_year_cum[(away_sp, year - 1)]
                a_car = pit_sub(pit_cum[away_sp], a_cur)
                a_cur_era,a_cur_whip,a_cur_k9,a_cur_bb9,a_cur_ip,_,_,_ = pitcher_metrics(a_cur)
                a_prev_era,a_prev_whip,a_prev_k9,a_prev_bb9,a_prev_ip,_,_,_ = pitcher_metrics(a_prev)
                a_car_era,a_car_whip,a_car_k9,a_car_bb9,a_car_ip,_,_,_ = pitcher_metrics(a_car)
                a_bera = blend_four(a_cur_era, a_cur_ip, a_prev_era, a_prev_ip, a_car_era, a_car_ip, l_era, PIT_PRIOR_IP)
                a_bwhip = blend_four(a_cur_whip, a_cur_ip, a_prev_whip, a_prev_ip, a_car_whip, a_car_ip, l_whip, PIT_PRIOR_IP)
                a_bk9 = blend_four(a_cur_k9, a_cur_ip, a_prev_k9, a_prev_ip, a_car_k9, a_car_ip, l_k9, PIT_PRIOR_IP)
                a_bbb9 = blend_four(a_cur_bb9, a_cur_ip, a_prev_bb9, a_prev_ip, a_car_bb9, a_car_ip, l_bb9, PIT_PRIOR_IP)
                a_hist_n = a_cur_ip + PREV_YEAR_W * a_prev_ip + CAREER_W * a_car_ip
                a_hist_rel = div(a_hist_n, a_hist_n + PIT_PRIOR_IP)
                a_pf, a_pr, a_prt = estimate_pitcher_cold_probs(
                    away_sp_name, a_cur_ip, a_prev_ip, a_car_ip, month, manual_tags.get(away_sp),
                    player_meta[away_sp]["first_year"], year, min_data_year
                )
                a_role_era, a_role_whip, a_role_k9, a_role_bb9 = sp_side_prior(away_sp_throw)
                a_fore_era = better_low(a_role_era, l_era, 0.96)
                a_fore_whip = better_low(a_role_whip, l_whip, 0.97)
                a_fore_k9 = worse_high(a_role_k9, l_k9, 1.05)
                a_fore_bb9 = better_low(a_role_bb9, l_bb9, 0.98)
                a_rook_era = worse_high(a_role_era, l_era, 1.08)
                a_rook_whip = worse_high(a_role_whip, l_whip, 1.08)
                a_rook_k9 = better_low(a_role_k9, l_k9, 0.95)
                a_rook_bb9 = worse_high(a_role_bb9, l_bb9, 1.08)
                a_ret_era, a_ret_whip, a_ret_k9, a_ret_bb9 = a_role_era, a_role_whip, a_role_k9, a_role_bb9
                a_cold_era = (a_pf * a_fore_era) + (a_pr * a_rook_era) + (a_prt * a_ret_era)
                a_cold_whip = (a_pf * a_fore_whip) + (a_pr * a_rook_whip) + (a_prt * a_ret_whip)
                a_cold_k9 = (a_pf * a_fore_k9) + (a_pr * a_rook_k9) + (a_prt * a_ret_k9)
                a_cold_bb9 = (a_pf * a_fore_bb9) + (a_pr * a_rook_bb9) + (a_prt * a_ret_bb9)
                a_bera = (a_hist_rel * a_bera) + ((1.0 - a_hist_rel) * a_cold_era)
                a_bwhip = (a_hist_rel * a_bwhip) + ((1.0 - a_hist_rel) * a_cold_whip)
                a_bk9 = (a_hist_rel * a_bk9) + ((1.0 - a_hist_rel) * a_cold_k9)
                a_bbb9 = (a_hist_rel * a_bbb9) + ((1.0 - a_hist_rel) * a_cold_bb9)
            else:
                a_bera,a_bwhip,a_bk9,a_bbb9 = 0.0,0.0,0.0,0.0
                a_pf, a_pr, a_prt, a_hist_rel = 0.0, 0.0, 0.0, 0.0

            h_sp_r5g,h_sp_r5era,h_sp_r5whip,h_sp_r5k9,h_sp_r5bb9 = pitcher_recent_metrics(pit_recent[home_sp]) if home_sp else (0,0,0,0,0)
            a_sp_r5g,a_sp_r5era,a_sp_r5whip,a_sp_r5k9,a_sp_r5bb9 = pitcher_recent_metrics(pit_recent[away_sp]) if away_sp else (0,0,0,0,0)
            h_sp_np_l1d, _, _ = recent_pitch_usage_by_days(pit_recent_usage[home_sp], d, 1) if home_sp else (0.0, 0.0, 0)
            h_sp_np_l3d, h_sp_ip_l3d, h_sp_app_l3d = recent_pitch_usage_by_days(pit_recent_usage[home_sp], d, 3) if home_sp else (0.0, 0.0, 0)
            a_sp_np_l1d, _, _ = recent_pitch_usage_by_days(pit_recent_usage[away_sp], d, 1) if away_sp else (0.0, 0.0, 0)
            a_sp_np_l3d, a_sp_ip_l3d, a_sp_app_l3d = recent_pitch_usage_by_days(pit_recent_usage[away_sp], d, 3) if away_sp else (0.0, 0.0, 0)
            h_sp_rest_days = rest_days(pit_last_game_date.get(home_sp), d) if home_sp else -1
            a_sp_rest_days = rest_days(pit_last_game_date.get(away_sp), d) if away_sp else -1
            h_sp_form_conf = min(1.0, div(h_sp_r5g, float(ROLL_N)))
            a_sp_form_conf = min(1.0, div(a_sp_r5g, float(ROLL_N)))
            h_sp_form_era_adj = (h_sp_r5era - h_bera) * h_sp_form_conf
            a_sp_form_era_adj = (a_sp_r5era - a_bera) * a_sp_form_conf
            h_sp_form_whip_adj = (h_sp_r5whip - h_bwhip) * h_sp_form_conf
            a_sp_form_whip_adj = (a_sp_r5whip - a_bwhip) * a_sp_form_conf
            h_sp_form_k9_adj = (h_sp_r5k9 - h_bk9) * h_sp_form_conf
            a_sp_form_k9_adj = (a_sp_r5k9 - a_bk9) * a_sp_form_conf

            def starter_fatigue(np_l3d, rest_days):
                if rest_days < 0:
                    return 0.0
                return div(np_l3d, max(1.0, float(rest_days + 1)))

            h_sp_fatigue_idx = starter_fatigue(h_sp_np_l3d, h_sp_rest_days)
            a_sp_fatigue_idx = starter_fatigue(a_sp_np_l3d, a_sp_rest_days)
            h_sp_short_rest = 1 if (0 <= h_sp_rest_days <= 3) else 0
            a_sp_short_rest = 1 if (0 <= a_sp_rest_days <= 3) else 0
            h_sp_expected_bp_ip = max(0.0, 9.0 - h_ipps)
            a_sp_expected_bp_ip = max(0.0, 9.0 - a_ipps)
            h_pitch_handoff_risk = h_sp_expected_bp_ip * h_bp_np_per_ip_l3
            a_pitch_handoff_risk = a_sp_expected_bp_ip * a_bp_np_per_ip_l3
            h_sp_time_era = h_bera
            h_sp_time_whip = h_bwhip
            a_sp_time_era = a_bera
            a_sp_time_whip = a_bwhip
            if home_sp:
                h_t_era, h_t_whip, _, _, h_t_ip, _, _, _ = pitcher_metrics(pit_time_cum[(home_sp, game_time_key)])
                h_sp_time_era = shrink(h_t_era, h_t_ip, h_bera, PIT_PRIOR_IP)
                h_sp_time_whip = shrink(h_t_whip, h_t_ip, h_bwhip, PIT_PRIOR_IP)
            if away_sp:
                a_t_era, a_t_whip, _, _, a_t_ip, _, _, _ = pitcher_metrics(pit_time_cum[(away_sp, game_time_key)])
                a_sp_time_era = shrink(a_t_era, a_t_ip, a_bera, PIT_PRIOR_IP)
                a_sp_time_whip = shrink(a_t_whip, a_t_ip, a_bwhip, PIT_PRIOR_IP)

            # lineup prior (batters 1~9)
            def lineup_agg(side, opp_sp_throw):
                batters = lineup_map[s_no][side]["batters"]
                bat_hands = lineup_map[s_no][side]["bat_hands"]
                bat_names = lineup_map[s_no][side]["bat_names"]
                bat_states = lineup_map[s_no][side]["bat_states"]
                p_info = []
                for i in range(1, 10):
                    p = batters.get(i)
                    if p:
                        p_info.append((i, p, bat_hands.get(i, ""), bat_names.get(i, ""), bat_states.get(i, "")))

                avgs=[]; obps=[]; slgs=[]; opss=[]; pas=[]
                hr_sum=0; so_sum=0; bb_sum=0; pa_sum=0
                prior_opss=[]; prior_hrs=[]; prior_sos=[]
                blend_opss=[]; blend_hrs=[]; blend_sos=[]
                r3_opss=[]; r3_hrs=[]; r3_sos=[]
                r7_opss=[]; r7_hrs=[]; r7_sos=[]
                time_opss=[]
                throw_opss=[]; throw_hrs=[]; throw_sos=[]
                r14_opss=[]; r14_hrs=[]; r14_sos=[]
                top3_blend=[]; bot6_blend=[]
                # v1: per-batter SLG, HR/AB for sum features
                all_slgs=[]; all_hr_abs=[]; all_so_abs=[]
                core5_opss=[]; core5_slgs=[]; core5_hr_abs=[]
                bottom4_opss=[]
                hand1=0; hand2=0; hand3=0
                samehand=0
                platoon=0
                state_y_cnt = 0
                nohist=0
                cold_foreign_sum=0.0; cold_rookie_sum=0.0; cold_returnee_sum=0.0; hist_rel_sum=0.0
                opp_sp_side = throw_side(opp_sp_throw)
                _,_,_,l_ops,l_hr,l_so,_ = batter_metrics(bat_lg_cum)

                for order, p, hand, name, state in p_info:
                    cum = bat_cum[p]
                    avg, obp, slg, ops, hr_rate, so_rate, pa = batter_metrics(cum)
                    if pa == 0:
                        nohist += 1
                    avgs.append(avg); obps.append(obp); slgs.append(slg); opss.append(ops); pas.append(pa)
                    prior_opss.append(shrink(ops, pa, l_ops, BAT_PRIOR_PA))
                    prior_hrs.append(shrink(hr_rate, pa, l_hr, BAT_PRIOR_PA))
                    prior_sos.append(shrink(so_rate, pa, l_so, BAT_PRIOR_PA))
                    _, _, _, t_ops, _, _, t_pa = batter_metrics(bat_time_cum[(p, game_time_key)])
                    time_opss.append(shrink(t_ops, t_pa, l_ops, BAT_PRIOR_PA))
                    if opp_sp_side:
                        _, _, _, s_ops, s_hr, s_so, s_pa = batter_metrics(bat_vs_throw_cum[(p, opp_sp_side)])
                        throw_opss.append(shrink(s_ops, s_pa, l_ops, BAT_PRIOR_PA))
                        throw_hrs.append(shrink(s_hr, s_pa, l_hr, BAT_PRIOR_PA))
                        throw_sos.append(shrink(s_so, s_pa, l_so, BAT_PRIOR_PA))
                    else:
                        throw_opss.append(shrink(ops, pa, l_ops, BAT_PRIOR_PA))
                        throw_hrs.append(shrink(hr_rate, pa, l_hr, BAT_PRIOR_PA))
                        throw_sos.append(shrink(so_rate, pa, l_so, BAT_PRIOR_PA))

                    cur = bat_year_cum[(p, year)]
                    prev = bat_year_cum[(p, year - 1)]
                    car = bat_sub(cum, cur)
                    _,_,_,cur_ops,cur_hr,cur_so,cur_pa = batter_metrics(cur)
                    _,_,_,prev_ops,prev_hr,prev_so,prev_pa = batter_metrics(prev)
                    _,_,_,car_ops,car_hr,car_so,car_pa = batter_metrics(car)
                    b_ops = blend_four(cur_ops, cur_pa, prev_ops, prev_pa, car_ops, car_pa, l_ops, BAT_PRIOR_PA)
                    b_hr = blend_four(cur_hr, cur_pa, prev_hr, prev_pa, car_hr, car_pa, l_hr, BAT_PRIOR_PA)
                    b_so = blend_four(cur_so, cur_pa, prev_so, prev_pa, car_so, car_pa, l_so, BAT_PRIOR_PA)

                    role = order_bucket(order)
                    role_cum = bat_order_cum[role]
                    _,_,_,role_ops,role_hr,role_so,role_pa = batter_metrics(role_cum)
                    if role_pa < BAT_PRIOR_PA:
                        role_ops, role_hr, role_so = l_ops, l_hr, l_so
                    hand_cum = bat_hand_cum[hand]
                    _,_,_,hand_ops,hand_hr,hand_so,hand_pa = batter_metrics(hand_cum)
                    if hand_pa < BAT_PRIOR_PA:
                        hand_ops, hand_hr, hand_so = l_ops, l_hr, l_so
                    base_ops = (0.6 * role_ops) + (0.4 * hand_ops)
                    base_hr = (0.6 * role_hr) + (0.4 * hand_hr)
                    base_so = (0.6 * role_so) + (0.4 * hand_so)

                    pf, pr, prt = estimate_batter_cold_probs(
                        name, order, state, cur_pa, prev_pa, car_pa, month, manual_tags.get(p),
                        player_meta[p]["first_year"], year, min_data_year
                    )
                    hist_n = cur_pa + (PREV_YEAR_W * prev_pa) + (CAREER_W * car_pa)
                    hist_rel = div(hist_n, hist_n + BAT_PRIOR_PA)

                    f_ops = worse_high(base_ops, l_ops, 1.04)
                    f_hr = worse_high(base_hr, l_hr, 1.10)
                    f_so = better_low(base_so, l_so, 0.98)
                    r_ops = better_low(base_ops, l_ops, 0.94)
                    r_hr = better_low(base_hr, l_hr, 0.90)
                    r_so = worse_high(base_so, l_so, 1.08)
                    rt_ops, rt_hr, rt_so = base_ops, base_hr, base_so
                    cold_ops = (pf * f_ops) + (pr * r_ops) + (prt * rt_ops)
                    cold_hr = (pf * f_hr) + (pr * r_hr) + (prt * rt_hr)
                    cold_so = (pf * f_so) + (pr * r_so) + (prt * rt_so)

                    b_ops = (hist_rel * b_ops) + ((1.0 - hist_rel) * cold_ops)
                    b_hr = (hist_rel * b_hr) + ((1.0 - hist_rel) * cold_hr)
                    b_so = (hist_rel * b_so) + ((1.0 - hist_rel) * cold_so)
                    blend_opss.append(b_ops)
                    blend_hrs.append(b_hr)
                    blend_sos.append(b_so)
                    cold_foreign_sum += pf
                    cold_rookie_sum += pr
                    cold_returnee_sum += prt
                    hist_rel_sum += hist_rel

                    _, r3_ops, r3_hr, r3_so, _ = batter_recent_metrics_by_games(bat_recent_games[p], 3)
                    _, r7_ops, r7_hr, r7_so, _ = batter_recent_metrics_by_games(bat_recent_games[p], 7)
                    _, r14_ops, r14_hr, r14_so, _ = batter_recent_metrics_by_games(bat_recent_games[p], 14)
                    r3_opss.append(r3_ops); r3_hrs.append(r3_hr); r3_sos.append(r3_so)
                    r7_opss.append(r7_ops); r7_hrs.append(r7_hr); r7_sos.append(r7_so)
                    r14_opss.append(r14_ops); r14_hrs.append(r14_hr); r14_sos.append(r14_so)
                    if order <= 3:
                        top3_blend.append(b_ops)
                    else:
                        bot6_blend.append(b_ops)
                    # v1: collect per-batter SLG, HR/AB for sum
                    all_slgs.append(slg)
                    all_hr_abs.append(div(cum["HR"], cum["AB"]) if cum["AB"] > 0 else 0.0)
                    all_so_abs.append(div(cum["SO"], cum["AB"]) if cum["AB"] > 0 else 0.0)
                    if order <= 5:
                        core5_opss.append(b_ops)
                        core5_slgs.append(slg)
                        core5_hr_abs.append(div(cum["HR"], cum["AB"]) if cum["AB"] > 0 else 0.0)
                    else:
                        bottom4_opss.append(b_ops)

                    if hand == "1":
                        hand1 += 1
                    elif hand == "2":
                        hand2 += 1
                    elif hand == "3":
                        hand3 += 1
                    bat_side = bat_side_vs_throw(hand, opp_sp_side)
                    if opp_sp_side and bat_side and bat_side == opp_sp_side:
                        samehand += 1
                    if opp_sp_side and bat_side and bat_side != opp_sp_side:
                        platoon += 1
                    if state == "Y":
                        state_y_cnt += 1

                    # rates는 합산해서 다시 계산(팀 기준)
                    hr_sum += cum["HR"]
                    so_sum += cum["SO"]
                    bb_sum += cum["BB"]
                    pa_sum += cum["PA"]

                n = len(p_info)
                avg_avg = sum(avgs)/n if n else 0.0
                avg_obp = sum(obps)/n if n else 0.0
                avg_slg = sum(slgs)/n if n else 0.0
                avg_ops = sum(opss)/n if n else 0.0
                hr_per_pa = div(hr_sum, pa_sum)
                so_per_pa = div(so_sum, pa_sum)
                bb_per_pa = div(bb_sum, pa_sum)
                avg_pa = sum(pas)/n if n else 0.0
                prior_avg_ops = sum(prior_opss)/n if n else 0.0
                prior_hr_per_pa = sum(prior_hrs)/n if n else 0.0
                prior_so_per_pa = sum(prior_sos)/n if n else 0.0
                blend_avg_ops = sum(blend_opss)/n if n else 0.0
                blend_hr_per_pa = sum(blend_hrs)/n if n else 0.0
                blend_so_per_pa = sum(blend_sos)/n if n else 0.0
                r3_avg_ops = sum(r3_opss)/n if n else 0.0
                r3_hr_per_pa = sum(r3_hrs)/n if n else 0.0
                r3_so_per_pa = sum(r3_sos)/n if n else 0.0
                r7_avg_ops = sum(r7_opss)/n if n else 0.0
                r7_hr_per_pa = sum(r7_hrs)/n if n else 0.0
                r7_so_per_pa = sum(r7_sos)/n if n else 0.0
                time_split_ops = sum(time_opss)/n if n else 0.0
                throw_split_ops = sum(throw_opss)/n if n else 0.0
                throw_split_hr_per_pa = sum(throw_hrs)/n if n else 0.0
                throw_split_so_per_pa = sum(throw_sos)/n if n else 0.0
                r14_avg_ops = sum(r14_opss)/n if n else 0.0
                r14_hr_per_pa = sum(r14_hrs)/n if n else 0.0
                r14_so_per_pa = sum(r14_sos)/n if n else 0.0
                known_cnt = n
                missing_cnt = 9 - known_cnt
                nohist_ratio = div(nohist, known_cnt)
                state_y_rate = div(state_y_cnt, known_cnt)
                samehand_rate = div(samehand, known_cnt)
                platoon_rate = div(platoon, known_cnt)
                cold_foreign_avg = div(cold_foreign_sum, known_cnt)
                cold_rookie_avg = div(cold_rookie_sum, known_cnt)
                cold_returnee_avg = div(cold_returnee_sum, known_cnt)
                hist_rel_avg = div(hist_rel_sum, known_cnt)
                top3_avg = mean_or_zero(top3_blend)
                bot6_avg = mean_or_zero(bot6_blend)
                top3_minus_bot6 = top3_avg - bot6_avg
                # v1: sum/core5/bottom4
                sum_ops = sum(blend_opss)
                sum_slg = sum(all_slgs)
                sum_hr_per_ab = sum(all_hr_abs)
                sum_so_per_ab = sum(all_so_abs)
                core5_ops = sum(core5_opss)
                core5_slg = sum(core5_slgs)
                core5_hr_per_ab = sum(core5_hr_abs)
                bottom4_ops = sum(bottom4_opss)
                return (
                    avg_avg, avg_obp, avg_slg, avg_ops, hr_per_pa, so_per_pa, bb_per_pa, avg_pa, nohist,
                    prior_avg_ops, prior_hr_per_pa, prior_so_per_pa,
                    blend_avg_ops, blend_hr_per_pa, blend_so_per_pa,
                    r3_avg_ops, r3_hr_per_pa, r3_so_per_pa,
                    r7_avg_ops, r7_hr_per_pa, r7_so_per_pa,
                    time_split_ops, throw_split_ops, throw_split_hr_per_pa, throw_split_so_per_pa,
                    r14_avg_ops, r14_hr_per_pa, r14_so_per_pa,
                    known_cnt, missing_cnt, nohist_ratio, state_y_cnt, state_y_rate,
                    hand1, hand2, hand3, samehand_rate, platoon_rate,
                    cold_foreign_avg, cold_rookie_avg, cold_returnee_avg, hist_rel_avg,
                    top3_avg, bot6_avg, top3_minus_bot6,
                    sum_ops, sum_slg, sum_hr_per_ab, sum_so_per_ab,
                    core5_ops, core5_slg, core5_hr_per_ab, bottom4_ops,
                )

            h_la, h_lo, h_ls, h_lops, h_hrpa, h_sopa, h_bbpa, h_avgpa, h_nohist, h_lops_p, h_hrpa_p, h_sopa_p, h_lops_b, h_hrpa_b, h_sopa_b, h_r3_ops, h_r3_hr, h_r3_so, h_r7_ops, h_r7_hr, h_r7_so, h_time_ops, h_vs_throw_ops, h_vs_throw_hr, h_vs_throw_so, h_r14_ops, h_r14_hr, h_r14_so, h_known, h_missing, h_nohist_ratio, h_state_y_cnt, h_state_y_rate, h_hand1, h_hand2, h_hand3, h_samehand, h_platoon, h_cold_f, h_cold_r, h_cold_rt, h_lu_hist_rel, h_top3_bops, h_bot6_bops, h_top3m_bops, h_sum_ops, h_sum_slg, h_sum_hr_ab, h_sum_so_ab, h_core5_ops, h_core5_slg, h_core5_hr_ab, h_bot4_ops = lineup_agg("home", away_sp_throw)
            a_la, a_lo, a_ls, a_lops, a_hrpa, a_sopa, a_bbpa, a_avgpa, a_nohist, a_lops_p, a_hrpa_p, a_sopa_p, a_lops_b, a_hrpa_b, a_sopa_b, a_r3_ops, a_r3_hr, a_r3_so, a_r7_ops, a_r7_hr, a_r7_so, a_time_ops, a_vs_throw_ops, a_vs_throw_hr, a_vs_throw_so, a_r14_ops, a_r14_hr, a_r14_so, a_known, a_missing, a_nohist_ratio, a_state_y_cnt, a_state_y_rate, a_hand1, a_hand2, a_hand3, a_samehand, a_platoon, a_cold_f, a_cold_r, a_cold_rt, a_lu_hist_rel, a_top3_bops, a_bot6_bops, a_top3m_bops, a_sum_ops, a_sum_slg, a_sum_hr_ab, a_sum_so_ab, a_core5_ops, a_core5_slg, a_core5_hr_ab, a_bot4_ops = lineup_agg("away", home_sp_throw)
            h_lineup_platoon_edge = h_platoon - h_samehand
            a_lineup_platoon_edge = a_platoon - a_samehand
            h_count_matchup_edge = (h_bbpa - h_sopa) - (div(a_bbb9, 9.0) - div(a_bk9, 9.0))
            a_count_matchup_edge = (a_bbpa - a_sopa) - (div(h_bbb9, 9.0) - div(h_bk9, 9.0))
            h_lineup_momentum_ops = h_r7_ops - h_lops_b
            a_lineup_momentum_ops = a_r7_ops - a_lops_b

            # v1: team cold start flag (1 if <COLD_TEAM_G games played THIS SEASON)
            h_season_g = team_year_g[(home, year)]
            a_season_g = team_year_g[(away, year)]
            h_cold_start = 1 if h_season_g < COLD_TEAM_G else 0
            a_cold_start = 1 if a_season_g < COLD_TEAM_G else 0

            out_rows.append({
                "date": d,
                "s_no": s_no,
                "s_code": s_code,
                "homeTeam": home,
                "awayTeam": away,
                "y_home_win": y,
                "homeScore": hs_raw if hs_raw is not None else "",
                "awayScore": as_raw if as_raw is not None else "",

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
                "home_team_pyth_winpct": round(h_pyth,6),
                "away_team_pyth_winpct": round(a_pyth,6),
                "diff_team_pyth_winpct": round(h_pyth-a_pyth,6),
                "home_team_prior_RS_perG": round(h_prs,6),
                "home_team_prior_RA_perG": round(h_pra,6),
                "home_team_prior_winpct": round(h_pwp,6),
                "away_team_prior_RS_perG": round(a_prs,6),
                "away_team_prior_RA_perG": round(a_pra,6),
                "away_team_prior_winpct": round(a_pwp,6),
                "diff_team_prior_RS_perG": round(h_prs-a_prs,6),
                "diff_team_prior_RA_perG": round(h_pra-a_pra,6),
                "diff_team_prior_winpct": round(h_pwp-a_pwp,6),
                "home_team_home_G": hhg,
                "home_team_home_RS_perG": round(hhrs,6),
                "home_team_home_RA_perG": round(hhra,6),
                "home_team_home_winpct": round(hhwp,6),
                "away_team_away_G": aag,
                "away_team_away_RS_perG": round(aars,6),
                "away_team_away_RA_perG": round(aara,6),
                "away_team_away_winpct": round(aawp,6),
                "diff_team_homeaway_RS_perG": round(hhrs-aars,6),
                "diff_team_homeaway_RA_perG": round(hhra-aara,6),
                "diff_team_homeaway_winpct": round(hhwp-aawp,6),
                "home_team_home_prior_winpct": round(h_home_pwp,6),
                "away_team_away_prior_winpct": round(a_away_pwp,6),
                "diff_team_homeaway_prior_winpct": round(h_home_pwp-a_away_pwp,6),
                "league_home_winpct": round(l_home_wp,6),
                "home_field_adv_to_date": round(home_field_adv_to_date,6),
                "league_type_regular": league_type_regular,
                "league_type_postseason": league_type_post,
                "league_type_exhibition": league_type_exhi,
                "league_type_other": league_type_other,
                "home_context_boost": round(home_context_boost,6),
                "expected_run_edge_homeaway": round(expected_run_edge_homeaway,6),
                "home_team_r5_G": h_r5g,
                "home_team_r5_RS_perG": round(h_r5rs,6),
                "home_team_r5_RA_perG": round(h_r5ra,6),
                "home_team_r5_winpct": round(h_r5wp,6),
                "away_team_r5_G": a_r5g,
                "away_team_r5_RS_perG": round(a_r5rs,6),
                "away_team_r5_RA_perG": round(a_r5ra,6),
                "away_team_r5_winpct": round(a_r5wp,6),
                "diff_team_r5_RS_perG": round(h_r5rs-a_r5rs,6),
                "diff_team_r5_RA_perG": round(h_r5ra-a_r5ra,6),
                "diff_team_r5_winpct": round(h_r5wp-a_r5wp,6),
                "home_team_rest_days": h_rest,
                "away_team_rest_days": a_rest,
                "diff_team_rest_days": h_rest-a_rest,
                "home_team_consec_days": h_consec,
                "away_team_consec_days": a_consec,
                "diff_team_consec_days": h_consec-a_consec,
                "home_team_games_last7": h_g7,
                "away_team_games_last7": a_g7,
                "diff_team_games_last7": h_g7-a_g7,
                "home_team_away_streak": h_away_streak,
                "away_team_away_streak": a_away_streak,
                "diff_team_away_streak": h_away_streak-a_away_streak,
                "home_team_oppwp_avg": round(h_opp_avg,6),
                "away_team_oppwp_avg": round(a_opp_avg,6),
                "diff_team_oppwp_avg": round(h_opp_avg-a_opp_avg,6),
                "home_team_oppwp_r5": round(h_opp_r5,6),
                "away_team_oppwp_r5": round(a_opp_r5,6),
                "diff_team_oppwp_r5": round(h_opp_r5-a_opp_r5,6),
                # v1: H2H
                "home_h2h_G": h2h_hg,
                "home_h2h_winpct": round(h2h_h_wp,6),
                "away_h2h_winpct": round(h2h_a_wp,6),
                "diff_h2h_winpct": round(h2h_h_wp-h2h_a_wp,6),
                "home_bp_np_l1": round(h_bp_np_l1,6),
                "home_bp_np_l3": round(h_bp_np_l3,6),
                "home_bp_ip_l3": round(h_bp_ip_l3,6),
                "home_bp_app_l3": h_bp_app_l3,
                "away_bp_np_l1": round(a_bp_np_l1,6),
                "away_bp_np_l3": round(a_bp_np_l3,6),
                "away_bp_ip_l3": round(a_bp_ip_l3,6),
                "away_bp_app_l3": a_bp_app_l3,
                "diff_bp_np_l3": round(h_bp_np_l3-a_bp_np_l3,6),
                "diff_bp_ip_l3": round(h_bp_ip_l3-a_bp_ip_l3,6),
                "diff_bp_app_l3": h_bp_app_l3-a_bp_app_l3,
                "home_bp_np_per_ip_l3": round(h_bp_np_per_ip_l3,6),
                "away_bp_np_per_ip_l3": round(a_bp_np_per_ip_l3,6),
                "diff_bp_np_per_ip_l3": round(h_bp_np_per_ip_l3-a_bp_np_per_ip_l3,6),
                "home_bp_np_per_app_l3": round(h_bp_np_per_app_l3,6),
                "away_bp_np_per_app_l3": round(a_bp_np_per_app_l3,6),
                "diff_bp_np_per_app_l3": round(h_bp_np_per_app_l3-a_bp_np_per_app_l3,6),
                "home_bp_whip_l3": round(h_bp_whip_l3,6),
                "away_bp_whip_l3": round(a_bp_whip_l3,6),
                "diff_bp_whip_l3": round(h_bp_whip_l3-a_bp_whip_l3,6),
                "home_bp_era_l3": round(h_bp_era_l3,6),
                "away_bp_era_l3": round(a_bp_era_l3,6),
                "diff_bp_era_l3": round(h_bp_era_l3-a_bp_era_l3,6),
                "home_bp_stress_index": round(h_bp_stress_idx,6),
                "away_bp_stress_index": round(a_bp_stress_idx,6),
                "diff_bp_stress_index": round(h_bp_stress_idx-a_bp_stress_idx,6),
                "home_bp_day_np_l1": round(h_bp_day_np_l1,6),
                "home_bp_day_np_l3": round(h_bp_day_np_l3,6),
                "home_bp_day_ip_l3": round(h_bp_day_ip_l3,6),
                "home_bp_day_app_l3": h_bp_day_app_l3,
                "away_bp_day_np_l1": round(a_bp_day_np_l1,6),
                "away_bp_day_np_l3": round(a_bp_day_np_l3,6),
                "away_bp_day_ip_l3": round(a_bp_day_ip_l3,6),
                "away_bp_day_app_l3": a_bp_day_app_l3,
                "diff_bp_day_np_l3": round(h_bp_day_np_l3-a_bp_day_np_l3,6),
                "diff_bp_day_ip_l3": round(h_bp_day_ip_l3-a_bp_day_ip_l3,6),
                "diff_bp_day_app_l3": h_bp_day_app_l3-a_bp_day_app_l3,
                "home_bp_day_np_per_ip_l3": round(h_bp_day_np_per_ip_l3,6),
                "away_bp_day_np_per_ip_l3": round(a_bp_day_np_per_ip_l3,6),
                "diff_bp_day_np_per_ip_l3": round(h_bp_day_np_per_ip_l3-a_bp_day_np_per_ip_l3,6),
                "home_bp_day_np_per_app_l3": round(h_bp_day_np_per_app_l3,6),
                "away_bp_day_np_per_app_l3": round(a_bp_day_np_per_app_l3,6),
                "diff_bp_day_np_per_app_l3": round(h_bp_day_np_per_app_l3-a_bp_day_np_per_app_l3,6),
                "home_bp_day_whip_l3": round(h_bp_day_whip_l3,6),
                "away_bp_day_whip_l3": round(a_bp_day_whip_l3,6),
                "diff_bp_day_whip_l3": round(h_bp_day_whip_l3-a_bp_day_whip_l3,6),
                "home_bp_day_era_l3": round(h_bp_day_era_l3,6),
                "away_bp_day_era_l3": round(a_bp_day_era_l3,6),
                "diff_bp_day_era_l3": round(h_bp_day_era_l3-a_bp_day_era_l3,6),
                # v1: key bullpen
                "home_key_bp_fatigue": round(h_key_bp_fatigue,6),
                "away_key_bp_fatigue": round(a_key_bp_fatigue,6),
                "diff_key_bp_fatigue": round(h_key_bp_fatigue-a_key_bp_fatigue,6),
                "home_key_bp_ops_against": round(h_key_bp_ops,6),
                "away_key_bp_ops_against": round(a_key_bp_ops,6),
                "diff_key_bp_ops_against": round(h_key_bp_ops-a_key_bp_ops,6),
                "weather_code": weather_code or 0,
                "weather_temperature": round(weather_temp,6),
                "weather_humidity": round(weather_humidity,6),
                "weather_wind_direction": weather_wdir or 0,
                "weather_wind_speed": round(weather_wspd,6),
                "weather_rain_probability": round(weather_rain,6),
                "weather_unknown": weather_unknown,
                "weather_hot": weather_hot,
                "weather_cold": weather_cold,
                "weather_windy": weather_windy,
                "weather_rainy": weather_rainy,
                "game_month": game_month,
                "game_weekday": game_weekday,
                "game_is_weekend": game_is_weekend,
                "game_hour": game_hour,
                "game_is_night": game_is_night,
                "game_is_day": game_is_day,
                "park_run_factor": round(park_factor,6),
                "park_hr_factor": round(park_hr_factor,6),

                "home_sp_sched_p_no": home_sp_sched or 0,
                "away_sp_sched_p_no": away_sp_sched or 0,
                "home_sp_sched_missing": home_sp_sched_missing,
                "away_sp_sched_missing": away_sp_sched_missing,
                "diff_sp_sched_missing": home_sp_sched_missing-away_sp_sched_missing,
                "home_sp_replaced": home_sp_replaced,
                "away_sp_replaced": away_sp_replaced,
                "diff_sp_replaced": home_sp_replaced-away_sp_replaced,
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
                # v1: SP SLG-against, BB/IP, SO/IP
                "home_sp_SLG_against": round(h_sp_slg_a,6),
                "away_sp_SLG_against": round(a_sp_slg_a,6),
                "diff_sp_SLG_against": round(h_sp_slg_a-a_sp_slg_a,6),
                "home_sp_BB_per_IP": round(h_sp_bb_per_ip,6),
                "away_sp_BB_per_IP": round(a_sp_bb_per_ip,6),
                "diff_sp_BB_per_IP": round(h_sp_bb_per_ip-a_sp_bb_per_ip,6),
                "home_sp_SO_per_IP": round(h_sp_so_per_ip,6),
                "away_sp_SO_per_IP": round(a_sp_so_per_ip,6),
                "diff_sp_SO_per_IP": round(h_sp_so_per_ip-a_sp_so_per_ip,6),
                "home_sp_blend_ERA": round(h_bera,6),
                "home_sp_blend_WHIP": round(h_bwhip,6),
                "home_sp_blend_K9": round(h_bk9,6),
                "home_sp_blend_BB9": round(h_bbb9,6),
                "away_sp_blend_ERA": round(a_bera,6),
                "away_sp_blend_WHIP": round(a_bwhip,6),
                "away_sp_blend_K9": round(a_bk9,6),
                "away_sp_blend_BB9": round(a_bbb9,6),
                "diff_sp_blend_ERA": round(h_bera-a_bera,6),
                "diff_sp_blend_WHIP": round(h_bwhip-a_bwhip,6),
                "diff_sp_blend_K9": round(h_bk9-a_bk9,6),
                "diff_sp_blend_BB9": round(h_bbb9-a_bbb9,6),
                "home_sp_prior_ERA": round(h_pera,6),
                "home_sp_prior_WHIP": round(h_pwhip,6),
                "home_sp_prior_K9": round(h_pk9,6),
                "home_sp_prior_BB9": round(h_pbb9,6),
                "away_sp_prior_ERA": round(a_pera,6),
                "away_sp_prior_WHIP": round(a_pwhip,6),
                "away_sp_prior_K9": round(a_pk9,6),
                "away_sp_prior_BB9": round(a_pbb9,6),
                "diff_sp_prior_ERA": round(h_pera-a_pera,6),
                "diff_sp_prior_WHIP": round(h_pwhip-a_pwhip,6),
                "diff_sp_prior_K9": round(h_pk9-a_pk9,6),
                "diff_sp_prior_BB9": round(h_pbb9-a_pbb9,6),
                "home_sp_r5_G": h_sp_r5g,
                "home_sp_r5_ERA": round(h_sp_r5era,6),
                "home_sp_r5_WHIP": round(h_sp_r5whip,6),
                "home_sp_r5_K9": round(h_sp_r5k9,6),
                "home_sp_r5_BB9": round(h_sp_r5bb9,6),
                "away_sp_r5_G": a_sp_r5g,
                "away_sp_r5_ERA": round(a_sp_r5era,6),
                "away_sp_r5_WHIP": round(a_sp_r5whip,6),
                "away_sp_r5_K9": round(a_sp_r5k9,6),
                "away_sp_r5_BB9": round(a_sp_r5bb9,6),
                "diff_sp_r5_ERA": round(h_sp_r5era-a_sp_r5era,6),
                "diff_sp_r5_WHIP": round(h_sp_r5whip-a_sp_r5whip,6),
                "diff_sp_r5_K9": round(h_sp_r5k9-a_sp_r5k9,6),
                "diff_sp_r5_BB9": round(h_sp_r5bb9-a_sp_r5bb9,6),
                "home_sp_form_conf": round(h_sp_form_conf,6),
                "away_sp_form_conf": round(a_sp_form_conf,6),
                "diff_sp_form_conf": round(h_sp_form_conf-a_sp_form_conf,6),
                "home_sp_form_ERA_adj": round(h_sp_form_era_adj,6),
                "away_sp_form_ERA_adj": round(a_sp_form_era_adj,6),
                "diff_sp_form_ERA_adj": round(h_sp_form_era_adj-a_sp_form_era_adj,6),
                "home_sp_form_WHIP_adj": round(h_sp_form_whip_adj,6),
                "away_sp_form_WHIP_adj": round(a_sp_form_whip_adj,6),
                "diff_sp_form_WHIP_adj": round(h_sp_form_whip_adj-a_sp_form_whip_adj,6),
                "home_sp_form_K9_adj": round(h_sp_form_k9_adj,6),
                "away_sp_form_K9_adj": round(a_sp_form_k9_adj,6),
                "diff_sp_form_K9_adj": round(h_sp_form_k9_adj-a_sp_form_k9_adj,6),
                "home_sp_time_split_ERA": round(h_sp_time_era,6),
                "away_sp_time_split_ERA": round(a_sp_time_era,6),
                "diff_sp_time_split_ERA": round(h_sp_time_era-a_sp_time_era,6),
                "home_sp_time_split_WHIP": round(h_sp_time_whip,6),
                "away_sp_time_split_WHIP": round(a_sp_time_whip,6),
                "diff_sp_time_split_WHIP": round(h_sp_time_whip-a_sp_time_whip,6),
                "home_sp_np_l1d": round(h_sp_np_l1d,6),
                "home_sp_np_l3d": round(h_sp_np_l3d,6),
                "home_sp_ip_l3d": round(h_sp_ip_l3d,6),
                "home_sp_app_l3d": h_sp_app_l3d,
                "home_sp_rest_days": h_sp_rest_days,
                "away_sp_np_l1d": round(a_sp_np_l1d,6),
                "away_sp_np_l3d": round(a_sp_np_l3d,6),
                "away_sp_ip_l3d": round(a_sp_ip_l3d,6),
                "away_sp_app_l3d": a_sp_app_l3d,
                "away_sp_rest_days": a_sp_rest_days,
                "diff_sp_np_l3d": round(h_sp_np_l3d-a_sp_np_l3d,6),
                "diff_sp_ip_l3d": round(h_sp_ip_l3d-a_sp_ip_l3d,6),
                "diff_sp_rest_days": h_sp_rest_days-a_sp_rest_days,
                "home_sp_fatigue_index": round(h_sp_fatigue_idx,6),
                "away_sp_fatigue_index": round(a_sp_fatigue_idx,6),
                "diff_sp_fatigue_index": round(h_sp_fatigue_idx-a_sp_fatigue_idx,6),
                "home_sp_short_rest": h_sp_short_rest,
                "away_sp_short_rest": a_sp_short_rest,
                "diff_sp_short_rest": h_sp_short_rest-a_sp_short_rest,
                "home_sp_expected_bp_ip": round(h_sp_expected_bp_ip,6),
                "away_sp_expected_bp_ip": round(a_sp_expected_bp_ip,6),
                "diff_sp_expected_bp_ip": round(h_sp_expected_bp_ip-a_sp_expected_bp_ip,6),
                "home_pitch_handoff_risk": round(h_pitch_handoff_risk,6),
                "away_pitch_handoff_risk": round(a_pitch_handoff_risk,6),
                "diff_pitch_handoff_risk": round(h_pitch_handoff_risk-a_pitch_handoff_risk,6),
                "home_sp_throw_unknown": 0 if throw_side(home_sp_throw) else 1,
                "away_sp_throw_unknown": 0 if throw_side(away_sp_throw) else 1,
                "home_sp_throw_R": 1 if throw_side(home_sp_throw) == "R" else 0,
                "home_sp_throw_L": 1 if throw_side(home_sp_throw) == "L" else 0,
                "away_sp_throw_R": 1 if throw_side(away_sp_throw) == "R" else 0,
                "away_sp_throw_L": 1 if throw_side(away_sp_throw) == "L" else 0,
                "home_sp_throw_under": throw_is_under(home_sp_throw),
                "away_sp_throw_under": throw_is_under(away_sp_throw),
                "diff_sp_throw_under": throw_is_under(home_sp_throw)-throw_is_under(away_sp_throw),
                "home_sp_cold_foreign_prob": round(h_pf,6),
                "home_sp_cold_rookie_prob": round(h_pr,6),
                "home_sp_cold_returnee_prob": round(h_prt,6),
                "home_sp_hist_reliability": round(h_hist_rel,6),
                "away_sp_cold_foreign_prob": round(a_pf,6),
                "away_sp_cold_rookie_prob": round(a_pr,6),
                "away_sp_cold_returnee_prob": round(a_prt,6),
                "away_sp_hist_reliability": round(a_hist_rel,6),
                "diff_sp_cold_foreign_prob": round(h_pf-a_pf,6),
                "diff_sp_hist_reliability": round(h_hist_rel-a_hist_rel,6),

                "home_lineup_avg_avg": round(h_la,6),
                "home_lineup_avg_obp": round(h_lo,6),
                "home_lineup_avg_slg": round(h_ls,6),
                "home_lineup_avg_ops": round(h_lops,6),
                "home_lineup_hr_per_pa": round(h_hrpa,6),
                "home_lineup_so_per_pa": round(h_sopa,6),
                "home_lineup_bb_per_pa": round(h_bbpa,6),
                "home_lineup_avg_pa": round(h_avgpa,6),
                "home_lineup_nohist_cnt": h_nohist,

                "away_lineup_avg_avg": round(a_la,6),
                "away_lineup_avg_obp": round(a_lo,6),
                "away_lineup_avg_slg": round(a_ls,6),
                "away_lineup_avg_ops": round(a_lops,6),
                "away_lineup_hr_per_pa": round(a_hrpa,6),
                "away_lineup_so_per_pa": round(a_sopa,6),
                "away_lineup_bb_per_pa": round(a_bbpa,6),
                "away_lineup_avg_pa": round(a_avgpa,6),
                "away_lineup_nohist_cnt": a_nohist,

                "diff_lineup_avg_ops": round(h_lops-a_lops,6),
                "diff_lineup_hr_per_pa": round(h_hrpa-a_hrpa,6),
                "diff_lineup_so_per_pa": round(h_sopa-a_sopa,6),
                "diff_lineup_bb_per_pa": round(h_bbpa-a_bbpa,6),
                "diff_lineup_nohist_cnt": h_nohist-a_nohist,
                "home_lineup_prior_avg_ops": round(h_lops_p,6),
                "home_lineup_prior_hr_per_pa": round(h_hrpa_p,6),
                "home_lineup_prior_so_per_pa": round(h_sopa_p,6),
                "away_lineup_prior_avg_ops": round(a_lops_p,6),
                "away_lineup_prior_hr_per_pa": round(a_hrpa_p,6),
                "away_lineup_prior_so_per_pa": round(a_sopa_p,6),
                "diff_lineup_prior_avg_ops": round(h_lops_p-a_lops_p,6),
                "diff_lineup_prior_hr_per_pa": round(h_hrpa_p-a_hrpa_p,6),
                "diff_lineup_prior_so_per_pa": round(h_sopa_p-a_sopa_p,6),
                "home_lineup_blend_ops": round(h_lops_b,6),
                "home_lineup_blend_hr_per_pa": round(h_hrpa_b,6),
                "home_lineup_blend_so_per_pa": round(h_sopa_b,6),
                "away_lineup_blend_ops": round(a_lops_b,6),
                "away_lineup_blend_hr_per_pa": round(a_hrpa_b,6),
                "away_lineup_blend_so_per_pa": round(a_sopa_b,6),
                "diff_lineup_blend_ops": round(h_lops_b-a_lops_b,6),
                "diff_lineup_blend_hr_per_pa": round(h_hrpa_b-a_hrpa_b,6),
                "diff_lineup_blend_so_per_pa": round(h_sopa_b-a_sopa_b,6),
                "home_lineup_r3_ops": round(h_r3_ops,6),
                "home_lineup_r3_hr_per_pa": round(h_r3_hr,6),
                "home_lineup_r3_so_per_pa": round(h_r3_so,6),
                "away_lineup_r3_ops": round(a_r3_ops,6),
                "away_lineup_r3_hr_per_pa": round(a_r3_hr,6),
                "away_lineup_r3_so_per_pa": round(a_r3_so,6),
                "diff_lineup_r3_ops": round(h_r3_ops-a_r3_ops,6),
                "diff_lineup_r3_hr_per_pa": round(h_r3_hr-a_r3_hr,6),
                "diff_lineup_r3_so_per_pa": round(h_r3_so-a_r3_so,6),
                "home_lineup_r7_ops": round(h_r7_ops,6),
                "home_lineup_r7_hr_per_pa": round(h_r7_hr,6),
                "home_lineup_r7_so_per_pa": round(h_r7_so,6),
                "away_lineup_r7_ops": round(a_r7_ops,6),
                "away_lineup_r7_hr_per_pa": round(a_r7_hr,6),
                "away_lineup_r7_so_per_pa": round(a_r7_so,6),
                "diff_lineup_r7_ops": round(h_r7_ops-a_r7_ops,6),
                "diff_lineup_r7_hr_per_pa": round(h_r7_hr-a_r7_hr,6),
                "diff_lineup_r7_so_per_pa": round(h_r7_so-a_r7_so,6),
                "home_lineup_time_split_ops": round(h_time_ops,6),
                "away_lineup_time_split_ops": round(a_time_ops,6),
                "diff_lineup_time_split_ops": round(h_time_ops-a_time_ops,6),
                "home_lineup_vs_throw_split_ops": round(h_vs_throw_ops,6),
                "away_lineup_vs_throw_split_ops": round(a_vs_throw_ops,6),
                "diff_lineup_vs_throw_split_ops": round(h_vs_throw_ops-a_vs_throw_ops,6),
                "home_lineup_vs_throw_split_hr_per_pa": round(h_vs_throw_hr,6),
                "away_lineup_vs_throw_split_hr_per_pa": round(a_vs_throw_hr,6),
                "diff_lineup_vs_throw_split_hr_per_pa": round(h_vs_throw_hr-a_vs_throw_hr,6),
                "home_lineup_vs_throw_split_so_per_pa": round(h_vs_throw_so,6),
                "away_lineup_vs_throw_split_so_per_pa": round(a_vs_throw_so,6),
                "diff_lineup_vs_throw_split_so_per_pa": round(h_vs_throw_so-a_vs_throw_so,6),
                "home_count_matchup_edge": round(h_count_matchup_edge,6),
                "away_count_matchup_edge": round(a_count_matchup_edge,6),
                "diff_count_matchup_edge": round(h_count_matchup_edge-a_count_matchup_edge,6),
                "home_lineup_momentum_ops": round(h_lineup_momentum_ops,6),
                "away_lineup_momentum_ops": round(a_lineup_momentum_ops,6),
                "diff_lineup_momentum_ops": round(h_lineup_momentum_ops-a_lineup_momentum_ops,6),
                "home_lineup_r14_ops": round(h_r14_ops,6),
                "home_lineup_r14_hr_per_pa": round(h_r14_hr,6),
                "home_lineup_r14_so_per_pa": round(h_r14_so,6),
                "away_lineup_r14_ops": round(a_r14_ops,6),
                "away_lineup_r14_hr_per_pa": round(a_r14_hr,6),
                "away_lineup_r14_so_per_pa": round(a_r14_so,6),
                "diff_lineup_r14_ops": round(h_r14_ops-a_r14_ops,6),
                "diff_lineup_r14_hr_per_pa": round(h_r14_hr-a_r14_hr,6),
                "diff_lineup_r14_so_per_pa": round(h_r14_so-a_r14_so,6),
                "home_lineup_known_cnt": h_known,
                "away_lineup_known_cnt": a_known,
                "diff_lineup_known_cnt": h_known-a_known,
                "home_lineup_missing_cnt": h_missing,
                "away_lineup_missing_cnt": a_missing,
                "diff_lineup_missing_cnt": h_missing-a_missing,
                "home_lineup_nohist_ratio": round(h_nohist_ratio,6),
                "away_lineup_nohist_ratio": round(a_nohist_ratio,6),
                "diff_lineup_nohist_ratio": round(h_nohist_ratio-a_nohist_ratio,6),
                "home_lineup_state_y_cnt": h_state_y_cnt,
                "away_lineup_state_y_cnt": a_state_y_cnt,
                "diff_lineup_state_y_cnt": h_state_y_cnt-a_state_y_cnt,
                "home_lineup_state_y_rate": round(h_state_y_rate,6),
                "away_lineup_state_y_rate": round(a_state_y_rate,6),
                "diff_lineup_state_y_rate": round(h_state_y_rate-a_state_y_rate,6),
                "home_lineup_hand1_cnt": h_hand1,
                "home_lineup_hand2_cnt": h_hand2,
                "home_lineup_hand3_cnt": h_hand3,
                "away_lineup_hand1_cnt": a_hand1,
                "away_lineup_hand2_cnt": a_hand2,
                "away_lineup_hand3_cnt": a_hand3,
                "home_lineup_vs_opp_sp_samehand_rate": round(h_samehand,6),
                "away_lineup_vs_opp_sp_samehand_rate": round(a_samehand,6),
                "diff_lineup_vs_opp_sp_samehand_rate": round(h_samehand-a_samehand,6),
                "home_lineup_vs_opp_sp_platoon_rate": round(h_platoon,6),
                "away_lineup_vs_opp_sp_platoon_rate": round(a_platoon,6),
                "diff_lineup_vs_opp_sp_platoon_rate": round(h_platoon-a_platoon,6),
                "home_lineup_platoon_edge": round(h_lineup_platoon_edge,6),
                "away_lineup_platoon_edge": round(a_lineup_platoon_edge,6),
                "diff_lineup_platoon_edge": round(h_lineup_platoon_edge-a_lineup_platoon_edge,6),
                "home_lineup_cold_foreign_prob_avg": round(h_cold_f,6),
                "home_lineup_cold_rookie_prob_avg": round(h_cold_r,6),
                "home_lineup_cold_returnee_prob_avg": round(h_cold_rt,6),
                "home_lineup_hist_reliability_avg": round(h_lu_hist_rel,6),
                "away_lineup_cold_foreign_prob_avg": round(a_cold_f,6),
                "away_lineup_cold_rookie_prob_avg": round(a_cold_r,6),
                "away_lineup_cold_returnee_prob_avg": round(a_cold_rt,6),
                "away_lineup_hist_reliability_avg": round(a_lu_hist_rel,6),
                "diff_lineup_cold_foreign_prob_avg": round(h_cold_f-a_cold_f,6),
                "diff_lineup_hist_reliability_avg": round(h_lu_hist_rel-a_lu_hist_rel,6),
                "home_lineup_top3_blend_ops": round(h_top3_bops,6),
                "home_lineup_bot6_blend_ops": round(h_bot6_bops,6),
                "home_lineup_top3_minus_bot6_blend_ops": round(h_top3m_bops,6),
                "away_lineup_top3_blend_ops": round(a_top3_bops,6),
                "away_lineup_bot6_blend_ops": round(a_bot6_bops,6),
                "away_lineup_top3_minus_bot6_blend_ops": round(a_top3m_bops,6),
                "diff_lineup_top3_minus_bot6_blend_ops": round(h_top3m_bops-a_top3m_bops,6),
                # v1: lineup sum
                "home_lineup_sum_ops": round(h_sum_ops,6),
                "away_lineup_sum_ops": round(a_sum_ops,6),
                "diff_lineup_sum_ops": round(h_sum_ops-a_sum_ops,6),
                "home_lineup_sum_slg": round(h_sum_slg,6),
                "away_lineup_sum_slg": round(a_sum_slg,6),
                "diff_lineup_sum_slg": round(h_sum_slg-a_sum_slg,6),
                "home_lineup_sum_hr_per_ab": round(h_sum_hr_ab,6),
                "away_lineup_sum_hr_per_ab": round(a_sum_hr_ab,6),
                "diff_lineup_sum_hr_per_ab": round(h_sum_hr_ab-a_sum_hr_ab,6),
                "home_lineup_sum_so_per_ab": round(h_sum_so_ab,6),
                "away_lineup_sum_so_per_ab": round(a_sum_so_ab,6),
                "diff_lineup_sum_so_per_ab": round(h_sum_so_ab-a_sum_so_ab,6),
                # v1: core5/bottom4
                "home_lineup_core5_ops": round(h_core5_ops,6),
                "away_lineup_core5_ops": round(a_core5_ops,6),
                "diff_lineup_core5_ops": round(h_core5_ops-a_core5_ops,6),
                "home_lineup_core5_slg": round(h_core5_slg,6),
                "away_lineup_core5_slg": round(a_core5_slg,6),
                "diff_lineup_core5_slg": round(h_core5_slg-a_core5_slg,6),
                "home_lineup_core5_hr_per_ab": round(h_core5_hr_ab,6),
                "away_lineup_core5_hr_per_ab": round(a_core5_hr_ab,6),
                "diff_lineup_core5_hr_per_ab": round(h_core5_hr_ab-a_core5_hr_ab,6),
                "home_lineup_bottom4_ops": round(h_bot4_ops,6),
                "away_lineup_bottom4_ops": round(a_bot4_ops,6),
                "diff_lineup_bottom4_ops": round(h_bot4_ops-a_bot4_ops,6),
                # v1: cold start
                "home_team_cold_start": h_cold_start,
                "away_team_cold_start": a_cold_start,
                "diff_team_cold_start": h_cold_start-a_cold_start,
            })

        today_game_ctx = {}
        for g in todays_games:
            s_no_ctx = safe_int(g["s_no"])
            if not s_no_ctx:
                continue
            today_game_ctx[s_no_ctx] = {
                "homeTeam": safe_int(g.get("homeTeam")),
                "awayTeam": safe_int(g.get("awayTeam")),
                "home_sp_throw": lineup_map[s_no_ctx]["home"]["P_throw"],
                "away_sp_throw": lineup_map[s_no_ctx]["away"]["P_throw"],
                "time_bucket": game_time_bucket(parse_game_hour(g.get("hm"))),
            }

        # 2) after all games of the day: update cumulative stats with TODAY results (so next day can use)
        for g in todays_games:
            s_no = safe_int(g["s_no"])
            home = safe_int(g["homeTeam"])
            away = safe_int(g["awayTeam"])
            s_code = safe_int(g.get("s_code"))
            hs_raw = safe_int_or_none(g.get("homeScore"))
            as_raw = safe_int_or_none(g.get("awayScore"))
            state_code = safe_int_or_none(g.get("state"))
            is_final_state = True if state_code is None else (state_code in (3, 5))
            if hs_raw is None or as_raw is None or not is_final_state:
                continue
            hs = hs_raw
            as_ = as_raw
            runs = hs + as_

            # team update
            team_cum[home]["G"] += 1
            team_cum[home]["RS"] += hs
            team_cum[home]["RA"] += as_
            team_cum[home]["W"]  += (1 if hs > as_ else 0)
            team_year_g[(home, year)] += 1  # v1
            team_home_cum[home]["G"] += 1
            team_home_cum[home]["RS"] += hs
            team_home_cum[home]["RA"] += as_
            team_home_cum[home]["W"]  += (1 if hs > as_ else 0)
            team_lg_cum["G"] += 1
            team_lg_cum["RS"] += hs
            team_lg_cum["RA"] += as_
            team_lg_cum["W"]  += (1 if hs > as_ else 0)
            team_lg_home_cum["G"] += 1
            team_lg_home_cum["RS"] += hs
            team_lg_home_cum["RA"] += as_
            team_lg_home_cum["W"] += (1 if hs > as_ else 0)
            team_recent[home].append((hs, as_, 1 if hs > as_ else 0))
            prev_h_date = team_last_game_date.get(home)
            team_consec_days[home] = (team_consec_days.get(home, 0) + 1) if (prev_h_date and day_diff(d, prev_h_date) == 1) else 1
            team_game_dates[home].append(d)
            team_last_side[home] = "home"
            team_away_streak[home] = 0
            team_last_game_date[home] = d

            team_cum[away]["G"] += 1
            team_year_g[(away, year)] += 1  # v1
            team_cum[away]["RS"] += as_
            team_cum[away]["RA"] += hs
            team_cum[away]["W"]  += (1 if as_ > hs else 0)
            team_away_cum[away]["G"] += 1
            team_away_cum[away]["RS"] += as_
            team_away_cum[away]["RA"] += hs
            team_away_cum[away]["W"] += (1 if as_ > hs else 0)
            team_lg_cum["G"] += 1
            team_lg_cum["RS"] += as_
            team_lg_cum["RA"] += hs
            team_lg_cum["W"]  += (1 if as_ > hs else 0)
            team_lg_away_cum["G"] += 1
            team_lg_away_cum["RS"] += as_
            team_lg_away_cum["RA"] += hs
            team_lg_away_cum["W"] += (1 if as_ > hs else 0)
            team_recent[away].append((as_, hs, 1 if as_ > hs else 0))
            prev_a_date = team_last_game_date.get(away)
            team_consec_days[away] = (team_consec_days.get(away, 0) + 1) if (prev_a_date and day_diff(d, prev_a_date) == 1) else 1
            team_game_dates[away].append(d)
            team_away_streak[away] = (team_away_streak.get(away, 0) + 1) if team_last_side.get(away) == "away" else 1
            team_last_side[away] = "away"
            team_last_game_date[away] = d

            hwp_pre, awp_pre = pregame_wp.get(s_no, (0.0, 0.0))
            team_opp_wp_hist[home].append(awp_pre)
            team_opp_wp_hist[away].append(hwp_pre)

            # v1: H2H update
            h2h_cum[(home, away)]["G"] += 1
            h2h_cum[(away, home)]["G"] += 1
            h2h_cum[(home, away)]["W"] += (1 if hs > as_ else 0)
            h2h_cum[(away, home)]["W"] += (1 if as_ > hs else 0)

            # append per-game bullpen usage for future game-window features
            h_bp = bp_usage_by_game.get((s_no, home), (0.0, 0.0, 0))
            a_bp = bp_usage_by_game.get((s_no, away), (0.0, 0.0, 0))
            team_bp_usage[home].append((d, h_bp[0], h_bp[1], h_bp[2]))
            team_bp_usage[away].append((d, a_bp[0], a_bp[1], a_bp[2]))
            h_bpq = bp_quality_by_game.get((s_no, home), (0.0, 0.0, 0.0, 0.0, 0.0))
            a_bpq = bp_quality_by_game.get((s_no, away), (0.0, 0.0, 0.0, 0.0, 0.0))
            team_bp_quality[home].append((d, h_bpq[0], h_bpq[1], h_bpq[2], h_bpq[3], h_bpq[4]))
            team_bp_quality[away].append((d, a_bpq[0], a_bpq[1], a_bpq[2], a_bpq[3], a_bpq[4]))

            if s_code:
                park_cum[s_code]["G"] += 1
                park_cum[s_code]["RUNS"] += runs
                park_hr_cum[s_code]["G"] += 1  # v1: HR counted from batter data below
            lg_game_cum["G"] += 1
            lg_game_cum["RUNS"] += runs
            lg_hr_cum["G"] += 1  # v1

            # v1: update key bullpen per-pitcher tracking
            for (bp_date, bp_team, bp_pno), bp_vals in bp_pitcher_map.items():
                if bp_date != str(d):
                    continue
                if bp_team not in (home, away):
                    continue
                # bp_vals: (np, s, hd, ip, h, bb, er, so, hr, ab, sf, hp)
                bp_np, bp_s, bp_hd, bp_ip, bp_h, bp_bb, bp_er, bp_so, bp_hr, bp_ab, bp_sf, bp_hp = bp_vals
                if bp_team == home or bp_team == away:
                    bp_career_sh[(bp_team, bp_pno)] += (bp_s + bp_hd)
                    bp_pitcher_daily_np[(bp_team, bp_pno, str(d))] += bp_np
                    c = bp_ops_cum[(bp_team, bp_pno)]
                    c["AB"] += int(bp_ab); c["H"] += int(bp_h); c["BB"] += int(bp_bb)
                    c["HP"] += int(bp_hp); c["SF"] += int(bp_sf)
                    c["TB"] += int(bp_h + bp_hr)  # rough: TB ≈ H + extra bases; we'll use H+HR as proxy since we don't have 2B/3B

            # v1: park HR factor - count HR from game scores (rough proxy: use game runs as source, but we don't have HR per game here)
            # We'll update park_hr_cum and lg_hr_cum in the batter/pitcher updates below

        # batter update
        for row in bat_by_date.get(d, []):
            p = safe_int(row.get("p_no"))
            if not p: continue
            c = bat_cum[p]
            y = safe_int(str(row.get("date"))[:4])
            cy = bat_year_cum[(p, y)]
            ab = safe_int(row.get("AB"))
            h = safe_int(row.get("H"))
            bb = safe_int(row.get("BB"))
            hp = safe_int(row.get("HP"))
            sf = safe_int(row.get("SF"))
            tb = safe_int(row.get("TB"))
            hr = safe_int(row.get("HR"))
            so = safe_int(row.get("SO"))
            c["AB"] += ab
            c["H"]  += h
            c["BB"] += bb
            c["HP"] += hp
            c["SF"] += sf
            c["TB"] += tb
            c["HR"] += hr
            c["SO"] += so
            cy["AB"] += ab
            cy["H"]  += h
            cy["BB"] += bb
            cy["HP"] += hp
            cy["SF"] += sf
            cy["TB"] += tb
            cy["HR"] += hr
            cy["SO"] += so
            # PA가 없으면 AB+BB+HP+SF로 근사
            pa = row.get("PA")
            pa_v = safe_int(pa, ab + bb + hp + sf)
            c["PA"] += pa_v
            cy["PA"] += pa_v

            bat_lg_cum["AB"] += ab
            bat_lg_cum["H"] += h
            bat_lg_cum["BB"] += bb
            bat_lg_cum["HP"] += hp
            bat_lg_cum["SF"] += sf
            bat_lg_cum["TB"] += tb
            bat_lg_cum["HR"] += hr
            bat_lg_cum["SO"] += so
            bat_lg_cum["PA"] += pa_v
            lg_hr_cum["HR"] += hr  # v1
            bat_recent_games[p].append((ab, h, bb, hp, sf, tb, hr, so, pa_v))

            # v1: park HR tracking
            s_no_bat = safe_int(row.get("s_no"))
            if not s_no_bat:
                s_no_bat = safe_int(row.get("situation"))
            for g_tmp in games_by_date.get(d, []):
                if safe_int(g_tmp["s_no"]) == s_no_bat:
                    sc = safe_int(g_tmp.get("s_code"))
                    if sc and hr > 0:
                        park_hr_cum[sc]["HR"] += hr
                    break

            # v1: foreign bat year tracking
            if manual_tags.get(p, {}).get("is_foreign") == 1:
                fy = foreign_bat_year[y]
                fy["AB"] += ab; fy["H"] += h; fy["BB"] += bb; fy["HP"] += hp
                fy["SF"] += sf; fy["TB"] += tb; fy["HR"] += hr; fy["SO"] += so; fy["PA"] += pa_v

            bo = safe_int(row.get("battingOrder"), 0)
            if 1 <= bo <= 9:
                rk = order_bucket(bo)
                rb = bat_order_cum[rk]
                rb["AB"] += ab
                rb["H"] += h
                rb["BB"] += bb
                rb["HP"] += hp
                rb["SF"] += sf
                rb["TB"] += tb
                rb["HR"] += hr
                rb["SO"] += so
                rb["PA"] += pa_v

            hand_code = player_meta[p]["bat"]
            if hand_code in ("1", "2", "3"):
                hb = bat_hand_cum[hand_code]
                hb["AB"] += ab
                hb["H"] += h
                hb["BB"] += bb
                hb["HP"] += hp
                hb["SF"] += sf
                hb["TB"] += tb
                hb["HR"] += hr
                hb["SO"] += so
                hb["PA"] += pa_v

            s_no_row = safe_int(row.get("s_no"))
            if not s_no_row:
                s_no_row = safe_int(row.get("situation"))
            ctx = today_game_ctx.get(s_no_row)
            if ctx:
                bt = bat_time_cum[(p, ctx["time_bucket"])]
                bt["AB"] += ab
                bt["H"] += h
                bt["BB"] += bb
                bt["HP"] += hp
                bt["SF"] += sf
                bt["TB"] += tb
                bt["HR"] += hr
                bt["SO"] += so
                bt["PA"] += pa_v

                team_code = safe_int(row.get("t_code"))
                opp_throw = ""
                if team_code == ctx["homeTeam"]:
                    opp_throw = ctx["away_sp_throw"]
                elif team_code == ctx["awayTeam"]:
                    opp_throw = ctx["home_sp_throw"]
                opp_side = throw_side(opp_throw)
                if opp_side:
                    bs = bat_vs_throw_cum[(p, opp_side)]
                    bs["AB"] += ab
                    bs["H"] += h
                    bs["BB"] += bb
                    bs["HP"] += hp
                    bs["SF"] += sf
                    bs["TB"] += tb
                    bs["HR"] += hr
                    bs["SO"] += so
                    bs["PA"] += pa_v

        # pitcher update
        for row in pit_by_date.get(d, []):
            p = safe_int(row.get("p_no"))
            if not p: continue
            c = pit_cum[p]
            y = safe_int(str(row.get("date"))[:4])
            cy = pit_year_cum[(p, y)]
            ip = parse_ip(row.get("IP"))
            er = safe_int(row.get("ER"))
            h = safe_int(row.get("H"))
            bb = safe_int(row.get("BB"))
            so = safe_int(row.get("SO"))
            hr = safe_int(row.get("HR"))
            np_v = safe_float(row.get("NP"))
            gs = safe_int(row.get("GS"))
            team_code = safe_int(row.get("t_code"))
            c["IP"] += ip
            c["ER"] += er
            c["H"]  += h
            c["BB"] += bb
            c["SO"] += so
            c["HR"] += hr
            c["G"]  += 1
            c["GS"] += gs
            cy["IP"] += ip
            cy["ER"] += er
            cy["H"] += h
            cy["BB"] += bb
            cy["SO"] += so
            cy["HR"] += hr
            cy["G"] += 1
            cy["GS"] += gs
            pit_lg_cum["IP"] += ip
            pit_lg_cum["ER"] += er
            pit_lg_cum["H"] += h
            pit_lg_cum["BB"] += bb
            pit_lg_cum["SO"] += so
            pit_lg_cum["HR"] += hr
            pit_lg_cum["G"] += 1
            pit_lg_cum["GS"] += gs
            pit_recent[p].append((ip, er, h, bb, so))
            pit_recent_usage[p].append((d, np_v, ip, 1))
            pit_last_game_date[p] = d

            # v1: pitcher SLG-against tracking (AB from row, TB = H + extra - approximate)
            pit_ab = safe_int(row.get("AB"))
            pit_tb_v = safe_int(row.get("TB"))
            if pit_tb_v == 0 and h > 0:
                pit_tb_v = h + hr  # rough fallback
            pit_slg_cum[p]["AB"] += pit_ab
            pit_slg_cum[p]["TB"] += pit_tb_v
            pit_slg_year_cum[(p, y)]["AB"] += pit_ab
            pit_slg_year_cum[(p, y)]["TB"] += pit_tb_v
            pit_slg_lg_cum["AB"] += pit_ab
            pit_slg_lg_cum["TB"] += pit_tb_v

            # v1: foreign pitcher year tracking
            if manual_tags.get(p, {}).get("is_foreign") == 1:
                fp = foreign_pit_year[y]
                fp["IP"] += ip; fp["ER"] += er; fp["H"] += h; fp["BB"] += bb
                fp["SO"] += so; fp["HR"] += hr; fp["G"] += 1; fp["GS"] += gs

            if gs > 0:
                s_no_row = safe_int(row.get("s_no"))
                if not s_no_row:
                    s_no_row = safe_int(row.get("situation"))
                ctx = today_game_ctx.get(s_no_row)
                if ctx:
                    pt = pit_time_cum[(p, ctx["time_bucket"])]
                    pt["IP"] += ip
                    pt["ER"] += er
                    pt["H"] += h
                    pt["BB"] += bb
                    pt["SO"] += so
                    pt["HR"] += hr
                    pt["G"] += 1
                    pt["GS"] += gs

                pit_sp_all_cum["IP"] += ip
                pit_sp_all_cum["ER"] += er
                pit_sp_all_cum["H"] += h
                pit_sp_all_cum["BB"] += bb
                pit_sp_all_cum["SO"] += so
                pit_sp_all_cum["HR"] += hr
                pit_sp_all_cum["G"] += 1
                pit_sp_all_cum["GS"] += gs
                sp_side = throw_side(player_meta[p]["throw"])
                sk = sp_side if sp_side else "U"
                ps = pit_sp_side_cum[sk]
                ps["IP"] += ip
                ps["ER"] += er
                ps["H"] += h
                ps["BB"] += bb
                ps["SO"] += so
                ps["HR"] += hr
                ps["G"] += 1
                ps["GS"] += gs

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print("DONE", "rows=", len(out_rows), "out=", OUT_CSV)

if __name__ == "__main__":
    main()
