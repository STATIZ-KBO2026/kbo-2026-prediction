"""
KBO 경기 일일 예측 파이프라인 (v2-rank1 모델)

하루의 전체 예측 흐름을 한 번에 처리합니다:
  1. gameSchedule API로 오늘 경기 목록 + 시작시간 조회
  2. gameLineup API로 각 경기의 선발 라인업 조회
  3. 기존 누적 데이터 기반으로 v2-rank1 3피처 계산
  4. 학습된 LR 모델로 홈팀 승리확률 예측
  5. savePrediction API로 예측 제출

[사용법]
  # 오늘 경기 예측 (기본)
  python3 predict_today.py

  # 특정 날짜 조회만 (제출하지 않음)
  python3 predict_today.py --date 20260321 --dry-run

  # 특정 날짜 실제 제출
  python3 predict_today.py --date 20260321

[환경변수 또는 하드코딩]
  API_KEY, SECRET은 스크립트 내에 직접 설정되어 있습니다.
"""

import os, sys, json, time, hmac, hashlib, urllib.parse, csv, argparse
from datetime import datetime, date, timedelta
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from collections import defaultdict, deque

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ══════════════════════════════════════════════
# API 설정 (.env 파일에서 로드)
# ══════════════════════════════════════════════
def _load_env(path=None):
    """별도 라이브러리 없이 .env 파일을 읽어 os.environ에 설정합니다."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_load_env()

API_KEY = os.environ.get("STATIZ_API_KEY", "")
SECRET  = os.environ.get("STATIZ_SECRET", "").encode("utf-8")
if not API_KEY or not SECRET:
    sys.exit("❌ STATIZ_API_KEY / STATIZ_SECRET 환경변수가 설정되지 않았습니다.\n"
             "   .env 파일을 만들거나 export로 설정해주세요.")
BASE    = "https://api.statiz.co.kr/baseballApi"

# ══════════════════════════════════════════════
# 데이터 경로
# ══════════════════════════════════════════════
DATA_DIR   = os.path.expanduser("~/statiz/data")
GAMES_CSV  = os.path.join(DATA_DIR, "game_index_played.csv")
LINEUP_CSV = os.path.join(DATA_DIR, "lineup_long.csv")
BAT_CSV    = os.path.join(DATA_DIR, "playerday_batter_long.csv")
PIT_CSV    = os.path.join(DATA_DIR, "playerday_pitcher_long.csv")
FEAT_CSV   = os.path.join(DATA_DIR, "features_v2_candidates.csv")
LOG_DIR    = os.path.expanduser("~/statiz/logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ══════════════════════════════════════════════
# 피처 파라미터 (build_features_v2_candidates.py와 동일)
# ══════════════════════════════════════════════
K_SMOOTH = 20
MIN_PA_LASTSEASON = 60
EARLY_BULLPEN_TEAM_GAMES = 20
RECENT_TEAM_GAMES_FOR_SP = 7
PREV_SEASON_GS_THRESHOLD = 5
FALLBACK_PRIOR_OPS = 0.700

# v2 추가 파라미터
K_BBIP_IP = 30.0              # BB/IP 베이즈 평활 강도
MIN_IP_LASTSEASON = 30.0      # 직전 시즌 BB/IP를 prior로 쓸 최소 이닝
FALLBACK_PRIOR_BBIP = 0.35    # BB/IP 기본값
PYTHAG_EXP = 1.83             # 피타고리안 승률 지수
K_PYTHAG_GAMES = 20.0         # 피타고리안 승률 평활 강도
MIN_PYTHAG_PRIOR_GAMES = 30   # 직전 시즌 prior 사용 최소 경기 수

# v2-rank1 피처 3개: 불펜피로도 + 투수제구력(BB/IP) + 피타고리안승률
FEATURE_COLS = [
    "diff_bullpen_fatigue",
    "diff_sp_bbip",
    "diff_pythag_winpct",
]


# ══════════════════════════════════════════════
# API 호출 함수
# ══════════════════════════════════════════════
def signed_get(path: str, params: dict, timeout=20):
    """HMAC-SHA256 서명이 포함된 GET 요청"""
    method = "GET"
    normalized_query = "&".join(
        f"{urllib.parse.quote(k)}={urllib.parse.quote(str(params[k]))}"
        for k in sorted(params)
    )
    ts = str(int(time.time()))
    payload = f"{method}|{path}|{normalized_query}|{ts}"
    sig = hmac.new(SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    url = f"{BASE}/{path}?{normalized_query}"
    req = Request(url, method=method, headers={
        "X-API-KEY": API_KEY,
        "X-TIMESTAMP": ts,
        "X-SIGNATURE": sig,
    })
    with urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8", errors="replace"))


def signed_post(path: str, params: dict, timeout=20):
    """
    HMAC-SHA256 서명이 포함된 POST 요청 (savePrediction용)

    핵심: 서명은 GET과 동일 구조(method="POST")이고,
    파라미터는 URL 쿼리스트링 + form body 양쪽에 전송합니다.
    """
    method = "POST"
    normalized_query = "&".join(
        f"{urllib.parse.quote(k)}={urllib.parse.quote(str(params[k]))}"
        for k in sorted(params)
    )
    ts = str(int(time.time()))
    payload = f"{method}|{path}|{normalized_query}|{ts}"
    sig = hmac.new(SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    url = f"{BASE}/{path}?{normalized_query}"
    body_data = urllib.parse.urlencode(params).encode("utf-8")

    req = Request(url, data=body_data, method=method, headers={
        "X-API-KEY": API_KEY,
        "X-TIMESTAMP": ts,
        "X-SIGNATURE": sig,
        "Content-Type": "application/x-www-form-urlencoded",
    })
    with urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8", errors="replace"))


# ══════════════════════════════════════════════
# 1단계: 오늘 경기 일정 조회
# ══════════════════════════════════════════════
def fetch_schedule(target_date: date):
    """
    gameSchedule API로 해당 날짜의 경기 목록을 조회합니다.

    Returns:
        [{"s_no": int, "homeTeam": int, "awayTeam": int, "hm": str, "s_state": int, ...}, ...]
    """
    y, m, d = target_date.year, target_date.month, target_date.day
    params = {"year": str(y), "month": str(m), "day": str(d)}

    print(f"\n[1/5] 📅 경기 일정 조회: {target_date.isoformat()}")
    status, data = signed_get("prediction/gameSchedule", params)

    if data.get("result_cd") != 100:
        print(f"  ⚠ API 응답 비정상: {data.get('result_msg')}")
        return []

    # 날짜 키 찾기 (MMDD 형식)
    date_key = f"{m:02d}{d:02d}"
    games = data.get(date_key, [])

    if not games:
        # 다른 키 형식 시도
        for k, v in data.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict) and "s_no" in v[0]:
                games = v
                break

    result = []
    for g in games:
        s_no = g.get("s_no")
        if not s_no:
            continue
        result.append({
            "s_no": int(s_no),
            "homeTeam": g.get("homeTeam"),
            "awayTeam": g.get("awayTeam"),
            "hm": g.get("hm", ""),
            "s_state": g.get("s_state"),
            "s_code": g.get("s_code"),
            "homeSPName": g.get("homeSPName", ""),
            "awaySPName": g.get("awaySPName", ""),
        })

    print(f"  ✅ {len(result)}개 경기 발견")
    for g in result:
        print(f"     s_no={g['s_no']}  {g.get('awaySPName','?')}(원정) vs {g.get('homeSPName','?')}(홈)  시작: {g['hm']}")

    return result


# ══════════════════════════════════════════════
# 2단계: 라인업 조회
# ══════════════════════════════════════════════
def fetch_lineup(s_no: int, home_team: int, away_team: int):
    """
    gameLineup API로 한 경기의 선발 라인업을 조회합니다.

    Returns:
        {"home": {"P": p_no, "batters": {1: p_no, ...}},
         "away": {"P": p_no, "batters": {1: p_no, ...}}}
    """
    status, data = signed_get("prediction/gameLineup", {"s_no": str(s_no)})

    if isinstance(data, dict) and data.get("result_cd") != 100:
        print(f"  ⚠ s_no={s_no} 라인업 조회 실패: {data.get('result_msg')}")
        return None

    lineup = {"home": {"P": None, "batters": {}}, "away": {"P": None, "batters": {}}}

    for team_key, players in data.items():
        if not (isinstance(team_key, str) and team_key.isdigit() and isinstance(players, list)):
            continue

        t_code = int(team_key)
        if t_code == home_team:
            side = "home"
        elif t_code == away_team:
            side = "away"
        else:
            continue

        for p in players:
            bo = str(p.get("battingOrder", "")).strip()
            p_no = p.get("p_no")
            if not p_no:
                continue

            if bo == "P":
                lineup[side]["P"] = int(p_no)
            else:
                try:
                    order = int(bo)
                    if 1 <= order <= 9:
                        lineup[side]["batters"][order] = int(p_no)
                except (ValueError, TypeError):
                    pass

    return lineup


def fetch_all_lineups(games: list):
    """모든 경기의 라인업을 조회합니다."""
    print(f"\n[2/5] 📋 라인업 조회 ({len(games)}경기)")
    lineups = {}
    for g in games:
        s_no = g["s_no"]
        lu = fetch_lineup(s_no, g["homeTeam"], g["awayTeam"])
        if lu:
            lineups[s_no] = lu
            home_batters = len(lu["home"]["batters"])
            away_batters = len(lu["away"]["batters"])
            home_sp = "✅" if lu["home"]["P"] else "❌"
            away_sp = "✅" if lu["away"]["P"] else "❌"
            print(f"  s_no={s_no}: 홈타자={home_batters}/9 홈선발={home_sp}  원정타자={away_batters}/9 원정선발={away_sp}")
        else:
            print(f"  s_no={s_no}: ⚠ 라인업 없음")
        time.sleep(0.15)
    return lineups


# ══════════════════════════════════════════════
# 3단계: 피처 생성 (누적 통계 기반)
# ══════════════════════════════════════════════
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


def calc_ops(H, BB, HP, AB, SF, TB):
    denom = AB + BB + HP + SF
    obp = (H + BB + HP) / denom if denom else 0.0
    slg = TB / AB if AB else 0.0
    return obp + slg, denom


def smooth(curr_val, curr_w, prior_val, K=K_SMOOTH):
    if curr_w < 0:
        curr_w = 0
    return (curr_w / (curr_w + K)) * curr_val + (K / (curr_w + K)) * prior_val


def ip_to_outs(ip_val):
    """KBO 이닝 표기(6.1→19아웃)를 아웃카운트로 변환"""
    s = str(ip_val).strip()
    if s == "" or s.lower() == "none":
        return 0
    if "." not in s:
        return max(0, safe_int(s)) * 3
    whole, frac = s.split(".", 1)
    w = max(0, safe_int(whole))
    f = frac[:1]
    add = 1 if f == "1" else (2 if f == "2" else 0)
    return w * 3 + add


def pythag_winpct(rs, ra, exp=PYTHAG_EXP):
    """피타고리안 기대 승률: 득점^exp / (득점^exp + 실점^exp)"""
    rs = max(0.0, float(rs))
    ra = max(0.0, float(ra))
    if rs == 0 and ra == 0:
        return 0.5
    rs_e = rs ** exp
    ra_e = ra ** exp
    denom = rs_e + ra_e
    return (rs_e / denom) if denom > 0 else 0.5


def get_svhld(row):
    sv = 0
    for key in ("SV", "sv", "Save", "save", "S"):
        if key in row:
            sv = safe_int(row.get(key))
            break
    hd = 0
    for key in ("HLD", "HD", "hld", "hd", "Hold", "hold"):
        if key in row:
            hd = safe_int(row.get(key))
            break
    return sv + hd


def build_cumulative_stats():
    """
    역대 모든 데이터(playerday CSV + game_index)를 읽어
    선수별, 팀별 누적 통계를 빌드합니다.
    v2-rank1 피처에 필요한 BB/IP, 피타고리안 승률 통계도 포함합니다.
    """
    print(f"\n[3/5] 📊 누적 통계 빌드 중...")

    # 투수 누적 (BB/IP 계산용으로 OUTS 추가)
    pit_cum = defaultdict(lambda: {"AB":0, "H":0, "BB":0, "HP":0, "SF":0, "TB":0, "BF":0, "OUTS":0})
    league_pit_tot = defaultdict(lambda: {"BB":0, "OUTS":0})  # 리그 평균 BB/IP 계산용
    pit_season_gs = defaultdict(int)
    pitcher_svhld_season = defaultdict(int)
    pitcher_np_by_date = defaultdict(int)
    team_game_cnt = defaultdict(int)
    team_recent_starters = defaultdict(lambda: deque(maxlen=RECENT_TEAM_GAMES_FOR_SP))
    team_pitchers_by_year = defaultdict(set)

    # 팀 레벨 득실점 (피타고리안 승률용)
    team_runs = defaultdict(lambda: {"RS":0, "RA":0, "G":0})

    # 투수 데이터 로드
    pit_count = 0
    with open(PIT_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p_no = safe_int(row.get("p_no"))
            y = safe_int(row.get("year"))
            d = row.get("date", "")
            t_code = safe_int(row.get("t_code"))
            if not p_no or not y:
                continue

            AB = safe_int(row.get("AB"))
            H  = safe_int(row.get("H"))
            BB = safe_int(row.get("BB"))
            HP = safe_int(row.get("HP"))
            SF = safe_int(row.get("SF"))
            TB = safe_int(row.get("TB"))
            TBF = safe_int(row.get("TBF"))
            GS = safe_int(row.get("GS"))
            NP = safe_int(row.get("NP"))
            SVHLD = get_svhld(row)
            outs = ip_to_outs(row.get("IP"))

            BF = TBF if TBF > 0 else (AB + BB + HP + SF)

            pit_cum[(p_no, y)]["AB"] += AB
            pit_cum[(p_no, y)]["H"]  += H
            pit_cum[(p_no, y)]["BB"] += BB
            pit_cum[(p_no, y)]["HP"] += HP
            pit_cum[(p_no, y)]["SF"] += SF
            pit_cum[(p_no, y)]["TB"] += TB
            pit_cum[(p_no, y)]["BF"] += BF
            pit_cum[(p_no, y)]["OUTS"] += outs

            league_pit_tot[y]["BB"] += BB
            league_pit_tot[y]["OUTS"] += outs

            pit_season_gs[(p_no, y)] += GS
            pitcher_svhld_season[(p_no, y)] += SVHLD
            pitcher_np_by_date[(p_no, d)] += NP

            if t_code:
                team_pitchers_by_year[(t_code, y)].add(p_no)
            pit_count += 1

    # 경기 인덱스에서 팀 경기 수 + 득실점 집계
    with open(GAMES_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = row.get("date", "")
            home = safe_int(row.get("homeTeam"))
            away = safe_int(row.get("awayTeam"))
            hs = safe_int(row.get("homeScore"))
            aws = safe_int(row.get("awayScore"))
            y = safe_int(d[:4]) if len(d) >= 4 else 0
            if not y:
                continue
            if home:
                team_game_cnt[(home, y)] += 1
                team_runs[(home, y)]["RS"] += hs
                team_runs[(home, y)]["RA"] += aws
                team_runs[(home, y)]["G"] += 1
            if away:
                team_game_cnt[(away, y)] += 1
                team_runs[(away, y)]["RS"] += aws
                team_runs[(away, y)]["RA"] += hs
                team_runs[(away, y)]["G"] += 1

    print(f"  ✅ 투수 기록 {pit_count:,}건 로드, 팀 득실점 집계 완료")

    return {
        "pit_cum": pit_cum,
        "league_pit_tot": league_pit_tot,
        "pit_season_gs": pit_season_gs,
        "pitcher_svhld_season": pitcher_svhld_season,
        "pitcher_np_by_date": pitcher_np_by_date,
        "team_game_cnt": team_game_cnt,
        "team_pitchers_by_year": team_pitchers_by_year,
        "team_recent_starters": team_recent_starters,
        "team_runs": team_runs,
    }


def compute_features(games, lineups, stats, target_year):
    """
    오늘 경기들의 v2-rank1 3피처를 계산합니다.

    Returns:
        {s_no: {"diff_bullpen_fatigue": ...,
                "diff_sp_bbip": ...,
                "diff_pythag_winpct": ...}, ...}
    """
    pit_cum = stats["pit_cum"]
    league_pit_tot = stats["league_pit_tot"]
    pit_season_gs = stats["pit_season_gs"]
    pitcher_svhld_season = stats["pitcher_svhld_season"]
    pitcher_np_by_date = stats["pitcher_np_by_date"]
    team_game_cnt = stats["team_game_cnt"]
    team_pitchers_by_year = stats["team_pitchers_by_year"]
    team_recent_starters = stats["team_recent_starters"]
    team_runs = stats["team_runs"]

    year = target_year

    # ── BB/IP 관련 함수 ──
    def league_bbip(y):
        """해당 연도 리그 평균 BB/IP"""
        tot = league_pit_tot.get(y)
        if not tot or tot["OUTS"] <= 0:
            return FALLBACK_PRIOR_BBIP
        ip = tot["OUTS"] / 3.0
        return (tot["BB"] / ip) if ip > 0 else FALLBACK_PRIOR_BBIP

    def pitcher_bbip_smooth(p_no, y):
        """투수 BB/IP(이닝당 볼넷)에 베이즈 평활 적용"""
        cur = pit_cum[(p_no, y)]
        cur_ip = cur["OUTS"] / 3.0 if cur["OUTS"] > 0 else 0.0
        cur_val = (cur["BB"] / cur_ip) if cur_ip > 0 else 0.0

        prev = pit_cum.get((p_no, y - 1))
        if prev:
            prev_ip = prev["OUTS"] / 3.0 if prev["OUTS"] > 0 else 0.0
            prev_val = (prev["BB"] / prev_ip) if prev_ip > 0 else 0.0
        else:
            prev_ip, prev_val = 0.0, 0.0

        prior = prev_val if prev_ip >= MIN_IP_LASTSEASON else league_bbip(y - 1)
        return smooth(cur_val, cur_ip, prior, K_BBIP_IP)

    # ── 피타고리안 승률 함수 ──
    def team_pythag_smooth(team, y):
        """팀 피타고리안 기대 승률에 베이즈 평활 적용"""
        cur = team_runs[(team, y)]
        cur_val = pythag_winpct(cur["RS"], cur["RA"])
        cur_w = cur["G"]

        prev = team_runs.get((team, y - 1))
        if prev and prev["G"] >= MIN_PYTHAG_PRIOR_GAMES:
            prior = pythag_winpct(prev["RS"], prev["RA"])
        else:
            prior = 0.5

        return smooth(cur_val, cur_w, prior, K_PYTHAG_GAMES)

    # ── 불펜 피로도 함수 ──
    def is_starter_group(team, p_no, y, team_game_no):
        if team_game_no <= EARLY_BULLPEN_TEAM_GAMES:
            return pit_season_gs.get((p_no, y - 1), 0) >= PREV_SEASON_GS_THRESHOLD
        return p_no in set(team_recent_starters[(team, y)])

    def pitcher_fatigue(p_no, date_str):
        d0 = datetime.strptime(date_str, "%Y%m%d")
        score = 0
        for lag in range(1, 6):
            w = 6 - lag
            dp = (d0 - timedelta(days=lag)).strftime("%Y%m%d")
            score += w * pitcher_np_by_date.get((p_no, dp), 0)
        return score

    def select_core_bullpen(team, y, team_game_no, sp_pno):
        candidates = set()
        candidates.update(team_pitchers_by_year.get((team, y - 1), set()))
        candidates.update(team_pitchers_by_year.get((team, y), set()))
        if sp_pno:
            candidates.discard(sp_pno)
        if not candidates:
            return []
        pool = [p for p in candidates if not is_starter_group(team, p, y, team_game_no)]
        if team_game_no <= EARLY_BULLPEN_TEAM_GAMES:
            ranked = sorted(pool, key=lambda p: (pitcher_svhld_season.get((p, y-1), 0), pitcher_svhld_season.get((p, y), 0), p), reverse=True)
        else:
            ranked = sorted(pool, key=lambda p: (pitcher_svhld_season.get((p, y), 0), pitcher_svhld_season.get((p, y-1), 0), p), reverse=True)
        return ranked[:4]

    # ── 피처 계산 ──
    today_str = datetime.now().strftime("%Y%m%d")
    features = {}

    for g in games:
        s_no = g["s_no"]
        lu = lineups.get(s_no)
        if not lu:
            continue

        home = g["homeTeam"]
        away = g["awayTeam"]
        home_game_no = team_game_cnt.get((home, year), 0) + 1
        away_game_no = team_game_cnt.get((away, year), 0) + 1

        # 선발투수
        home_sp = lu["home"]["P"]
        away_sp = lu["away"]["P"]

        # (1) 불펜 피로도
        core4_home = select_core_bullpen(home, year, home_game_no, home_sp)
        core4_away = select_core_bullpen(away, year, away_game_no, away_sp)
        home_fat = sum(pitcher_fatigue(p, today_str) for p in core4_home)
        away_fat = sum(pitcher_fatigue(p, today_str) for p in core4_away)

        # (2) 선발투수 BB/IP (제구력)
        home_sp_bbip = pitcher_bbip_smooth(home_sp, year) if home_sp else league_bbip(year - 1)
        away_sp_bbip = pitcher_bbip_smooth(away_sp, year) if away_sp else league_bbip(year - 1)

        # (3) 피타고리안 기대 승률
        home_pyth = team_pythag_smooth(home, year) if home else 0.5
        away_pyth = team_pythag_smooth(away, year) if away else 0.5

        features[s_no] = {
            "diff_bullpen_fatigue": home_fat - away_fat,
            "diff_sp_bbip": round(home_sp_bbip - away_sp_bbip, 6),
            "diff_pythag_winpct": round(home_pyth - away_pyth, 6),
        }

    return features


# ══════════════════════════════════════════════
# 4단계: 모델 예측
# ══════════════════════════════════════════════
def train_model():
    """features_v2_candidates.csv의 전체 데이터로 v2-rank1 LR 모델을 학습합니다."""
    print(f"\n[4/5] 🤖 v2-rank1 모델 학습 중...")

    import pandas as pd
    df = pd.read_csv(FEAT_CSV)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLS + ["y_home_win"]).copy()
    df["y_home_win"] = df["y_home_win"].astype(int)

    X = df[FEATURE_COLS].values
    y = df["y_home_win"].values

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=42)),
    ])
    model.fit(X, y)
    print(f"  ✅ 학습 완료 (n={len(df)})")
    return model


def predict_games(model, features_dict):
    """각 경기의 피처로 홈팀 승리확률을 예측합니다."""
    predictions = {}

    for s_no, feat in features_dict.items():
        X = np.array([[feat[c] for c in FEATURE_COLS]])
        prob = model.predict_proba(X)[0, 1]
        # 소수점 두 자리까지 (API 스펙)
        prob_pct = round(prob * 100, 2)
        predictions[s_no] = prob_pct

    return predictions


# ══════════════════════════════════════════════
# 5단계: 예측 제출
# ══════════════════════════════════════════════
def submit_predictions(predictions: dict, dry_run=False):
    """savePrediction API로 각 경기의 예측을 제출합니다."""
    print(f"\n[5/5] 📤 예측 제출 {'(DRY RUN - 실제 전송 안 함)' if dry_run else ''}")

    results = []
    for s_no, prob in predictions.items():
        print(f"  s_no={s_no}  홈팀 승리확률={prob:.2f}%", end="")

        if dry_run:
            print("  → SKIP (dry-run)")
            results.append({"s_no": s_no, "percent": prob, "status": "DRY_RUN", "msg": ""})
            continue

        try:
            status, data = signed_post("prediction/savePrediction", {
                "s_no": str(s_no),
                "percent": f"{prob:.2f}",
            })
            result_cd = data.get("result_cd", data.get("cdoe"))
            result_msg = data.get("result_msg", "")
            ok = (result_cd == 100)
            print(f"  → {'✅ 성공' if ok else '❌ 실패'}: {result_msg}")
            results.append({"s_no": s_no, "percent": prob, "status": "OK" if ok else "FAIL", "msg": result_msg})

        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            try:
                err_data = json.loads(body)
                result_msg = err_data.get("result_msg", body)
            except:
                result_msg = body
            print(f"  → ❌ HTTP {e.code}: {result_msg}")
            results.append({"s_no": s_no, "percent": prob, "status": f"HTTP_{e.code}", "msg": result_msg})

        except Exception as e:
            print(f"  → ❌ Error: {e}")
            results.append({"s_no": s_no, "percent": prob, "status": "ERROR", "msg": str(e)})

        time.sleep(0.2)

    return results


def load_submitted(target_date):
    """
    오늘 이미 제출 성공한 경기의 s_no 세트를 반환합니다.
    하루에 여러 번 실행해도 이미 성공한 경기는 다시 제출하지 않습니다.
    """
    log_file = os.path.join(LOG_DIR, f"predict_{target_date.strftime('%Y%m%d')}.csv")
    submitted = set()
    if not os.path.exists(log_file):
        return submitted
    with open(log_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "OK":
                submitted.add(int(row["s_no"]))
    return submitted


def save_log(target_date, predictions, results):
    """제출 결과를 로그 파일에 추가(append)합니다."""
    log_file = os.path.join(LOG_DIR, f"predict_{target_date.strftime('%Y%m%d')}.csv")
    fieldnames = ["datetime", "date", "s_no", "percent", "status", "msg"]

    file_exists = os.path.exists(log_file)
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for r in results:
            w.writerow({
                "datetime": now,
                "date": target_date.strftime("%Y%m%d"),
                "s_no": r["s_no"],
                "percent": r["percent"],
                "status": r["status"],
                "msg": r["msg"],
            })

    print(f"\n📝 로그 저장: {log_file}")


# ══════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="KBO 일일 예측 파이프라인")
    p.add_argument("--date", type=str, default=None,
                   help="예측 대상 날짜 (YYYYMMDD, 기본: 오늘)")
    p.add_argument("--dry-run", action="store_true",
                   help="API 제출 없이 예측까지만 실행")
    return p.parse_args()


def main():
    args = parse_args()

    # 대상 날짜 결정
    if args.date:
        target = datetime.strptime(args.date, "%Y%m%d").date()
    else:
        target = date.today()

    target_year = target.year

    print("=" * 60)
    print(f"  KBO 일일 예측 파이프라인 (v2-rank1)")
    print(f"  피처: 불펜피로도 + 투수BB/IP + 피타고리안승률")
    print(f"  대상 날짜: {target.isoformat()}")
    print(f"  모드: {'DRY RUN' if args.dry_run else '실제 제출'}")
    print("=" * 60)

    # 이미 제출 성공한 경기 확인
    already_submitted = load_submitted(target)
    if already_submitted:
        print(f"\nℹ️  이미 제출된 경기: {sorted(already_submitted)}")

    # 1단계: 경기 일정 조회
    games = fetch_schedule(target)
    if not games:
        print("\n⚠ 오늘 경기가 없습니다.")
        return

    # 이미 제출한 경기 제외
    remaining_games = [g for g in games if g["s_no"] not in already_submitted]
    if not remaining_games:
        print(f"\n✅ 모든 {len(games)}경기 이미 제출 완료!")
        return

    skipped = len(games) - len(remaining_games)
    if skipped > 0:
        print(f"\n⏭️  {skipped}경기 이미 제출됨 → 나머지 {len(remaining_games)}경기만 처리")

    # 2단계: 라인업 조회 (남은 경기만)
    lineups = fetch_all_lineups(remaining_games)
    if not lineups:
        print("\n⚠ 라인업을 조회할 수 없습니다. (아직 발표 전일 수 있음)")
        return

    # 3단계: 누적 통계 빌드 + 피처 계산
    stats = build_cumulative_stats()
    features = compute_features(remaining_games, lineups, stats, target_year)

    if not features:
        print("\n⚠ 피처를 계산할 수 없습니다.")
        return

    # 4단계: 모델 학습 + 예측
    model = train_model()
    predictions = predict_games(model, features)

    print(f"\n📊 예측 결과:")
    for s_no, prob in predictions.items():
        print(f"  s_no={s_no}  홈팀 승리확률={prob:.2f}%")

    # 5단계: 제출
    results = submit_predictions(predictions, dry_run=args.dry_run)

    # 로그 저장 (append)
    save_log(target, predictions, results)

    # 결과 요약
    ok_cnt = sum(1 for r in results if r["status"] == "OK")
    total_submitted = len(already_submitted) + ok_cnt
    print(f"\n✅ 파이프라인 완료! (오늘 제출: {total_submitted}/{len(games)}경기)")


if __name__ == "__main__":
    main()
