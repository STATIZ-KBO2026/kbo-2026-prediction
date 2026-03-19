"""
v1 핵심 피처 생성기 — 논문 기반 4개 피처

경기별로 홈/원정팀의 공격력·투수력·불펜 피로도를 나타내는
4가지 핵심 피처를 계산하여 CSV로 저장합니다.

[생성되는 4가지 피처]
  1. diff_sum_ops_smooth   — 타순 1~9번 OPS_smooth 합의 홈-원정 차이
  2. diff_sum_ops_recent5  — 타순 1~9번 최근 5경기 OPS 합의 홈-원정 차이
  3. diff_sp_oops          — 선발투수 피안타 OPS(상대에게 허용한 OPS)의 홈-원정 차이
  4. diff_bullpen_fatigue  — 핵심 불펜 4인의 피로도 점수 합의 홈-원정 차이

[핵심 로직]
  - 경기 D일의 피처는 반드시 D-1일까지의 데이터만 사용합니다 (미래 정보 누수 방지).
  - "하루 치리 → 피처 뽑고 → 그날 기록 업데이트" 순으로 날짜별 순회합니다.
  - OPS_smooth = (누적PA / (누적PA + K)) × 누적OPS + (K / (누적PA + K)) × 사전OPS
    시즌 초반 적은 타석에서도 안정적인 값을 만들기 위한 베이즈 평활(smoothing) 기법입니다.
  - 사전(prior) OPS: 작년 성적이 60PA 이상이면 작년 OPS, 아니면 리그 평균 OPS를 사용합니다.

[파이프라인 위치]
  8a단계 — build_playerday_tables_v2.py 이후에 실행합니다.

[입력]
  - ~/statiz/data/game_index_played.csv
  - ~/statiz/data/lineup_long.csv
  - ~/statiz/data/playerday_batter_long.csv
  - ~/statiz/data/playerday_pitcher_long.csv

[출력]
  - ~/statiz/data/features_v1_paper.csv (2024년 이후 경기만 포함)
"""
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

# ──────────────────────────────────────────────
# 하이퍼파라미터
# ──────────────────────────────────────────────
K_SMOOTH = 20                    # 베이즈 평활 강도 (클수록 prior 쪽으로 더 끌림)
MIN_PA_LASTSEASON = 60           # 작년 OPS를 prior로 쓸 최소 타석 수
MIN_PA_RECENT = 10               # 최근 5경기 OPS를 사용할 최소 타석 수
RECENT_GAMES = 5                 # 최근 경기 윈도우 크기

EARLY_BULLPEN_TEAM_GAMES = 20    # 시즌 초반(1~20경기): 직전 시즌 기준으로 선발/불펜 구분
RECENT_TEAM_GAMES_FOR_SP = 7     # 시즌 중반 이후: 최근 7경기 내 선발 등판하면 선발군
PREV_SEASON_GS_THRESHOLD = 5    # 시즌 초반 선발군 기준: 직전 시즌 선발 등판(GS) ≥ 5

MIN_FEATURE_YEAR = 2024          # 피처 CSV에 출력할 최소 연도
FALLBACK_PRIOR_OPS = 0.700       # 리그 평균 OPS 계산이 불가할 때 사용할 기본값


# ──────────────────────────────────────────────
# 유틸리티 함수
# ──────────────────────────────────────────────
def safe_int(x, default=0):
    """문자열이나 None을 안전하게 정수로 변환합니다. 실패 시 default 반환."""
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
    """YYYYMMDD 형식 문자열을 datetime 객체로 변환합니다."""
    return datetime.strptime(s, "%Y%m%d")

def dt_to_yyyymmdd(d: datetime) -> str:
    """datetime 객체를 YYYYMMDD 형식 문자열로 변환합니다."""
    return d.strftime("%Y%m%d")

def calc_ops_from_counts(H, BB, HP, AB, SF, TB):
    """
    개수 기반으로 OPS(출루율 + 장타율)와 타석 수(PA)를 계산합니다.

    - 출루율(OBP) = (안타 + 볼넷 + 사구) / (타수 + 볼넷 + 사구 + 희비)
    - 장타율(SLG) = 루타 / 타수
    - OPS = OBP + SLG

    Returns:
        (OPS 값, PA 수) 튜플
    """
    denom_obp = AB + BB + HP + SF
    obp = (H + BB + HP) / denom_obp if denom_obp else 0.0
    slg = TB / AB if AB else 0.0
    ops = obp + slg
    pa = denom_obp
    return ops, pa

def smooth_value(curr_val, curr_w, prior_val, K):
    """
    베이즈 평활(Bayesian Smoothing) 공식으로 안정적인 추정값을 만듭니다.

    공식: (현재가중치 / (현재가중치 + K)) × 현재값 + (K / (현재가중치 + K)) × 사전값
    - 데이터가 적으면(curr_w가 작으면) prior_val 쪽으로 끌립니다.
    - 데이터가 충분하면 curr_val을 그대로 사용합니다.

    Args:
        curr_val: 현재까지의 관측 값 (예: 시즌 누적 OPS)
        curr_w: 현재 가중치 (예: 누적 타석 수)
        prior_val: 사전 값 (예: 작년 OPS 또는 리그 평균)
        K: 평활 강도 (클수록 사전값에 기댐)
    """
    if curr_w < 0:
        curr_w = 0
    return (curr_w / (curr_w + K)) * curr_val + (K / (curr_w + K)) * prior_val

def get_save_from_row(row):
    """CSV 행에서 세이브(SV) 값을 찾아 반환합니다. 키 이름이 다를 수 있어서 여러 후보를 탐색."""
    for key in ("SV", "sv", "Save", "save"):
        if key in row:
            return safe_int(row.get(key))
    return 0

def get_hold_from_row(row):
    """CSV 행에서 홀드(HD) 값을 찾아 반환합니다."""
    for key in ("HLD", "HD", "hld", "hd", "Hold", "hold"):
        if key in row:
            return safe_int(row.get(key))
    return 0

def get_svhld_from_row(row):
    """세이브 + 홀드 합계를 반환합니다. 핵심 불펜 투수를 식별하는 데 사용."""
    return get_save_from_row(row) + get_hold_from_row(row)


# ──────────────────────────────────────────────
# 데이터 로더
# ──────────────────────────────────────────────
def load_games():
    """game_index_played.csv에서 경기 목록을 읽어 날짜+경기번호 순으로 정렬합니다."""
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
    """
    lineup_long.csv에서 경기별·팀별 선발 라인업 정보를 구조화합니다.

    Returns:
        lineup_map[s_no][side]["P"]            → 선발투수 p_no
        lineup_map[s_no][side]["batters"][1~9]  → 해당 타순 타자 p_no
    """
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
    """
    CSV 파일을 날짜별로 그룹핑합니다.
    D일의 피처 생성 후 D일 기록으로 누적 통계를 업데이트하는 패턴에 사용됩니다.

    Returns:
        {날짜 문자열: [해당 날짜의 행 리스트, ...]} 형태의 dict
    """
    by_date = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            d = row.get("date")
            if d:
                by_date[d].append(row)
    return by_date


# ──────────────────────────────────────────────
# 메인 로직
# ──────────────────────────────────────────────
def main():
    games = load_games()
    lineup_map = load_lineup_map()

    bat_by_date = group_by_date(BAT_CSV)
    pit_by_date = group_by_date(PIT_CSV)

    games_by_date = defaultdict(list)
    for g in games:
        games_by_date[g["date"]].append(g)
    dates = sorted(games_by_date.keys())

    # ── 시즌 누적 통계 저장소 ──
    bat_cum = defaultdict(lambda: {"AB":0, "H":0, "BB":0, "HP":0, "SF":0, "TB":0})
    pit_cum = defaultdict(lambda: {"AB":0, "H":0, "BB":0, "HP":0, "SF":0, "TB":0, "BF":0})

    # 리그 전체 타격 합계 (prior OPS 계산용)
    league_tot = defaultdict(lambda: {"AB":0, "H":0, "BB":0, "HP":0, "SF":0, "TB":0})

    # 최근 5경기 타격 기록 윈도우
    bat_recent = defaultdict(lambda: deque(maxlen=RECENT_GAMES))

    # 투수 시즌 누적 정보
    pit_season_gs = defaultdict(int)        # (p_no, year) → 선발 등판 수(GS)
    pitcher_svhld_season = defaultdict(int)  # (p_no, year) → 세이브+홀드 합계
    pitcher_np_by_date = defaultdict(int)    # (p_no, date) → 해당 날짜 투구 수

    # 팀 레벨 컨텍스트
    team_game_cnt = defaultdict(int)           # (team, year) → 해당 시즌 누적 경기 수
    team_recent_starters = defaultdict(        # (team, year) → 최근 7경기 선발투수 기록
        lambda: deque(maxlen=RECENT_TEAM_GAMES_FOR_SP)
    )
    team_pitchers_by_year = defaultdict(set)   # (team, year) → 해당 시즌 등판한 투수 집합


    def league_ops(year: int) -> float:
        """해당 연도의 리그 평균 OPS를 계산합니다. 계산 불가 시 기본값(0.700) 반환."""
        tot = league_tot.get(year)
        if not tot:
            return FALLBACK_PRIOR_OPS
        ops, _ = calc_ops_from_counts(tot["H"], tot["BB"], tot["HP"], tot["AB"], tot["SF"], tot["TB"])
        return ops if ops > 0 else FALLBACK_PRIOR_OPS

    def batter_ops_smooth(p_no: int, year: int):
        """
        타자의 시즌 누적 OPS에 베이즈 평활을 적용합니다.

        - prior(사전값): 작년 PA ≥ 60이면 작년 OPS, 아니면 리그 평균 OPS
        - 시즌 초반 타석이 적을 때 극단적인 값을 방지합니다.

        Returns:
            (smoothed OPS, 현재 시즌 PA 수) 튜플
        """
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
        """
        최근 5경기 OPS가 충분하면(PA ≥ 10) 그 값을, 아니면 OPS_smooth를 반환합니다.
        최근 폼을 반영하기 위한 피처입니다.

        Returns:
            (최근 OPS 또는 smoothed OPS, 최근 PA 수) 튜플
        """
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
        """
        투수의 피안타 OPS(상대에게 허용한 OPS)에 베이즈 평활을 적용합니다.
        타자 OPS_smooth와 같은 원리이지만, 가중치로 타석 대신 상대 타자 수(BF)를 사용합니다.

        Returns:
            (smoothed 피안타 OPS, 현재 시즌 BF 수) 튜플
        """
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
        """
        해당 투수가 선발군에 속하는지 판별합니다.

        - 시즌 초반(≤20경기): 직전 시즌에 선발 등판(GS) ≥ 5이면 선발군
        - 시즌 중반 이후: 최근 7팀경기 내에 선발로 등판한 적이 있으면 선발군
        """
        if team_game_no <= EARLY_BULLPEN_TEAM_GAMES:
            return pit_season_gs.get((p_no, year - 1), 0) >= PREV_SEASON_GS_THRESHOLD

        recent_sp_set = set(team_recent_starters[(team, year)])
        return p_no in recent_sp_set

    def pitcher_fatigue_score(p_no: int, date_str: str) -> int:
        """
        투수의 최근 5일간 피로도 점수를 계산합니다.

        D-1~D-5일의 투구 수에 가중치(5,4,3,2,1)를 곱한 합입니다.
        최근에 많이 던졌을수록 높은 점수 → 피로 누적을 나타냅니다.
        """
        d0 = yyyymmdd_to_dt(date_str)
        score = 0
        for lag in range(1, 6):
            w = 6 - lag  # D-1:5, D-2:4, ..., D-5:1
            dp = dt_to_yyyymmdd(d0 - timedelta(days=lag))
            score += w * pitcher_np_by_date.get((p_no, dp), 0)
        return score

    def select_core_bullpen(team: int, year: int, team_game_no: int, today_starter_p_no: int):
        """
        팀의 핵심 불펜 4인을 선정합니다.

        올/작년에 등판한 투수 풀에서 선발군을 제외하고,
        세이브+홀드 합계가 높은 순으로 상위 4명을 뽑습니다.

        Returns:
            핵심 불펜 4인의 p_no 리스트
        """
        candidates = set()
        candidates.update(team_pitchers_by_year[(team, year - 1)])
        candidates.update(team_pitchers_by_year[(team, year)])

        # 오늘 선발투수는 불펜 풀에서 제외
        if today_starter_p_no:
            candidates.discard(today_starter_p_no)

        if not candidates:
            return []

        bullpen_pool = [p for p in candidates if not is_starter_group(team, p, year, team_game_no)]

        # 시즌 초반에는 작년 → 올해 순, 이후에는 올해 → 작년 순으로 정렬
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
        """핵심 불펜 4인의 피로도 점수 합계를 반환합니다."""
        core4 = select_core_bullpen(team, year, team_game_no, today_starter_p_no)
        return sum(pitcher_fatigue_score(p_no, date_str) for p_no in core4)


    # ── 출력 CSV 컬럼 정의 ──
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

        # ═══════════════════════════════════════════════
        # (1) 피처 추출 단계
        #     D일 경기의 피처를 D-1일까지의 누적 데이터로 계산합니다.
        #     아직 오늘 기록은 업데이트하지 않았으므로 미래 정보가 섞이지 않습니다.
        # ═══════════════════════════════════════════════
        for g in todays_games:
            s_no = safe_int(g.get("s_no"))
            if not s_no:
                continue

            home = safe_int(g.get("homeTeam"))
            away = safe_int(g.get("awayTeam"))
            s_code = safe_int(g.get("s_code"))
            hs = safe_int(g.get("homeScore"))
            aw = safe_int(g.get("awayScore"))
            y = 1 if hs > aw else 0  # 정답 레이블: 홈팀 승리 여부

            home_game_no = team_game_cnt[(home, year)] + 1 if home else 9999
            away_game_no = team_game_cnt[(away, year)] + 1 if away else 9999

            def lineup_sums(side: str):
                """해당 팀(home/away)의 타순 1~9번 OPS_smooth 합과 최근5경기 OPS 합을 계산."""
                batters = lineup_map[s_no][side]["batters"]
                sum_smooth = 0.0
                sum_recent = 0.0
                for order in range(1, 10):
                    p_no = safe_int(batters.get(order))
                    if not p_no:
                        # 라인업 정보가 없으면 리그 평균으로 대체
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

            # 선발투수 피안타 OPS
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

            # 불펜 피로도
            home_fat = team_core_bullpen_fatigue(home, year, home_game_no, date, home_sp) if home else 0
            away_fat = team_core_bullpen_fatigue(away, year, away_game_no, date, away_sp) if away else 0

            # 2024년 이후 경기만 출력 CSV에 포함
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

        # ═══════════════════════════════════════════════
        # (2) 팀 경기 수 업데이트 (피처 추출 이후에 수행)
        #     순서가 중요합니다! 먼저 피처를 뽑고, 그 다음 오늘 경기를 카운트합니다.
        # ═══════════════════════════════════════════════
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

        # ═══════════════════════════════════════════════
        # (3) 타자 시즌 누적 통계 업데이트 + 최근 5경기 윈도우 갱신
        # ═══════════════════════════════════════════════
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

            # 시즌 누적 통계에 오늘 기록 더하기
            bat_cum[(p_no, y)]["AB"] += AB
            bat_cum[(p_no, y)]["H"]  += H
            bat_cum[(p_no, y)]["BB"] += BB
            bat_cum[(p_no, y)]["HP"] += HP
            bat_cum[(p_no, y)]["SF"] += SF
            bat_cum[(p_no, y)]["TB"] += TB

            # 리그 전체 합계에도 더하기 (리그 평균 OPS 계산용)
            league_tot[y]["AB"] += AB
            league_tot[y]["H"]  += H
            league_tot[y]["BB"] += BB
            league_tot[y]["HP"] += HP
            league_tot[y]["SF"] += SF
            league_tot[y]["TB"] += TB

            # 최근 5경기 윈도우에 추가 (오래된 건 자동으로 빠짐)
            bat_recent[(p_no, y)].append({
                "AB": AB, "H": H, "BB": BB, "HP": HP, "SF": SF, "TB": TB
            })

        # ═══════════════════════════════════════════════
        # (4) 투수 시즌 누적 통계 업데이트 + 피로도/선발/불펜 정보 갱신
        # ═══════════════════════════════════════════════
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

            # 투수 시즌 누적 통계(피안타 OPS 계산용)
            pit_cum[(p_no, y)]["AB"] += AB
            pit_cum[(p_no, y)]["H"]  += H
            pit_cum[(p_no, y)]["BB"] += BB
            pit_cum[(p_no, y)]["HP"] += HP
            pit_cum[(p_no, y)]["SF"] += SF
            pit_cum[(p_no, y)]["TB"] += TB
            pit_cum[(p_no, y)]["BF"] += BF

            # 선발/불펜 분류 및 피로도 계산에 필요한 정보
            pit_season_gs[(p_no, y)] += GS
            pitcher_svhld_season[(p_no, y)] += SVHLD
            pitcher_np_by_date[(p_no, date)] += NP

            if t_code:
                team_pitchers_by_year[(t_code, y)].add(p_no)

    # ── 결과 CSV 저장 ──
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print(f"[OK] wrote: {OUT_CSV} rows={len(rows_out)} (year>={MIN_FEATURE_YEAR})")

if __name__ == "__main__":
    main()
