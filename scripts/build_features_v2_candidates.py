"""
v2 후보 피처 생성기 — 12개 피처 세트

v1의 핵심 4피처에 추가로, 피처 subset 탐색을 위한 후보 피처들을 생성합니다.
v1의 모든 피처를 포함하면서 8개의 새로운 피처를 추가하여 총 12개의 diff 피처를 만듭니다.

[생성되는 12가지 diff 피처]
  ── v1 그대로 ──
  1. diff_sum_ops_smooth        — 타순 1~9번 OPS_smooth 합의 홈-원정 차이
  2. diff_sum_ops_recent5       — 최근 5경기 OPS 합의 차이
  3. diff_sp_oops               — 선발투수 피안타 OPS 차이
  4. diff_bullpen_fatigue       — 핵심 불펜 피로도 차이

  ── v2 추가 ──
  5. diff_top5_ops_smooth       — 상위 5번 타순(1~5번) OPS_smooth 합의 차이
  6. diff_sp_bbip               — 선발투수 BB/IP(이닝당 볼넷)의 차이
  7. diff_opp_sp_platoon_cnt    — 상대 선발 대비 유리한 타석(플래툰 이점) 수 차이
  8. diff_pythag_winpct         — 피타고리안 기대 승률의 차이
  9. diff_recent10_winpct       — 최근 10경기 승률의 차이
  10. diff_team_stadium_winpct  — 해당 구장에서의 팀 승률 차이
  11. park_factor_stadium       — 구장 파크 팩터 (타자 친화도)
  12. diff_team_stadium_winpct_pfadj — 구장 승률 × 파크 팩터 보정값

[핵심 로직]
  - v1과 동일하게 D일 피처는 D-1일까지의 데이터만 사용합니다.
  - 모든 smoothing에는 베이즈 평활 공식을 사용합니다.
  - 구간 누적(2023~)을 통해 시계열 정보를 점진적으로 반영합니다.

[파이프라인 위치]
  8b단계 — build_playerday_tables_v2.py 이후에 실행합니다.

[입력]
  - ~/statiz/data/game_index_played.csv
  - ~/statiz/data/lineup_long.csv
  - ~/statiz/data/playerday_batter_long.csv
  - ~/statiz/data/playerday_pitcher_long.csv

[출력]
  - ~/statiz/data/features_v2_candidates.csv (2024년 이후 경기만 포함)
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

# ──────────────────────────────────────────────
# 연도 범위 설정
# ──────────────────────────────────────────────
MIN_INTERNAL_YEAR = 2023   # 내부 누적 통계는 2023년부터 시작
MIN_FEATURE_YEAR = 2024    # 출력 CSV에는 2024년 이후만 포함

# ──────────────────────────────────────────────
# v1 호환 OPS / 선발-불펜 구분 파라미터
# ──────────────────────────────────────────────
K_SMOOTH = 20.0                     # OPS 베이즈 평활 강도
MIN_PA_LASTSEASON = 60              # 직전 시즌 OPS를 prior로 쓸 최소 PA
MIN_PA_RECENT = 10                  # 최근 5경기 OPS 사용 최소 PA
RECENT_GAMES = 5                    # 최근 경기 윈도우 크기

EARLY_BULLPEN_TEAM_GAMES = 20       # 시즌 초반(1~20경기): 직전 시즌 기준 선발/불펜 구분
RECENT_TEAM_GAMES_FOR_SP = 7        # 시즌 중반 이후: 최근 7경기 내 선발 이력으로 구분
PREV_SEASON_GS_THRESHOLD = 5       # 직전 시즌 선발 등판 ≥ 5 → 선발군

# ──────────────────────────────────────────────
# v2 추가 피처 전용 파라미터
# ──────────────────────────────────────────────
TOP_ORDER_N = 5                     # 상위 N번 타순만 따로 집계 (1~5번)

K_BBIP_IP = 30.0                    # BB/IP 베이즈 평활 강도
MIN_IP_LASTSEASON = 30.0            # 직전 시즌 BB/IP를 prior로 쓸 최소 이닝
FALLBACK_PRIOR_BBIP = 0.35         # BB/IP 기본값

PYTHAG_EXP = 1.83                   # 피타고리안 승률 지수
K_PYTHAG_GAMES = 20.0               # 피타고리안 승률 평활 강도
MIN_PYTHAG_PRIOR_GAMES = 30        # 직전 시즌 피타고리안 prior 사용 최소 경기 수

K_RECENT10_GAMES = 10.0             # 최근 10경기 승률 평활 강도

K_STADIUM_WIN_GAMES = 10.0          # 구장별 승률 평활 강도
MIN_STADIUM_PRIOR_GAMES = 5        # 직전 시즌 구장별 승률 prior 사용 최소 경기 수

K_PF_GAMES = 30.0                   # 파크 팩터 평활 강도
MIN_PF_PRIOR_GAMES = 30            # 직전 시즌 파크 팩터 prior 사용 최소 경기 수

FALLBACK_PRIOR_OPS = 0.700         # 리그 평균 OPS 기본값


# ══════════════════════════════════════════════
# 유틸리티 함수
# ══════════════════════════════════════════════
def safe_int(x, default=0):
    """문자열이나 None을 안전하게 정수로 변환합니다."""
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
    """문자열이나 None을 안전하게 실수로 변환합니다."""
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
    """YYYYMMDD 문자열 → datetime 변환"""
    return datetime.strptime(s, "%Y%m%d")


def dt_to_yyyymmdd(d):
    """datetime → YYYYMMDD 문자열 변환"""
    return d.strftime("%Y%m%d")


def calc_ops_from_counts(H, BB, HP, AB, SF, TB):
    """개수 기반 OPS(출루율+장타율)와 PA(타석 수)를 계산합니다."""
    denom_obp = AB + BB + HP + SF
    obp = (H + BB + HP) / denom_obp if denom_obp else 0.0
    slg = TB / AB if AB else 0.0
    return obp + slg, denom_obp


def smooth_value(curr_val, curr_w, prior_val, k):
    """베이즈 평활 공식: 데이터가 적으면 prior 쪽으로, 충분하면 관측값 쪽으로 끌림."""
    if curr_w < 0:
        curr_w = 0
    return (curr_w / (curr_w + k)) * curr_val + (k / (curr_w + k)) * prior_val


def ip_to_outs(ip_val):
    """
    KBO 이닝 표기(예: 6.1, 6.2)를 아웃카운트로 변환합니다.

    KBO 규칙:
      6.0 → 18아웃 (6이닝 완료)
      6.1 → 19아웃 (6이닝 + 1아웃)
      6.2 → 20아웃 (6이닝 + 2아웃)
    """
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
    """
    피타고리안 기대 승률을 계산합니다.

    공식: 득점^exp / (득점^exp + 실점^exp)
    팀의 득점/실점 비율로부터 "기대되는" 승률을 추정합니다.
    """
    rs = max(0.0, float(rs))
    ra = max(0.0, float(ra))
    if rs == 0 and ra == 0:
        return 0.5
    rs_e = rs ** exp
    ra_e = ra ** exp
    denom = rs_e + ra_e
    return (rs_e / denom) if denom > 0 else 0.5


def result_point(score_for, score_against):
    """경기 결과를 점수로 변환합니다: 승=1.0, 패=0.0, 무=0.5"""
    if score_for > score_against:
        return 1.0
    if score_for < score_against:
        return 0.0
    return 0.5


def batter_hand_from_p_bat(p_bat):
    """
    API의 타석 코드를 R/L/S 문자로 변환합니다.
    1=우타(R), 2=좌타(L), 3=스위치(S)
    """
    if p_bat == 1:
        return "R"
    if p_bat == 2:
        return "L"
    if p_bat == 3:
        return "S"
    return None


def pitcher_hand_from_p_throw(p_throw):
    """
    API의 투구 손 코드를 R/L 문자로 변환합니다.
    1,2=우투(R), 3,4=좌투(L)
    """
    if p_throw in (1, 2):
        return "R"
    if p_throw in (3, 4):
        return "L"
    return None


def get_save_from_row(row):
    """CSV 행에서 세이브(SV) 값을 찾아 반환합니다."""
    for key in ("SV", "sv", "Save", "save", "S"):
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
    """세이브 + 홀드 합계를 반환합니다."""
    return get_save_from_row(row) + get_hold_from_row(row)


# ══════════════════════════════════════════════
# 데이터 로더
# ══════════════════════════════════════════════
def load_games():
    """game_index_played.csv에서 2023년 이후 경기 목록을 정렬하여 반환합니다."""
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
    """
    경기별·팀별 선발 라인업 정보를 구조화합니다.
    v1과 달리 타석(p_bat)과 투구 손(p_throw) 정보도 함께 저장합니다 (플래툰 피처용).

    Returns:
        lineup_map[s_no][side]["P"]            → {"p_no", "p_bat", "p_throw"}
        lineup_map[s_no][side]["batters"][1~9]  → {"p_no", "p_bat", "p_throw"}
    """
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
                "p_bat": safe_int(row.get("p_bat")),      # 타석 정보
                "p_throw": safe_int(row.get("p_throw")),   # 투구 손 정보
            }

            if bo == "P":
                lineup_map[s_no][side]["P"] = info
            else:
                order = safe_int(bo, 0)
                if 1 <= order <= 9:
                    lineup_map[s_no][side]["batters"][order] = info
    return lineup_map


def group_by_date(csv_path):
    """CSV를 날짜별로 그룹핑합니다. D일 피처 → D일 업데이트 패턴에 사용."""
    by_date = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            d = row.get("date")
            if d:
                by_date[d].append(row)
    return by_date


# ══════════════════════════════════════════════
# 메인 로직
# ══════════════════════════════════════════════
def main():
    games = load_games()
    lineup_map = load_lineup_map()
    bat_by_date = group_by_date(BAT_CSV)
    pit_by_date = group_by_date(PIT_CSV)

    games_by_date = defaultdict(list)
    for g in games:
        games_by_date[g["date"]].append(g)
    dates = sorted(games_by_date.keys())

    # ── 누적 통계 저장소 ──
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
            "OUTS": 0,   # v2 추가: BB/IP 계산에 필요
        }
    )

    # 리그 전체 합계 (prior 계산용)
    league_bat_tot = defaultdict(lambda: {"AB": 0, "H": 0, "BB": 0, "HP": 0, "SF": 0, "TB": 0})
    league_pit_tot = defaultdict(lambda: {"BB": 0, "OUTS": 0})       # BB/IP 리그 평균용
    league_game_runs = defaultdict(lambda: {"R": 0, "G": 0})         # 파크 팩터 계산용

    # 최근 윈도우
    bat_recent = defaultdict(lambda: deque(maxlen=RECENT_GAMES))     # 타자 최근 5경기
    team_recent_results = defaultdict(lambda: deque(maxlen=10))      # 팀 최근 10경기 결과

    # 불펜 컨텍스트
    pit_season_gs = defaultdict(int)                                 # (p_no, year) → 선발 등판 수
    pitcher_svhld_season = defaultdict(int)                          # (p_no, year) → SV+HD 합계
    pitcher_np_by_date = defaultdict(int)                            # (p_no, date) → 투구 수
    team_game_cnt = defaultdict(int)                                 # (team, year) → 경기 수
    team_recent_starters = defaultdict(lambda: deque(maxlen=RECENT_TEAM_GAMES_FOR_SP))
    team_pitchers_by_year = defaultdict(set)                         # (team, year) → 등판 투수 집합

    # 팀 레벨 스코어/승률 컨텍스트 (v2 추가)
    team_runs = defaultdict(lambda: {"RS": 0, "RA": 0, "G": 0})            # 피타고리안용
    team_wins = defaultdict(lambda: {"WPTS": 0.0, "G": 0})                 # 시즌 누적 승률
    team_stadium_wins = defaultdict(lambda: {"WPTS": 0.0, "G": 0})         # 구장별 승률
    stadium_runs = defaultdict(lambda: {"R": 0, "G": 0})                   # 구장별 득점 (파크 팩터용)


    # ── 내부 계산 함수들 ──

    def league_ops(year):
        """해당 연도 리그 평균 OPS를 계산합니다."""
        tot = league_bat_tot.get(year)
        if not tot:
            return FALLBACK_PRIOR_OPS
        val, _ = calc_ops_from_counts(
            tot["H"], tot["BB"], tot["HP"], tot["AB"], tot["SF"], tot["TB"]
        )
        return val if val > 0 else FALLBACK_PRIOR_OPS

    def league_bbip(year):
        """해당 연도 리그 평균 BB/IP(이닝당 볼넷 수)를 계산합니다."""
        tot = league_pit_tot.get(year)
        if not tot:
            return FALLBACK_PRIOR_BBIP
        outs = tot["OUTS"]
        if outs <= 0:
            return FALLBACK_PRIOR_BBIP
        ip = outs / 3.0
        return (tot["BB"] / ip) if ip > 0 else FALLBACK_PRIOR_BBIP

    def team_prev_winpct(team, year):
        """직전 시즌 팀 승률을 반환합니다. 없으면 0.5(균등) 반환."""
        prev = team_wins.get((team, year - 1))
        if not prev or prev["G"] <= 0:
            return 0.5
        return prev["WPTS"] / prev["G"]

    def league_rpg(year):
        """리그 경기당 평균 득점을 계산합니다. 파크 팩터 산출에 사용."""
        rec = league_game_runs.get(year)
        if not rec or rec["G"] <= 0:
            return None
        return rec["R"] / rec["G"]

    def batter_ops_smooth(p_no, year):
        """타자 시즌 누적 OPS에 베이즈 평활 적용. (v1과 동일)"""
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
        """최근 5경기 OPS (PA ≥ 10이면 사용, 아니면 smooth 값 반환)."""
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
        """투수 피안타 OPS에 베이즈 평활 적용. (v1과 동일)"""
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
        """
        투수의 BB/IP(이닝당 볼넷)에 베이즈 평활을 적용합니다.
        제구력 지표로, 낮을수록 제구가 좋은 투수입니다.

        Returns:
            (smoothed BB/IP, 현재 시즌 이닝 수) 튜플
        """
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
        """팀 피타고리안 기대 승률에 베이즈 평활 적용. prior는 직전 시즌 피타고리안 승률."""
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
        """
        최근 10경기 승률에 베이즈 평활 적용.
        최근 폼(momentum)을 반영하는 피처입니다.
        """
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
        """
        특정 구장에서의 팀 승률에 베이즈 평활 적용.
        홈 어드밴티지나 구장 적성을 반영합니다.
        """
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
        """
        구장 파크 팩터를 계산합니다.

        파크 팩터 = (이 구장의 경기당 평균 득점) / (리그 경기당 평균 득점)
        1.0 이상이면 타자 친화 구장, 미만이면 투수 친화 구장입니다.
        """
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
        """해당 투수가 선발군인지 불펜군인지 판별합니다."""
        if team_game_no <= EARLY_BULLPEN_TEAM_GAMES:
            return pit_season_gs.get((p_no, year - 1), 0) >= PREV_SEASON_GS_THRESHOLD
        recent_sp_set = set(team_recent_starters[(team, year)])
        return p_no in recent_sp_set

    def pitcher_fatigue_score(p_no, date_str):
        """D-1~D-5일 투구 수의 가중합으로 피로도를 계산합니다."""
        d0 = yyyymmdd_to_dt(date_str)
        score = 0
        for lag in range(1, 6):
            w = 6 - lag
            dp = dt_to_yyyymmdd(d0 - timedelta(days=lag))
            score += w * pitcher_np_by_date.get((p_no, dp), 0)
        return score

    def select_core_bullpen(team, year, team_game_no, today_starter_p_no):
        """SV+HD 기반으로 핵심 불펜 4인을 선정합니다."""
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
        """핵심 불펜 4인의 피로도 합계를 반환합니다."""
        core4 = select_core_bullpen(team, year, team_game_no, today_starter_p_no)
        return sum(pitcher_fatigue_score(p_no, date_str) for p_no in core4)

    def count_platoon_advantage(batters, opp_sp_throw_code):
        """
        상대 선발투수 대비 플래툰 이점(유리한 타석)의 수를 셉니다.

        플래툰 이점이란:
          - 좌투 vs 우타 → 우타에게 유리
          - 우투 vs 좌타 → 좌타에게 유리
          - 스위치 히터는 항상 유리

        Returns:
            1~9번 타순 중 플래툰 이점을 가진 타자의 수 (0~9)
        """
        p_hand = pitcher_hand_from_p_throw(opp_sp_throw_code)
        cnt = 0
        for order in range(1, 10):
            info = batters.get(order)
            if not info:
                continue
            b_hand = batter_hand_from_p_bat(safe_int(info.get("p_bat")))
            if b_hand == "S":
                cnt += 1  # 스위치히터는 항상 유리
            elif p_hand == "L" and b_hand == "R":
                cnt += 1  # 좌투 vs 우타
            elif p_hand == "R" and b_hand == "L":
                cnt += 1  # 우투 vs 좌타
        return cnt


    # ── 출력 CSV 컬럼 정의 (v1 4개 + v2 추가 8개 = 12개 diff 피처) ──
    fieldnames = [
        "date",
        "s_no",
        "s_code",
        "homeTeam",
        "awayTeam",
        "y_home_win",
        "homeScore",
        "awayScore",
        # v1 피처들
        "home_sum_ops_smooth",
        "away_sum_ops_smooth",
        "diff_sum_ops_smooth",
        "home_sum_ops_recent5",
        "away_sum_ops_recent5",
        "diff_sum_ops_recent5",
        # v2 추가: 상위 5번 타순
        "home_top5_ops_smooth",
        "away_top5_ops_smooth",
        "diff_top5_ops_smooth",
        # 선발투수 피안타 OPS
        "home_sp_oops",
        "away_sp_oops",
        "diff_sp_oops",
        # v2 추가: 선발투수 BB/IP
        "home_sp_bbip",
        "away_sp_bbip",
        "diff_sp_bbip",
        # v2 추가: 플래툰 이점
        "home_platoon_cnt_vs_opp_sp",
        "away_platoon_cnt_vs_opp_sp",
        "diff_opp_sp_platoon_cnt",
        # 불펜 피로도
        "home_bullpen_fatigue",
        "away_bullpen_fatigue",
        "diff_bullpen_fatigue",
        # v2 추가: 피타고리안 승률
        "home_pythag_winpct",
        "away_pythag_winpct",
        "diff_pythag_winpct",
        # v2 추가: 최근 10경기 승률
        "home_recent10_winpct",
        "away_recent10_winpct",
        "diff_recent10_winpct",
        # v2 추가: 구장별 승률
        "home_stadium_winpct",
        "away_stadium_winpct",
        "diff_team_stadium_winpct",
        # v2 추가: 파크 팩터 + 보정값
        "park_factor_stadium",
        "diff_team_stadium_winpct_pfadj",
        # 메타 정보
        "home_sp_p_no",
        "away_sp_p_no",
    ]

    rows_out = []

    for date in dates:
        todays_games = games_by_date[date]
        todays_games.sort(key=lambda x: safe_int(x.get("s_no")))
        year = safe_int(date[:4])

        # ═══════════════════════════════════════════════
        # (1) 피처 추출: D-1까지의 누적 데이터로 D일 경기 피처를 계산
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
            y = 1 if hs > aw else 0

            home_game_no = team_game_cnt[(home, year)] + 1 if home else 9999
            away_game_no = team_game_cnt[(away, year)] + 1 if away else 9999

            def lineup_sums(side):
                """홈/원정 타선의 OPS_smooth 합, 최근5경기 합, 상위5타순 합을 계산."""
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

            # 선발투수 정보
            home_sp_info = lineup_map[s_no]["home"]["P"] or {}
            away_sp_info = lineup_map[s_no]["away"]["P"] or {}

            home_sp = safe_int(home_sp_info.get("p_no"))
            away_sp = safe_int(away_sp_info.get("p_no"))
            home_sp_throw = safe_int(home_sp_info.get("p_throw"))
            away_sp_throw = safe_int(away_sp_info.get("p_throw"))

            # 선발투수 피안타 OPS + BB/IP
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

            # 플래툰 이점: 홈 타자 vs 원정 선발, 원정 타자 vs 홈 선발
            home_batters = lineup_map[s_no]["home"]["batters"]
            away_batters = lineup_map[s_no]["away"]["batters"]
            home_platoon = count_platoon_advantage(home_batters, away_sp_throw)
            away_platoon = count_platoon_advantage(away_batters, home_sp_throw)

            # 불펜 피로도
            home_fat = team_core_bullpen_fatigue(home, year, home_game_no, date, home_sp) if home else 0
            away_fat = team_core_bullpen_fatigue(away, year, away_game_no, date, away_sp) if away else 0

            # 팀 레벨 피처들
            home_pyth = team_pythag_smooth(home, year) if home else 0.5
            away_pyth = team_pythag_smooth(away, year) if away else 0.5

            home_recent10 = team_recent10_winpct_smooth(home, year) if home else 0.5
            away_recent10 = team_recent10_winpct_smooth(away, year) if away else 0.5

            home_stadium_wr = team_stadium_winpct_smooth(home, year, s_code) if home and s_code else 0.5
            away_stadium_wr = team_stadium_winpct_smooth(away, year, s_code) if away and s_code else 0.5
            diff_stadium_wr = home_stadium_wr - away_stadium_wr

            park_factor = stadium_park_factor_smooth(s_code, year) if s_code else 1.0
            diff_stadium_wr_pfadj = diff_stadium_wr * park_factor

            # 2024년 이후 경기만 CSV에 포함
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

        # ═══════════════════════════════════════════════
        # (2) 팀 경기 수 / 선발 이력 업데이트
        # ═══════════════════════════════════════════════
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

        # (2b) 팀 득점/실점, 승률, 구장별 승률, 파크 팩터 원천 업데이트
        for g in todays_games:
            home = safe_int(g.get("homeTeam"))
            away = safe_int(g.get("awayTeam"))
            s_code = safe_int(g.get("s_code"))
            hs = safe_int(g.get("homeScore"))
            aw = safe_int(g.get("awayScore"))

            # 홈팀 관점에서 득점/실점, 승점 업데이트
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

            # 원정팀 관점에서 업데이트
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

            # 구장별 총 득점 (파크 팩터 계산용)
            if s_code:
                stadium_runs[(s_code, year)]["R"] += (hs + aw)
                stadium_runs[(s_code, year)]["G"] += 1
            league_game_runs[year]["R"] += (hs + aw)
            league_game_runs[year]["G"] += 1

        # ═══════════════════════════════════════════════
        # (3) 타자 시즌 누적 통계 업데이트
        # ═══════════════════════════════════════════════
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

        # ═══════════════════════════════════════════════
        # (4) 투수 시즌 누적 통계 업데이트 (피안타 OPS, BB/IP, 피로도 등)
        # ═══════════════════════════════════════════════
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

    # ── 결과 CSV 저장 ──
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
