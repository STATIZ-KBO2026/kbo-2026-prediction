"""
playerDay raw JSON → 타자/투수 경기별 기록 long CSV 변환기 (v2)

download_playerday.py가 저장한 선수별 raw JSON을 읽어서,
타자와 투수를 자동으로 분류하고, 각각의 long 형태 CSV를 생성합니다.

[타자/투수 구분 방법]
  JSON 기록에 IP, TBF, WHIP, ERA 같은 투수 전용 키가 있으면 투수로 분류하고,
  없으면 타자로 분류합니다.

[파이프라인 위치]
  7단계 — download_playerday.py 이후에 실행합니다.
  이 스크립트의 출력은 피처 생성(build_features_v1_paper.py 등)의 핵심 입력입니다.

[입력]
  - ~/statiz/data/raw_playerday/{p_no}_{year}.json (선수별 경기 기록)
  - ~/statiz/data/game_index_played.csv (경기 메타정보: 날짜, 홈/원정팀)

[출력]
  - ~/statiz/data/playerday_batter_long.csv  — 타자 경기별 기록
  - ~/statiz/data/playerday_pitcher_long.csv — 투수 경기별 기록
"""

import os, csv, json, glob

# 입력/출력 경로
RAW_DIR   = os.path.expanduser("~/statiz/data/raw_playerday")
INDEX_CSV = os.path.expanduser("~/statiz/data/game_index_played.csv")

OUT_BAT   = os.path.expanduser("~/statiz/data/playerday_batter_long.csv")
OUT_PIT   = os.path.expanduser("~/statiz/data/playerday_pitcher_long.csv")

# ──────────────────────────────────────────────
# CSV 컬럼 정의
# ──────────────────────────────────────────────

# 타자 CSV 컬럼: 기본 메타 + 타격 성적
BATTER_COLS = [
    "date","s_no","p_no","year","homeTeam","awayTeam","side",
    "t_code","vs_tCode","awayTeam_rec","awayScore","homeScore","gameDate",
    "G","GS","PA","ePA","AB","R","H","1B","2B","3B","HR","TB","RBI","SB","CS","BB","HP","IB","SO","GDP","SH","SF",
    "AVG","OBP","SLG","OPS","NP","situation","battingOrder","position"
]

# 투수 CSV 컬럼: 기본 메타 + 투구 성적
PITCHER_COLS = [
    "date","s_no","p_no","year","homeTeam","awayTeam","side",
    "t_code","vs_tCode","awayTeam_rec","awayScore","homeScore","gameDate",
    "G","GS","IP","R","rRA","ER","TBF","AB","H","2B","3B","HR","BB","IB","HP","SO","NP","TB","SF",
    "W","L","CG","SHO","S","HD","BS","BH",
    "WHIP","AVG","OBP","SLG","OPS","ERA","situation"
]

# 투수인지 판별할 때 사용하는 키 목록
# 이 키들 중 하나라도 값이 있으면 투수 기록으로 분류
PITCHER_HINT_KEYS = {"IP","TBF","WHIP","ERA","W","L","S","HD","BS","SHO","CG"}


def load_game_index():
    """
    game_index_played.csv에서 경기별 메타정보를 읽어옵니다.

    Returns:
        {s_no: {"date": "YYYYMMDD", "homeTeam": int, "awayTeam": int}} 형태의 dict
    """
    idx = {}
    with open(INDEX_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            s_no = int(row["s_no"])
            idx[s_no] = {
                "date": row["date"],
                "homeTeam": int(row["homeTeam"]),
                "awayTeam": int(row["awayTeam"]),
            }
    return idx


def is_pitcher(rec: dict):
    """
    경기 기록(dict)이 투수 기록인지 판별합니다.
    PITCHER_HINT_KEYS 중 하나라도 값이 있으면 투수로 분류합니다.
    """
    return any(k in rec and rec.get(k) not in (None, "") for k in PITCHER_HINT_KEYS)


def pick(merged: dict, cols: list):
    """주어진 컬럼 목록에 해당하는 값만 추출하여 dict로 반환합니다."""
    return {c: merged.get(c, None) for c in cols}


def iter_records(data: dict):
    """
    playerDay API 응답에서 경기별 기록을 하나씩 꺼냅니다.

    API 응답 구조 예시:
      {
        "20250003": { ...타격/투구 기록... },  ← 8자리 숫자 키 = s_no(경기번호)
        "20250008": { ... },
        "result_cd": 100,                      ← 메타 필드 (건너뜀)
        "result_msg": "...",
      }

    Yields:
        (s_no, 기록 dict) 튜플
    """
    for k, v in data.items():
        # 메타 필드는 건너뛰기
        if k in ("result_cd","result_msg","update_time"):
            continue
        # 8자리 숫자 키만 경기 기록으로 인식
        if not (isinstance(k, str) and k.isdigit() and len(k) == 8):
            continue

        # 보통 v는 dict(한 경기 기록), 간혹 list인 경우도 처리
        if isinstance(v, dict):
            yield int(k), v
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    yield int(k), item


def main():
    """playerDay raw JSON들을 읽어 타자/투수 long CSV를 생성합니다."""
    idx = load_game_index()
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))

    bat_rows, pit_rows = [], []

    for fp in files:
        # 파일명에서 선수번호와 연도 추출 (예: 12345_2024.json)
        base = os.path.basename(fp).replace(".json","")
        try:
            p_no_file, year_file = base.split("_")
            p_no_file = int(p_no_file)
            year_file = int(year_file)
        except Exception:
            continue

        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 비정상 응답 스킵
        if not isinstance(data, dict) or data.get("result_cd") != 100:
            continue

        for s_no_key, rec in iter_records(data):
            s_no = s_no_key

            meta = idx.get(s_no)
            if not meta:
                # 정규시즌 경기 목록에 없는 경기면 스킵
                continue

            homeTeam = meta["homeTeam"]
            awayTeam = meta["awayTeam"]
            date = meta["date"]

            # 홈/원정 구분: 기록의 t_code와 경기 메타의 팀 코드를 비교
            side = "unknown"
            try:
                t_code_int = int(rec.get("t_code"))
                if t_code_int == homeTeam:
                    side = "home"
                elif t_code_int == awayTeam:
                    side = "away"
            except Exception:
                pass

            # 경기 메타정보와 선수 기록을 합침
            merged = {
                **rec,
                "date": date,
                "s_no": s_no,
                "p_no": int(rec.get("p_no") or p_no_file),
                "year": year_file,
                "homeTeam": homeTeam,
                "awayTeam": awayTeam,
                "side": side,
                "awayTeam_rec": rec.get("awayTeam"),
            }

            # 투수/타자 분류 후 해당 리스트에 추가
            if is_pitcher(rec):
                pit_rows.append(pick(merged, PITCHER_COLS))
            else:
                bat_rows.append(pick(merged, BATTER_COLS))

    # CSV 파일로 저장
    os.makedirs(os.path.dirname(OUT_BAT), exist_ok=True)

    with open(OUT_BAT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BATTER_COLS)
        w.writeheader()
        w.writerows(bat_rows)

    with open(OUT_PIT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PITCHER_COLS)
        w.writeheader()
        w.writerows(pit_rows)

    print("DONE", "batter_rows=", len(bat_rows), "pitcher_rows=", len(pit_rows))
    print("OUT", OUT_BAT)
    print("OUT", OUT_PIT)

if __name__ == "__main__":
    main()
