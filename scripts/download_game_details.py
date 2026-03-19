"""
경기별 라인업(gameLineup) + 박스스코어(gameBoxscore) 다운로더

game_index_played.csv에 기록된 경기들의 s_no(경기 고유번호)를 읽어서,
각 경기별로 라인업과 박스스코어 API를 호출하여 raw JSON을 저장합니다.

[파이프라인 위치]
  3단계 — build_game_index.py (→ game_index_played.csv 필터링) 이후에 실행합니다.
  이 스크립트가 저장한 raw JSON을 build_lineup_table.py 등에서 활용합니다.

[입력]
  - ~/statiz/data/game_index_played.csv (s_no 목록)
  - STATIZ API (환경변수 STATIZ_API_KEY, STATIZ_SECRET 필요)

[출력]
  - ~/statiz/data/raw_lineup/{s_no}.json   — 라인업 정보
  - ~/statiz/data/raw_boxscore/{s_no}.json — 박스스코어 정보

[참고]
  - 이미 다운받은 경기는 자동 스킵합니다.
  - --limit 옵션으로 다운로드 건수를 제한할 수 있습니다 (기본 20건, 테스트용).
  - API 실패 시 최대 3회 재시도합니다.
"""

import os, csv, json, time, hmac, hashlib, urllib.parse, argparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# ──────────────────────────────────────────────
# API 인증 정보
# ──────────────────────────────────────────────
API_KEY = os.environ["STATIZ_API_KEY"]
SECRET  = os.environ["STATIZ_SECRET"].encode("utf-8")

BASE = "https://api.statiz.co.kr/baseballApi"
INDEX_CSV = os.path.expanduser("~/statiz/data/game_index_played.csv")

# 저장 폴더
OUT_LINEUP = os.path.expanduser("~/statiz/data/raw_lineup")
OUT_BOXS   = os.path.expanduser("~/statiz/data/raw_boxscore")
os.makedirs(OUT_LINEUP, exist_ok=True)
os.makedirs(OUT_BOXS, exist_ok=True)


def signed_get(path: str, params: dict, timeout=20):
    """
    STATIZ API에 서명(HMAC-SHA256) 인증이 포함된 GET 요청을 보냅니다.

    download_schedule.py의 signed_get과 같은 방식이지만,
    여기서는 path(엔드포인트)도 인자로 받아서 여러 API를 호출할 수 있습니다.

    Args:
        path: API 엔드포인트 (예: "prediction/gameLineup")
        params: 쿼리 파라미터 (예: {"s_no": "20240003"})
        timeout: 요청 제한시간 (초)

    Returns:
        (HTTP 상태코드, 응답 본문 문자열) 튜플
    """
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
        body = resp.read().decode("utf-8", errors="replace")
        return resp.status, body


def read_snos(limit=None):
    """
    game_index_played.csv에서 경기 번호(s_no) 목록을 읽어옵니다.

    Args:
        limit: 읽을 최대 건수 (None이면 전체)

    Returns:
        s_no 정수 리스트
    """
    snos = []
    with open(INDEX_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            snos.append(int(row["s_no"]))
            if limit and len(snos) >= limit:
                break
    return snos


def save_json(out_dir, s_no, body):
    """API 응답 본문(JSON 문자열)을 파일로 저장합니다."""
    fp = os.path.join(out_dir, f"{s_no}.json")
    with open(fp, "w", encoding="utf-8") as f:
        f.write(body)


def is_ok(body: str):
    """
    API 응답이 정상인지 확인합니다.

    STATIZ API는 HTTP 200을 반환하더라도 result_cd 필드로 실패를 알리는 경우가 있습니다.
    result_cd == 100이면 정상, 없으면 일단 통과(원본 저장이 목적이므로).
    """
    try:
        j = json.loads(body)
        if isinstance(j, dict) and "result_cd" in j:
            return j.get("result_cd") == 100
    except Exception:
        pass
    return True


def fetch_one(s_no, sleep_sec=0.15, retries=3):
    """
    한 경기의 라인업과 박스스코어 raw JSON을 각각 다운로드합니다.

    두 API(gameLineup, gameBoxscore)를 순서대로 호출하며,
    이미 파일이 존재하는 건은 건너뜁니다.

    Args:
        s_no: 경기 고유번호
        sleep_sec: 호출 후 대기 시간 (API 부하 방지)
        retries: 최대 재시도 횟수
    """
    # 호출할 API 목록: (엔드포인트, 파라미터, 저장 폴더)
    tasks = [
        ("prediction/gameLineup", {"s_no": str(s_no)}, OUT_LINEUP),
        ("prediction/gameBoxscore", {"s_no": str(s_no)}, OUT_BOXS),
    ]

    for path, params, out_dir in tasks:
        out_fp = os.path.join(out_dir, f"{s_no}.json")

        # 이미 다운로드된 파일이 있으면 건너뛰기
        if os.path.exists(out_fp):
            continue

        last_err = None
        for attempt in range(1, retries + 1):
            try:
                status, body = signed_get(path, params)
                save_json(out_dir, s_no, body)

                ok = (status == 200 and is_ok(body))
                if not ok:
                    print("FAIL", path, "s_no=", s_no, "status=", status)
                time.sleep(sleep_sec)
                break
            except (HTTPError, URLError, TimeoutError) as e:
                last_err = e
                time.sleep(0.5 * attempt)
        else:
            # 모든 재시도가 실패한 경우
            print("ERROR", path, "s_no=", s_no, repr(last_err))


def main():
    """
    경기 목록을 읽어 라인업/박스스코어 raw JSON을 일괄 다운로드합니다.

    사용법:
      python download_game_details.py --limit 20    # 처음 20건만 (테스트용)
      python download_game_details.py --limit 0     # 전체 다운로드
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20, help="처음에는 20으로 테스트 추천")
    args = ap.parse_args()

    snos = read_snos(limit=args.limit)
    print("targets:", len(snos))

    for i, s_no in enumerate(snos, 1):
        fetch_one(s_no)
        if i % 10 == 0:
            print("progress", i, "/", len(snos))

    print("DONE")

if __name__ == "__main__":
    main()
