"""
선수별 일별 기록(playerDay) 다운로더

player_year_index.csv에서 (선수번호, 연도) 조합 목록을 읽어,
각 조합별로 STATIZ playerDay API를 호출하여 raw JSON을 저장합니다.

playerDay는 해당 선수의 해당 연도 경기별 세부 기록(타격/투구 성적)을 담고 있어서,
피처 생성(OPS, 피안타율 등)의 핵심 원천 데이터가 됩니다.

[파이프라인 위치]
  6단계 — build_player_year_index.py 이후에 실행합니다.

[입력]
  - ~/statiz/data/player_year_index.csv  (선수-연도 목록)
  - STATIZ API (환경변수 STATIZ_API_KEY, STATIZ_SECRET 필요)

[출력]
  - ~/statiz/data/raw_playerday/{p_no}_{year}.json (선수-연도별 경기 기록)

[참고]
  - 이미 다운받은 파일은 자동 스킵합니다.
  - --limit 옵션으로 건수 제한 가능 (기본 20건, 테스트용).
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
PATH = "prediction/playerDay"

# 입력/출력 경로
INDEX_CSV = os.path.expanduser("~/statiz/data/player_year_index.csv")
OUT_DIR   = os.path.expanduser("~/statiz/data/raw_playerday")
os.makedirs(OUT_DIR, exist_ok=True)


def signed_get(params: dict, timeout=25):
    """
    playerDay API에 서명(HMAC-SHA256) 인증이 포함된 GET 요청을 보냅니다.

    Args:
        params: 쿼리 파라미터 (예: {"p_no": "12345", "year": "2024"})
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
    payload = f"{method}|{PATH}|{normalized_query}|{ts}"
    sig = hmac.new(SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    url = f"{BASE}/{PATH}?{normalized_query}"
    req = Request(url, method=method, headers={
        "X-API-KEY": API_KEY,
        "X-TIMESTAMP": ts,
        "X-SIGNATURE": sig,
    })
    with urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", errors="replace")


def is_ok(body: str):
    """
    API 응답이 정상인지 result_cd로 확인합니다.
    result_cd == 100이면 정상.
    result_cd 필드가 없으면 일단 통과 (원본 저장이 목적이므로).
    """
    try:
        j = json.loads(body)
        if isinstance(j, dict) and "result_cd" in j:
            return j.get("result_cd") == 100
    except Exception:
        pass
    return True


def load_pairs(limit=None):
    """
    player_year_index.csv에서 (선수번호, 연도) 쌍 목록을 읽어옵니다.

    Args:
        limit: 읽을 최대 건수 (None이면 전체)

    Returns:
        [(p_no, year), ...] 형태의 정수 튜플 리스트
    """
    pairs = []
    with open(INDEX_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pairs.append((int(row["p_no"]), int(row["year"])))
            if limit and len(pairs) >= limit:
                break
    return pairs


def out_path(p_no, year):
    """저장할 파일 경로를 생성합니다. (예: ~/statiz/data/raw_playerday/12345_2024.json)"""
    return os.path.join(OUT_DIR, f"{p_no}_{year}.json")


def main():
    """
    선수-연도 목록을 순회하며 playerDay raw JSON을 일괄 다운로드합니다.

    사용법:
      python download_playerday.py --limit 20    # 처음 20건 (테스트용)
      python download_playerday.py --limit 0     # 전체 다운로드
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20, help="처음엔 20으로 테스트 추천")
    args = ap.parse_args()

    pairs = load_pairs(limit=args.limit if args.limit > 0 else None)
    print("targets:", len(pairs))

    ok = 0
    fail = 0

    for i, (p_no, year) in enumerate(pairs, 1):
        fp = out_path(p_no, year)

        # 이미 다운로드된 파일이 있으면 건너뛰기
        if os.path.exists(fp):
            continue

        params = {"p_no": str(p_no), "year": str(year)}
        last_err = None

        # 최대 3회 재시도
        for attempt in range(1, 4):
            try:
                status, body = signed_get(params)
                with open(fp, "w", encoding="utf-8") as f:
                    f.write(body)

                if status == 200 and is_ok(body):
                    ok += 1
                else:
                    fail += 1
                    print("FAIL", p_no, year, "status=", status)
                break

            except (HTTPError, URLError, TimeoutError) as e:
                last_err = e
                time.sleep(0.5 * attempt)
        else:
            # 모든 재시도 실패
            fail += 1
            print("ERROR", p_no, year, repr(last_err))

        # API 부하 방지용 대기
        time.sleep(0.12)

        # 50건마다 진행 상황 출력
        if i % 50 == 0:
            print("progress", i, "/", len(pairs), "ok=", ok, "fail=", fail)

    print("DONE", "ok=", ok, "fail=", fail, "dir=", OUT_DIR)

if __name__ == "__main__":
    main()
