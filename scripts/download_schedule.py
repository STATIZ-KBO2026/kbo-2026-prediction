"""
STATIZ 경기 일정(gameSchedule) 다운로더

이 스크립트는 STATIZ 예측 API에서 날짜별 KBO 경기 일정을 수집합니다.
2022~2025년의 매일을 순회하며, 각 날짜의 경기 정보를 JSON 파일로 저장합니다.

[파이프라인 위치]
  1단계 - 가장 먼저 실행해야 하는 스크립트입니다.
  이 스크립트가 저장한 raw JSON을 build_game_index.py가 읽어서 경기 목록 CSV를 만듭니다.

[입력]
  - STATIZ API (환경변수 STATIZ_API_KEY, STATIZ_SECRET 필요)

[출력]
  - ~/statiz/data/raw_schedule/YYYYMMDD.json (날짜별 JSON 파일)

[참고]
  - 이미 다운받은 날짜는 자동 스킵하므로 중간에 끊겨도 안전하게 재실행할 수 있습니다.
  - API 실패 시 최대 3회까지 재시도하며, 재시도 간격이 점점 길어집니다(백오프).
"""

import os, json, time, hmac, hashlib, urllib.parse
from datetime import date, timedelta
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# ──────────────────────────────────────────────
# API 인증 정보 (환경변수에서 읽어옴)
# ──────────────────────────────────────────────
API_KEY = os.environ["STATIZ_API_KEY"]
SECRET  = os.environ["STATIZ_SECRET"].encode("utf-8")

# API 기본 URL과 엔드포인트
BASE = "https://api.statiz.co.kr/baseballApi"
PATH = "prediction/gameSchedule"   # ✅ 앞에 / 없이

# 저장 폴더 (없으면 자동 생성)
OUT_DIR = os.path.expanduser("~/statiz/data/raw_schedule")
os.makedirs(OUT_DIR, exist_ok=True)


def signed_get(params: dict, timeout=20):
    """
    STATIZ API 전용 서명(HMAC-SHA256) 인증이 포함된 GET 요청을 보냅니다.

    STATIZ API는 요청마다 다음 형태의 서명을 요구합니다:
      서명 = HMAC-SHA256(시크릿, "GET|엔드포인트|쿼리|타임스탬프")

    Args:
        params: API에 보낼 쿼리 파라미터 (예: {"year":"2024", "month":"5", "day":"3"})
        timeout: 요청 제한시간 (초)

    Returns:
        (HTTP 상태코드, 응답 본문 문자열) 튜플
    """
    method = "GET"

    # 쿼리 파라미터를 키 이름 순으로 정렬하여 문자열로 만듦
    normalized_query = "&".join(
        f"{urllib.parse.quote(k)}={urllib.parse.quote(str(params[k]))}"
        for k in sorted(params)
    )

    # 현재 시각(유닉스 타임스탬프)으로 서명 생성
    ts = str(int(time.time()))
    payload = f"{method}|{PATH}|{normalized_query}|{ts}"
    sig = hmac.new(SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    # HTTP 요청 전송
    url = f"{BASE}/{PATH}?{normalized_query}"
    req = Request(url, method=method, headers={
        "X-API-KEY": API_KEY,
        "X-TIMESTAMP": ts,
        "X-SIGNATURE": sig,
    })
    with urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", errors="replace")


def daterange(d0: date, d1: date):
    """
    시작일(d0)부터 종료일(d1)까지 하루씩 증가시키며 날짜를 반환합니다.

    예: daterange(date(2024,1,1), date(2024,1,3))
        → 2024-01-01, 2024-01-02, 2024-01-03
    """
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def fetch_one_day(d: date, sleep_sec=0.2, retries=3):
    """
    특정 날짜의 경기 일정 JSON을 다운로드하여 파일로 저장합니다.

    - 이미 파일이 존재하면 스킵합니다 (재실행 안전).
    - 실패 시 최대 retries회 재시도하며, 재시도 간격이 점점 늘어납니다.

    Args:
        d: 다운로드할 날짜
        sleep_sec: 성공 후 대기 시간 (API 부하 방지용)
        retries: 최대 재시도 횟수

    Returns:
        (성공 여부, 메시지) 튜플
    """
    y, m, dd = d.year, d.month, d.day
    fname = f"{y:04d}{m:02d}{dd:02d}.json"
    fpath = os.path.join(OUT_DIR, fname)

    # 이미 다운로드된 파일이 있으면 건너뛰기
    if os.path.exists(fpath):
        return True, "SKIP"

    params = {"year": str(y), "month": str(m), "day": str(dd)}

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            status, body = signed_get(params)

            # 응답 본문을 그대로 파일로 저장
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(body)

            # result_cd == 100이면 정상 응답
            data = json.loads(body)
            ok = (status == 200 and data.get("result_cd") == 100)
            time.sleep(sleep_sec)
            return ok, data.get("result_msg")

        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            # 재시도 간격을 점점 늘림 (0.5초, 1초, 1.5초...)
            time.sleep(0.5 * attempt)

    return False, repr(last_err)


def main():
    """
    2022~2025년 전체 날짜를 순회하며 경기 일정 raw JSON을 수집합니다.

    참고: 실제 모델링은 2023년 이후 데이터만 사용하지만,
    2022년부터 수집하는 이유는 시즌 초반 prior(사전 데이터) 계산에 필요하기 때문입니다.
    """
    start = date(2022, 1, 1)
    end   = date(2025, 12, 31)

    ok_days = 0
    fail_days = 0

    for d in daterange(start, end):
        ok, msg = fetch_one_day(d)
        if ok:
            ok_days += 1
        else:
            fail_days += 1
            print("FAIL", d.isoformat(), msg)

    print("DONE", "ok_days=", ok_days, "fail_days=", fail_days, "dir=", OUT_DIR)

if __name__ == "__main__":
    main()
