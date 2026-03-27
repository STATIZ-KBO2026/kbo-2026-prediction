import os, json, time, hmac, hashlib, urllib.parse
from datetime import date, timedelta
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

API_KEY = os.environ["STATIZ_API_KEY"]
SECRET  = os.environ["STATIZ_SECRET"].encode("utf-8")

BASE = "https://api.statiz.co.kr/baseballApi"
PATH = "prediction/gameSchedule"   # ✅ 앞에 / 없이

OUT_DIR = os.path.expanduser("~/statiz/data/raw_schedule")
os.makedirs(OUT_DIR, exist_ok=True)

def signed_get(params: dict, timeout=20):
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

def daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)

def fetch_one_day(d: date, sleep_sec=0.2, retries=3):
    y, m, dd = d.year, d.month, d.day
    fname = f"{y:04d}{m:02d}{dd:02d}.json"
    fpath = os.path.join(OUT_DIR, fname)

    # 이미 있으면 스킵(재실행 안전)
    if os.path.exists(fpath):
        return True, "SKIP"

    params = {"year": str(y), "month": str(m), "day": str(dd)}

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            status, body = signed_get(params)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(body)

            data = json.loads(body)
            ok = (status == 200 and data.get("result_cd") == 100)
            time.sleep(sleep_sec)
            return ok, data.get("result_msg")

        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(0.5 * attempt)  # 가벼운 백오프

    return False, repr(last_err)

def main():
    # 대회 공지 기준: 2023년부터 사용
    start = date(2023, 1, 1)
    end   = date.today()

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
