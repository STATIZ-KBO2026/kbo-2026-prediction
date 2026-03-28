import os, csv, json, time, hmac, hashlib, urllib.parse, argparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

API_KEY = os.environ["STATIZ_API_KEY"]
SECRET  = os.environ["STATIZ_SECRET"].encode("utf-8")

BASE = "https://api.statiz.co.kr/baseballApi"
PATH = "prediction/playerDay"

INDEX_CSV = os.path.expanduser("~/statiz/data/player_year_index.csv")
OUT_DIR   = os.path.expanduser("~/statiz/data/raw_playerday")
os.makedirs(OUT_DIR, exist_ok=True)

def signed_get(params: dict, timeout=25):
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
    try:
        j = json.loads(body)
        if isinstance(j, dict) and "result_cd" in j:
            return j.get("result_cd") == 100
    except Exception:
        pass
    return True

def load_pairs(limit=None):
    pairs = []
    with open(INDEX_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pairs.append((int(row["p_no"]), int(row["year"])))
            if limit and len(pairs) >= limit:
                break
    return pairs

def out_path(p_no, year):
    return os.path.join(OUT_DIR, f"{p_no}_{year}.json")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0이면 전체, 양수면 앞에서 N개만 수집")
    args = ap.parse_args()

    pairs = load_pairs(limit=args.limit if args.limit > 0 else None)
    print("targets:", len(pairs))

    ok = 0
    fail = 0

    for i, (p_no, year) in enumerate(pairs, 1):
        fp = out_path(p_no, year)
        if os.path.exists(fp):
            continue

        params = {"p_no": str(p_no), "year": str(year)}
        last_err = None

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
            fail += 1
            print("ERROR", p_no, year, repr(last_err))

        time.sleep(0.12)
        if i % 50 == 0:
            print("progress", i, "/", len(pairs), "ok=", ok, "fail=", fail)

    print("DONE", "ok=", ok, "fail=", fail, "dir=", OUT_DIR)

if __name__ == "__main__":
    main()
