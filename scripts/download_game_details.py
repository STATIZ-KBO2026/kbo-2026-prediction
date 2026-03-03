import os, csv, json, time, hmac, hashlib, urllib.parse, argparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

API_KEY = os.environ["STATIZ_API_KEY"]
SECRET  = os.environ["STATIZ_SECRET"].encode("utf-8")

BASE = "https://api.statiz.co.kr/baseballApi"
INDEX_CSV = os.path.expanduser("~/statiz/data/game_index_played.csv")

OUT_LINEUP = os.path.expanduser("~/statiz/data/raw_lineup")
OUT_BOXS   = os.path.expanduser("~/statiz/data/raw_boxscore")
os.makedirs(OUT_LINEUP, exist_ok=True)
os.makedirs(OUT_BOXS, exist_ok=True)

def signed_get(path: str, params: dict, timeout=20):
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
    snos = []
    with open(INDEX_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            snos.append(int(row["s_no"]))
            if limit and len(snos) >= limit:
                break
    return snos

def save_json(out_dir, s_no, body):
    fp = os.path.join(out_dir, f"{s_no}.json")
    with open(fp, "w", encoding="utf-8") as f:
        f.write(body)

def is_ok(body: str):
    # 어떤 API는 HTTP 200이어도 result_cd로 실패를 줌
    try:
        j = json.loads(body)
        if isinstance(j, dict) and "result_cd" in j:
            return j.get("result_cd") == 100
    except Exception:
        pass
    return True  # result_cd가 없으면 일단 통과(원본 저장이 목적)

def fetch_one(s_no, sleep_sec=0.15, retries=3):
    tasks = [
        ("prediction/gameLineup", {"s_no": str(s_no)}, OUT_LINEUP),
        ("prediction/gameBoxscore", {"s_no": str(s_no)}, OUT_BOXS),
    ]

    for path, params, out_dir in tasks:
        out_fp = os.path.join(out_dir, f"{s_no}.json")
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
            print("ERROR", path, "s_no=", s_no, repr(last_err))

def main():
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
