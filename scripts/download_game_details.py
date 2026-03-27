import os, csv, json, time, hmac, hashlib, urllib.parse, argparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

API_KEY = os.environ["STATIZ_API_KEY"]
SECRET  = os.environ["STATIZ_SECRET"].encode("utf-8")

BASE = "https://api.statiz.co.kr/baseballApi"
INDEX_CSV = os.path.expanduser("~/statiz/data/game_index.csv")

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

def existing_ok(out_fp):
    if not os.path.exists(out_fp):
        return False
    try:
        with open(out_fp, "r", encoding="utf-8") as f:
            body = f.read()
        return is_ok(body)
    except Exception:
        return False


def read_snos(index_csv, limit=None):
    snos = []
    with open(index_csv, "r", encoding="utf-8") as f:
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
        if existing_ok(out_fp):
            continue

        last_err = None
        for attempt in range(1, retries + 1):
            try:
                status, body = signed_get(path, params)
                ok = (status == 200 and is_ok(body))
                if ok:
                    save_json(out_dir, s_no, body)
                    time.sleep(sleep_sec)
                    break
                if attempt == retries:
                    print("FAIL", path, "s_no=", s_no, "status=", status)
                else:
                    time.sleep(0.4 * attempt)
            except (HTTPError, URLError, TimeoutError) as e:
                last_err = e
                time.sleep(0.5 * attempt)
        else:
            print("ERROR", path, "s_no=", s_no, repr(last_err))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", default=INDEX_CSV, help="game index csv path")
    ap.add_argument("--limit", type=int, default=0, help="0이면 전체, 양수면 앞에서 N개만 수집")
    args = ap.parse_args()

    snos = read_snos(args.index_csv, limit=args.limit)
    print("index_csv:", args.index_csv)
    print("targets:", len(snos))

    for i, s_no in enumerate(snos, 1):
        fetch_one(s_no)
        if i % 10 == 0:
            print("progress", i, "/", len(snos))

    print("DONE")

if __name__ == "__main__":
    main()
