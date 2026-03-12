import os, time, hmac, hashlib, urllib.parse
from urllib.request import Request, urlopen
from urllib.error import HTTPError

api_key = os.environ["STATIZ_API_KEY"]
secret  = os.environ["STATIZ_SECRET"].encode("utf-8")

method = "GET"
base = "https://api.statiz.co.kr/baseballApi"
path = "prediction/gameSchedule"  # 앞에 / 없이!

# ✅ 파라미터(테스트용 날짜)
params = {"year": "2025", "month": "5", "day": "4"}

# query normalize: key 정렬 + URL 인코딩
normalized_query = "&".join(
    f"{urllib.parse.quote(k)}={urllib.parse.quote(str(params[k]))}"
    for k in sorted(params)
)

timestamp = str(int(time.time()))
payload = f"{method}|{path}|{normalized_query}|{timestamp}"
signature = hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()

url = f"{base}/{path}?{normalized_query}"
req = Request(url, method=method, headers={
    "X-API-KEY": api_key,
    "X-TIMESTAMP": timestamp,
    "X-SIGNATURE": signature,
})

try:
    with urlopen(req, timeout=20) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        print("STATUS:", resp.status)
        print(body)
except HTTPError as e:
    body = e.read().decode("utf-8", errors="replace")
    print("STATUS:", e.code)
    print(body)
