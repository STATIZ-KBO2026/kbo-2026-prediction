"""
fetch_today_and_predict.py
==========================
경기 시작 50분 전 실행:
  1) 오늘 스케줄 API → game_index에 오늘 경기 추가
  2) 오늘 경기 라인업 API (강제 갱신)
  3) 전날 boxscore가 없으면 수집 (결과 반영용)
  4) 인덱스 & 라인업 & playerday 재빌드
  5) features_v1 빌드 → 학습 → 예측 → 제출

사용법:
  python3 scripts/fetch_today_and_predict.py                    # 오늘
  python3 scripts/fetch_today_and_predict.py --date 20260329    # 특정 날짜
  python3 scripts/fetch_today_and_predict.py --dry-run          # 제출 안 함
"""
import os
import sys
import csv
import json
import time
import hmac
import hashlib
import urllib.parse
import argparse
from datetime import date, timedelta
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

API_KEY = os.environ.get("STATIZ_API_KEY", "").strip()
SECRET = os.environ.get("STATIZ_SECRET", "").strip().encode("utf-8")
BASE = "https://api.statiz.co.kr/baseballApi"

RAW_SCHEDULE = os.path.expanduser("~/statiz/data/raw_schedule")
RAW_LINEUP = os.path.expanduser("~/statiz/data/raw_lineup")
RAW_BOXSCORE = os.path.expanduser("~/statiz/data/raw_boxscore")
INDEX_CSV = os.path.expanduser("~/statiz/data/game_index.csv")

for d in [RAW_SCHEDULE, RAW_LINEUP, RAW_BOXSCORE]:
    os.makedirs(d, exist_ok=True)


def signed_get(path, params, timeout=20):
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
        return resp.status, resp.read().decode("utf-8", errors="replace")


def fetch_schedule(d, force=False):
    """하루 스케줄 JSON 다운로드."""
    fname = f"{d.strftime('%Y%m%d')}.json"
    fpath = os.path.join(RAW_SCHEDULE, fname)
    if os.path.exists(fpath) and not force:
        print(f"  schedule {fname}: cached")
        return fpath
    params = {"year": str(d.year), "month": str(d.month), "day": str(d.day)}
    status, body = signed_get("prediction/gameSchedule", params)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(body)
    data = json.loads(body)
    ok = status == 200 and data.get("result_cd") == 100
    print(f"  schedule {fname}: {'OK' if ok else 'FAIL'}")
    return fpath if ok else None


def get_snos_from_schedule(fpath):
    """스케줄 JSON에서 s_no 목록 추출."""
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.loads(f.read())
    snos = []
    # Format: {"MMDD": [games...], ...} or {"result": [games...]}
    games = []
    for k, v in data.items():
        if isinstance(k, str) and k.isdigit() and len(k) == 4 and isinstance(v, list):
            games.extend(v)
    if not games and isinstance(data.get("result"), list):
        games = data["result"]
    for game in games:
        s_no = game.get("s_no")
        if s_no:
            snos.append(int(s_no))
    return snos


def get_snos_from_index(target_date_str):
    """game_index.csv에서 특정 날짜의 s_no 목록."""
    snos = []
    if not os.path.exists(INDEX_CSV):
        return snos
    with open(INDEX_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("date") == target_date_str:
                snos.append(int(row["s_no"]))
    return snos


def fetch_lineup(s_no, force=False):
    """경기별 라인업 JSON 다운로드 (force=True면 캐시 무시)."""
    fpath = os.path.join(RAW_LINEUP, f"{s_no}.json")
    if os.path.exists(fpath) and not force:
        return True
    try:
        status, body = signed_get("prediction/gameLineup", {"s_no": str(s_no)})
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(body)
        time.sleep(0.15)
        return status == 200
    except Exception as e:
        print(f"    lineup {s_no}: ERROR {e}")
        return False


def fetch_boxscore(s_no, force=False):
    """경기별 박스스코어 JSON 다운로드."""
    fpath = os.path.join(RAW_BOXSCORE, f"{s_no}.json")
    if os.path.exists(fpath) and not force:
        return True
    try:
        status, body = signed_get("prediction/gameBoxscore", {"s_no": str(s_no)})
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(body)
        time.sleep(0.15)
        return status == 200
    except Exception as e:
        print(f"    boxscore {s_no}: ERROR {e}")
        return False


def run_py(script_name, *extra_args):
    """Python 스크립트 실행."""
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / script_name)] + list(extra_args)
    print(f"  + {' '.join(cmd)}")
    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"    STDERR: {result.stderr[-500:]}" if result.stderr else "")
        print(f"    (exit code {result.returncode})")
    else:
        # 마지막 줄만 출력
        last_line = (result.stdout.strip().split('\n') or [''])[-1]
        if last_line:
            print(f"    {last_line}")
    return result.returncode == 0


def main():
    ap = argparse.ArgumentParser(description="Fetch today's lineups → predict → submit")
    ap.add_argument("--date", default="", help="YYYYMMDD (default: today)")
    ap.add_argument("--dry-run", action="store_true", help="예측만, 제출 안 함")
    ap.add_argument("--skip-submit", action="store_true", help="제출 스킵")
    ap.add_argument("--force-lineup", action="store_true",
                    help="라인업 캐시 무시하고 무조건 새로 받기 (기본: True)")
    ap.add_argument("--season-mode", default="all")
    args = ap.parse_args()

    if args.date:
        target = date(int(args.date[:4]), int(args.date[4:6]), int(args.date[6:8]))
    else:
        target = date.today()
    target_str = target.strftime("%Y%m%d")
    yesterday = target - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y%m%d")

    if not API_KEY or not SECRET:
        print("WARNING: STATIZ_API_KEY / STATIZ_SECRET not set. Skipping API calls.")
        api_available = False
    else:
        api_available = True

    print(f"{'='*50}")
    print(f" KBO v1 Fetch & Predict")
    print(f" Target: {target_str}  Yesterday: {yesterday_str}")
    print(f"{'='*50}")

    # ── 1. 오늘 스케줄 다운로드 ──
    print(f"\n[1] Fetching schedule for {target_str} ...")
    if api_available:
        sched_path = fetch_schedule(target, force=True)
    else:
        sched_path = os.path.join(RAW_SCHEDULE, f"{target_str}.json")
        if not os.path.exists(sched_path):
            sched_path = None

    # ── 2. 전날 boxscore 수집 (아직 없는 것만) ──
    print(f"\n[2] Fetching yesterday's boxscores ({yesterday_str}) ...")
    if api_available:
        fetch_schedule(yesterday, force=False)
    yesterday_snos = get_snos_from_index(yesterday_str)
    if not yesterday_snos:
        # index가 아직 없으면 스케줄에서
        ysched = os.path.join(RAW_SCHEDULE, f"{yesterday_str}.json")
        if os.path.exists(ysched):
            yesterday_snos = get_snos_from_schedule(ysched)
    for s_no in yesterday_snos:
        if api_available:
            fetch_boxscore(s_no, force=False)
            fetch_lineup(s_no, force=False)  # 전날 라인업도 확보
    print(f"  Yesterday games: {len(yesterday_snos)}")

    # ── 3. 인덱스 재빌드 (전날 + 오늘 포함) ──
    print(f"\n[3] Rebuilding indexes ...")
    run_py("build_game_index.py")
    run_py("build_lineup_table.py")
    run_py("build_playerday_tables_v2.py")

    # ── 4. 오늘 경기 라인업 수집 (강제 갱신) ──
    print(f"\n[4] Fetching today's lineups (force refresh) ...")
    today_snos = get_snos_from_index(target_str)
    if not today_snos and sched_path and os.path.exists(sched_path):
        today_snos = get_snos_from_schedule(sched_path)
    print(f"  Today's games: {len(today_snos)}")
    for s_no in today_snos:
        if api_available:
            ok = fetch_lineup(s_no, force=True)  # 항상 새로 받기
            print(f"    lineup {s_no}: {'OK' if ok else 'FAIL'}")

    # ── 5. 라인업 테이블 재빌드 (오늘 라인업 반영) ──
    print(f"\n[5] Rebuilding lineup table with today's data ...")
    run_py("build_lineup_table.py")

    # ── 6. features + 학습 + 예측 + 제출 ──
    print(f"\n[6] Running prediction pipeline ...")
    pipeline_args = ["--date", target_str, "--season-mode", args.season_mode]
    if args.dry_run or args.skip_submit:
        pipeline_args.append("--skip-submit")
    run_py("run_submit_pipeline_v1.py", *pipeline_args)

    print(f"\n{'='*50}")
    print(f" Done!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
