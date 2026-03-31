"""
kbo_scheduler.py
================
Dynamic per-game scheduler for KBO predictions.

Flow:
  1) At launch (00:00 via cron): fetch today's schedule, get game times
  2) Update model with yesterday's data (rebuild indexes, features)
  3) Per game at T-50min: fetch lineup (retry every 5min if fail)
  4) Per game: run prediction & submit by T-15min deadline

Usage:
  python3 scripts/kbo_scheduler.py                 # today
  python3 scripts/kbo_scheduler.py --date 20260329  # specific date
  python3 scripts/kbo_scheduler.py --dry-run        # no submission

Crontab (single entry):
  0 0 * * * /home/ubuntu/kbo-2026-prediction/scripts/launch_scheduler.sh >> /home/ubuntu/statiz/logs/scheduler.log 2>&1
"""
import os
import sys
import json
import time
import hmac
import hashlib
import urllib.parse
import argparse
import subprocess
import threading
from datetime import datetime, date, timedelta
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

REPO_ROOT = Path(__file__).resolve().parents[1]

API_KEY = os.environ.get("STATIZ_API_KEY", "").strip()
SECRET = os.environ.get("STATIZ_SECRET", "").strip().encode("utf-8")
BASE = "https://api.statiz.co.kr/baseballApi"

RAW_SCHEDULE = os.path.expanduser("~/statiz/data/raw_schedule")
RAW_LINEUP = os.path.expanduser("~/statiz/data/raw_lineup")
RAW_BOXSCORE = os.path.expanduser("~/statiz/data/raw_boxscore")
LOG_DIR = os.path.expanduser("~/statiz/logs")

for d in [RAW_SCHEDULE, RAW_LINEUP, RAW_BOXSCORE, LOG_DIR]:
    os.makedirs(d, exist_ok=True)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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
    fname = f"{d.strftime('%Y%m%d')}.json"
    fpath = os.path.join(RAW_SCHEDULE, fname)
    if os.path.exists(fpath) and not force:
        log(f"  schedule {fname}: cached")
        return fpath
    params = {"year": str(d.year), "month": str(d.month), "day": str(d.day)}
    status, body = signed_get("prediction/gameSchedule", params)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(body)
    data = json.loads(body)
    ok = status == 200 and data.get("result_cd") == 100
    log(f"  schedule {fname}: {'OK' if ok else 'FAIL'}")
    return fpath if ok else None


def parse_schedule_games(fpath):
    """Parse schedule JSON → list of {s_no, hm, homeTeam, awayTeam, state}."""
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    games = []
    for key, val in data.items():
        if isinstance(val, list):
            for g in val:
                games.append({
                    "s_no": int(g["s_no"]),
                    "hm": g.get("hm", "14:00:00"),
                    "homeTeam": g.get("homeTeam"),
                    "awayTeam": g.get("awayTeam"),
                    "state": g.get("state"),
                })
    return games


def fetch_lineup(s_no, force=False):
    fpath = os.path.join(RAW_LINEUP, f"{s_no}.json")
    if os.path.exists(fpath) and not force:
        return True
    try:
        status, body = signed_get("prediction/gameLineup", {"s_no": str(s_no)})
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(body)
        return status == 200
    except Exception as e:
        log(f"    lineup {s_no}: ERROR {e}")
        return False


def lineup_batters_confirmed(s_no, min_batters_per_team=7):
    """배터 라인업이 확정됐는지 확인.
    lineupState='Y'인 타자(battingOrder 1-9)가 팀당 min_batters_per_team명 이상이어야 함.
    SP만 있는 경우(battingOrder='P')는 미확정으로 판단.
    """
    fpath = os.path.join(RAW_LINEUP, f"{s_no}.json")
    if not os.path.exists(fpath):
        return False
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            ldata = json.load(f)
        team_batter_counts = {}
        for key, val in ldata.items():
            if not str(key).isdigit():
                continue
            if not isinstance(val, list):
                continue
            confirmed = 0
            for player in val:
                if not isinstance(player, dict):
                    continue
                bo = player.get("battingOrder", "P")
                ls = player.get("lineupState", "N")
                # battingOrder가 숫자(1-9)인 타자만 카운트
                try:
                    bo_int = int(bo)
                    if 1 <= bo_int <= 9 and ls == "Y":
                        confirmed += 1
                except (ValueError, TypeError):
                    pass
            team_batter_counts[key] = confirmed
        if not team_batter_counts:
            return False
        # 모든 팀이 min_batters_per_team명 이상 확정돼야 함
        return all(cnt >= min_batters_per_team for cnt in team_batter_counts.values())
    except Exception:
        return False


def fetch_boxscore(s_no, force=False):
    fpath = os.path.join(RAW_BOXSCORE, f"{s_no}.json")
    if os.path.exists(fpath) and not force:
        return True
    try:
        status, body = signed_get("prediction/gameBoxscore", {"s_no": str(s_no)})
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(body)
        return status == 200
    except Exception as e:
        log(f"    boxscore {s_no}: ERROR {e}")
        return False


def run_py(script_name, *extra_args, timeout=300):
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / script_name)] + list(extra_args)
    log(f"  + {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        if result.stderr:
            log(f"    STDERR: {result.stderr[-500:]}")
        log(f"    (exit code {result.returncode})")
    else:
        last = (result.stdout.strip().split('\n') or [''])[-1]
        if last:
            log(f"    {last}")
    return result.returncode == 0


def hm_to_minutes(hm_str):
    """'14:00:00' → minutes since midnight (840)."""
    parts = hm_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def current_minutes():
    """Current time as minutes since midnight."""
    now = datetime.now()
    return now.hour * 60 + now.minute


def sleep_until_minutes(target_min):
    """Sleep until target minutes-since-midnight. Returns False if target already passed."""
    now_min = current_minutes()
    if target_min <= now_min:
        return False
    wait_sec = (target_min - now_min) * 60
    # Adjust for seconds within the minute
    now = datetime.now()
    wait_sec = wait_sec - now.second
    if wait_sec <= 0:
        return False
    log(f"  Sleeping {wait_sec // 60}m {wait_sec % 60}s until {target_min // 60:02d}:{target_min % 60:02d}")
    time.sleep(wait_sec)
    return True


def phase1_daily_update(target, yesterday):
    """Phase 1: Run at startup (00:00). Fetch schedule, update yesterday's data, rebuild."""
    target_str = target.strftime("%Y%m%d")
    yesterday_str = yesterday.strftime("%Y%m%d")

    log("=" * 50)
    log(f"Phase 1: Daily update  target={target_str}  yesterday={yesterday_str}")
    log("=" * 50)

    # 1a. Fetch today's schedule
    log("\n[1a] Fetching today's schedule ...")
    sched_path = fetch_schedule(target, force=True)

    # 1b. Fetch yesterday's schedule + boxscores + lineups
    log(f"\n[1b] Fetching yesterday's data ({yesterday_str}) ...")
    fetch_schedule(yesterday, force=False)
    ysched = os.path.join(RAW_SCHEDULE, f"{yesterday_str}.json")
    if os.path.exists(ysched):
        ygames = parse_schedule_games(ysched)
        for g in ygames:
            fetch_boxscore(g["s_no"], force=False)
            fetch_lineup(g["s_no"], force=False)
            time.sleep(0.15)
        log(f"  Yesterday: {len(ygames)} games processed")
    else:
        log("  Yesterday: no schedule found")

    # 1c. Rebuild indexes & features
    log("\n[1c] Rebuilding indexes ...")
    run_py("build_game_index.py")
    run_py("build_lineup_table.py")
    run_py("build_playerday_tables_v2.py")

    # 1d. Build features (includes yesterday's results)
    log("\n[1d] Building features_v1.csv ...")
    run_py("build_features_v1.py", timeout=600)

    return sched_path


def phase2_per_game(game, target_str, dry_run=False, season_mode="all"):
    """Phase 2: Per-game lineup fetch + predict + submit.
    Called at T-50min for each game.
    """
    s_no = game["s_no"]
    hm = game["hm"]
    game_min = hm_to_minutes(hm)
    deadline_min = game_min - 15  # must submit by T-15

    log(f"\n{'='*50}")
    log(f"Phase 2: Game s_no={s_no}  start={hm}  deadline={deadline_min//60:02d}:{deadline_min%60:02d}")
    log(f"{'='*50}")

    # 2a. Fetch lineup with retry (every 5min, up to deadline)
    # 배터 라인업(battingOrder 1-9, lineupState='Y')이 확정될 때까지 대기
    log("[2a] Fetching lineup (waiting for batting lineup confirmation)...")
    retry_interval_sec = 300  # 5분
    while True:
        now_min = current_minutes()
        if now_min >= deadline_min:
            log(f"  Deadline {deadline_min//60:02d}:{deadline_min%60:02d} reached. "
                f"Proceeding with current lineup data.")
            break

        fetch_lineup(s_no, force=True)
        if lineup_batters_confirmed(s_no):
            log(f"  Batting lineup confirmed for s_no={s_no}!")
            break

        remaining = (deadline_min - now_min) * 60
        wait_sec = min(retry_interval_sec, max(0, remaining - 30))
        if wait_sec <= 0:
            log(f"  No time left to retry. Proceeding.")
            break
        log(f"  Batting lineup not ready (SP only?). Retrying in {wait_sec//60}m {wait_sec%60}s "
            f"(deadline in {remaining//60}m)...")
        time.sleep(wait_sec)

    if not lineup_batters_confirmed(s_no):
        log(f"  WARNING: Batting lineup not fully confirmed for s_no={s_no}. "
            f"Lineup features may be incomplete.")

    # 2b. Rebuild lineup table with new data
    log("[2b] Rebuilding lineup table ...")
    run_py("build_lineup_table.py")

    # 2c. Run prediction pipeline for this date
    log("[2c] Running prediction pipeline ...")
    pipeline_args = ["--date", target_str, "--season-mode", season_mode]
    if dry_run:
        pipeline_args.append("--skip-submit")
    run_py("run_submit_pipeline_v1.py", *pipeline_args, timeout=300)

    log(f"Phase 2 complete for s_no={s_no}")


def main():
    ap = argparse.ArgumentParser(description="KBO Dynamic Scheduler")
    ap.add_argument("--date", default="", help="YYYYMMDD (default: today)")
    ap.add_argument("--dry-run", action="store_true", help="Skip submission")
    ap.add_argument("--season-mode", default="all", choices=["regular", "all"])
    ap.add_argument("--skip-daily-update", action="store_true",
                    help="Skip phase 1 (daily update), go straight to per-game scheduling")
    args = ap.parse_args()

    if args.date:
        target = date(int(args.date[:4]), int(args.date[4:6]), int(args.date[6:8]))
    else:
        target = date.today()
    target_str = target.strftime("%Y%m%d")
    yesterday = target - timedelta(days=1)

    if not API_KEY or not SECRET:
        log("FATAL: STATIZ_API_KEY / STATIZ_SECRET not set.")
        sys.exit(1)

    log(f"KBO Scheduler starting: target={target_str}")

    # ── Phase 1: Daily update (indexes, features, yesterday's data) ──
    if not args.skip_daily_update:
        sched_path = phase1_daily_update(target, yesterday)
    else:
        sched_path = os.path.join(RAW_SCHEDULE, f"{target_str}.json")
        if not os.path.exists(sched_path):
            log("No cached schedule found. Fetching ...")
            sched_path = fetch_schedule(target, force=True)

    # ── Parse today's games and their start times ──
    if not sched_path or not os.path.exists(sched_path):
        log("No schedule available. Exiting.")
        return

    games = parse_schedule_games(sched_path)
    if not games:
        log("No games today. Exiting.")
        return

    log(f"\nToday's games ({len(games)}):")
    for g in games:
        log(f"  s_no={g['s_no']}  start={g['hm']}  {g['awayTeam']} @ {g['homeTeam']}")

    # ── Group games by start time ──
    # Games starting at the same time share a single prediction run
    time_groups = {}
    for g in games:
        hm = g["hm"]
        if hm not in time_groups:
            time_groups[hm] = []
        time_groups[hm].append(g)

    sorted_times = sorted(time_groups.keys())
    log(f"\nGame time groups: {sorted_times}")

    # ── Phase 2: Per time-group scheduling ──
    submitted_times = set()
    for hm in sorted_times:
        group = time_groups[hm]
        game_min = hm_to_minutes(hm)
        fetch_min = game_min - 50  # T-50 minutes
        deadline_min = game_min - 15

        log(f"\n--- Time group {hm}: {len(group)} game(s) ---")
        log(f"  Lineup fetch at: {fetch_min//60:02d}:{fetch_min%60:02d}")
        log(f"  Deadline:        {deadline_min//60:02d}:{deadline_min%60:02d}")

        # Sleep until T-50
        now_min = current_minutes()
        if now_min < fetch_min:
            sleep_until_minutes(fetch_min)
        elif now_min >= deadline_min:
            log(f"  Deadline already passed for {hm} games. Skipping.")
            continue

        # ── Lineup fetch + 배터 확정 대기 루프 ──
        # T-50min부터 5분 간격으로 재시도, T-15min(deadline)까지 배터 라인업이
        # 확정되길 기다렸다가 파이프라인 실행.
        log(f"[Fetch] Getting lineups for {hm} games (waiting for batting lineup confirmation)...")

        retry_interval_sec = 300  # 5분
        while True:
            now_min = current_minutes()
            if now_min >= deadline_min:
                log(f"  Deadline {deadline_min//60:02d}:{deadline_min%60:02d} reached. "
                    f"Proceeding with current lineup data.")
                break

            # 모든 경기 라인업 갱신
            for g in group:
                fetch_lineup(g["s_no"], force=True)

            # 배터 확정 여부 체크
            all_confirmed = all(lineup_batters_confirmed(g["s_no"]) for g in group)
            if all_confirmed:
                log(f"  All batting lineups confirmed for {hm} group!")
                break

            # 미확정 경기 출력
            for g in group:
                confirmed = lineup_batters_confirmed(g["s_no"])
                log(f"  s_no={g['s_no']}: batting lineup {'confirmed' if confirmed else 'NOT confirmed (SP only?)'}")

            remaining = (deadline_min - now_min) * 60
            wait_sec = min(retry_interval_sec, max(0, remaining - 30))
            if wait_sec <= 0:
                log(f"  No time left to retry. Proceeding.")
                break
            log(f"  Batting lineup not ready yet. Retrying in {wait_sec//60}m {wait_sec%60}s "
                f"(deadline in {remaining//60}m)...")
            time.sleep(wait_sec)

        # Rebuild lineup table & run prediction for all today's games
        log(f"\n[Predict] Rebuilding lineup table & running pipeline ...")
        run_py("build_lineup_table.py")

        pipeline_args = ["--date", target_str, "--season-mode", args.season_mode]
        if args.dry_run:
            pipeline_args.append("--skip-submit")
        run_py("run_submit_pipeline_v1.py", *pipeline_args, timeout=300)

        submitted_times.add(hm)
        log(f"Prediction submitted for {hm} group")

    log(f"\n{'='*50}")
    log(f"Scheduler complete. Submitted for time groups: {sorted(submitted_times)}")
    log(f"{'='*50}")


if __name__ == "__main__":
    main()
