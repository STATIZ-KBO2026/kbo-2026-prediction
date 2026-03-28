import os
import csv
import json
import time
import hmac
import hashlib
import argparse
import urllib.parse
from collections import OrderedDict
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

BASE = "https://api.statiz.co.kr/baseballApi"
PATH = "prediction/savePrediction"

DEFAULT_IN_CSV = os.path.expanduser("~/statiz/data/backtest_pred_v5_best.csv")
DEFAULT_OUT_CSV = os.path.expanduser("~/statiz/data/save_prediction_result.csv")


def load_auth():
    api_key = os.getenv("STATIZ_API_KEY", "").strip()
    secret = os.getenv("STATIZ_SECRET", "").strip()
    if not api_key or not secret:
        raise RuntimeError("Missing STATIZ_API_KEY / STATIZ_SECRET environment variables")
    return api_key, secret.encode("utf-8")


def safe_float(x):
    try:
        if x is None:
            return 0.0
        s = str(x).strip()
        if s == "":
            return 0.0
        return float(s)
    except Exception:
        return 0.0


def safe_int(x):
    try:
        if x is None:
            return 0
        s = str(x).strip()
        if s == "":
            return 0
        return int(float(s))
    except Exception:
        return 0


def to_int_or_none(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None


def nonempty(x):
    return x is not None and str(x).strip() != ""


def detect_scale(values):
    if not values:
        return "0-1"
    mx = max(abs(v) for v in values)
    if mx <= 1.2:
        return "0-1"
    return "0-100"


def normalize_percent(raw_value, scale):
    if scale == "0-1":
        pct = raw_value * 100.0
    else:
        pct = raw_value
    pct = max(0.0, min(100.0, pct))
    return round(pct + 1e-12, 2)


def unique_columns(cols):
    out = []
    seen = set()
    for c in cols:
        if not c:
            continue
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def pick_input_value(row, percent_cols, prob_cols):
    for c in percent_cols:
        if nonempty(row.get(c)):
            return safe_float(row[c]), c, "percent"
    for c in prob_cols:
        if nonempty(row.get(c)):
            return safe_float(row[c]), c, "prob"
    return None, "", ""


def load_prediction_rows(path, sno_col, percent_col, prob_col, input_scale, keep_duplicates, date_col):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"No header found: {path}")
        if sno_col not in reader.fieldnames:
            raise RuntimeError(f"Missing s_no column: {sno_col}")
        if date_col and date_col not in reader.fieldnames:
            raise RuntimeError(f"Missing date column: {date_col}")

        percent_cols = unique_columns([percent_col, "percent"])
        prob_cols = unique_columns([prob_col, "p_homewin", "prob", "prob_homewin", "prediction", "pred"])

        recs = []
        for line_no, row in enumerate(reader, start=2):
            s_no = safe_int(row.get(sno_col))
            if s_no <= 0:
                continue
            raw_val, src_col, src_kind = pick_input_value(row, percent_cols, prob_cols)
            if src_col == "":
                continue
            recs.append(
                {
                    "line_no": line_no,
                    "s_no": s_no,
                    "date": str(row.get(date_col, "")).strip() if date_col else "",
                    "raw_val": raw_val,
                    "src_col": src_col,
                    "src_kind": src_kind,
                }
            )

    if not recs:
        raise RuntimeError("No usable rows found. Check s_no/percent/p_homewin columns.")

    if not keep_duplicates:
        dedup = OrderedDict()
        for r in recs:
            if r["s_no"] in dedup:
                del dedup[r["s_no"]]
            dedup[r["s_no"]] = r
        recs = list(dedup.values())

    scale = input_scale
    if scale == "auto":
        if all(r["src_kind"] == "percent" for r in recs):
            scale = "0-100"
        else:
            scale = detect_scale([r["raw_val"] for r in recs])

    for r in recs:
        r["percent"] = normalize_percent(r["raw_val"], scale)

    recs.sort(key=lambda x: x["line_no"])
    return recs, scale


def in_date_window(date_val, exact, date_from, date_to):
    if not exact and not date_from and not date_to:
        return True
    d = str(date_val or "").strip()
    if exact and d != exact:
        return False
    if date_from and d < date_from:
        return False
    if date_to and d > date_to:
        return False
    return True


def sign_payload(method, path, params, ts, secret):
    normalized = "&".join(
        f"{urllib.parse.quote(k)}={urllib.parse.quote(str(params[k]))}"
        for k in sorted(params)
    )
    payload = f"{method}|{path}|{normalized}|{ts}"
    sig = hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return normalized, sig


def signed_post(params, timeout=20):
    api_key, secret = load_auth()
    method = "POST"
    ts = str(int(time.time()))
    normalized, sig = sign_payload(method, PATH, params, ts, secret)
    req = Request(
        f"{BASE}/{PATH}",
        method=method,
        data=normalized.encode("utf-8"),
        headers={
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            "X-API-KEY": api_key,
            "X-TIMESTAMP": ts,
            "X-SIGNATURE": sig,
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", errors="replace")


def parse_response_fields(body):
    cdoe = ""
    result_msg = ""
    try:
        data = json.loads(body)
        if isinstance(data, dict):
            cdoe = data.get("cdoe", data.get("code", data.get("result_cd", "")))
            result_msg = data.get("result_msg", "")
    except Exception:
        pass
    return cdoe, result_msg


def is_success(http_status, cdoe):
    if http_status != 200:
        return False
    if cdoe in ("", None):
        return True
    code = to_int_or_none(cdoe)
    if code is None:
        return True
    return code in (0, 100)


def load_existing_rows(path):
    if not os.path.exists(path):
        return [], set()
    rows = []
    ok_snos = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            ok_flag = str(row.get("ok", "")).strip().lower()
            if ok_flag in {"1", "true", "y", "yes"}:
                ok_snos.add(safe_int(row.get("s_no")))
    return rows, ok_snos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", default=DEFAULT_IN_CSV, help="model prediction csv path")
    ap.add_argument("--out-csv", default=DEFAULT_OUT_CSV, help="submission response log csv path")
    ap.add_argument("--sno-col", default="s_no", help="game id column name")
    ap.add_argument("--date-col", default="date", help="date column name (YYYYMMDD)")
    ap.add_argument("--date", default="", help="submit only exact date YYYYMMDD")
    ap.add_argument("--date-from", default="", help="submit date >= YYYYMMDD")
    ap.add_argument("--date-to", default="", help="submit date <= YYYYMMDD")
    ap.add_argument("--percent-col", default="", help="already-percent column (0~100)")
    ap.add_argument("--prob-col", default="p_homewin", help="probability column")
    ap.add_argument("--input-scale", choices=["auto", "0-1", "0-100"], default="auto")
    ap.add_argument("--limit", type=int, default=0, help="0 means all rows")
    ap.add_argument("--sleep-sec", type=float, default=0.15)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--resume", action="store_true", help="skip s_no already marked ok in out-csv")
    ap.add_argument("--keep-duplicates", action="store_true", help="do not deduplicate by s_no")
    ap.add_argument("--dry-run", action="store_true", help="show payloads only, do not POST")
    args = ap.parse_args()

    rows, scale = load_prediction_rows(
        args.in_csv,
        args.sno_col,
        args.percent_col,
        args.prob_col,
        args.input_scale,
        args.keep_duplicates,
        args.date_col,
    )

    if args.date or args.date_from or args.date_to:
        before = len(rows)
        rows = [
            r
            for r in rows
            if in_date_window(r.get("date", ""), args.date.strip(), args.date_from.strip(), args.date_to.strip())
        ]
        print("date_filter_keep:", len(rows), "/", before)

    if args.limit > 0:
        rows = rows[: args.limit]

    existing_rows = []
    if args.resume and not args.dry_run:
        existing_rows, ok_snos = load_existing_rows(args.out_csv)
        before = len(rows)
        rows = [r for r in rows if r["s_no"] not in ok_snos]
        print("resume_skip:", before - len(rows))

    print("targets:", len(rows))
    print("input_scale:", scale)
    print("in_csv:", args.in_csv)
    print("out_csv:", args.out_csv)
    if args.date:
        print("date:", args.date)
    if args.date_from or args.date_to:
        print("date_range:", args.date_from or "-", "~", args.date_to or "-")

    if not rows:
        print("DONE", "no rows to submit")
        return

    if args.dry_run:
        for r in rows:
            print("DRYRUN", {"date": r.get("date", ""), "s_no": r["s_no"], "percent": f"{r['percent']:.2f}"})
        print("DONE dry-run only")
        return

    out_rows = []
    for i, r in enumerate(rows, start=1):
        payload = {"s_no": str(r["s_no"]), "percent": f"{r['percent']:.2f}"}
        sent = False

        for attempt in range(1, args.retries + 1):
            try:
                status, body = signed_post(payload, timeout=args.timeout)
                cdoe, result_msg = parse_response_fields(body)
                ok = is_success(status, cdoe)
                out_rows.append(
                    {
                        "date": r.get("date", ""),
                        "s_no": payload["s_no"],
                        "percent": payload["percent"],
                        "cdoe": cdoe,
                        "result_msg": result_msg,
                        "http_status": status,
                        "ok": 1 if ok else 0,
                        "error": "",
                        "raw_body": body,
                    }
                )
                if not ok:
                    print("FAIL", payload, "status=", status, "cdoe=", cdoe, "msg=", result_msg)
                sent = True
                break

            except HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                cdoe, result_msg = parse_response_fields(body)
                out_rows.append(
                    {
                        "date": r.get("date", ""),
                        "s_no": payload["s_no"],
                        "percent": payload["percent"],
                        "cdoe": cdoe,
                        "result_msg": result_msg,
                        "http_status": e.code,
                        "ok": 0,
                        "error": f"HTTPError:{e.code}",
                        "raw_body": body,
                    }
                )
                print("HTTPFAIL", payload, "status=", e.code)
                sent = True
                break

            except (URLError, TimeoutError) as e:
                if attempt == args.retries:
                    out_rows.append(
                        {
                            "date": r.get("date", ""),
                            "s_no": payload["s_no"],
                            "percent": payload["percent"],
                            "cdoe": "",
                            "result_msg": "",
                            "http_status": "",
                            "ok": 0,
                            "error": repr(e),
                            "raw_body": "",
                        }
                    )
                    print("ERROR", payload, repr(e))
                    sent = True
                    break
                time.sleep(0.5 * attempt)

        if not sent:
            out_rows.append(
                {
                    "date": r.get("date", ""),
                    "s_no": payload["s_no"],
                    "percent": payload["percent"],
                    "cdoe": "",
                    "result_msg": "",
                    "http_status": "",
                    "ok": 0,
                    "error": "unknown",
                    "raw_body": "",
                }
            )

        if i % 20 == 0:
            print("progress", i, "/", len(rows))
        time.sleep(args.sleep_sec)

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    merged_rows = existing_rows + out_rows if args.resume else out_rows
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["date", "s_no", "percent", "cdoe", "result_msg", "http_status", "ok", "error", "raw_body"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(merged_rows)

    ok_n = sum(1 for r in out_rows if str(r.get("ok")) == "1")
    fail_n = len(out_rows) - ok_n
    print("DONE", "ok=", ok_n, "fail=", fail_n, "out=", args.out_csv)


if __name__ == "__main__":
    main()
