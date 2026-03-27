import os, csv, math, argparse

IN_CSV = os.path.expanduser("~/statiz/data/features_v0.csv")
OUT_CSV = os.path.expanduser("~/statiz/data/feature_noise_report.csv")

EXCLUDE = {
    "date", "s_no", "homeTeam", "awayTeam",
    "y_home_win", "homeScore", "awayScore",
    "home_sp_p_no", "away_sp_p_no",
}

COUNTLIKE_HINTS = (
    "_G", "_IP", "_PA", "_app_", "games_last7", "consec_days", "away_streak",
)

def safe_float(x):
    try:
        if x is None:
            return 0.0
        s = str(x).strip()
        if s == "" or s.lower() == "none":
            return 0.0
        return float(s)
    except Exception:
        return 0.0

def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0

def variance(vals, mu=None):
    if not vals:
        return 0.0
    if mu is None:
        mu = mean(vals)
    return sum((v - mu) * (v - mu) for v in vals) / len(vals)

def corr_with_binary(x, y):
    # Pearson corr for numeric x and binary y in {0,1}
    n = len(x)
    if n == 0:
        return 0.0
    mx = mean(x)
    my = mean(y)
    vx = variance(x, mx)
    vy = variance(y, my)
    if vx <= 1e-15 or vy <= 1e-15:
        return 0.0
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    return cov / math.sqrt(vx * vy)

def is_countlike(col):
    return any(h in col for h in COUNTLIKE_HINTS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-start", default="20240101")
    ap.add_argument("--train-end", default="20241231")
    ap.add_argument("--test-start", default="20250101")
    ap.add_argument("--test-end", default="20251231")
    args = ap.parse_args()

    with open(IN_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
        cols = r.fieldnames or []

    feat_cols = [c for c in cols if c not in EXCLUDE]
    train = [row for row in rows if args.train_start <= row["date"] <= args.train_end]
    test = [row for row in rows if args.test_start <= row["date"] <= args.test_end]

    train_y = [int(float(row["y_home_win"])) for row in train]
    report = []
    for c in feat_cols:
        tx = [safe_float(row[c]) for row in train]
        sx = [safe_float(row[c]) for row in test]
        tmu = mean(tx)
        smu = mean(sx)
        tvar = variance(tx, tmu)
        svar = variance(sx, smu)
        tsd = math.sqrt(tvar)
        ssd = math.sqrt(svar)
        pooled = max(tsd, ssd, 1e-9)
        shift_z = abs(tmu - smu) / pooled
        zero_ratio = (sum(1 for v in tx if abs(v) < 1e-12) / len(tx)) if tx else 0.0
        corr = corr_with_binary(tx, train_y) if train else 0.0

        tags = []
        if tvar <= 1e-12:
            tags.append("constant")
        if zero_ratio >= 0.995:
            tags.append("near_all_zero")
        if shift_z >= 2.0:
            tags.append("high_train_test_shift")
        if is_countlike(c):
            tags.append("count_like")

        drop = 1 if ("constant" in tags or "near_all_zero" in tags) else 0
        report.append({
            "feature": c,
            "train_zero_ratio": round(zero_ratio, 6),
            "train_variance": round(tvar, 10),
            "train_test_shift_z": round(shift_z, 6),
            "train_mean": round(tmu, 8),
            "test_mean": round(smu, 8),
            "train_corr_with_y": round(corr, 8),
            "is_count_like": 1 if is_countlike(c) else 0,
            "drop_recommended": drop,
            "tags": "|".join(tags),
        })

    # sort by priority: drop candidates first, then high shift
    report.sort(key=lambda x: (-(x["drop_recommended"]), -x["train_test_shift_z"], x["feature"]))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "feature",
                "train_zero_ratio",
                "train_variance",
                "train_test_shift_z",
                "train_mean",
                "test_mean",
                "train_corr_with_y",
                "is_count_like",
                "drop_recommended",
                "tags",
            ],
        )
        w.writeheader()
        w.writerows(report)

    drop_n = sum(1 for r in report if r["drop_recommended"] == 1)
    high_shift_n = sum(1 for r in report if r["train_test_shift_z"] >= 2.0)
    print("DONE")
    print("rows:", len(rows), "train:", len(train), "test:", len(test))
    print("features:", len(feat_cols))
    print("drop_recommended:", drop_n)
    print("high_shift_features:", high_shift_n)
    print("out:", OUT_CSV)

if __name__ == "__main__":
    main()
