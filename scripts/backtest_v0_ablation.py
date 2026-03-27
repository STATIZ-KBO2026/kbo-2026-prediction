import os, csv, math

IN_CSV = os.path.expanduser("~/statiz/data/features_v0.csv")
OUT_CSV = os.path.expanduser("~/statiz/data/backtest_ablation_v0.csv")

EXCLUDE = {
    "date", "s_no", "homeTeam", "awayTeam",
    "y_home_win", "homeScore", "awayScore",
    "home_sp_p_no", "away_sp_p_no",
    "s_code",
}

FAMILY_KEYWORDS = {
    "prior": ["_prior_", "_blend_"],
    "rolling": ["_r3_", "_r5_", "_r7_", "_r14_", "_l1d", "_l3d"],
    "schedule": ["rest_days", "games_last7", "consec_days", "away_streak", "_bp_"],
    "matchup": ["samehand", "top3_", "_hand1_", "_hand2_", "_hand3_"],
    "park": ["park_run_factor"],
    "opponent": ["oppwp"],
    "uncertainty": ["nohist", "missing", "known", "throw_unknown"],
}

EPS = 1e-12

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

def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)

def logloss(y, p):
    p = min(max(p, EPS), 1.0 - EPS)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))

def auc_score(y_list, p_list):
    pairs = sorted(zip(p_list, y_list), key=lambda x: x[0])
    n = len(pairs)
    n_pos = sum(y_list)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    rank = 1
    pos_rank_sum = 0.0
    i = 0
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                pos_rank_sum += avg_rank
        rank += (j - i)
        i = j

    u = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)

def has_any_keyword(col, keywords):
    return any(k in col for k in keywords)

def run_online_backtest(rows, feat_cols):
    d = len(feat_cols)
    mean = [0.0] * d
    m2 = [0.0] * d
    n_seen = 0

    w = [0.0] * d
    b = 0.0
    base_lr = 0.05
    l2 = 1e-4
    step = 0

    def current_std():
        if n_seen < 2:
            return [1.0] * d
        return [math.sqrt(m2[i] / (n_seen - 1) + 1e-9) for i in range(d)]

    def standardize(x_raw, std):
        return [(x_raw[i] - mean[i]) / std[i] for i in range(d)]

    def update_scaler(x_raw):
        nonlocal n_seen
        n_seen += 1
        for i in range(d):
            delta = x_raw[i] - mean[i]
            mean[i] += delta / n_seen
            delta2 = x_raw[i] - mean[i]
            m2[i] += delta * delta2

    def predict_proba(x):
        z = b
        for i in range(d):
            z += w[i] * x[i]
        return sigmoid(z)

    total_loss = 0.0
    total_acc = 0
    total_n = 0
    all_y = []
    all_p = []

    cur_date = None
    day_buf = []

    def flush_day(buf):
        nonlocal b, step, total_loss, total_acc, total_n
        if not buf:
            return
        std = current_std()
        day_x = []
        day_y = []
        for _, y, x_raw in buf:
            x = standardize(x_raw, std) if n_seen >= 2 else x_raw[:]
            p = predict_proba(x)
            all_y.append(y)
            all_p.append(p)
            total_loss += logloss(y, p)
            total_acc += (1 if ((p >= 0.5) == (y == 1)) else 0)
            total_n += 1
            day_x.append(x)
            day_y.append(y)

        for x, y in zip(day_x, day_y):
            step += 1
            lr = base_lr / (1.0 + 0.001 * step)
            p = predict_proba(x)
            err = (p - y)
            for i in range(d):
                grad = err * x[i] + l2 * w[i]
                w[i] -= lr * grad
            b -= lr * err

        for _, _, x_raw in buf:
            update_scaler(x_raw)

    for row in rows:
        date = row["date"]
        y = int(float(row["y_home_win"]))
        x_raw = [safe_float(row[c]) for c in feat_cols]
        if cur_date is None:
            cur_date = date
        if date != cur_date:
            flush_day(day_buf)
            day_buf = []
            cur_date = date
        day_buf.append((date, y, x_raw))
    flush_day(day_buf)

    return {
        "games": total_n,
        "accuracy": (total_acc / max(1, total_n)),
        "logloss": (total_loss / max(1, total_n)),
        "auc": auc_score(all_y, all_p),
    }

def main():
    with open(IN_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
        cols = r.fieldnames or []

    base_cols = [c for c in cols if c not in EXCLUDE]

    scenarios = [("all", [])]
    for fam in FAMILY_KEYWORDS:
        scenarios.append((f"minus_{fam}", [fam]))

    out_rows = []
    for name, remove_fams in scenarios:
        remove_keywords = []
        for fam in remove_fams:
            remove_keywords.extend(FAMILY_KEYWORDS[fam])
        feat_cols = [c for c in base_cols if not has_any_keyword(c, remove_keywords)]
        m = run_online_backtest(rows, feat_cols)
        out_rows.append({
            "scenario": name,
            "removed_families": ",".join(remove_fams),
            "n_features": len(feat_cols),
            "games": m["games"],
            "accuracy": round(m["accuracy"], 6),
            "logloss": round(m["logloss"], 6),
            "auc": round(m["auc"], 6) if m["auc"] == m["auc"] else "nan",
        })
        print(name, "features=", len(feat_cols), "logloss=", round(m["logloss"], 6), "auc=", round(m["auc"], 6) if m["auc"] == m["auc"] else "nan")

    out_rows.sort(key=lambda x: x["logloss"])
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "removed_families", "n_features", "games", "accuracy", "logloss", "auc"])
        w.writeheader()
        w.writerows(out_rows)

    print("DONE", "out=", OUT_CSV)

if __name__ == "__main__":
    main()
