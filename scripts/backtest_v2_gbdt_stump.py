import os, csv, math, argparse

IN_CSV = os.path.expanduser("~/statiz/data/features_v0.csv")
OUT_CSV = os.path.expanduser("~/statiz/data/backtest_pred_v2_gbdt.csv")

EXCLUDE = {
    "date", "s_no", "homeTeam", "awayTeam",
    "y_home_win", "homeScore", "awayScore",
    "home_sp_p_no", "away_sp_p_no",
}

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

def safe_int(x):
    try:
        if x is None:
            return 0
        s = str(x).strip()
        if s == "" or s.lower() == "none":
            return 0
        return int(float(s))
    except Exception:
        return 0

def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)

def logloss(y, p):
    eps = 1e-12
    p = min(max(p, eps), 1.0 - eps)
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

def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0

def variance(vals, mu=None):
    if not vals:
        return 0.0
    if mu is None:
        mu = mean(vals)
    return sum((v - mu) * (v - mu) for v in vals) / len(vals)

def filter_features(train_rows, feat_cols, zero_thr=0.995, var_thr=1e-12):
    keep = []
    dropped = []
    for c in feat_cols:
        vals = [safe_float(r[c]) for r in train_rows]
        z = (sum(1 for v in vals if abs(v) < 1e-12) / len(vals)) if vals else 1.0
        v = variance(vals)
        if z >= zero_thr or v <= var_thr:
            dropped.append((c, z, v))
        else:
            keep.append(c)
    return keep, dropped

def prepare_xy(rows, feat_cols):
    X = [[safe_float(r[c]) for c in feat_cols] for r in rows]
    y = [int(float(r["y_home_win"])) for r in rows]
    return X, y

def quantile_positions(n, max_bins):
    pos = set()
    for b in range(1, max_bins):
        p = int(n * b / max_bins)
        if 1 <= p <= n - 1:
            pos.add(p)
    return sorted(pos)

def build_feature_cache(X, max_bins):
    n = len(X)
    d = len(X[0]) if n else 0
    cache = []
    for j in range(d):
        order = sorted(range(n), key=lambda i: X[i][j])
        vals = [X[i][j] for i in order]
        cand = []
        for p in quantile_positions(n, max_bins):
            if vals[p - 1] < vals[p]:
                cand.append(p)
        cache.append((order, vals, cand))
    return cache

def fit_stump_from_residual(resid, feature_cache, min_leaf):
    n = len(resid)
    best = {
        "feature": -1,
        "threshold": 0.0,
        "left_val": 0.0,
        "right_val": 0.0,
        "gain": -1e18,
    }
    for j, (order, vals, cand_pos) in enumerate(feature_cache):
        if not cand_pos:
            continue
        r_sorted = [resid[i] for i in order]
        prefix = [0.0] * n
        s = 0.0
        for i, rv in enumerate(r_sorted):
            s += rv
            prefix[i] = s
        total = prefix[-1]

        for p in cand_pos:
            n_l = p
            n_r = n - p
            if n_l < min_leaf or n_r < min_leaf:
                continue
            sum_l = prefix[p - 1]
            sum_r = total - sum_l
            gain = (sum_l * sum_l / n_l) + (sum_r * sum_r / n_r)
            if gain > best["gain"]:
                thr = (vals[p - 1] + vals[p]) / 2.0
                best = {
                    "feature": j,
                    "threshold": thr,
                    "left_val": (sum_l / n_l),
                    "right_val": (sum_r / n_r),
                    "gain": gain,
                }
    return best

def predict_raw(model, x):
    s = model["bias"]
    for stump in model["stumps"]:
        j = stump["feature"]
        s += stump["lr"] * (stump["left_val"] if x[j] <= stump["threshold"] else stump["right_val"])
    return s

def predict_proba(model, x):
    return sigmoid(predict_raw(model, x))

def fit_gbdt_stump(X_train, y_train, X_val, y_val, n_estimators=300, lr=0.05, min_leaf=20, max_bins=16, early_stopping_rounds=30):
    n = len(X_train)
    pos = sum(y_train)
    base_p = min(max(pos / max(1, n), 1e-6), 1 - 1e-6)
    bias = math.log(base_p / (1 - base_p))
    model = {"bias": bias, "stumps": []}

    f_train = [bias] * n
    f_val = [bias] * len(X_val)

    cache = build_feature_cache(X_train, max_bins)
    best_val = 1e18
    best_round = -1
    best_stumps = []

    for t in range(n_estimators):
        p_train = [sigmoid(z) for z in f_train]
        resid = [y - p for y, p in zip(y_train, p_train)]

        stump = fit_stump_from_residual(resid, cache, min_leaf=min_leaf)
        if stump["feature"] < 0:
            break
        stump["lr"] = lr
        model["stumps"].append(stump)

        j = stump["feature"]
        thr = stump["threshold"]
        lv = stump["left_val"]
        rv = stump["right_val"]

        for i, x in enumerate(X_train):
            f_train[i] += lr * (lv if x[j] <= thr else rv)
        for i, x in enumerate(X_val):
            f_val[i] += lr * (lv if x[j] <= thr else rv)

        p_val = [sigmoid(z) for z in f_val]
        ll = sum(logloss(y, p) for y, p in zip(y_val, p_val)) / max(1, len(y_val))
        if ll < best_val:
            best_val = ll
            best_round = t
            best_stumps = list(model["stumps"])
        elif t - best_round >= early_stopping_rounds:
            break

    model["stumps"] = best_stumps
    return model, best_val, len(best_stumps)

def metrics(y, p):
    n = len(y)
    ll = sum(logloss(yy, pp) for yy, pp in zip(y, p)) / max(1, n)
    acc = sum(1 for yy, pp in zip(y, p) if ((pp >= 0.5) == (yy == 1))) / max(1, n)
    auc = auc_score(y, p)
    return ll, acc, auc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-start", default="20240101")
    ap.add_argument("--train-end", default="20241231")
    ap.add_argument("--test-start", default="20250101")
    ap.add_argument("--test-end", default="20251231")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--min-leaf", type=int, default=20)
    ap.add_argument("--max-bins", type=int, default=16)
    ap.add_argument("--early-stopping-rounds", type=int, default=30)
    args = ap.parse_args()

    with open(IN_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
        cols = r.fieldnames or []
    rows.sort(key=lambda x: (x["date"], safe_int(x["s_no"])))

    feat_cols = [c for c in cols if c not in EXCLUDE]
    train_rows = [r for r in rows if args.train_start <= r["date"] <= args.train_end]
    test_rows = [r for r in rows if args.test_start <= r["date"] <= args.test_end]
    if len(train_rows) < 50 or len(test_rows) < 50:
        raise RuntimeError("Not enough rows for train/test split")

    # feature filtering using train only
    feat_cols, dropped = filter_features(train_rows, feat_cols)

    # chronological validation split inside train
    cut = int(len(train_rows) * 0.8)
    fit_rows = train_rows[:cut]
    val_rows = train_rows[cut:]

    X_fit, y_fit = prepare_xy(fit_rows, feat_cols)
    X_val, y_val = prepare_xy(val_rows, feat_cols)
    X_train_all, y_train_all = prepare_xy(train_rows, feat_cols)
    X_test, y_test = prepare_xy(test_rows, feat_cols)

    model_cv, best_val, best_round = fit_gbdt_stump(
        X_fit, y_fit, X_val, y_val,
        n_estimators=args.n_estimators,
        lr=args.lr,
        min_leaf=args.min_leaf,
        max_bins=args.max_bins,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    # retrain on full train with chosen rounds
    model, _, _ = fit_gbdt_stump(
        X_train_all, y_train_all, X_val, y_val,
        n_estimators=max(1, best_round),
        lr=args.lr,
        min_leaf=args.min_leaf,
        max_bins=args.max_bins,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    p_test = [predict_proba(model, x) for x in X_test]
    ll, acc, auc = metrics(y_test, p_test)

    pred_rows = []
    for row, y, p in zip(test_rows, y_test, p_test):
        pred_rows.append({
            "date": row["date"],
            "s_no": row["s_no"],
            "y": y,
            "p_homewin": p,
        })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "s_no", "y", "p_homewin"])
        w.writeheader()
        w.writerows(pred_rows)

    print("DONE")
    print("mode: split_gbdt_stump")
    print("train_range:", args.train_start, "~", args.train_end, "train_games:", len(train_rows))
    print("test_range:", args.test_start, "~", args.test_end, "test_games:", len(test_rows))
    print("total_features:", len([c for c in cols if c not in EXCLUDE]))
    print("used_features:", len(feat_cols))
    print("dropped_features:", len(dropped))
    print("best_val_logloss:", round(best_val, 6), "best_rounds:", best_round)
    print("test_accuracy:", round(acc, 4))
    print("test_logloss:", round(ll, 5))
    print("test_auc:", round(auc, 4) if auc == auc else "nan")
    print("pred_out:", OUT_CSV)

if __name__ == "__main__":
    main()
