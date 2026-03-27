import os
import csv
import math
import random
import argparse

IN_CSV = os.path.expanduser("~/statiz/data/features_v0.csv")
OUT_PRED = os.path.expanduser("~/statiz/data/backtest_pred_v3_best.csv")
OUT_REPORT = os.path.expanduser("~/statiz/data/backtest_v3_model_report.csv")
OUT_DROP = os.path.expanduser("~/statiz/data/backtest_v3_dropped_features.csv")

EXCLUDE = {
    "date",
    "s_no",
    "homeTeam",
    "awayTeam",
    "y_home_win",
    "homeScore",
    "awayScore",
    "home_sp_p_no",
    "away_sp_p_no",
}

COUNTLIKE_HINTS = (
    "_G",
    "_PA",
    "_IP",
    "_AB",
    "_H",
    "_HR",
    "_BB",
    "_SO",
    "_R",
    "_app_",
    "games_last7",
    "consec_days",
    "away_streak",
)

# Features repeatedly identified as no-signal or harmful in two-fold diagnostics.
DROP_ALWAYS = {
    "home_sp_nohist",
    "away_sp_nohist",
    "diff_sp_np_l3d",
    "diff_sp_ip_l3d",
    "home_lineup_nohist_cnt",
    "home_lineup_nohist_ratio",
    "away_lineup_hand3_cnt",
    "diff_lineup_top3_minus_bot6_blend_ops",
    "home_lineup_avg_obp",
    "away_lineup_avg_obp",
    "away_lineup_blend_ops",
    # cold-start helper probabilities are used to shape blend features; raw probs are noisy as direct predictors
    "home_sp_cold_foreign_prob",
    "home_sp_cold_rookie_prob",
    "home_sp_cold_returnee_prob",
    "away_sp_cold_foreign_prob",
    "away_sp_cold_rookie_prob",
    "away_sp_cold_returnee_prob",
    "diff_sp_cold_foreign_prob",
    "home_lineup_cold_foreign_prob_avg",
    "home_lineup_cold_rookie_prob_avg",
    "home_lineup_cold_returnee_prob_avg",
    "away_lineup_cold_foreign_prob_avg",
    "away_lineup_cold_rookie_prob_avg",
    "away_lineup_cold_returnee_prob_avg",
    "diff_lineup_cold_foreign_prob_avg",
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


def metrics(y, p):
    n = len(y)
    ll = sum(logloss(yy, pp) for yy, pp in zip(y, p)) / max(1, n)
    acc = sum(1 for yy, pp in zip(y, p) if ((pp >= 0.5) == (yy == 1))) / max(1, n)
    auc = auc_score(y, p)
    return ll, acc, auc


def is_countlike(col):
    return any(h in col for h in COUNTLIKE_HINTS)


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
        cols = r.fieldnames or []
    rows.sort(key=lambda x: (x["date"], safe_int(x["s_no"])))
    return rows, cols


def label_int(row):
    y_raw = str(row.get("y_home_win", "")).strip()
    if y_raw == "" or y_raw.lower() == "none":
        return None
    try:
        y = int(float(y_raw))
    except Exception:
        return None
    if y not in (0, 1):
        return None
    return y


def has_label(row):
    return label_int(row) is not None


def is_regular_row(row):
    # features_v0 stores league_type_regular (1: regular-season, 0: others)
    v = str(row.get("league_type_regular", "")).strip()
    if v != "":
        try:
            return int(float(v)) == 1
        except Exception:
            pass
    # fallback if raw leagueType exists
    lt = str(row.get("leagueType", "")).strip()
    if lt != "":
        try:
            return int(float(lt)) == 10100
        except Exception:
            pass
    return True


def is_exhibition_row(row):
    # features_v0 stores league_type_exhibition (1: exhibition, 0: others)
    v = str(row.get("league_type_exhibition", "")).strip()
    if v != "":
        try:
            return int(float(v)) == 1
        except Exception:
            pass
    lt = str(row.get("leagueType", "")).strip()
    if lt != "":
        try:
            return int(float(lt)) == 10400
        except Exception:
            pass
    return False


def season_match(row, season_mode):
    if season_mode == "regular":
        return is_regular_row(row)
    if season_mode == "exhibition":
        return is_exhibition_row(row)
    return True


def split_date(rows, start, end, labeled_only=False, regular_only=False, season_mode="all"):
    if regular_only and season_mode == "all":
        season_mode = "regular"
    out = [r for r in rows if start <= r["date"] <= end]
    out = [r for r in out if season_match(r, season_mode)]
    if labeled_only:
        out = [r for r in out if has_label(r)]
    return out


def build_feature_filter(
    fit_rows,
    val_rows,
    feat_cols,
    zero_thr=0.995,
    var_thr=1e-12,
    shift_thr=2.0,
    drop_countlike_shift=True,
):
    kept = []
    dropped = []
    for c in feat_cols:
        if c in DROP_ALWAYS:
            dropped.append(
                {
                    "feature": c,
                    "fit_zero_ratio": 0.0,
                    "fit_variance": 0.0,
                    "fit_val_shift_z": 0.0,
                    "tags": "manual_drop",
                }
            )
            continue
        fit_vals = [safe_float(r[c]) for r in fit_rows]
        val_vals = [safe_float(r[c]) for r in val_rows] if val_rows else fit_vals
        fit_mu = mean(fit_vals)
        val_mu = mean(val_vals)
        fit_var = variance(fit_vals, fit_mu)
        val_var = variance(val_vals, val_mu)
        fit_sd = math.sqrt(max(fit_var, 0.0))
        val_sd = math.sqrt(max(val_var, 0.0))
        pooled = max(fit_sd, val_sd, 1e-9)
        shift_z = abs(fit_mu - val_mu) / pooled
        zero_ratio = (sum(1 for v in fit_vals if abs(v) < 1e-12) / len(fit_vals)) if fit_vals else 1.0
        tags = []
        drop = False
        if fit_var <= var_thr:
            drop = True
            tags.append("constant")
        if zero_ratio >= zero_thr:
            drop = True
            tags.append("near_all_zero")
        if drop_countlike_shift and is_countlike(c) and shift_z >= shift_thr:
            drop = True
            tags.append("countlike_high_shift")
        if drop:
            dropped.append(
                {
                    "feature": c,
                    "fit_zero_ratio": round(zero_ratio, 6),
                    "fit_variance": round(fit_var, 10),
                    "fit_val_shift_z": round(shift_z, 6),
                    "tags": "|".join(tags),
                }
            )
        else:
            kept.append(c)
    return kept, dropped


def make_matrix(rows, feat_cols):
    X = [[safe_float(r[c]) for c in feat_cols] for r in rows]
    y = []
    for r in rows:
        yy = label_int(r)
        if yy is None:
            raise RuntimeError(f"Unlabeled row detected in make_matrix: date={r.get('date')} s_no={r.get('s_no')}")
        y.append(yy)
    return X, y


def fit_standardizer(X):
    if not X:
        return [], []
    d = len(X[0])
    mu = [0.0] * d
    var = [0.0] * d
    n = len(X)
    for j in range(d):
        vals = [x[j] for x in X]
        mu[j] = mean(vals)
        var[j] = variance(vals, mu[j])
    sd = [math.sqrt(v + 1e-9) for v in var]
    return mu, sd


def transform_standardize(X, mu, sd):
    out = []
    for x in X:
        out.append([(x[j] - mu[j]) / sd[j] for j in range(len(x))])
    return out


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


def predict_gbdt(model, X):
    out = []
    for x in X:
        s = model["bias"]
        for stump in model["stumps"]:
            j = stump["feature"]
            s += stump["lr"] * (stump["left_val"] if x[j] <= stump["threshold"] else stump["right_val"])
        out.append(sigmoid(s))
    return out


def fit_lr_batch(X_train, y_train, X_val, y_val, base_lr=0.05, l2=1e-4, epochs=500, patience=40):
    n = len(X_train)
    d = len(X_train[0]) if X_train else 0
    pos = sum(y_train)
    base_p = min(max(pos / max(1, n), 1e-6), 1 - 1e-6)
    b = math.log(base_p / (1 - base_p))
    w = [0.0] * d

    best = {
        "w": list(w),
        "b": b,
        "val_ll": 1e18,
        "epoch": 0,
    }

    no_improve = 0
    for ep in range(1, epochs + 1):
        lr = base_lr / (1.0 + 0.01 * ep)
        grad_w = [0.0] * d
        grad_b = 0.0
        for x, y in zip(X_train, y_train):
            z = b
            for j in range(d):
                z += w[j] * x[j]
            p = sigmoid(z)
            err = p - y
            grad_b += err
            for j in range(d):
                grad_w[j] += err * x[j]
        inv_n = 1.0 / max(1, n)
        for j in range(d):
            grad = grad_w[j] * inv_n + l2 * w[j]
            w[j] -= lr * grad
        b -= lr * (grad_b * inv_n)

        p_val = []
        for x in X_val:
            z = b
            for j in range(d):
                z += w[j] * x[j]
            p_val.append(sigmoid(z))
        val_ll = sum(logloss(y, p) for y, p in zip(y_val, p_val)) / max(1, len(y_val))
        if val_ll < best["val_ll"]:
            best = {"w": list(w), "b": b, "val_ll": val_ll, "epoch": ep}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    return best


def predict_lr(model, X):
    w = model["w"]
    b = model["b"]
    d = len(w)
    out = []
    for x in X:
        z = b
        for j in range(d):
            z += w[j] * x[j]
        out.append(sigmoid(z))
    return out


def fit_gaussian_nb(X_train, y_train, var_smooth=1e-6):
    n = len(X_train)
    d = len(X_train[0]) if X_train else 0
    idx0 = [i for i, y in enumerate(y_train) if y == 0]
    idx1 = [i for i, y in enumerate(y_train) if y == 1]
    n0 = max(1, len(idx0))
    n1 = max(1, len(idx1))
    p1 = min(max(len(idx1) / max(1, n), 1e-6), 1 - 1e-6)

    mu0 = [0.0] * d
    mu1 = [0.0] * d
    var0 = [0.0] * d
    var1 = [0.0] * d

    for j in range(d):
        v0 = [X_train[i][j] for i in idx0] if idx0 else [0.0]
        v1 = [X_train[i][j] for i in idx1] if idx1 else [0.0]
        mu0[j] = mean(v0)
        mu1[j] = mean(v1)
        var0[j] = variance(v0, mu0[j]) + var_smooth
        var1[j] = variance(v1, mu1[j]) + var_smooth

    return {"p1": p1, "mu0": mu0, "mu1": mu1, "var0": var0, "var1": var1}


def predict_gaussian_nb(model, X):
    p1 = model["p1"]
    mu0 = model["mu0"]
    mu1 = model["mu1"]
    var0 = model["var0"]
    var1 = model["var1"]
    d = len(mu0)
    log_prior1 = math.log(p1)
    log_prior0 = math.log(1.0 - p1)
    out = []
    for x in X:
        l0 = log_prior0
        l1 = log_prior1
        for j in range(d):
            v = x[j]
            vv0 = var0[j]
            vv1 = var1[j]
            l0 += -0.5 * math.log(2.0 * math.pi * vv0) - ((v - mu0[j]) ** 2) / (2.0 * vv0)
            l1 += -0.5 * math.log(2.0 * math.pi * vv1) - ((v - mu1[j]) ** 2) / (2.0 * vv1)
        m = max(l0, l1)
        p_1 = math.exp(l1 - m)
        p_0 = math.exp(l0 - m)
        out.append(p_1 / (p_1 + p_0))
    return out


def class_entropy(pos, total):
    if total <= 0:
        return 0.0
    p = min(max(pos / total, 1e-12), 1.0 - 1e-12)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def sample_feature_indices(d, mtry, rnd):
    if mtry >= d:
        return list(range(d))
    pool = list(range(d))
    rnd.shuffle(pool)
    return pool[:mtry]


def build_thresholds(vals, max_bins):
    n = len(vals)
    if n < 2:
        return []
    sorted_vals = sorted(vals)
    out = []
    for p in quantile_positions(n, max_bins):
        a = sorted_vals[p - 1]
        b = sorted_vals[p]
        if a < b:
            out.append((a + b) / 2.0)
    return out


def best_split_entropy(X, y, idxs, feat_ids, min_leaf, max_bins):
    best = None
    n = len(idxs)
    if n < 2 * min_leaf:
        return None
    for j in feat_ids:
        vals = [X[i][j] for i in idxs]
        thrs = build_thresholds(vals, max_bins)
        if not thrs:
            continue
        for thr in thrs:
            l_idx = []
            r_idx = []
            l_pos = 0
            r_pos = 0
            for i in idxs:
                if X[i][j] <= thr:
                    l_idx.append(i)
                    l_pos += y[i]
                else:
                    r_idx.append(i)
                    r_pos += y[i]
            nl = len(l_idx)
            nr = len(r_idx)
            if nl < min_leaf or nr < min_leaf:
                continue
            loss = (nl * class_entropy(l_pos, nl) + nr * class_entropy(r_pos, nr)) / n
            if best is None or loss < best["loss"]:
                best = {
                    "feature": j,
                    "threshold": thr,
                    "left_idxs": l_idx,
                    "right_idxs": r_idx,
                    "left_prob": l_pos / nl,
                    "right_prob": r_pos / nr,
                    "loss": loss,
                }
    return best


def fit_rf_tree(X, y, rnd, max_depth=3, min_leaf=16, mtry=16, max_bins=16):
    d = len(X[0]) if X else 0
    n = len(X)
    if n == 0 or d == 0:
        return {"leaf": True, "prob": 0.5}
    root_idxs = [rnd.randrange(0, n) for _ in range(n)]

    def build_node(idxs, depth):
        pos = sum(y[i] for i in idxs)
        total = len(idxs)
        prob = pos / max(1, total)
        if depth >= max_depth or total < 2 * min_leaf:
            return {"leaf": True, "prob": prob}
        feat_ids = sample_feature_indices(d, mtry, rnd)
        split = best_split_entropy(X, y, idxs, feat_ids, min_leaf, max_bins)
        if split is None:
            return {"leaf": True, "prob": prob}
        return {
            "leaf": False,
            "feature": split["feature"],
            "threshold": split["threshold"],
            "left": build_node(split["left_idxs"], depth + 1),
            "right": build_node(split["right_idxs"], depth + 1),
            "prob": prob,
        }

    return build_node(root_idxs, 0)


def predict_rf_tree(tree, x):
    node = tree
    while not node["leaf"]:
        if x[node["feature"]] <= node["threshold"]:
            node = node["left"]
        else:
            node = node["right"]
    return node["prob"]


def fit_random_forest(X_train, y_train, n_trees=300, max_depth=3, min_leaf=16, mtry=16, max_bins=16, seed=2026):
    rnd = random.Random(seed)
    trees = []
    for _ in range(n_trees):
        trees.append(
            fit_rf_tree(
                X_train,
                y_train,
                rnd=rnd,
                max_depth=max_depth,
                min_leaf=min_leaf,
                mtry=mtry,
                max_bins=max_bins,
            )
        )
    return {"trees": trees}


def predict_random_forest(model, X):
    trees = model["trees"]
    nt = max(1, len(trees))
    out = []
    for x in X:
        s = 0.0
        for t in trees:
            s += predict_rf_tree(t, x)
        out.append(s / nt)
    return out


def fit_blend_greedy(val_pred_map, y_val, rounds=100):
    names = list(val_pred_map.keys())
    if not names:
        return {}
    counts = {n: 0 for n in names}
    denom = 0
    best_ll = 1e18
    for _ in range(rounds):
        best_name = None
        best_cur_ll = 1e18
        for n in names:
            c = dict(counts)
            c[n] += 1
            d = denom + 1
            p = []
            for i in range(len(y_val)):
                s = 0.0
                for k in names:
                    if c[k] > 0:
                        s += (c[k] / d) * val_pred_map[k][i]
                p.append(s)
            ll = sum(logloss(y, pp) for y, pp in zip(y_val, p)) / max(1, len(y_val))
            if ll < best_cur_ll:
                best_cur_ll = ll
                best_name = n
        if best_name is None:
            break
        counts[best_name] += 1
        denom += 1
        best_ll = best_cur_ll
    if denom == 0:
        best = min(names, key=lambda n: sum(logloss(y, p) for y, p in zip(y_val, val_pred_map[n])) / max(1, len(y_val)))
        return {best: 1.0}
    return {k: (v / denom) for k, v in counts.items() if v > 0}


def blend_predict(pred_map, weights):
    names = list(weights.keys())
    if not names:
        return []
    n = len(pred_map[names[0]])
    out = []
    for i in range(n):
        s = 0.0
        for name in names:
            s += weights[name] * pred_map[name][i]
        out.append(s)
    return out


def report_add(report_rows, name, split_name, y, p, extra=None):
    ll, acc, auc = metrics(y, p)
    row = {
        "model": name,
        "split": split_name,
        "logloss": round(ll, 6),
        "accuracy": round(acc, 6),
        "auc": round(auc, 6) if auc == auc else "",
    }
    if extra:
        row.update(extra)
    report_rows.append(row)
    return ll, acc, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-start", default="20240101")
    ap.add_argument("--train-end", default="20241231")
    ap.add_argument("--test-start", default="20250101")
    ap.add_argument("--test-end", default="20251231")
    ap.add_argument("--fit-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--blend-rounds", type=int, default=120)
    args = ap.parse_args()

    random.seed(args.seed)

    rows, cols = load_rows(IN_CSV)
    feat_cols_all = [c for c in cols if c not in EXCLUDE]
    train_rows = split_date(rows, args.train_start, args.train_end, labeled_only=True)
    test_rows = split_date(rows, args.test_start, args.test_end, labeled_only=True)
    if len(train_rows) < 200 or len(test_rows) < 100:
        raise RuntimeError("Not enough rows for train/test split")

    cut = int(len(train_rows) * args.fit_ratio)
    cut = min(max(cut, 100), len(train_rows) - 50)
    fit_rows = train_rows[:cut]
    val_rows = train_rows[cut:]

    feat_cols, dropped = build_feature_filter(fit_rows, val_rows, feat_cols_all)

    X_fit_raw, y_fit = make_matrix(fit_rows, feat_cols)
    X_val_raw, y_val = make_matrix(val_rows, feat_cols)
    X_train_raw, y_train = make_matrix(train_rows, feat_cols)
    X_test_raw, y_test = make_matrix(test_rows, feat_cols)

    mu_fit, sd_fit = fit_standardizer(X_fit_raw)
    X_fit_std = transform_standardize(X_fit_raw, mu_fit, sd_fit)
    X_val_std = transform_standardize(X_val_raw, mu_fit, sd_fit)

    mu_train, sd_train = fit_standardizer(X_train_raw)
    X_train_std = transform_standardize(X_train_raw, mu_train, sd_train)
    X_test_std = transform_standardize(X_test_raw, mu_train, sd_train)

    report_rows = []
    val_pred = {}
    test_pred = {}

    # 1) LR batch
    lr_grid = [
        {"base_lr": 0.03, "l2": 1e-4, "epochs": 500},
        {"base_lr": 0.05, "l2": 1e-4, "epochs": 500},
        {"base_lr": 0.08, "l2": 5e-4, "epochs": 700},
        {"base_lr": 0.03, "l2": 1e-3, "epochs": 700},
    ]
    best_lr_cfg = None
    best_lr_model = None
    best_lr_ll = 1e18
    for cfg in lr_grid:
        model = fit_lr_batch(X_fit_std, y_fit, X_val_std, y_val, patience=50, **cfg)
        p = predict_lr(model, X_val_std)
        ll, _, _ = metrics(y_val, p)
        if ll < best_lr_ll:
            best_lr_ll = ll
            best_lr_cfg = cfg
            best_lr_model = model
    val_pred["lr_batch"] = predict_lr(best_lr_model, X_val_std)
    report_add(report_rows, "lr_batch", "val", y_val, val_pred["lr_batch"], extra={"params": str(best_lr_cfg)})

    model_lr_full = fit_lr_batch(X_train_std, y_train, X_val_std, y_val, patience=50, **best_lr_cfg)
    test_pred["lr_batch"] = predict_lr(model_lr_full, X_test_std)
    report_add(report_rows, "lr_batch", "test", y_test, test_pred["lr_batch"], extra={"params": str(best_lr_cfg)})

    # 2) GBDT stump
    gbdt_grid = [
        {"n_estimators": 300, "lr": 0.05, "min_leaf": 12, "max_bins": 16, "early_stopping_rounds": 35},
        {"n_estimators": 400, "lr": 0.05, "min_leaf": 16, "max_bins": 20, "early_stopping_rounds": 40},
        {"n_estimators": 500, "lr": 0.03, "min_leaf": 12, "max_bins": 24, "early_stopping_rounds": 45},
    ]
    best_gbdt_cfg = None
    best_gbdt_round = 0
    best_gbdt_ll = 1e18
    for cfg in gbdt_grid:
        model, ll, rounds = fit_gbdt_stump(X_fit_raw, y_fit, X_val_raw, y_val, **cfg)
        if ll < best_gbdt_ll:
            best_gbdt_ll = ll
            best_gbdt_cfg = cfg
            best_gbdt_round = rounds
    model_gbdt_val, _, _ = fit_gbdt_stump(
        X_fit_raw,
        y_fit,
        X_val_raw,
        y_val,
        n_estimators=max(1, best_gbdt_round),
        lr=best_gbdt_cfg["lr"],
        min_leaf=best_gbdt_cfg["min_leaf"],
        max_bins=best_gbdt_cfg["max_bins"],
        early_stopping_rounds=best_gbdt_cfg["early_stopping_rounds"],
    )
    val_pred["gbdt_stump"] = predict_gbdt(model_gbdt_val, X_val_raw)
    report_add(
        report_rows,
        "gbdt_stump",
        "val",
        y_val,
        val_pred["gbdt_stump"],
        extra={"params": str(best_gbdt_cfg), "best_round": best_gbdt_round},
    )

    model_gbdt_full, _, _ = fit_gbdt_stump(
        X_train_raw,
        y_train,
        X_val_raw,
        y_val,
        n_estimators=max(1, best_gbdt_round),
        lr=best_gbdt_cfg["lr"],
        min_leaf=best_gbdt_cfg["min_leaf"],
        max_bins=best_gbdt_cfg["max_bins"],
        early_stopping_rounds=best_gbdt_cfg["early_stopping_rounds"],
    )
    test_pred["gbdt_stump"] = predict_gbdt(model_gbdt_full, X_test_raw)
    report_add(
        report_rows,
        "gbdt_stump",
        "test",
        y_test,
        test_pred["gbdt_stump"],
        extra={"params": str(best_gbdt_cfg), "best_round": best_gbdt_round},
    )

    # 3) Random forest (pure python)
    d = len(feat_cols)
    rf_grid = [
        {"n_trees": 260, "max_depth": 3, "min_leaf": 12, "mtry": max(10, int(math.sqrt(d))), "max_bins": 12},
        {"n_trees": 360, "max_depth": 4, "min_leaf": 12, "mtry": max(12, int(math.sqrt(d) * 1.4)), "max_bins": 12},
        {"n_trees": 420, "max_depth": 4, "min_leaf": 16, "mtry": max(12, int(math.sqrt(d))), "max_bins": 16},
    ]
    best_rf_cfg = None
    best_rf_ll = 1e18
    for idx, cfg in enumerate(rf_grid):
        model = fit_random_forest(X_fit_raw, y_fit, seed=args.seed + 100 + idx, **cfg)
        p = predict_random_forest(model, X_val_raw)
        ll, _, _ = metrics(y_val, p)
        if ll < best_rf_ll:
            best_rf_ll = ll
            best_rf_cfg = cfg
    model_rf_val = fit_random_forest(X_fit_raw, y_fit, seed=args.seed + 199, **best_rf_cfg)
    val_pred["random_forest"] = predict_random_forest(model_rf_val, X_val_raw)
    report_add(report_rows, "random_forest", "val", y_val, val_pred["random_forest"], extra={"params": str(best_rf_cfg)})

    model_rf_full = fit_random_forest(X_train_raw, y_train, seed=args.seed + 299, **best_rf_cfg)
    test_pred["random_forest"] = predict_random_forest(model_rf_full, X_test_raw)
    report_add(report_rows, "random_forest", "test", y_test, test_pred["random_forest"], extra={"params": str(best_rf_cfg)})

    # 4) Gaussian NB
    nb_grid = [
        {"var_smooth": 1e-6},
        {"var_smooth": 1e-5},
        {"var_smooth": 1e-4},
        {"var_smooth": 1e-3},
    ]
    best_nb_cfg = None
    best_nb_ll = 1e18
    for cfg in nb_grid:
        model = fit_gaussian_nb(X_fit_std, y_fit, **cfg)
        p = predict_gaussian_nb(model, X_val_std)
        ll, _, _ = metrics(y_val, p)
        if ll < best_nb_ll:
            best_nb_ll = ll
            best_nb_cfg = cfg
    model_nb_val = fit_gaussian_nb(X_fit_std, y_fit, **best_nb_cfg)
    val_pred["gaussian_nb"] = predict_gaussian_nb(model_nb_val, X_val_std)
    report_add(report_rows, "gaussian_nb", "val", y_val, val_pred["gaussian_nb"], extra={"params": str(best_nb_cfg)})

    model_nb_full = fit_gaussian_nb(X_train_std, y_train, **best_nb_cfg)
    test_pred["gaussian_nb"] = predict_gaussian_nb(model_nb_full, X_test_std)
    report_add(report_rows, "gaussian_nb", "test", y_test, test_pred["gaussian_nb"], extra={"params": str(best_nb_cfg)})

    # 5) Greedy blend
    blend_weights = fit_blend_greedy(val_pred, y_val, rounds=args.blend_rounds)
    p_val_blend = blend_predict(val_pred, blend_weights)
    report_add(report_rows, "blend_greedy", "val", y_val, p_val_blend, extra={"weights": str(blend_weights)})

    p_test_blend = blend_predict(test_pred, blend_weights)
    ll_blend, acc_blend, auc_blend = report_add(
        report_rows,
        "blend_greedy",
        "test",
        y_test,
        p_test_blend,
        extra={"weights": str(blend_weights)},
    )

    pred_rows = []
    for row, y, p in zip(test_rows, y_test, p_test_blend):
        pred_rows.append(
            {
                "date": row["date"],
                "s_no": row["s_no"],
                "y": y,
                "p_homewin": p,
            }
        )

    os.makedirs(os.path.dirname(OUT_PRED), exist_ok=True)
    with open(OUT_PRED, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "s_no", "y", "p_homewin"])
        w.writeheader()
        w.writerows(pred_rows)

    with open(OUT_REPORT, "w", newline="", encoding="utf-8") as f:
        cols_out = ["model", "split", "logloss", "accuracy", "auc", "params", "best_round", "weights"]
        w = csv.DictWriter(f, fieldnames=cols_out)
        w.writeheader()
        for r in report_rows:
            w.writerow(r)

    with open(OUT_DROP, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "fit_zero_ratio", "fit_variance", "fit_val_shift_z", "tags"])
        w.writeheader()
        w.writerows(sorted(dropped, key=lambda x: (x["tags"], -x["fit_val_shift_z"], x["feature"])))

    print("DONE")
    print("mode: split_model_zoo")
    print("train_range:", args.train_start, "~", args.train_end, "train_games:", len(train_rows))
    print("test_range:", args.test_start, "~", args.test_end, "test_games:", len(test_rows))
    print("total_features:", len(feat_cols_all))
    print("used_features:", len(feat_cols))
    print("dropped_features:", len(dropped))
    print("blend_weights:", blend_weights)
    print("test_accuracy:", round(acc_blend, 4))
    print("test_logloss:", round(ll_blend, 5))
    print("test_auc:", round(auc_blend, 4) if auc_blend == auc_blend else "nan")
    print("pred_out:", OUT_PRED)
    print("report_out:", OUT_REPORT)
    print("drop_out:", OUT_DROP)


if __name__ == "__main__":
    main()
