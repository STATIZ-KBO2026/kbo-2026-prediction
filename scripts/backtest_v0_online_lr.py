import os, csv, math, argparse

IN_CSV = os.path.expanduser("~/statiz/data/features_v0.csv")
OUT_CSV = os.path.expanduser("~/statiz/data/backtest_pred_v0.csv")

EXCLUDE = {
    "date", "s_no", "homeTeam", "awayTeam",
    "y_home_win", "homeScore", "awayScore",
    "home_sp_p_no", "away_sp_p_no",
    "s_code",
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

def load_rows(in_csv):
    with open(in_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
        cols = r.fieldnames or []
    rows.sort(key=lambda x: (x["date"], safe_int(x["s_no"])))
    return rows, cols

def build_online_model(dim):
    return {
        "mean": [0.0] * dim,
        "m2": [0.0] * dim,
        "n_seen": 0,
        "w": [0.0] * dim,
        "b": 0.0,
        "base_lr": 0.05,
        "l2": 1e-4,
        "step": 0,
    }

def current_std(model):
    if model["n_seen"] < 2:
        return [1.0] * len(model["w"])
    return [math.sqrt(model["m2"][i] / (model["n_seen"] - 1) + 1e-9) for i in range(len(model["w"]))]

def standardize(model, x_raw, std):
    return [(x_raw[i] - model["mean"][i]) / std[i] for i in range(len(model["w"]))]

def update_scaler(model, x_raw):
    model["n_seen"] += 1
    n_seen = model["n_seen"]
    for i in range(len(model["w"])):
        delta = x_raw[i] - model["mean"][i]
        model["mean"][i] += delta / n_seen
        delta2 = x_raw[i] - model["mean"][i]
        model["m2"][i] += delta * delta2

def predict_proba(model, x):
    z = model["b"]
    for i in range(len(model["w"])):
        z += model["w"][i] * x[i]
    return sigmoid(z)

def train_one(model, x, y):
    model["step"] += 1
    lr = model["base_lr"] / (1.0 + 0.001 * model["step"])
    p = predict_proba(model, x)
    err = (p - y)
    for i in range(len(model["w"])):
        grad = err * x[i] + model["l2"] * model["w"][i]
        model["w"][i] -= lr * grad
    model["b"] -= lr * err

def run_expanding(rows, feat_cols):
    d = len(feat_cols)
    model = build_online_model(d)

    pred_rows = []
    all_y, all_p = [], []
    total_loss = 0.0
    total_acc = 0
    total_n = 0

    cur_date = None
    day_buf = []  # (date, s_no, y, x_raw)

    def flush_day(buf):
        nonlocal total_loss, total_acc, total_n
        if not buf:
            return
        std = current_std(model)
        day_x = []
        day_y = []
        for date, s_no, y, x_raw in buf:
            x = standardize(model, x_raw, std) if model["n_seen"] >= 2 else x_raw[:]
            p = predict_proba(model, x)
            pred_rows.append({"date": date, "s_no": s_no, "y": y, "p_homewin": p})
            all_y.append(y)
            all_p.append(p)
            total_loss += logloss(y, p)
            total_acc += (1 if ((p >= 0.5) == (y == 1)) else 0)
            total_n += 1
            day_x.append(x)
            day_y.append(y)

        for x, y in zip(day_x, day_y):
            train_one(model, x, y)
        for _, _, _, x_raw in buf:
            update_scaler(model, x_raw)

    for row in rows:
        date = row["date"]
        s_no = row["s_no"]
        y = int(float(row["y_home_win"]))
        x_raw = [safe_float(row[c]) for c in feat_cols]

        if cur_date is None:
            cur_date = date
        if date != cur_date:
            flush_day(day_buf)
            day_buf = []
            cur_date = date
        day_buf.append((date, s_no, y, x_raw))

    flush_day(day_buf)
    return pred_rows, total_n, total_acc, total_loss, all_y, all_p

def run_split(rows, feat_cols, train_start, train_end, test_start, test_end):
    d = len(feat_cols)
    model = build_online_model(d)

    train_rows = [r for r in rows if train_start <= r["date"] <= train_end]
    test_rows = [r for r in rows if test_start <= r["date"] <= test_end]

    for row in train_rows:
        y = int(float(row["y_home_win"]))
        x_raw = [safe_float(row[c]) for c in feat_cols]
        std = current_std(model)
        x = standardize(model, x_raw, std) if model["n_seen"] >= 2 else x_raw[:]
        train_one(model, x, y)
        update_scaler(model, x_raw)

    pred_rows = []
    all_y, all_p = [], []
    total_loss = 0.0
    total_acc = 0
    total_n = 0

    std = current_std(model)
    for row in test_rows:
        date = row["date"]
        s_no = row["s_no"]
        y = int(float(row["y_home_win"]))
        x_raw = [safe_float(row[c]) for c in feat_cols]
        x = standardize(model, x_raw, std) if model["n_seen"] >= 2 else x_raw[:]
        p = predict_proba(model, x)
        pred_rows.append({"date": date, "s_no": s_no, "y": y, "p_homewin": p})
        all_y.append(y)
        all_p.append(p)
        total_loss += logloss(y, p)
        total_acc += (1 if ((p >= 0.5) == (y == 1)) else 0)
        total_n += 1

    return pred_rows, total_n, total_acc, total_loss, all_y, all_p, len(train_rows), len(test_rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["expanding", "split"], default="expanding")
    ap.add_argument("--train-start", default="20240101")
    ap.add_argument("--train-end", default="20241231")
    ap.add_argument("--test-start", default="20250101")
    ap.add_argument("--test-end", default="20251231")
    args = ap.parse_args()

    rows, cols = load_rows(IN_CSV)
    feat_cols = [c for c in cols if c not in EXCLUDE]

    if args.mode == "expanding":
        pred_rows, total_n, total_acc, total_loss, all_y, all_p = run_expanding(rows, feat_cols)
        print("mode: expanding")
    else:
        pred_rows, total_n, total_acc, total_loss, all_y, all_p, tr_n, te_n = run_split(
            rows, feat_cols, args.train_start, args.train_end, args.test_start, args.test_end
        )
        print("mode: split")
        print("train_range:", args.train_start, "~", args.train_end, "train_games:", tr_n)
        print("test_range:", args.test_start, "~", args.test_end, "test_games:", te_n)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=["date", "s_no", "y", "p_homewin"])
        wcsv.writeheader()
        wcsv.writerows(pred_rows)

    acc = total_acc / max(1, total_n)
    ll = total_loss / max(1, total_n)
    auc = auc_score(all_y, all_p)

    print("DONE")
    print("games:", total_n)
    print("accuracy:", round(acc, 4))
    print("logloss:", round(ll, 5))
    print("auc:", round(auc, 4) if auc == auc else "nan")
    print("pred_out:", OUT_CSV)

if __name__ == "__main__":
    main()
