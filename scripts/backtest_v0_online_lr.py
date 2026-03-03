import os, csv, math

IN_CSV  = os.path.expanduser("~/statiz/data/features_v0.csv")
OUT_CSV = os.path.expanduser("~/statiz/data/backtest_pred_v0.csv")

# ✅ baseline에선 "ID성 컬럼"은 빼고, 연속형 피처만 쓴다
EXCLUDE = {
    "date","s_no","homeTeam","awayTeam",
    "y_home_win","homeScore","awayScore",
    "home_sp_p_no","away_sp_p_no",
    "s_code",  # 구장/코드(숫자 카테고리)라 baseline에서는 제외
}

EPS = 1e-12

def safe_float(x):
    try:
        if x is None: return 0.0
        s = str(x).strip()
        if s == "" or s.lower() == "none": return 0.0
        return float(s)
    except:
        return 0.0

def sigmoid(z):
    # overflow-safe sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

def logloss(y, p):
    p = min(max(p, EPS), 1.0 - EPS)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))

def auc_score(y_list, p_list):
    # rank-based AUC with tie handling (O(n log n))
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
        # tie block
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        # average rank for ties
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        # add ranks for positives in this block
        for k in range(i, j):
            if pairs[k][1] == 1:
                pos_rank_sum += avg_rank
        rank += (j - i)
        i = j

    # Mann–Whitney U
    u = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)

def main():
    with open(IN_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames

        feat_cols = [c for c in cols if c not in EXCLUDE]
        # label
        y_col = "y_home_win"

        # online standardization (using training history only)
        d = len(feat_cols)
        mean = [0.0] * d
        m2   = [0.0] * d
        n_seen = 0  # number of training samples incorporated into scaler

        # online logistic regression weights
        w = [0.0] * d
        b = 0.0
        base_lr = 0.05
        l2 = 1e-4
        step = 0

        def current_std():
            if n_seen < 2:
                return [1.0] * d
            return [math.sqrt(m2[i] / (n_seen - 1) + 1e-9) for i in range(d)]

        def standardize(x, std):
            return [(x[i] - mean[i]) / std[i] for i in range(d)]

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

        # process by date (expanding window daily)
        pred_rows = []
        all_y, all_p = [], []
        total_loss = 0.0
        total_acc = 0
        total_n = 0

        cur_date = None
        day_buf = []  # list of (date,s_no,y,x_raw)

        def flush_day(buf):
            nonlocal b, step, total_loss, total_acc, total_n

            if not buf:
                return

            std = current_std()

            # 1) predict using model trained up to previous day
            day_xnorm = []
            day_y = []
            for (date, s_no, y, x_raw) in buf:
                x = standardize(x_raw, std) if n_seen >= 2 else x_raw[:]  # early days: no scaling
                p = predict_proba(x)

                pred_rows.append({"date": date, "s_no": s_no, "y": y, "p_homewin": p})
                all_y.append(y)
                all_p.append(p)

                total_loss += logloss(y, p)
                total_acc += (1 if ((p >= 0.5) == (y == 1)) else 0)
                total_n += 1

                day_xnorm.append(x)
                day_y.append(y)

            # 2) after predicting the day, train on the same day (so next day can use)
            for x, y in zip(day_xnorm, day_y):
                step += 1
                lr = base_lr / (1.0 + 0.001 * step)

                p = predict_proba(x)
                err = (p - y)

                # weights
                for i in range(d):
                    grad = err * x[i] + l2 * w[i]
                    w[i] -= lr * grad
                # bias
                b -= lr * err

            # 3) update scaler with raw x of the day (so next day uses it)
            for (_, _, _, x_raw) in buf:
                update_scaler(x_raw)

        for row in r:
            date = row["date"]
            s_no = row["s_no"]
            y = int(float(row[y_col]))

            x_raw = [safe_float(row[c]) for c in feat_cols]

            if cur_date is None:
                cur_date = date

            if date != cur_date:
                flush_day(day_buf)
                day_buf = []
                cur_date = date

            day_buf.append((date, s_no, y, x_raw))

        # last day
        flush_day(day_buf)

    # write predictions
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=["date","s_no","y","p_homewin"])
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
