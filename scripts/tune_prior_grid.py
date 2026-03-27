import os, csv, math, itertools, argparse, subprocess

PRED_CSV = os.path.expanduser("~/statiz/data/backtest_pred_v0.csv")

GRID = {
    "PREV_YEAR_W": [0.4, 0.8, 1.2],
    "CAREER_W": [0.2, 0.5, 0.8],
    "BAT_PRIOR_PA": [40, 80, 120],
    "PIT_PRIOR_IP": [15, 30, 45],
}

def eval_pred(path):
    eps = 1e-12
    y, p = [], []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            yy = int(float(row["y"]))
            pp = float(row["p_homewin"])
            pp = min(max(pp, eps), 1.0 - eps)
            y.append(yy)
            p.append(pp)

    n = len(y)
    ll = sum(-(yy * math.log(pp) + (1 - yy) * math.log(1 - pp)) for yy, pp in zip(y, p)) / max(1, n)
    acc = sum(1 if ((pp >= 0.5) == (yy == 1)) else 0 for yy, pp in zip(y, p)) / max(1, n)

    pairs = sorted(zip(p, y), key=lambda x: x[0])
    n_pos = sum(y)
    n_neg = n - n_pos
    auc = float("nan")
    if n_pos > 0 and n_neg > 0:
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
        auc = u / (n_pos * n_neg)

    return ll, acc, auc

def run_once(params):
    env = os.environ.copy()
    env.update({k: str(v) for k, v in params.items()})
    p1 = subprocess.run(["python3", "scripts/build_features_v0.py"], env=env, capture_output=True, text=True)
    if p1.returncode != 0:
        raise RuntimeError("build_features_v0.py failed")
    p2 = subprocess.run(["python3", "scripts/backtest_v0_online_lr.py"], env=env, capture_output=True, text=True)
    if p2.returncode != 0:
        raise RuntimeError("backtest_v0_online_lr.py failed")
    return eval_pred(PRED_CSV)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    keys = list(GRID.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[GRID[k] for k in keys])]

    rows = []
    for i, params in enumerate(combos, 1):
        ll, acc, auc = run_once(params)
        rows.append((ll, acc, auc, params))
        print(f"[{i}/{len(combos)}] ll={ll:.10f} acc={acc:.6f} auc={auc:.6f} {params}")

    rows.sort(key=lambda x: x[0])
    print("\nTOP")
    for ll, acc, auc, params in rows[:args.topk]:
        print(f"ll={ll:.10f} acc={acc:.6f} auc={auc:.6f} {params}")

if __name__ == "__main__":
    main()
