import os
import sys
import csv
import math
import argparse

sys.path.append(os.path.dirname(__file__))
import backtest_v3_model_zoo as core

OUT_PRED = os.path.expanduser("~/statiz/data/backtest_pred_v4_champion_rf.csv")
OUT_USAGE = os.path.expanduser("~/statiz/data/backtest_v4_rf_feature_usage.csv")


def collect_split_usage(tree, counts):
    if tree.get("leaf", False):
        return
    j = tree["feature"]
    counts[j] = counts.get(j, 0) + 1
    collect_split_usage(tree["left"], counts)
    collect_split_usage(tree["right"], counts)


def feature_usage(model, feat_cols):
    counts = {}
    for t in model["trees"]:
        collect_split_usage(t, counts)
    rows = []
    total = sum(counts.values())
    for j, c in counts.items():
        rows.append(
            {
                "feature": feat_cols[j],
                "split_count": c,
                "split_ratio": (c / total) if total > 0 else 0.0,
            }
        )
    rows.sort(key=lambda x: (-x["split_count"], x["feature"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-start", default="20240101")
    ap.add_argument("--train-end", default="20241231")
    ap.add_argument("--test-start", default="20250101")
    ap.add_argument("--test-end", default="20251231")
    ap.add_argument("--fit-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-trees", type=int, default=280)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--min-leaf", type=int, default=10)
    ap.add_argument("--mtry-mult", type=float, default=1.0)
    ap.add_argument("--max-bins", type=int, default=16)
    ap.add_argument("--season-mode", choices=["regular", "exhibition", "all"], default="regular")
    ap.add_argument("--include-non-regular", action="store_true", help="include exhibition/postseason rows")
    args = ap.parse_args()

    season_mode = args.season_mode
    if args.include_non_regular and season_mode == "regular":
        season_mode = "all"

    rows, cols = core.load_rows(core.IN_CSV)
    feat_cols_all = [c for c in cols if c not in core.EXCLUDE]

    train_rows = core.split_date(
        rows,
        args.train_start,
        args.train_end,
        labeled_only=True,
        season_mode=season_mode,
    )
    test_rows = core.split_date(
        rows,
        args.test_start,
        args.test_end,
        labeled_only=True,
        season_mode=season_mode,
    )
    if len(train_rows) < 200 or len(test_rows) < 100:
        raise RuntimeError("Not enough rows for train/test split")

    cut = int(len(train_rows) * args.fit_ratio)
    cut = min(max(cut, 100), len(train_rows) - 50)
    fit_rows = train_rows[:cut]
    val_rows = train_rows[cut:]

    feat_cols, dropped = core.build_feature_filter(fit_rows, val_rows, feat_cols_all)
    X_train, y_train = core.make_matrix(train_rows, feat_cols)
    X_test, y_test = core.make_matrix(test_rows, feat_cols)

    d = len(feat_cols)
    mtry = max(8, int(math.sqrt(d) * args.mtry_mult))

    model = core.fit_random_forest(
        X_train,
        y_train,
        n_trees=args.n_trees,
        max_depth=args.max_depth,
        min_leaf=args.min_leaf,
        mtry=mtry,
        max_bins=args.max_bins,
        seed=args.seed,
    )
    p_test = core.predict_random_forest(model, X_test)
    ll, acc, auc = core.metrics(y_test, p_test)

    pred_rows = []
    for row, y, p in zip(test_rows, y_test, p_test):
        pred_rows.append({"date": row["date"], "s_no": row["s_no"], "y": y, "p_homewin": p})

    usage_rows = feature_usage(model, feat_cols)

    os.makedirs(os.path.dirname(OUT_PRED), exist_ok=True)
    with open(OUT_PRED, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "s_no", "y", "p_homewin"])
        w.writeheader()
        w.writerows(pred_rows)

    with open(OUT_USAGE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "split_count", "split_ratio"])
        w.writeheader()
        w.writerows(usage_rows)

    print("DONE")
    print("mode: split_champion_rf")
    print("train_range:", args.train_start, "~", args.train_end, "train_games:", len(train_rows))
    print("test_range:", args.test_start, "~", args.test_end, "test_games:", len(test_rows))
    print("total_features:", len(feat_cols_all))
    print("used_features:", len(feat_cols))
    print("dropped_features:", len(dropped))
    print("season_mode:", season_mode)
    print("rf_params:", {"n_trees": args.n_trees, "max_depth": args.max_depth, "min_leaf": args.min_leaf, "mtry": mtry, "max_bins": args.max_bins})
    print("test_accuracy:", round(acc, 4))
    print("test_logloss:", round(ll, 5))
    print("test_auc:", round(auc, 4) if auc == auc else "nan")
    print("pred_out:", OUT_PRED)
    print("usage_out:", OUT_USAGE)
    print("top_features:")
    for r in usage_rows[:15]:
        print(" ", r["feature"], r["split_count"], round(r["split_ratio"], 4))


if __name__ == "__main__":
    main()
