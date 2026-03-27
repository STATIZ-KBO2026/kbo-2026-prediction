import os
import sys
import csv
import math
import json
import argparse
import warnings

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.dirname(__file__))
import backtest_v3_model_zoo as core

OUT_PRED = os.path.expanduser("~/statiz/data/backtest_pred_v5_best.csv")
OUT_REPORT = os.path.expanduser("~/statiz/data/backtest_v5_model_report.csv")
OUT_DROP = os.path.expanduser("~/statiz/data/backtest_v5_dropped_features.csv")


def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def to_proba(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if len(p.shape) == 2 and p.shape[1] >= 2:
            return p[:, 1]
        return p.reshape(-1)
    if hasattr(model, "decision_function"):
        d = model.decision_function(X)
        if hasattr(d, "__len__"):
            return np.array([sigmoid(float(x)) for x in d], dtype=float)
        return np.array([sigmoid(float(d))], dtype=float)
    y = model.predict(X)
    return np.array([float(v) for v in y], dtype=float)


def eval_metrics(y, p):
    y_list = [int(v) for v in y]
    p_list = [float(v) for v in p]
    return core.metrics(y_list, p_list)


def add_report(rows, model_name, split_name, y, p, params=None, status="ok", note=""):
    ll, acc, auc = eval_metrics(y, p)
    rows.append(
        {
            "model": model_name,
            "split": split_name,
            "logloss": round(ll, 6),
            "accuracy": round(acc, 6),
            "auc": round(auc, 6) if auc == auc else "",
            "params": json.dumps(params or {}, ensure_ascii=True),
            "status": status,
            "note": note,
        }
    )
    return ll, acc, auc


def make_rows_for_error(rows, model_name, split_name, params, err):
    rows.append(
        {
            "model": model_name,
            "split": split_name,
            "logloss": "",
            "accuracy": "",
            "auc": "",
            "params": json.dumps(params or {}, ensure_ascii=True),
            "status": "error",
            "note": str(err),
        }
    )


def add_skip_row(rows, model_name, split_name, params=None, note=""):
    rows.append(
        {
            "model": model_name,
            "split": split_name,
            "logloss": "",
            "accuracy": "",
            "auc": "",
            "params": json.dumps(params or {}, ensure_ascii=True),
            "status": "skip",
            "note": note,
        }
    )


def masked_pred(preds, mask):
    return np.array([float(p) for p, keep in zip(preds, mask) if keep], dtype=float)


def normalize_01(vals, invert=False):
    if not vals:
        return []
    lo = min(vals)
    hi = max(vals)
    if abs(hi - lo) <= 1e-12:
        return [0.5 for _ in vals]
    if invert:
        return [(hi - v) / (hi - lo) for v in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def finite_auc(v):
    if v != v:
        return 0.5
    return float(v)


def metric_scores(metric_rows, w_acc, w_auc, w_logloss):
    acc_vals = [float(r["acc"]) for r in metric_rows]
    auc_vals = [finite_auc(r["auc"]) for r in metric_rows]
    ll_vals = [float(r["ll"]) for r in metric_rows]
    acc_n = normalize_01(acc_vals, invert=False)
    auc_n = normalize_01(auc_vals, invert=False)
    ll_n = normalize_01(ll_vals, invert=True)
    out = []
    for i in range(len(metric_rows)):
        out.append(w_acc * acc_n[i] + w_auc * auc_n[i] + w_logloss * ll_n[i])
    return out


def choose_best_metric_row(metric_rows, objective, w_acc, w_auc, w_logloss):
    if not metric_rows:
        return None, []
    if objective == "logloss":
        best = min(metric_rows, key=lambda r: (r["ll"], -r["acc"], -finite_auc(r["auc"])))
        return best, []
    if objective == "accuracy":
        best = max(metric_rows, key=lambda r: (r["acc"], finite_auc(r["auc"]), -r["ll"]))
        return best, []
    scores = metric_scores(metric_rows, w_acc=w_acc, w_auc=w_auc, w_logloss=w_logloss)
    best_i = max(
        range(len(metric_rows)),
        key=lambda i: (scores[i], metric_rows[i]["acc"], finite_auc(metric_rows[i]["auc"]), -metric_rows[i]["ll"]),
    )
    return metric_rows[best_i], scores


def select_topk_by_logloss(metric_map, k=3):
    items = sorted(metric_map.items(), key=lambda x: x[1]["ll"])
    return [name for name, _ in items[:k]]


def inv_logloss_weights(model_names, metric_map):
    raw = []
    for n in model_names:
        ll = max(1e-6, float(metric_map[n]["ll"]))
        raw.append((n, 1.0 / ll))
    s = sum(w for _, w in raw)
    if s <= 0:
        w = 1.0 / max(1, len(model_names))
        return {n: w for n in model_names}
    return {n: (w / s) for n, w in raw}


def score_weights(model_names, score_map):
    raw = []
    for n in model_names:
        raw.append((n, max(1e-6, float(score_map.get(n, 0.0)))))
    s = sum(w for _, w in raw)
    if s <= 0:
        w = 1.0 / max(1, len(model_names))
        return {n: w for n in model_names}
    return {n: (w / s) for n, w in raw}


def blend(pred_map, weights):
    names = list(weights.keys())
    n = len(pred_map[names[0]])
    out = []
    for i in range(n):
        s = 0.0
        for name in names:
            s += weights[name] * float(pred_map[name][i])
        out.append(s)
    return np.array(out, dtype=float)


def add_blend_candidate(
    report_rows,
    candidates,
    name,
    weights,
    val_pred,
    test_pred,
    y_val,
    y_test_eval,
    test_eval_mask,
):
    p_val = blend(val_pred, weights)
    p_test = blend(test_pred, weights)
    ll, acc, auc = add_report(
        report_rows,
        name,
        "val",
        y_val,
        p_val,
        params={"weights": weights},
        note="blend",
    )
    if y_test_eval:
        add_report(
            report_rows,
            name,
            "test",
            y_test_eval,
            masked_pred(p_test, test_eval_mask),
            params={"weights": weights},
            note="blend|eval_labeled_only",
        )
    else:
        add_skip_row(
            report_rows,
            name,
            "test",
            params={"weights": weights},
            note="no_labeled_targets_in_test",
        )
    candidates[name] = {
        "val": p_val,
        "test": p_test,
        "ll": float(ll),
        "acc": float(acc),
        "auc": float(auc),
        "params": {"weights": weights},
    }


def compute_fit_cut(n_rows, fit_ratio):
    if n_rows <= 2:
        return 1
    target = int(n_rows * fit_ratio)
    if n_rows >= 150:
        cut = min(max(target, 100), n_rows - 50)
        return min(max(cut, 1), n_rows - 1)
    # small-sample fallback for pipeline smoke tests
    min_fit = max(5, n_rows // 3)
    max_fit = max(min_fit, n_rows - max(5, n_rows // 4))
    cut = min(max(target, min_fit), max_fit)
    return min(max(cut, 1), n_rows - 1)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-start", default="20230101")
    ap.add_argument("--train-end", default="20261231")
    ap.add_argument("--test-start", default="20260101")
    ap.add_argument("--test-end", default="20261231")
    ap.add_argument("--fit-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--feature-filter", choices=["strict", "balanced", "none"], default="strict")
    ap.add_argument("--select-objective", choices=["composite", "logloss", "accuracy"], default="composite")
    ap.add_argument("--w-acc", type=float, default=0.45)
    ap.add_argument("--w-auc", type=float, default=0.30)
    ap.add_argument("--w-logloss", type=float, default=0.25)
    ap.add_argument("--allow-blend-select", action="store_true")
    ap.add_argument("--selection-pool", choices=["stable_tree", "all"], default="stable_tree")
    ap.add_argument("--model-pool", choices=["full", "stable_tree_only"], default="full")
    ap.add_argument("--season-mode", choices=["regular", "exhibition", "all"], default="regular")
    ap.add_argument("--pipeline-test", action="store_true", help="simple smoke test mode (preseason-focused)")
    ap.add_argument("--min-train-rows", type=int, default=200)
    ap.add_argument("--include-non-regular", action="store_true", help="include exhibition/postseason rows")
    args = ap.parse_args()
    full_pool = args.model_pool == "full"

    season_mode = args.season_mode
    if args.include_non_regular and season_mode == "regular":
        season_mode = "all"
    if args.pipeline_test and season_mode == "regular":
        season_mode = "exhibition"
    if args.pipeline_test:
        args.min_train_rows = min(args.min_train_rows, 30)
        if args.model_pool == "full":
            args.model_pool = "stable_tree_only"
            full_pool = False
        if args.selection_pool == "all":
            args.selection_pool = "stable_tree"

    w_sum = max(1e-9, args.w_acc + args.w_auc + args.w_logloss)
    w_acc = args.w_acc / w_sum
    w_auc = args.w_auc / w_sum
    w_logloss = args.w_logloss / w_sum

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
        labeled_only=False,
        season_mode=season_mode,
    )
    if len(train_rows) < args.min_train_rows or len(test_rows) < 1:
        raise RuntimeError("Not enough rows for train/test split")

    cut = compute_fit_cut(len(train_rows), args.fit_ratio)
    fit_rows = train_rows[:cut]
    val_rows = train_rows[cut:]

    if args.feature_filter == "none":
        feat_cols = list(feat_cols_all)
        dropped = []
    elif args.feature_filter == "strict":
        feat_cols, dropped = core.build_feature_filter(
            fit_rows,
            val_rows,
            feat_cols_all,
            drop_countlike_shift=True,
        )
    else:
        feat_cols, dropped = core.build_feature_filter(
            fit_rows,
            val_rows,
            feat_cols_all,
            drop_countlike_shift=False,
        )

    X_fit_raw, y_fit = core.make_matrix(fit_rows, feat_cols)
    X_val_raw, y_val = core.make_matrix(val_rows, feat_cols)
    X_train_raw, y_train = core.make_matrix(train_rows, feat_cols)
    X_test_raw = [[core.safe_float(r[c]) for c in feat_cols] for r in test_rows]
    test_eval_mask = [core.has_label(r) for r in test_rows]
    y_test_eval = [core.label_int(r) for r in test_rows if core.has_label(r)]

    X_fit = np.array(X_fit_raw, dtype=np.float64)
    X_val = np.array(X_val_raw, dtype=np.float64)
    X_train = np.array(X_train_raw, dtype=np.float64)
    X_test = np.array(X_test_raw, dtype=np.float64)
    y_fit = np.array(y_fit, dtype=np.int64)
    y_val = np.array(y_val, dtype=np.int64)
    y_train = np.array(y_train, dtype=np.int64)

    report_rows = []
    val_pred = {}
    test_pred = {}
    val_metric = {}

    model_specs = []
    model_specs.append(
        (
            "sk_lr",
            [
                {"C": 0.5},
                {"C": 1.0},
                {"C": 2.0},
            ],
            lambda p: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=p["C"],
                            max_iter=5000,
                            solver="lbfgs",
                            random_state=args.seed,
                        ),
                    ),
                ]
            ),
        )
    )
    model_specs.append(
        (
            "sk_sgd_log",
            [
                {"alpha": 1e-4},
                {"alpha": 5e-4},
                {"alpha": 1e-3},
            ],
            lambda p: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        SGDClassifier(
                            loss="log_loss",
                            penalty="l2",
                            alpha=p["alpha"],
                            max_iter=5000,
                            tol=1e-4,
                            random_state=args.seed,
                        ),
                    ),
                ]
            ),
        )
    )
    model_specs.append(
        (
            "sk_random_forest",
            [
                {"n_estimators": 800, "max_depth": 8, "min_samples_leaf": 3},
                {"n_estimators": 1200, "max_depth": None, "min_samples_leaf": 3},
                {"n_estimators": 1200, "max_depth": 10, "min_samples_leaf": 5},
                {"n_estimators": 1800, "max_depth": 12, "min_samples_leaf": 6},
            ],
            lambda p: RandomForestClassifier(
                n_estimators=p["n_estimators"],
                max_depth=p["max_depth"],
                min_samples_leaf=p["min_samples_leaf"],
                max_features="sqrt",
                random_state=args.seed,
                n_jobs=-1,
            ),
        )
    )
    model_specs.append(
        (
            "sk_extra_trees",
            [
                {"n_estimators": 1000, "max_depth": None, "min_samples_leaf": 2},
                {"n_estimators": 1400, "max_depth": 10, "min_samples_leaf": 3},
                {"n_estimators": 1800, "max_depth": 12, "min_samples_leaf": 6},
                {"n_estimators": 2500, "max_depth": 12, "min_samples_leaf": 6},
            ],
            lambda p: ExtraTreesClassifier(
                n_estimators=p["n_estimators"],
                max_depth=p["max_depth"],
                min_samples_leaf=p["min_samples_leaf"],
                max_features="sqrt",
                random_state=args.seed,
                n_jobs=-1,
            ),
        )
    )
    model_specs.append(
        (
            "sk_hist_gbdt",
            [
                {"learning_rate": 0.05, "max_iter": 500, "max_depth": 4, "min_samples_leaf": 20},
                {"learning_rate": 0.03, "max_iter": 700, "max_depth": 5, "min_samples_leaf": 20},
            ],
            lambda p: HistGradientBoostingClassifier(
                learning_rate=p["learning_rate"],
                max_iter=p["max_iter"],
                max_depth=p["max_depth"],
                min_samples_leaf=p["min_samples_leaf"],
                random_state=args.seed,
            ),
        )
    )
    model_specs.append(
        (
            "sk_gbdt",
            [
                {"learning_rate": 0.03, "n_estimators": 600, "max_depth": 3},
                {"learning_rate": 0.05, "n_estimators": 500, "max_depth": 2},
            ],
            lambda p: GradientBoostingClassifier(
                learning_rate=p["learning_rate"],
                n_estimators=p["n_estimators"],
                max_depth=p["max_depth"],
                random_state=args.seed,
            ),
        )
    )
    model_specs.append(
        (
            "sk_adaboost",
            [
                {"n_estimators": 500, "learning_rate": 0.03, "depth": 2},
                {"n_estimators": 800, "learning_rate": 0.02, "depth": 2},
            ],
            lambda p: AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=p["depth"], random_state=args.seed),
                n_estimators=p["n_estimators"],
                learning_rate=p["learning_rate"],
                random_state=args.seed,
            ),
        )
    )

    if full_pool:
        try:
            from xgboost import XGBClassifier

            model_specs.append(
                (
                    "xgboost",
                    [
                        {"n_estimators": 800, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 3},
                        {"n_estimators": 600, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 5},
                        {"n_estimators": 1200, "learning_rate": 0.02, "max_depth": 5, "min_child_weight": 5},
                    ],
                    lambda p: XGBClassifier(
                        n_estimators=p["n_estimators"],
                        learning_rate=p["learning_rate"],
                        max_depth=p["max_depth"],
                        min_child_weight=p["min_child_weight"],
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=args.seed,
                        n_jobs=0,
                    ),
                )
            )
        except Exception as e:
            make_rows_for_error(report_rows, "xgboost", "val", {}, e)

    if full_pool:
        try:
            from lightgbm import LGBMClassifier

            model_specs.append(
                (
                    "lightgbm",
                    [
                        {"n_estimators": 1200, "learning_rate": 0.03, "num_leaves": 31},
                        {"n_estimators": 800, "learning_rate": 0.05, "num_leaves": 63},
                    ],
                    lambda p: LGBMClassifier(
                        n_estimators=p["n_estimators"],
                        learning_rate=p["learning_rate"],
                        num_leaves=p["num_leaves"],
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=args.seed,
                        objective="binary",
                        n_jobs=-1,
                        verbose=-1,
                    ),
                )
            )
        except Exception as e:
            make_rows_for_error(report_rows, "lightgbm", "val", {}, e)

    if full_pool:
        try:
            from catboost import CatBoostClassifier

            model_specs.append(
                (
                    "catboost",
                    [
                        {"iterations": 1000, "learning_rate": 0.03, "depth": 6},
                        {"iterations": 700, "learning_rate": 0.05, "depth": 6},
                    ],
                    lambda p: CatBoostClassifier(
                        iterations=p["iterations"],
                        learning_rate=p["learning_rate"],
                        depth=p["depth"],
                        loss_function="Logloss",
                        eval_metric="Logloss",
                        random_seed=args.seed,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                )
            )
        except Exception as e:
            make_rows_for_error(report_rows, "catboost", "val", {}, e)

    if not full_pool:
        keep = {"sk_random_forest", "sk_extra_trees"}
        model_specs = [spec for spec in model_specs if spec[0] in keep]

    for model_name, param_grid, factory in model_specs:
        cfg_rows = []
        for cfg in param_grid:
            try:
                model = factory(cfg)
                model.fit(X_fit, y_fit)
                p_val = to_proba(model, X_val)
                ll, acc, auc = add_report(report_rows, model_name, "val", y_val, p_val, params=cfg)
                cfg_rows.append(
                    {
                        "cfg": cfg,
                        "ll": float(ll),
                        "acc": float(acc),
                        "auc": float(auc),
                        "p_val": np.array(p_val, dtype=float),
                    }
                )
            except Exception as e:
                make_rows_for_error(report_rows, model_name, "val", cfg, e)

        if not cfg_rows:
            continue

        best_cfg_row, _ = choose_best_metric_row(
            cfg_rows,
            objective=args.select_objective,
            w_acc=w_acc,
            w_auc=w_auc,
            w_logloss=w_logloss,
        )
        best_cfg = best_cfg_row["cfg"]
        val_pred[model_name] = np.array(best_cfg_row["p_val"], dtype=float)
        val_metric[model_name] = {
            "ll": float(best_cfg_row["ll"]),
            "acc": float(best_cfg_row["acc"]),
            "auc": float(best_cfg_row["auc"]),
            "cfg": best_cfg,
        }

        try:
            model_final = factory(best_cfg)
            model_final.fit(X_train, y_train)
            p_test = to_proba(model_final, X_test)
            test_pred[model_name] = np.array(p_test, dtype=float)
            if y_test_eval:
                add_report(
                    report_rows,
                    model_name,
                    "test",
                    y_test_eval,
                    masked_pred(p_test, test_eval_mask),
                    params=best_cfg,
                    note=f"cfg_select={args.select_objective}|eval_labeled_only",
                )
            else:
                add_skip_row(
                    report_rows,
                    model_name,
                    "test",
                    params=best_cfg,
                    note="no_labeled_targets_in_test",
                )
        except Exception as e:
            make_rows_for_error(report_rows, model_name, "test", best_cfg, e)

    if not val_metric:
        raise RuntimeError("No external model trained successfully")

    candidates = {}
    for model_name in sorted(test_pred.keys()):
        m = val_metric[model_name]
        candidates[model_name] = {
            "val": val_pred[model_name],
            "test": test_pred[model_name],
            "ll": m["ll"],
            "acc": m["acc"],
            "auc": m["auc"],
            "params": m["cfg"],
        }

    top_ll = select_topk_by_logloss(val_metric, k=min(3, len(val_metric)))
    w_ll = inv_logloss_weights(top_ll, val_metric)
    add_blend_candidate(
        report_rows,
        candidates,
        "blend_topk_logloss",
        w_ll,
        val_pred,
        test_pred,
        y_val,
        y_test_eval,
        test_eval_mask,
    )

    metric_rows = [
        {"name": n, "ll": float(v["ll"]), "acc": float(v["acc"]), "auc": float(v["auc"])}
        for n, v in val_metric.items()
    ]
    score_list = metric_scores(metric_rows, w_acc=w_acc, w_auc=w_auc, w_logloss=w_logloss)
    score_map = {row["name"]: float(s) for row, s in zip(metric_rows, score_list)}
    top_comp = [n for n, _ in sorted(score_map.items(), key=lambda x: x[1], reverse=True)[: min(3, len(score_map))]]
    w_comp = score_weights(top_comp, score_map)
    add_blend_candidate(
        report_rows,
        candidates,
        "blend_topk_composite",
        w_comp,
        val_pred,
        test_pred,
        y_val,
        y_test_eval,
        test_eval_mask,
    )

    select_names = set(val_metric.keys())
    if args.selection_pool == "stable_tree":
        stable = {"sk_random_forest", "sk_extra_trees"}
        stable_names = set(n for n in select_names if n in stable)
        if stable_names:
            select_names = stable_names
    if args.allow_blend_select:
        select_names = set(candidates.keys())

    candidate_rows = []
    for name, data in candidates.items():
        if name not in select_names:
            continue
        candidate_rows.append(
            {
                "name": name,
                "ll": float(data["ll"]),
                "acc": float(data["acc"]),
                "auc": float(data["auc"]),
            }
        )

    best_row, candidate_scores = choose_best_metric_row(
        candidate_rows,
        objective=args.select_objective,
        w_acc=w_acc,
        w_auc=w_auc,
        w_logloss=w_logloss,
    )
    chosen = best_row["name"]
    p_out = candidates[chosen]["test"]

    pred_rows = []
    for row, p in zip(test_rows, p_out):
        y_lab = core.label_int(row)
        pred_rows.append(
            {
                "date": row["date"],
                "s_no": row["s_no"],
                "y": y_lab if y_lab is not None else "",
                "p_homewin": float(p),
            }
        )

    os.makedirs(os.path.dirname(OUT_PRED), exist_ok=True)
    with open(OUT_PRED, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "s_no", "y", "p_homewin"])
        w.writeheader()
        w.writerows(pred_rows)

    with open(OUT_REPORT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "split", "logloss", "accuracy", "auc", "params", "status", "note"],
        )
        w.writeheader()
        w.writerows(report_rows)

    with open(OUT_DROP, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "fit_zero_ratio", "fit_variance", "fit_val_shift_z", "tags"])
        w.writeheader()
        w.writerows(sorted(dropped, key=lambda x: (x["tags"], -x["fit_val_shift_z"], x["feature"])))

    ll = float("nan")
    acc = float("nan")
    auc = float("nan")
    if y_test_eval:
        ll, acc, auc = eval_metrics(y_test_eval, masked_pred(p_out, test_eval_mask))
    print("DONE")
    print("mode: split_external_ml")
    print("train_range:", args.train_start, "~", args.train_end, "train_games:", len(train_rows))
    print("test_range:", args.test_start, "~", args.test_end, "test_games:", len(test_rows))
    print("feature_filter:", args.feature_filter)
    print("selection_objective:", args.select_objective)
    print("selection_pool:", args.selection_pool)
    print("model_pool:", args.model_pool)
    print("season_mode:", season_mode)
    print("pipeline_test:", bool(args.pipeline_test))
    print("allow_blend_select:", bool(args.allow_blend_select))
    print("selection_weights:", {"acc": round(w_acc, 4), "auc": round(w_auc, 4), "logloss": round(w_logloss, 4)})
    print("total_features:", len(feat_cols_all))
    print("used_features:", len(feat_cols))
    print("dropped_features:", len(dropped))
    print("models_trained:", len(val_metric))
    print("test_labeled_games:", len(y_test_eval))
    print("test_unlabeled_games:", len(test_rows) - len(y_test_eval))
    print("topk_logloss:", top_ll)
    print("topk_composite:", top_comp)
    print("chosen_for_output:", chosen)
    if candidate_scores:
        score_map_out = {r["name"]: round(s, 6) for r, s in zip(candidate_rows, candidate_scores)}
        print("candidate_scores:", score_map_out)
    if y_test_eval:
        print("test_accuracy:", round(acc, 4))
        print("test_logloss:", round(ll, 5))
        print("test_auc:", round(auc, 4) if auc == auc else "nan")
    else:
        print("test_accuracy:", "nan")
        print("test_logloss:", "nan")
        print("test_auc:", "nan")
    print("pred_out:", OUT_PRED)
    print("report_out:", OUT_REPORT)
    print("drop_out:", OUT_DROP)


if __name__ == "__main__":
    main()
