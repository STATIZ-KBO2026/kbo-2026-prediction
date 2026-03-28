"""
backtest_v1_expanding.py
========================
Expanding-window backtest for features_v1.csv
- Trains on all data up to date T, predicts date T+1
- Specifically tests cold-start performance (opening day etc.)
- Uses RandomForest (primary) + LogisticRegression + HistGBDT for comparison
"""
import os
import sys
import csv
import math
import json
import argparse
import warnings
from collections import defaultdict

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
)

sys.path.append(os.path.dirname(__file__))
import backtest_v3_model_zoo as core

IN_CSV = os.path.expanduser("~/statiz/data/features_v1.csv")
OUT_PRED = os.path.expanduser("~/statiz/data/backtest_v1_expanding_pred.csv")
OUT_REPORT = os.path.expanduser("~/statiz/data/backtest_v1_expanding_report.csv")

# features to exclude from model input
EXCLUDE = set(core.EXCLUDE) | {"s_code", "home_sp_sched_p_no", "away_sp_sched_p_no"}


def safe_float(x):
    try:
        if x is None:
            return 0.0
        s = str(x).strip()
        if s == "" or s.lower() == "none":
            return 0.0
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v
    except Exception:
        return 0.0


def load_data(path):
    rows, cols = core.load_rows(path)
    feat_cols = [c for c in cols if c not in EXCLUDE]
    return rows, feat_cols


def make_xy(rows, feat_cols):
    X = np.array([[safe_float(r[c]) for c in feat_cols] for r in rows], dtype=np.float64)
    y = np.array([core.label_int(r) for r in rows], dtype=np.int64)
    return X, y


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
        return np.array([sigmoid(float(x)) for x in np.atleast_1d(d)], dtype=float)
    return np.array([float(v) for v in model.predict(X)], dtype=float)


def get_models(seed, lightweight=False):
    if lightweight:
        # low-memory mode for constrained environments
        return {
            "RF_200_d6": RandomForestClassifier(
                n_estimators=200, max_depth=6, min_samples_leaf=5,
                max_features="sqrt", random_state=seed, n_jobs=1,
            ),
            "HGBDT": HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=300, max_depth=4,
                min_samples_leaf=20, random_state=seed,
            ),
            "LR": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=seed)),
            ]),
        }
    return {
        "RF_800_d8": RandomForestClassifier(
            n_estimators=800, max_depth=8, min_samples_leaf=3,
            max_features="sqrt", random_state=seed, n_jobs=-1,
        ),
        "RF_1200_d10": RandomForestClassifier(
            n_estimators=1200, max_depth=10, min_samples_leaf=5,
            max_features="sqrt", random_state=seed, n_jobs=-1,
        ),
        "ET_1400_d10": ExtraTreesClassifier(
            n_estimators=1400, max_depth=10, min_samples_leaf=3,
            max_features="sqrt", random_state=seed, n_jobs=-1,
        ),
        "HGBDT": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=500, max_depth=4,
            min_samples_leaf=20, random_state=seed,
        ),
        "LR": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=seed)),
        ]),
    }


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=IN_CSV)
    ap.add_argument("--min-train", type=int, default=100,
                    help="minimum training rows before starting predictions")
    ap.add_argument("--retrain-every", type=int, default=1,
                    help="retrain every N dates (1=daily)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--season-mode", choices=["regular", "all"], default="regular")
    ap.add_argument("--year-start", default="20250101",
                    help="start predicting from this date")
    ap.add_argument("--year-end", default="20261231")
    ap.add_argument("--cold-analysis", action="store_true",
                    help="output separate metrics for cold-start dates")
    ap.add_argument("--lightweight", action="store_true",
                    help="use smaller models for low-memory environments")
    args = ap.parse_args()

    rows, feat_cols = load_data(args.input)

    # filter by season mode
    if args.season_mode == "regular":
        rows = [r for r in rows if core.season_match(r, "regular")]

    # labeled only
    rows = [r for r in rows if core.has_label(r)]

    # group by date
    dates_set = sorted(set(r["date"] for r in rows))
    date_to_rows = defaultdict(list)
    for r in rows:
        date_to_rows[r["date"]].append(r)

    print(f"Total dates: {len(dates_set)}, Total rows: {len(rows)}, Features: {len(feat_cols)}")

    # expanding window
    train_pool = []
    models = get_models(args.seed, lightweight=args.lightweight)
    model_names = list(models.keys())
    current_models = {}

    # per-model predictions
    all_preds = {name: [] for name in model_names}
    all_labels = []
    all_dates = []
    all_meta = []

    pred_count = 0
    retrain_counter = 0

    for date_idx, d in enumerate(dates_set):
        day_rows = date_to_rows[d]

        if d < args.year_start or d > args.year_end:
            # accumulate training data but don't predict
            train_pool.extend(day_rows)
            continue

        # predict today's games using model trained on all past data
        if len(train_pool) >= args.min_train:
            need_retrain = (not current_models) or (retrain_counter >= args.retrain_every)

            if need_retrain:
                X_train, y_train = make_xy(train_pool, feat_cols)
                for name in model_names:
                    try:
                        m = get_models(args.seed, lightweight=args.lightweight)[name]  # fresh model
                        m.fit(X_train, y_train)
                        current_models[name] = m
                    except Exception as e:
                        print(f"  [WARN] {name} fit failed: {e}")
                retrain_counter = 0

            if current_models:
                X_pred, y_true = make_xy(day_rows, feat_cols)
                for name in model_names:
                    if name in current_models:
                        try:
                            preds = to_proba(current_models[name], X_pred)
                            all_preds[name].extend(preds.tolist())
                        except Exception:
                            all_preds[name].extend([0.5] * len(day_rows))
                    else:
                        all_preds[name].extend([0.5] * len(day_rows))

                all_labels.extend(y_true.tolist())
                all_dates.extend([d] * len(day_rows))
                for r in day_rows:
                    # cold start: check team game counts
                    h_cold = safe_float(r.get("home_team_cold_start", 0))
                    a_cold = safe_float(r.get("away_team_cold_start", 0))
                    is_cold = int(h_cold > 0 or a_cold > 0)
                    all_meta.append({
                        "date": d,
                        "s_no": r.get("s_no", ""),
                        "homeTeam": r.get("homeTeam", ""),
                        "awayTeam": r.get("awayTeam", ""),
                        "is_cold": is_cold,
                        "home_G": r.get("home_team_G", ""),
                    })
                pred_count += len(day_rows)

        # add today's data to training pool for future
        train_pool.extend(day_rows)
        retrain_counter += 1

    print(f"\nPredictions made: {pred_count}")

    if pred_count == 0:
        print("No predictions generated. Check date range and data.")
        return

    # compute metrics
    y_arr = np.array(all_labels, dtype=np.int64)
    report_rows = []

    # per-model metrics
    best_name = None
    best_ll = 999.0
    for name in model_names:
        p_arr = np.array(all_preds[name], dtype=np.float64)
        ll, acc, auc = core.metrics(y_arr.tolist(), p_arr.tolist())
        report_rows.append({
            "model": name, "split": "all",
            "logloss": round(ll, 6), "accuracy": round(acc, 6),
            "auc": round(auc, 6) if auc == auc else "",
            "n": pred_count,
        })
        print(f"  {name:20s}  LL={ll:.4f}  ACC={acc:.4f}  AUC={auc:.4f}")
        if ll < best_ll:
            best_ll = ll
            best_name = name

    # cold-start analysis
    if args.cold_analysis:
        cold_mask = [m["is_cold"] for m in all_meta]
        warm_mask = [1 - c for c in cold_mask]
        for name in model_names:
            p_arr = np.array(all_preds[name], dtype=np.float64)
            # cold
            y_cold = [y for y, c in zip(y_arr.tolist(), cold_mask) if c]
            p_cold = [p for p, c in zip(p_arr.tolist(), cold_mask) if c]
            if len(y_cold) >= 5:
                ll_c, acc_c, auc_c = core.metrics(y_cold, p_cold)
                report_rows.append({
                    "model": name, "split": "cold_start",
                    "logloss": round(ll_c, 6), "accuracy": round(acc_c, 6),
                    "auc": round(auc_c, 6) if auc_c == auc_c else "",
                    "n": len(y_cold),
                })
                print(f"  {name:20s}  COLD  LL={ll_c:.4f}  ACC={acc_c:.4f}  n={len(y_cold)}")
            # warm
            y_warm = [y for y, w in zip(y_arr.tolist(), warm_mask) if w]
            p_warm = [p for p, w in zip(p_arr.tolist(), warm_mask) if w]
            if len(y_warm) >= 5:
                ll_w, acc_w, auc_w = core.metrics(y_warm, p_warm)
                report_rows.append({
                    "model": name, "split": "warm",
                    "logloss": round(ll_w, 6), "accuracy": round(acc_w, 6),
                    "auc": round(auc_w, 6) if auc_w == auc_w else "",
                    "n": len(y_warm),
                })

        # opening day analysis (month == 3 or first 5 days of season per year)
        opening_mask = []
        for m in all_meta:
            month = int(str(m["date"])[4:6]) if len(str(m["date"])) >= 6 else 0
            g = safe_float(m["home_G"])
            opening_mask.append(1 if (month <= 4 and g < 10) else 0)
        for name in model_names:
            p_arr = np.array(all_preds[name], dtype=np.float64)
            y_op = [y for y, o in zip(y_arr.tolist(), opening_mask) if o]
            p_op = [p for p, o in zip(p_arr.tolist(), opening_mask) if o]
            if len(y_op) >= 5:
                ll_o, acc_o, auc_o = core.metrics(y_op, p_op)
                report_rows.append({
                    "model": name, "split": "opening_month",
                    "logloss": round(ll_o, 6), "accuracy": round(acc_o, 6),
                    "auc": round(auc_o, 6) if auc_o == auc_o else "",
                    "n": len(y_op),
                })
                print(f"  {name:20s}  OPEN  LL={ll_o:.4f}  ACC={acc_o:.4f}  n={len(y_op)}")

    # ensemble: simple average of top-2 models
    if len(model_names) >= 2:
        sorted_models = sorted(model_names, key=lambda n: next(
            (r["logloss"] for r in report_rows if r["model"] == n and r["split"] == "all"), 999
        ))
        top2 = sorted_models[:2]
        p_ens = np.mean([np.array(all_preds[n], dtype=np.float64) for n in top2], axis=0)
        ll_e, acc_e, auc_e = core.metrics(y_arr.tolist(), p_ens.tolist())
        report_rows.append({
            "model": f"ensemble_{top2[0]}+{top2[1]}", "split": "all",
            "logloss": round(ll_e, 6), "accuracy": round(acc_e, 6),
            "auc": round(auc_e, 6) if auc_e == auc_e else "",
            "n": pred_count,
        })
        print(f"\n  {'Ensemble(top2)':20s}  LL={ll_e:.4f}  ACC={acc_e:.4f}  AUC={auc_e:.4f}")

    # save report
    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    with open(OUT_REPORT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "split", "logloss", "accuracy", "auc", "n"])
        w.writeheader()
        w.writerows(report_rows)
    print(f"\nReport saved: {OUT_REPORT}")

    # save predictions
    pred_rows = []
    for i, meta in enumerate(all_meta):
        row = dict(meta)
        row["y_true"] = all_labels[i]
        for name in model_names:
            row[f"pred_{name}"] = round(all_preds[name][i], 6)
        if best_name:
            row["pred_best"] = round(all_preds[best_name][i], 6)
        pred_rows.append(row)

    pred_fields = list(pred_rows[0].keys()) if pred_rows else []
    with open(OUT_PRED, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pred_fields)
        w.writeheader()
        w.writerows(pred_rows)
    print(f"Predictions saved: {OUT_PRED}")

    # feature importance (best model)
    if best_name and best_name in current_models:
        m = current_models[best_name]
        if hasattr(m, "feature_importances_"):
            imp = list(zip(feat_cols, m.feature_importances_))
            imp.sort(key=lambda x: -x[1])
            print(f"\nTop-20 features ({best_name}):")
            for fname, fval in imp[:20]:
                print(f"  {fname:50s}  {fval:.4f}")


if __name__ == "__main__":
    main()
