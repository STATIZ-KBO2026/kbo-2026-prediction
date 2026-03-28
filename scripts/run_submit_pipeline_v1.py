"""
run_submit_pipeline_v1.py
=========================
v1 제출 파이프라인:
  1) 전날 데이터까지 포함하여 features_v1.csv 재빌드
  2) 전체 labeled 데이터로 모델 학습
  3) 오늘 경기 예측 → pred CSV 생성
  4) submit_predictions.py로 API 제출

사용법:
  python3 scripts/run_submit_pipeline_v1.py --date 20260328
  python3 scripts/run_submit_pipeline_v1.py --date 20260328 --dry-run
  python3 scripts/run_submit_pipeline_v1.py --date 20260328 --skip-build  # features 재빌드 스킵
"""
import os
import sys
import csv
import math
import json
import argparse
import subprocess
import warnings
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_V1_CSV = os.path.expanduser("~/statiz/data/features_v1.csv")
PRED_CSV = os.path.expanduser("~/statiz/data/pred_v1_today.csv")
SUBMIT_SCRIPT = REPO_ROOT / "scripts" / "submit_predictions.py"
BUILD_FEATURES = REPO_ROOT / "scripts" / "build_features_v1.py"

sys.path.append(str(REPO_ROOT / "scripts"))
import backtest_v3_model_zoo as core

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
    y = np.array([core.label_int(r) or 0 for r in rows], dtype=np.int64)
    return X, y


def make_x(rows, feat_cols):
    return np.array([[safe_float(r[c]) for c in feat_cols] for r in rows], dtype=np.float64)


def to_proba(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if len(p.shape) == 2 and p.shape[1] >= 2:
            return p[:, 1]
        return p.reshape(-1)
    if hasattr(model, "decision_function"):
        d = model.decision_function(X)
        return np.array([1.0 / (1.0 + math.exp(-float(x))) for x in np.atleast_1d(d)], dtype=float)
    return np.array([float(v) for v in model.predict(X)], dtype=float)


def get_models(seed):
    """풀 모델 (백테스트 최적 설정, n_jobs=1 for EC2 안정성)"""
    return {
        "RF_1200_d10": RandomForestClassifier(
            n_estimators=1200, max_depth=10, min_samples_leaf=5,
            max_features="sqrt", random_state=seed, n_jobs=1,
        ),
        "ET_1400_d10": ExtraTreesClassifier(
            n_estimators=1400, max_depth=10, min_samples_leaf=3,
            max_features="sqrt", random_state=seed, n_jobs=1,
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


def run_cmd(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="v1 daily prediction & submission pipeline")
    ap.add_argument("--date", required=True, help="예측할 날짜 YYYYMMDD (오늘 경기)")
    ap.add_argument("--skip-build", action="store_true", help="features_v1.csv 재빌드 스킵")
    ap.add_argument("--skip-submit", action="store_true", help="제출 스킵 (예측만)")
    ap.add_argument("--dry-run", action="store_true", help="제출 dry-run")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--season-mode", choices=["regular", "all"], default="regular")
    ap.add_argument("--ensemble-method", choices=["best", "avg", "weighted"], default="weighted",
                    help="앙상블 방식: best=단일최적, avg=단순평균, weighted=가중평균")
    ap.add_argument("--pred-csv", default=PRED_CSV, help="예측 결과 저장 경로")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--sleep-sec", type=float, default=0.15)
    args = ap.parse_args()

    target_date = args.date
    print(f"=== v1 Pipeline: target date = {target_date} ===")

    # ── Step 1: features_v1.csv 재빌드 (전날까지의 결과 반영) ──
    if not args.skip_build:
        print("\n[Step 1] Building features_v1.csv ...")
        py = sys.executable
        run_cmd([py, str(BUILD_FEATURES)])
    else:
        print("\n[Step 1] Skipped (--skip-build)")

    if not os.path.exists(FEATURES_V1_CSV):
        raise RuntimeError(f"features_v1.csv not found: {FEATURES_V1_CSV}")

    # ── Step 2: 데이터 로드 & 분할 ──
    print("\n[Step 2] Loading data ...")
    rows, feat_cols = load_data(FEATURES_V1_CSV)

    if args.season_mode == "regular":
        rows = [r for r in rows if core.season_match(r, "regular")]

    # train: target_date 이전의 모든 labeled 데이터
    # test: target_date의 모든 경기 (labeled 여부 무관)
    train_rows = [r for r in rows if r["date"] < target_date and core.has_label(r)]
    test_rows = [r for r in rows if r["date"] == target_date]

    print(f"  Train rows: {len(train_rows)}")
    print(f"  Test rows (today): {len(test_rows)}")

    if len(train_rows) < 50:
        raise RuntimeError(f"Train data too small ({len(train_rows)}). Need at least 50 rows.")

    if len(test_rows) == 0:
        print(f"  ⚠️  No games found for {target_date}. Check if data includes today's schedule.")
        # 아직 오늘 경기가 features에 없을 수 있음 → 가장 가까운 미래 날짜 탐색
        future_dates = sorted(set(r["date"] for r in rows if r["date"] >= target_date))
        if future_dates:
            print(f"  Available future dates: {future_dates[:5]}")
        return

    # ── Step 3: 모델 학습 ──
    print("\n[Step 3] Training models ...")
    X_train, y_train = make_xy(train_rows, feat_cols)
    X_test = make_x(test_rows, feat_cols)

    models = get_models(args.seed)
    trained = {}
    val_scores = {}

    # 간이 validation: 최근 100경기로 logloss 추정
    val_n = min(100, len(train_rows) // 5)
    if val_n >= 20:
        val_split = train_rows[-val_n:]
        fit_split = train_rows[:-val_n]
        X_fit, y_fit = make_xy(fit_split, feat_cols)
        X_val, y_val = make_xy(val_split, feat_cols)
    else:
        X_fit, y_fit = X_train, y_train
        X_val, y_val = None, None

    for name, model in models.items():
        try:
            # 1) fit on fit_split for validation score
            if X_val is not None:
                model.fit(X_fit, y_fit)
                p_val = to_proba(model, X_val)
                ll, acc, auc = core.metrics(y_val.tolist(), p_val.tolist())
                val_scores[name] = {"ll": ll, "acc": acc, "auc": auc}
                print(f"  {name:10s}  val LL={ll:.4f}  ACC={acc:.4f}  AUC={auc:.4f}")

            # 2) retrain on ALL train data for final prediction
            fresh = get_models(args.seed)[name]
            fresh.fit(X_train, y_train)
            trained[name] = fresh
        except Exception as e:
            print(f"  [WARN] {name} failed: {e}")

    if not trained:
        raise RuntimeError("All models failed to train")

    # ── Step 4: 예측 & 앙상블 ──
    print("\n[Step 4] Predicting ...")
    preds = {}
    for name, model in trained.items():
        preds[name] = to_proba(model, X_test)

    # 앙상블
    model_names = list(preds.keys())
    if args.ensemble_method == "best":
        if val_scores:
            best = min(val_scores, key=lambda n: val_scores[n]["ll"])
        else:
            best = model_names[0]
        final_pred = preds[best]
        print(f"  Using best model: {best}")
    elif args.ensemble_method == "avg":
        final_pred = np.mean([preds[n] for n in model_names], axis=0)
        print(f"  Simple average of {model_names}")
    else:  # weighted
        if val_scores:
            # inverse logloss weights
            weights = {}
            for n in model_names:
                if n in val_scores:
                    weights[n] = 1.0 / max(val_scores[n]["ll"], 0.01)
                else:
                    weights[n] = 1.0
            w_sum = sum(weights.values())
            weights = {n: w / w_sum for n, w in weights.items()}
        else:
            weights = {n: 1.0 / len(model_names) for n in model_names}
        final_pred = np.zeros(len(test_rows))
        for n in model_names:
            final_pred += weights[n] * preds[n]
        print(f"  Weighted ensemble: {json.dumps({k: round(v, 3) for k, v in weights.items()})}")

    # ── Step 5: 예측 CSV 저장 ──
    print(f"\n[Step 5] Saving predictions → {args.pred_csv}")
    pred_rows = []
    for i, r in enumerate(test_rows):
        pred_rows.append({
            "date": r.get("date", ""),
            "s_no": r.get("s_no", ""),
            "homeTeam": r.get("homeTeam", ""),
            "awayTeam": r.get("awayTeam", ""),
            "p_homewin": round(float(final_pred[i]), 6),
            "percent": round(float(final_pred[i]) * 100.0, 2),
        })
        # 개별 모델 예측도 추가
        for name in model_names:
            pred_rows[-1][f"pred_{name}"] = round(float(preds[name][i]), 6)

    os.makedirs(os.path.dirname(args.pred_csv), exist_ok=True)
    fieldnames = list(pred_rows[0].keys()) if pred_rows else []
    with open(args.pred_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(pred_rows)

    print(f"  {len(pred_rows)} predictions saved")
    for pr in pred_rows:
        print(f"    s_no={pr['s_no']}  {pr['homeTeam']} vs {pr['awayTeam']}  "
              f"home_win={pr['p_homewin']:.3f} ({pr['percent']:.1f}%)")

    # ── Step 6: 제출 ──
    if args.skip_submit:
        print("\n[Step 6] Skipped (--skip-submit)")
    else:
        print(f"\n[Step 6] Submitting predictions ...")
        py = sys.executable
        sub_cmd = [
            py, str(SUBMIT_SCRIPT),
            "--in-csv", args.pred_csv,
            "--retries", str(args.retries),
            "--timeout", str(args.timeout),
            "--sleep-sec", str(args.sleep_sec),
            "--date", target_date,
        ]
        if args.dry_run:
            sub_cmd.append("--dry-run")
        run_cmd(sub_cmd)

    print("\n=== Pipeline complete ===")

    # feature importance 출력
    if "RF" in trained and hasattr(trained["RF"], "feature_importances_"):
        imp = sorted(zip(feat_cols, trained["RF"].feature_importances_), key=lambda x: -x[1])
        print("\nTop-10 features:")
        for fname, fval in imp[:10]:
            print(f"  {fname:50s}  {fval:.4f}")


if __name__ == "__main__":
    main()
