"""
?? v1 ?? ??? ???? expanding-window ????? ???? ??????.

?? ??
- 2024 ????? ?? ?? ???? ????.
- 2025 ??? 7? ?? ???? ??? ????? ????.
- ? ?? ?? ???? ?? ???? ??? ?? ????.

?, ? ????? ??? `???? ?? ??? ???` ??? ???? ? ??.
"""
import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score


DATA_DIR = os.path.expanduser("~/statiz/data")
FEATURES_CSV = os.path.join(DATA_DIR, "features_v1_paper.csv")
OUT_PRED_CSV = os.path.join(DATA_DIR, "backtest_pred_v1.csv")
OUT_BLOCK_CSV = os.path.join(DATA_DIR, "backtest_block_metrics_v1.csv")

FEATURE_COLS = [
    "diff_sum_ops_smooth",
    "diff_sum_ops_recent5",
    "diff_sp_oops",
    "diff_bullpen_fatigue",
]

BLOCK_DAYS = 7
TEST_YEAR = 2025


def safe_auc(y_true, y_prob):
    """?? ???? ?? ????? AUC? ???? ???? ???? ????."""
    y_unique = set(pd.Series(y_true).astype(int).tolist())
    if len(y_unique) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def main():
    """v1 feature CSV? ?? expanding-window LR ????? ????."""
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"not found: {FEATURES_CSV}")

    df = pd.read_csv(FEATURES_CSV)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")

    needed = FEATURE_COLS + ["date", "y_home_win", "s_no", "homeTeam", "awayTeam"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    # ???/???? ??? ?? ??? ??? ???.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLS + ["date", "y_home_win"]).copy()
    df["y_home_win"] = df["y_home_win"].astype(int)
    df = df.sort_values(["date", "s_no"]).reset_index(drop=True)

    # 2024??? ?? ?? ???, 2025? ?? ?? ???? ??.
    seed_train = df[df["date"].dt.year <= (TEST_YEAR - 1)].copy()
    test_df = df[df["date"].dt.year == TEST_YEAR].copy()

    if len(seed_train) == 0:
        raise ValueError("seed_train is empty. Need at least 2024 data.")
    if len(test_df) == 0:
        raise ValueError(f"test_df is empty. No rows for TEST_YEAR={TEST_YEAR}.")

    first_test_date = test_df["date"].min().normalize()
    last_test_date = test_df["date"].max().normalize()

    pred_rows = []
    block_rows = []

    block_idx = 0
    block_start = first_test_date

    while block_start <= last_test_date:
        block_end = block_start + pd.Timedelta(days=BLOCK_DAYS - 1)

        train_mask = df["date"] < block_start
        test_mask = (df["date"] >= block_start) & (df["date"] <= block_end)

        tr = df.loc[train_mask].copy()
        te = df.loc[test_mask].copy()

        if len(te) == 0:
            block_start = block_start + pd.Timedelta(days=BLOCK_DAYS)
            continue

        X_train = tr[FEATURE_COLS].values
        y_train = tr["y_home_win"].values
        X_test = te[FEATURE_COLS].values
        y_test = te["y_home_win"].values

        # ??? ? ???? ??? ????.
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="liblinear",
                max_iter=1000,
                random_state=42,
            )),
        ])

        model.fit(X_train, y_train)

        prob_home = model.predict_proba(X_test)[:, 1]
        pred_home = (prob_home >= 0.5).astype(int)

        acc = accuracy_score(y_test, pred_home)
        ll = log_loss(y_test, prob_home, labels=[0, 1])
        brier = brier_score_loss(y_test, prob_home)
        auc = safe_auc(y_test, prob_home)

        print(
            f"[Block {block_idx+1:02d}] "
            f"{block_start.date()}~{block_end.date()} | "
            f"train={len(tr)} test={len(te)} | "
            f"ACC={acc:.4f} LOGLOSS={ll:.4f} BRIER={brier:.4f} "
            f"AUC={auc:.4f}" if not np.isnan(auc) else
            f"[Block {block_idx+1:02d}] "
            f"{block_start.date()}~{block_end.date()} | "
            f"train={len(tr)} test={len(te)} | "
            f"ACC={acc:.4f} LOGLOSS={ll:.4f} BRIER={brier:.4f} AUC=nan"
        )

        block_rows.append({
            "block_idx": block_idx + 1,
            "block_start": block_start.strftime("%Y-%m-%d"),
            "block_end": block_end.strftime("%Y-%m-%d"),
            "n_train": len(tr),
            "n_test": len(te),
            "acc": acc,
            "logloss": ll,
            "brier": brier,
            "auc": auc,
        })

        tmp = te.copy()
        tmp["block_idx"] = block_idx + 1
        tmp["block_start"] = block_start.strftime("%Y-%m-%d")
        tmp["block_end"] = block_end.strftime("%Y-%m-%d")
        tmp["pred_home_win_proba"] = prob_home
        tmp["pred_home_win"] = pred_home
        pred_rows.append(tmp)

        block_idx += 1
        block_start = block_start + pd.Timedelta(days=BLOCK_DAYS)

    pred_df = pd.concat(pred_rows, axis=0).sort_values(["date", "s_no"]).reset_index(drop=True)
    block_df = pd.DataFrame(block_rows)

    keep_cols = [
        "date", "s_no", "s_code", "homeTeam", "awayTeam",
        "y_home_win", "homeScore", "awayScore",
        "diff_sum_ops_smooth", "diff_sum_ops_recent5",
        "diff_sp_oops", "diff_bullpen_fatigue",
        "pred_home_win_proba", "pred_home_win",
        "block_idx", "block_start", "block_end",
    ]
    keep_cols = [c for c in keep_cols if c in pred_df.columns]
    pred_df = pred_df[keep_cols].copy()

    pred_df["date"] = pred_df["date"].dt.strftime("%Y%m%d")

    os.makedirs(DATA_DIR, exist_ok=True)
    pred_df.to_csv(OUT_PRED_CSV, index=False, encoding="utf-8-sig")
    block_df.to_csv(OUT_BLOCK_CSV, index=False, encoding="utf-8-sig")

    y_all = pred_df["y_home_win"].values
    p_all = pred_df["pred_home_win_proba"].values
    yhat_all = pred_df["pred_home_win"].values

    acc_all = accuracy_score(y_all, yhat_all)
    ll_all = log_loss(y_all, p_all, labels=[0, 1])
    brier_all = brier_score_loss(y_all, p_all)
    auc_all = safe_auc(y_all, p_all)

    print("\n=== OVERALL ===")
    print(f"rows       : {len(pred_df)}")
    print(f"ACC        : {acc_all:.4f}")
    print(f"LOGLOSS    : {ll_all:.4f}")
    print(f"BRIER      : {brier_all:.4f}")
    print(f"AUC        : {auc_all:.4f}" if not np.isnan(auc_all) else "AUC        : nan")
    print(f"[OK] wrote : {OUT_PRED_CSV}")
    print(f"[OK] wrote : {OUT_BLOCK_CSV}")


if __name__ == "__main__":
    main()
