"""
backtest_v1_online_lr.py ? ?? ???? ??? LR ??? ???? ??????.

? ????? ?? ????? `?? ??? ?? ???? ?????`? ???? ? ??? ??.
"""
import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DATA_DIR = os.path.expanduser("~/statiz/data")
FEATURES_CSV = os.path.join(DATA_DIR, "features_v1_paper.csv")
OUT_COEF_CSV = os.path.join(DATA_DIR, "backtest_lr_coef_v1.csv")
OUT_SUMMARY_CSV = os.path.join(DATA_DIR, "backtest_lr_coef_summary_v1.csv")

FEATURE_COLS = [
    "diff_sum_ops_smooth",
    "diff_sum_ops_recent5",
    "diff_sp_oops",
    "diff_bullpen_fatigue",
]

BLOCK_DAYS = 7
TEST_YEAR = 2025


def main():
    """???? ???? ???? ?? ??? ???? ????."""
    df = pd.read_csv(FEATURES_CSV)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")

    needed = FEATURE_COLS + ["date", "y_home_win", "s_no"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLS + ["date", "y_home_win"]).copy()
    df["y_home_win"] = df["y_home_win"].astype(int)
    df = df.sort_values(["date", "s_no"]).reset_index(drop=True)

    test_df = df[df["date"].dt.year == TEST_YEAR].copy()
    if len(test_df) == 0:
        raise ValueError(f"No rows for TEST_YEAR={TEST_YEAR}")

    first_test_date = test_df["date"].min().normalize()
    last_test_date = test_df["date"].max().normalize()

    coef_rows = []

    block_idx = 0
    block_start = first_test_date

    while block_start <= last_test_date:
        block_end = block_start + pd.Timedelta(days=BLOCK_DAYS - 1)

        tr = df[df["date"] < block_start].copy()
        te = df[(df["date"] >= block_start) & (df["date"] <= block_end)].copy()

        if len(te) == 0:
            block_start = block_start + pd.Timedelta(days=BLOCK_DAYS)
            continue

        X_train = tr[FEATURE_COLS].values
        y_train = tr["y_home_win"].values

        # ????? ??? ???/?? ??? ?? ?? ??? ?? ??? ??.
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=1.0,
                solver="liblinear",
                max_iter=1000,
                random_state=42,
            )),
        ])
        model.fit(X_train, y_train)

        lr = model.named_steps["lr"]
        for feat, coef in zip(FEATURE_COLS, lr.coef_[0]):
            coef_rows.append({
                "block_idx": block_idx + 1,
                "block_start": block_start.strftime("%Y-%m-%d"),
                "block_end": block_end.strftime("%Y-%m-%d"),
                "feature": feat,
                "coef": float(coef),
            })

        block_idx += 1
        block_start = block_start + pd.Timedelta(days=BLOCK_DAYS)

    coef_df = pd.DataFrame(coef_rows)

    summary_df = (
        coef_df.groupby("feature")
        .agg(
            mean_coef=("coef", "mean"),
            mean_abs_coef=("coef", lambda s: float(np.mean(np.abs(s)))),
            std_coef=("coef", "std"),
            pos_ratio=("coef", lambda s: float(np.mean(s > 0))),
            n_blocks=("coef", "size"),
        )
        .sort_values("mean_abs_coef", ascending=False)
        .reset_index()
    )

    coef_df.to_csv(OUT_COEF_CSV, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("=== COEF SUMMARY ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\n[OK] wrote : {OUT_COEF_CSV}")
    print(f"[OK] wrote : {OUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
