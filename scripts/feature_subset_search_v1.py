"""
피처 조합(Subset) 자동 탐색기

v2 후보 12개 피처 중에서 가능한 모든 조합을 만들고,
각 조합에 대해 expanding-window 백테스트를 실행하여
최적의 피처 세트를 찾습니다.

[동작 방식]
  1. min_k~max_k 크기의 모든 피처 조합을 생성합니다.
     (예: 12개 중 3개 선택 → C(12,3) = 220개 조합)
  2. 각 조합으로 backtest_v1_online_lr.py와 동일한 expanding-window 백테스트를 수행합니다.
  3. LOGLOSS > BRIER > AUC 순으로 정렬하여 상위 조합을 찾습니다.

[파이프라인 위치]
  9b단계 — build_features_v2_candidates.py 이후에 실행합니다.

[입력]
  - ~/statiz/data/features_v2_candidates.csv

[출력]
  - ~/statiz/data/feature_subset_search_v1.csv      — 전체 조합 결과
  - ~/statiz/data/feature_subset_top20_v1.csv        — 상위 20개 조합

[사용법]
  python feature_subset_search_v1.py                              # 기본 (1~12개 조합 전탐색)
  python feature_subset_search_v1.py --min-k 3 --max-k 5          # 3~5개 조합만 탐색
  python feature_subset_search_v1.py --feature-cols "col1,col2"   # 특정 피처만 대상
"""
import os
import argparse
import itertools
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score


DATA_DIR = os.path.expanduser("~/statiz/data")
DEFAULT_FEATURES_CSV = os.path.join(DATA_DIR, "features_v2_candidates.csv")
DEFAULT_OUT_CSV = os.path.join(DATA_DIR, "feature_subset_search_v1.csv")
DEFAULT_OUT_TOP_CSV = os.path.join(DATA_DIR, "feature_subset_top20_v1.csv")

# 탐색 대상 피처 12개 (v2 후보 전체)
DEFAULT_FEATURE_COLS = [
    "diff_sum_ops_smooth",
    "diff_sum_ops_recent5",
    "diff_sp_oops",
    "diff_bullpen_fatigue",
    "diff_opp_sp_platoon_cnt",
    "diff_sp_bbip",
    "diff_pythag_winpct",
    "diff_recent10_winpct",
    "diff_top5_ops_smooth",
    "diff_team_stadium_winpct",
    "park_factor_stadium",
    "diff_team_stadium_winpct_pfadj",
]


def safe_auc(y_true, y_prob):
    """클래스가 하나뿐이면 NaN을 반환하는 안전한 AUC 계산."""
    y_unique = set(pd.Series(y_true).astype(int).tolist())
    if len(y_unique) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def run_backtest(df_all, feature_cols, block_days=7, test_year=2025):
    """
    주어진 피처 조합으로 expanding-window 백테스트를 실행합니다.

    Args:
        df_all: 전체 피처 DataFrame (date, y_home_win, s_no 포함)
        feature_cols: 사용할 피처 컬럼 이름 튜플
        block_days: 테스트 블록 크기 (기본 7일)
        test_year: 평가 대상 연도 (기본 2025)

    Returns:
        {"n_rows", "n_blocks", "acc", "logloss", "brier", "auc"} dict
        또는 데이터 부족 시 None
    """
    needed = list(feature_cols) + ["date", "y_home_win", "s_no"]
    missing = [c for c in needed if c not in df_all.columns]
    if missing:
        raise ValueError(f"missing columns for subset {feature_cols}: {missing}")

    df = df_all.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=needed).copy()
    if len(df) == 0:
        return None

    df["y_home_win"] = df["y_home_win"].astype(int)
    df = df.sort_values(["date", "s_no"]).reset_index(drop=True)

    test_df = df[df["date"].dt.year == test_year].copy()
    if len(test_df) == 0:
        return None

    first_test_date = test_df["date"].min().normalize()
    last_test_date = test_df["date"].max().normalize()

    y_all = []
    p_all = []
    yhat_all = []
    n_blocks_used = 0

    block_start = first_test_date
    while block_start <= last_test_date:
        block_end = block_start + pd.Timedelta(days=block_days - 1)

        tr = df[df["date"] < block_start].copy()
        te = df[(df["date"] >= block_start) & (df["date"] <= block_end)].copy()

        if len(te) == 0:
            block_start = block_start + pd.Timedelta(days=block_days)
            continue

        # 학습 데이터에 승/패 클래스가 모두 있어야 LR 학습 가능
        if tr["y_home_win"].nunique() < 2:
            block_start = block_start + pd.Timedelta(days=block_days)
            continue

        X_train = tr[list(feature_cols)].values
        y_train = tr["y_home_win"].values
        X_test = te[list(feature_cols)].values
        y_test = te["y_home_win"].values

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        C=1.0,
                        solver="liblinear",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)

        prob_home = model.predict_proba(X_test)[:, 1]
        pred_home = (prob_home >= 0.5).astype(int)

        y_all.extend(y_test.tolist())
        p_all.extend(prob_home.tolist())
        yhat_all.extend(pred_home.tolist())
        n_blocks_used += 1

        block_start = block_start + pd.Timedelta(days=block_days)

    if len(y_all) == 0:
        return None

    y_all = np.array(y_all)
    p_all = np.array(p_all)
    yhat_all = np.array(yhat_all)

    return {
        "n_rows": int(len(y_all)),
        "n_blocks": int(n_blocks_used),
        "acc": float(accuracy_score(y_all, yhat_all)),
        "logloss": float(log_loss(y_all, p_all, labels=[0, 1])),
        "brier": float(brier_score_loss(y_all, p_all)),
        "auc": float(safe_auc(y_all, p_all)) if not np.isnan(safe_auc(y_all, p_all)) else np.nan,
    }


def parse_args():
    """커맨드라인 인자를 파싱합니다."""
    p = argparse.ArgumentParser()
    p.add_argument("--features-csv", default=DEFAULT_FEATURES_CSV)
    p.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    p.add_argument("--out-top-csv", default=DEFAULT_OUT_TOP_CSV)
    p.add_argument("--test-year", type=int, default=2025)
    p.add_argument("--block-days", type=int, default=7)
    p.add_argument("--min-k", type=int, default=1)
    p.add_argument(
        "--max-k",
        type=int,
        default=0,
        help="0 means all features",
    )
    p.add_argument(
        "--feature-cols",
        default=",".join(DEFAULT_FEATURE_COLS),
        help="comma-separated feature columns",
    )
    return p.parse_args()


def main():
    """모든 피처 subset을 순회하며 백테스트하고 결과를 CSV로 저장합니다."""
    args = parse_args()

    if not os.path.exists(args.features_csv):
        raise FileNotFoundError(f"not found: {args.features_csv}")

    feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    if not feature_cols:
        raise ValueError("feature list is empty")

    # 탐색할 조합 크기 범위 설정
    max_k = len(feature_cols) if args.max_k <= 0 else min(args.max_k, len(feature_cols))
    min_k = max(1, args.min_k)
    if min_k > max_k:
        raise ValueError(f"invalid k range: min_k={min_k}, max_k={max_k}")

    df = pd.read_csv(args.features_csv)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns in csv: {missing}")

    # 전체 조합 수 계산 (진행률 표시용)
    rows = []
    subset_total = 0
    for k in range(min_k, max_k + 1):
        subset_total += len(list(itertools.combinations(feature_cols, k)))

    # 모든 조합에 대해 백테스트 실행
    done = 0
    for k in range(min_k, max_k + 1):
        for comb in itertools.combinations(feature_cols, k):
            done += 1
            metrics = run_backtest(df, comb, block_days=args.block_days, test_year=args.test_year)
            if metrics is None:
                continue

            rows.append(
                {
                    "k": k,
                    "subset": "+".join(comb),
                    "n_rows": metrics["n_rows"],
                    "n_blocks": metrics["n_blocks"],
                    "acc": metrics["acc"],
                    "logloss": metrics["logloss"],
                    "brier": metrics["brier"],
                    "auc": metrics["auc"],
                }
            )

            # 50개마다 진행률 출력
            if done % 50 == 0 or done == subset_total:
                print(f"[progress] {done}/{subset_total}")

    if not rows:
        raise RuntimeError("no valid subset result")

    # LOGLOSS → BRIER → AUC 순으로 정렬 (낫을 수록 좋은 순)
    res = pd.DataFrame(rows)
    res["auc_sort"] = res["auc"].fillna(-1.0)
    res = res.sort_values(["logloss", "brier", "auc_sort"], ascending=[True, True, False]).reset_index(drop=True)
    res = res.drop(columns=["auc_sort"])

    # 상위 20개 추출
    top = res.head(20).copy()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    res.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    top.to_csv(args.out_top_csv, index=False, encoding="utf-8-sig")

    print("=== TOP 20 (by LOGLOSS, then BRIER, then AUC) ===")
    print(top.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"[OK] wrote : {args.out_csv}")
    print(f"[OK] wrote : {args.out_top_csv}")


if __name__ == "__main__":
    main()
