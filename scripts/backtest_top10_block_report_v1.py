"""
상위 N개 피처 조합 상세 블록 리포트

feature_subset_search_v1.py가 찾은 상위 조합들에 대해
블록별 상세 성능 지표 + 경기별 예측 결과를 생성합니다.

[목적]
  - subset 탐색은 전체 요약만 보여주는 반면, 이 스크립트는
    각 조합의 블록별 성능 추이와 개별 경기 예측값을 확인할 수 있습니다.
  - "어떤 시기에 어떤 조합이 잘/못 맞추는지" 분석하는 데 사용합니다.

[파이프라인 위치]
  9c단계 — feature_subset_search_v1.py 이후에 실행합니다.

[입력]
  - ~/statiz/data/features_v2_candidates.csv    — 피처 데이터
  - ~/statiz/data/feature_subset_search_v1.csv  — 조합 순위 결과

[출력]
  - ~/statiz/data/top10_subset_summary_v1.csv      — 각 조합의 전체 요약 성능
  - ~/statiz/data/top10_subset_block_metrics_v1.csv — 각 조합의 블록별 상세 성능
  - ~/statiz/data/top10_subset_pred_v1.csv          — 각 조합의 경기별 예측 결과
"""
import os
import argparse
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score


DATA_DIR = os.path.expanduser("~/statiz/data")

DEFAULT_FEATURES_CSV = os.path.join(DATA_DIR, "features_v2_candidates.csv")
DEFAULT_SUBSET_RANK_CSV = os.path.join(DATA_DIR, "feature_subset_search_v1.csv")
DEFAULT_OUT_BLOCK_CSV = os.path.join(DATA_DIR, "top10_subset_block_metrics_v1.csv")
DEFAULT_OUT_SUMMARY_CSV = os.path.join(DATA_DIR, "top10_subset_summary_v1.csv")
DEFAULT_OUT_PRED_CSV = os.path.join(DATA_DIR, "top10_subset_pred_v1.csv")


def safe_auc(y_true, y_prob):
    """클래스가 하나뿐이면 NaN을 반환하는 안전한 AUC 계산."""
    uniq = set(pd.Series(y_true).astype(int).tolist())
    if len(uniq) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def run_one_subset(df_all, feature_cols, block_days=7, test_year=2025):
    """
    한 개의 피처 조합에 대해 expanding-window 백테스트를 수행하고
    전체 요약, 블록별 지표, 경기별 예측을 반환합니다.

    Returns:
        (overall_dict, block_rows_list, pred_df) 튜플
        데이터 부족 시 (None, [], []) 반환
    """
    need = list(feature_cols) + ["date", "s_no", "y_home_win"]
    missing = [c for c in need if c not in df_all.columns]
    if missing:
        raise ValueError(f"missing columns for subset={feature_cols}: {missing}")

    df = df_all.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=need).copy()
    if len(df) == 0:
        return None, [], []

    df["y_home_win"] = df["y_home_win"].astype(int)
    df = df.sort_values(["date", "s_no"]).reset_index(drop=True)

    test_df = df[df["date"].dt.year == test_year].copy()
    if len(test_df) == 0:
        return None, [], []

    # 예측 결과에 추가로 포함할 메타 컬럼
    extra_cols = [c for c in ["homeTeam", "awayTeam", "s_code", "homeScore", "awayScore"] if c in df.columns]

    first_test_date = test_df["date"].min().normalize()
    last_test_date = test_df["date"].max().normalize()

    block_rows = []
    pred_rows = []

    y_all = []
    p_all = []
    yhat_all = []

    block_idx = 0
    block_start = first_test_date
    while block_start <= last_test_date:
        block_end = block_start + pd.Timedelta(days=block_days - 1)

        tr = df[df["date"] < block_start].copy()
        te = df[(df["date"] >= block_start) & (df["date"] <= block_end)].copy()

        if len(te) == 0:
            block_start = block_start + pd.Timedelta(days=block_days)
            continue

        # 학습 데이터에 승/패가 모두 있어야 LR 학습 가능
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

        # 블록별 성능 측정
        acc = accuracy_score(y_test, pred_home)
        ll = log_loss(y_test, prob_home, labels=[0, 1])
        br = brier_score_loss(y_test, prob_home)
        auc = safe_auc(y_test, prob_home)

        block_rows.append(
            {
                "block_idx": block_idx + 1,
                "block_start": block_start.strftime("%Y-%m-%d"),
                "block_end": block_end.strftime("%Y-%m-%d"),
                "n_train": len(tr),
                "n_test": len(te),
                "acc": float(acc),
                "logloss": float(ll),
                "brier": float(br),
                "auc": float(auc) if not np.isnan(auc) else np.nan,
            }
        )

        # 경기별 예측 결과 저장
        tmp = te[["date", "s_no", "y_home_win"] + extra_cols].copy()
        tmp["block_idx"] = block_idx + 1
        tmp["block_start"] = block_start.strftime("%Y-%m-%d")
        tmp["block_end"] = block_end.strftime("%Y-%m-%d")
        tmp["pred_home_win_proba"] = prob_home
        tmp["pred_home_win"] = pred_home
        pred_rows.append(tmp)

        y_all.extend(y_test.tolist())
        p_all.extend(prob_home.tolist())
        yhat_all.extend(pred_home.tolist())

        block_idx += 1
        block_start = block_start + pd.Timedelta(days=block_days)

    if len(y_all) == 0:
        return None, [], []

    y_all = np.array(y_all)
    p_all = np.array(p_all)
    yhat_all = np.array(yhat_all)

    # 전체 기간 통합 성능
    auc_all = safe_auc(y_all, p_all)
    overall = {
        "n_rows": int(len(y_all)),
        "n_blocks": int(len(block_rows)),
        "acc": float(accuracy_score(y_all, yhat_all)),
        "logloss": float(log_loss(y_all, p_all, labels=[0, 1])),
        "brier": float(brier_score_loss(y_all, p_all)),
        "auc": float(auc_all) if not np.isnan(auc_all) else np.nan,
    }

    pred_df = pd.concat(pred_rows, axis=0).sort_values(["date", "s_no"]).reset_index(drop=True)
    pred_df["date"] = pred_df["date"].dt.strftime("%Y%m%d")

    return overall, block_rows, pred_df


def parse_args():
    """커맨드라인 인자를 파싱합니다."""
    p = argparse.ArgumentParser()
    p.add_argument("--features-csv", default=DEFAULT_FEATURES_CSV)
    p.add_argument("--subset-rank-csv", default=DEFAULT_SUBSET_RANK_CSV)
    p.add_argument("--out-block-csv", default=DEFAULT_OUT_BLOCK_CSV)
    p.add_argument("--out-summary-csv", default=DEFAULT_OUT_SUMMARY_CSV)
    p.add_argument("--out-pred-csv", default=DEFAULT_OUT_PRED_CSV)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--block-days", type=int, default=7)
    p.add_argument("--test-year", type=int, default=2025)
    return p.parse_args()


def main():
    """상위 subset들에 대해 상세 블록 리포트를 생성합니다."""
    args = parse_args()

    if not os.path.exists(args.features_csv):
        raise FileNotFoundError(f"not found: {args.features_csv}")
    if not os.path.exists(args.subset_rank_csv):
        raise FileNotFoundError(f"not found: {args.subset_rank_csv}")

    feat_df = pd.read_csv(args.features_csv)
    feat_df["date"] = pd.to_datetime(feat_df["date"].astype(str), format="%Y%m%d")

    # 순위 파일에서 상위 N개 조합 읽기
    rank_df = pd.read_csv(args.subset_rank_csv)
    if "subset" not in rank_df.columns:
        raise ValueError("subset rank csv must include 'subset' column")

    top_n = max(1, int(args.top_n))
    top_df = rank_df.head(top_n).copy()
    if len(top_df) == 0:
        raise ValueError("subset rank csv is empty")

    summary_rows = []
    block_all_rows = []
    pred_all = []

    # 각 조합에 대해 상세 백테스트 실행
    for i, row in top_df.reset_index(drop=True).iterrows():
        subset_rank = i + 1
        subset = str(row["subset"]).strip()
        cols = [c for c in subset.split("+") if c]  # "col1+col2+col3" → ["col1","col2","col3"]
        k = len(cols)
        if k == 0:
            continue

        overall, block_rows, pred_df = run_one_subset(
            feat_df,
            cols,
            block_days=args.block_days,
            test_year=args.test_year,
        )
        if overall is None:
            continue

        # 요약 행 추가
        summary_rows.append(
            {
                "subset_rank": subset_rank,
                "k": k,
                "subset": subset,
                "n_rows": overall["n_rows"],
                "n_blocks": overall["n_blocks"],
                "acc": overall["acc"],
                "logloss": overall["logloss"],
                "brier": overall["brier"],
                "auc": overall["auc"],
            }
        )

        # 블록별 상세 지표에 조합 정보 추가
        for b in block_rows:
            b["subset_rank"] = subset_rank
            b["k"] = k
            b["subset"] = subset
            block_all_rows.append(b)

        # 경기별 예측에 조합 정보 추가
        pred_df["subset_rank"] = subset_rank
        pred_df["k"] = k
        pred_df["subset"] = subset
        pred_all.append(pred_df)

        print(
            f"[{subset_rank:02d}/{len(top_df):02d}] "
            f"k={k} subset={subset} | "
            f"LOGLOSS={overall['logloss']:.4f} "
            f"BRIER={overall['brier']:.4f} "
            f"AUC={overall['auc']:.4f}"
        )

    if not summary_rows:
        raise RuntimeError("no valid top subset results were produced")

    # CSV 저장
    summary_out = pd.DataFrame(summary_rows).sort_values("subset_rank").reset_index(drop=True)
    block_out = pd.DataFrame(block_all_rows).sort_values(["subset_rank", "block_idx"]).reset_index(drop=True)
    pred_out = pd.concat(pred_all, axis=0).sort_values(["subset_rank", "date", "s_no"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out_summary_csv), exist_ok=True)
    summary_out.to_csv(args.out_summary_csv, index=False, encoding="utf-8-sig")
    block_out.to_csv(args.out_block_csv, index=False, encoding="utf-8-sig")
    pred_out.to_csv(args.out_pred_csv, index=False, encoding="utf-8-sig")

    print("\n=== TOP SUBSET OVERALL ===")
    print(summary_out.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"[OK] wrote : {args.out_summary_csv}")
    print(f"[OK] wrote : {args.out_block_csv}")
    print(f"[OK] wrote : {args.out_pred_csv}")


if __name__ == "__main__":
    main()
