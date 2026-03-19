"""
v1 피처 기반 Expanding-Window 백테스트 (로지스틱 회귀)

v1 핵심 4피처(OPS_smooth, OPS_recent5, 선발투수 피안타OPS, 불펜 피로도)를 사용하여
KBO 경기 승패를 예측하는 모델의 성능을 평가합니다.

[백테스트 방식: Expanding Window]
  - 2024 시즌 데이터로 초기 학습 세트를 구성합니다.
  - 2025 시즌을 7일 단위 블록으로 나눕니다.
  - 각 블록에서:
    1) 해당 블록 이전의 모든 데이터로 모델을 학습 (→ 점점 커지는 학습 세트)
    2) 해당 블록의 경기를 예측
    3) 정확도(ACC), 로그손실(LOGLOSS), 브라이어(BRIER), AUC를 측정

  이 방식은 "시간 순서를 지키면서 모델을 평가"하는 현실적인 백테스트입니다.

[모델]
  StandardScaler → LogisticRegression (L2 정규화, C=1.0)

[파이프라인 위치]
  9a단계 — build_features_v1_paper.py 이후에 실행합니다.

[입력]
  - ~/statiz/data/features_v1_paper.csv

[출력]
  - ~/statiz/data/backtest_pred_v1.csv           — 경기별 예측 결과
  - ~/statiz/data/backtest_block_metrics_v1.csv   — 블록별 성능 지표
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

# 입력 피처 4개 (v1 핵심 피처)
FEATURE_COLS = [
    "diff_sum_ops_smooth",       # 타선 OPS_smooth 합 차이
    "diff_sum_ops_recent5",      # 최근 5경기 OPS 합 차이
    "diff_sp_oops",              # 선발투수 피안타 OPS 차이
    "diff_bullpen_fatigue",      # 불펜 피로도 차이
]

BLOCK_DAYS = 7     # 테스트 블록 크기 (7일 단위)
TEST_YEAR = 2025   # 평가 대상 연도


def safe_auc(y_true, y_prob):
    """
    AUC를 안전하게 계산합니다.
    테스트 블록에 클래스가 하나만 있으면(전부 승 또는 전부 패) AUC 계산이 불가능하므로,
    그 경우 NaN을 반환합니다.
    """
    y_unique = set(pd.Series(y_true).astype(int).tolist())
    if len(y_unique) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def main():
    """v1 피처로 expanding-window 백테스트를 실행하고 결과를 CSV로 저장합니다."""

    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"not found: {FEATURES_CSV}")

    df = pd.read_csv(FEATURES_CSV)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")

    # 필수 컬럼 확인
    needed = FEATURE_COLS + ["date", "y_home_win", "s_no", "homeTeam", "awayTeam"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    # 무한대/결측치 제거 후 정렬
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLS + ["date", "y_home_win"]).copy()
    df["y_home_win"] = df["y_home_win"].astype(int)
    df = df.sort_values(["date", "s_no"]).reset_index(drop=True)

    # 2024 시즌 = 초기 학습 세트, 2025 시즌 = 테스트 대상
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

    # 7일 단위 블록으로 순회하며 백테스트
    while block_start <= last_test_date:
        block_end = block_start + pd.Timedelta(days=BLOCK_DAYS - 1)

        # 학습 세트: 블록 시작일 이전의 모든 데이터 (expanding window)
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

        # 모델 학습: 표준화 → 로지스틱 회귀
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

        # 예측
        prob_home = model.predict_proba(X_test)[:, 1]    # 홈팀 승리 확률
        pred_home = (prob_home >= 0.5).astype(int)       # 0.5 기준 이진 예측

        # 블록별 성능 지표 계산
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

        # 경기별 예측 결과 저장
        tmp = te.copy()
        tmp["block_idx"] = block_idx + 1
        tmp["block_start"] = block_start.strftime("%Y-%m-%d")
        tmp["block_end"] = block_end.strftime("%Y-%m-%d")
        tmp["pred_home_win_proba"] = prob_home
        tmp["pred_home_win"] = pred_home
        pred_rows.append(tmp)

        block_idx += 1
        block_start = block_start + pd.Timedelta(days=BLOCK_DAYS)

    # ── 결과 합산 및 저장 ──
    pred_df = pd.concat(pred_rows, axis=0).sort_values(["date", "s_no"]).reset_index(drop=True)
    block_df = pd.DataFrame(block_rows)

    # 출력할 컬럼 선택
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

    # 전체 기간 통합 성능 지표
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
