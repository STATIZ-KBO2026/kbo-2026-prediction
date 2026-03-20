"""
로지스틱 회귀 계수(coefficient) 분석기

backtest_v1_online_lr.py와 동일한 expanding-window 방식으로
매 블록마다 LR 모델을 학습한 뒤, 각 피처의 회귀 계수를 기록합니다.

[분석 목적]
  - 각 피처가 모델 예측에 얼마나 기여하는지 확인합니다.
  - 계수의 부호(+/-)가 블록마다 일관적인지 확인합니다.
  - 크기가 큰 계수 = 예측에 영향력이 큰 피처

[출력물 해석]
  - mean_coef:     전체 블록의 평균 계수
  - mean_abs_coef: 절대값 평균 (영향력 크기)
  - std_coef:      계수의 표준편차 (안정성)
  - pos_ratio:     양수(+) 비율 (방향 일관성,  1.0에 가까우면 항상 양수)

[파이프라인 위치]
  10단계 — backtest_v1_online_lr.py 기능의 해석/보조 분석입니다.

[입력]
  - ~/statiz/data/features_v1_paper.csv

[출력]
  - ~/statiz/data/backtest_lr_coef_v1.csv         — 블록별 계수 상세
  - ~/statiz/data/backtest_lr_coef_summary_v1.csv  — 피처별 요약 통계
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

# 분석 대상 피처 (v1 핵심 4피처)
FEATURE_COLS = [
    "diff_sum_ops_smooth",       # 타선 OPS_smooth 합 차이
    "diff_sum_ops_recent5",      # 최근 5경기 OPS 합 차이
    "diff_sp_oops",              # 선발투수 피안타 OPS 차이
    "diff_bullpen_fatigue",      # 불펜 피로도 차이
]

BLOCK_DAYS = 7     # 블록 크기
TEST_YEAR = 2025   # 평가 대상 연도


def main():
    """블록별로 LR 계수를 추출하고 피처별 요약 통계를 산출합니다."""

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

        # 스케일링 후 LR 학습 (백테스트와 동일한 설정)
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

        # 학습된 LR 모델에서 계수 추출
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

    # 피처별 요약 통계 계산
    summary_df = (
        coef_df.groupby("feature")
        .agg(
            mean_coef=("coef", "mean"),                                    # 평균 계수
            mean_abs_coef=("coef", lambda s: float(np.mean(np.abs(s)))),   # 절대값 평균
            std_coef=("coef", "std"),                                      # 표준편차
            pos_ratio=("coef", lambda s: float(np.mean(s > 0))),           # 양수 비율
            n_blocks=("coef", "size"),                                     # 블록 수
        )
        .sort_values("mean_abs_coef", ascending=False)  # 영향력 큰 순으로 정렬
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
