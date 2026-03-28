#!/bin/bash
# daily_update_v1.sh
# ==================
# 매일 실행: 전날 데이터 수집 → features 재빌드 → 오늘 경기 예측 & 제출
#
# 사용법:
#   bash scripts/daily_update_v1.sh               # 오늘 날짜 자동 감지
#   bash scripts/daily_update_v1.sh 20260328      # 특정 날짜
#   bash scripts/daily_update_v1.sh 20260328 --dry-run  # 제출 없이 예측만

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Python 실행 경로
if [ -f ".venv311/bin/python" ]; then
    PY=".venv311/bin/python"
else
    PY="python3"
fi

# 날짜 결정
if [ -n "$1" ] && [[ "$1" =~ ^[0-9]{8}$ ]]; then
    TARGET_DATE="$1"
    shift
else
    TARGET_DATE=$(date +%Y%m%d)
fi

YESTERDAY=$(date -d "$TARGET_DATE - 1 day" +%Y%m%d 2>/dev/null || date -v-1d -j -f "%Y%m%d" "$TARGET_DATE" +%Y%m%d)
EXTRA_ARGS="$@"

echo "========================================"
echo " KBO v1 Daily Update"
echo " Target date: $TARGET_DATE"
echo " Yesterday:   $YESTERDAY"
echo " Extra args:  $EXTRA_ARGS"
echo "========================================"

# Step 1: 전날 데이터 수집 (game_details + playerday)
echo ""
echo "[1/4] Downloading yesterday's data ($YESTERDAY) ..."
$PY scripts/download_game_details.py --date-from "$YESTERDAY" --date-to "$YESTERDAY" 2>&1 || echo "  (game_details download skipped or failed)"
$PY scripts/download_playerday.py --date-from "$YESTERDAY" --date-to "$YESTERDAY" 2>&1 || echo "  (playerday download skipped or failed)"

# Step 2: 인덱스 & 라인업 & playerday 테이블 재빌드
echo ""
echo "[2/4] Rebuilding indexes ..."
$PY scripts/build_game_index.py 2>&1 || echo "  (game_index rebuild skipped)"
$PY scripts/build_lineup_table.py 2>&1 || echo "  (lineup rebuild skipped)"
$PY scripts/build_playerday_tables_v2.py 2>&1 || echo "  (playerday tables rebuild skipped)"

# Step 3: 오늘 스케줄 다운로드 (라인업 포함)
echo ""
echo "[3/4] Downloading today's schedule ($TARGET_DATE) ..."
$PY scripts/download_schedule.py --date-from "$TARGET_DATE" --date-to "$TARGET_DATE" 2>&1 || echo "  (schedule download skipped)"
$PY scripts/download_game_details.py --date-from "$TARGET_DATE" --date-to "$TARGET_DATE" 2>&1 || echo "  (today's game_details skipped)"

# Rebuild indexes again with today's data
$PY scripts/build_game_index.py 2>&1 || true
$PY scripts/build_lineup_table.py 2>&1 || true

# Step 4: features 빌드 + 모델 학습 + 예측 + 제출
echo ""
echo "[4/4] Running v1 pipeline ..."
$PY scripts/run_submit_pipeline_v1.py --date "$TARGET_DATE" $EXTRA_ARGS

echo ""
echo "========================================"
echo " Daily update complete!"
echo "========================================"
