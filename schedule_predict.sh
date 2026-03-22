#!/bin/bash
# KBO 일일 예측 자동 실행 스크립트
#
# 30분 간격으로 cron에서 호출됩니다.
# predict_today.py가 이미 제출된 경기를 자동 건너뛰므로,
# 여러 번 실행해도 중복 제출 없이 안전합니다.
#
# cron 설정 (UTC 기준, KST=UTC+9):
#   */30 2-9 * * * /home/ubuntu/statiz/schedule_predict.sh
#   → KST 11:00~18:00 사이 매 30분 실행

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

DATE_TAG=$(date +%Y%m%d)
LOG_FILE="$LOG_DIR/cron_${DATE_TAG}.log"

echo "=== $(date) ===" >> "$LOG_FILE" 2>&1
cd "$SCRIPT_DIR"
/usr/bin/python3 "$SCRIPT_DIR/predict_today.py" >> "$LOG_FILE" 2>&1
echo "" >> "$LOG_FILE" 2>&1
