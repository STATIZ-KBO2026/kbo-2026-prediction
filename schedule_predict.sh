#!/bin/bash
# KBO 일일 예측 자동 실행 스크립트
# cron에서 호출됩니다.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

DATE_TAG=$(date +%Y%m%d)
LOG_FILE="$LOG_DIR/cron_${DATE_TAG}.log"

echo "=== predict_today.py 시작: $(date) ===" >> "$LOG_FILE" 2>&1

cd "$SCRIPT_DIR"
/usr/bin/python3 "$SCRIPT_DIR/predict_today.py" >> "$LOG_FILE" 2>&1

echo "=== 완료: $(date) ===" >> "$LOG_FILE" 2>&1
