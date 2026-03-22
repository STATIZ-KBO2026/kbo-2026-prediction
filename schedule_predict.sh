#!/bin/bash
# KBO 일일 예측 자동 실행 스크립트 (재시도 포함)
#
# 라인업은 경기 ~1시간 전에 발표되지만 정확한 시간은 매일 다릅니다.
# 12:05에 첫 시도 후, 라인업이 없으면 5분 간격으로 최대 8회 재시도합니다.
# → 12:05, 12:10, 12:15, 12:20, 12:25, 12:30, 12:35, 12:40 (마감 12:45 전)
#
# cron 설정: 5 3 * * * /home/ubuntu/statiz/schedule_predict.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

DATE_TAG=$(date +%Y%m%d)
LOG_FILE="$LOG_DIR/cron_${DATE_TAG}.log"

MAX_RETRIES=8
RETRY_INTERVAL=300  # 5분 (초)

for attempt in $(seq 1 $MAX_RETRIES); do
    echo "=== 시도 $attempt/$MAX_RETRIES: $(date) ===" >> "$LOG_FILE" 2>&1

    cd "$SCRIPT_DIR"
    /usr/bin/python3 "$SCRIPT_DIR/predict_today.py" >> "$LOG_FILE" 2>&1
    EXIT_CODE=$?

    # 성공 여부 확인: predict CSV 로그에 "OK" 또는 "저장 성공"이 있는지
    PREDICT_LOG="$LOG_DIR/predict_${DATE_TAG}.csv"
    if [ -f "$PREDICT_LOG" ] && grep -q "OK" "$PREDICT_LOG"; then
        echo "=== ✅ 제출 성공! 재시도 중단: $(date) ===" >> "$LOG_FILE" 2>&1
        exit 0
    fi

    if [ $attempt -lt $MAX_RETRIES ]; then
        echo "=== ⏳ ${RETRY_INTERVAL}초 후 재시도... ===" >> "$LOG_FILE" 2>&1
        sleep $RETRY_INTERVAL
    else
        echo "=== ❌ 최대 재시도 초과: $(date) ===" >> "$LOG_FILE" 2>&1
    fi
done
