#!/bin/bash
# launch_scheduler.sh - cron에서 00:00에 호출
# crontab:
#   0 0 * * * /home/ubuntu/kbo-2026-prediction/scripts/launch_scheduler.sh >> /home/ubuntu/statiz/logs/scheduler.log 2>&1

set -e

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
export HOME="/home/ubuntu"

# .env 파일에서 API 키 로드
if [ -f "$HOME/.statiz_env" ]; then
    source "$HOME/.statiz_env"
fi

REPO="/home/ubuntu/kbo-2026-prediction"
LOG_DIR="$HOME/statiz/logs"
mkdir -p "$LOG_DIR"

TODAY=$(date +%Y%m%d)
LOG="$LOG_DIR/scheduler_${TODAY}.log"

echo "=== launch_scheduler.sh start: $(date) ===" | tee -a "$LOG"

cd "$REPO"
python3 scripts/kbo_scheduler.py --season-mode all 2>&1 | tee -a "$LOG"

echo "=== launch_scheduler.sh end: $(date) ===" | tee -a "$LOG"
