#!/usr/bin/env bash
# Daily Agmarknet snapshot for Submission 2. Installed in crontab; safe to run by hand.
# Writes data_submission2/mandi/mandi_<date>.jsonl plus a hashed manifest, and logs one line.
cd /home/Debz/Research/Finetune_AgriFair/Codes || exit 1
set -a; . ./.env; set +a
LOG=data_submission2/mandi/mandi_cron.log
for attempt in 1 2 3; do
  if out=$(/home/Debz/Research/Finetune_AgriFair/.venv/bin/python -m Submission2_Run.fetch_mandi 2>&1); then
    echo "$(date '+%F %H:%M') ok   attempt=$attempt $(echo "$out" | tail -1)" >> "$LOG"; exit 0
  fi
  sleep 600
done
echo "$(date '+%F %H:%M') FAIL after 3 attempts: $(echo "$out" | tail -1 | cut -c1-200)" >> "$LOG"
exit 1
