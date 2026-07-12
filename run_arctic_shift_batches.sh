#!/usr/bin/env bash
set -u

SCRIPT="/Users/qichenzhao/Documents/Codex/2026-06-26/hi/outputs/arctic_shift_subreddit_counts.py"
INPUT="/Users/qichenzhao/Documents/subreddit_screening.numbers"
OUTPUT="/Users/qichenzhao/Documents/Codex/2026-06-26/hi/outputs/subreddit_chatgpt_ai_counts_as.csv"

START_INDEX="${START_INDEX:-66}"
END_INDEX="${END_INDEX:-221}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DELAY_MINUTES="${DELAY_MINUTES:-10}"
RETRIES="${RETRIES:-1}"
MONTHLY_FALLBACK_RETRIES="${MONTHLY_FALLBACK_RETRIES:-1}"
MAX_MONTH_FAILURES="${MAX_MONTH_FAILURES:-3}"
TIMEOUT="${TIMEOUT:-90}"
PAUSE="${PAUSE:-0.5}"

current="$START_INDEX"
while [ "$current" -le "$END_INDEX" ]; do
  echo "=== Running subreddit batch starting at ${current}, batch size ${BATCH_SIZE} ==="
  python3 "$SCRIPT" \
    --input "$INPUT" \
    --yes-column final_decision \
    --subreddit-column subreddit \
    --start-month 2023-03 \
    --end-month 2026-01 \
    --start-index "$current" \
    --batch-size "$BATCH_SIZE" \
    --output "$OUTPUT" \
    --resume \
    --retries "$RETRIES" \
    --monthly-fallback-retries "$MONTHLY_FALLBACK_RETRIES" \
    --max-month-failures "$MAX_MONTH_FAILURES" \
    --timeout "$TIMEOUT" \
    --pause "$PAUSE"

  status="$?"
  if [ "$status" -ne 0 ]; then
    echo "Batch starting at ${current} failed with status ${status}."
    echo "Fix the issue, then rerun this script with START_INDEX=${current}."
    exit "$status"
  fi

  current=$((current + BATCH_SIZE))
  if [ "$current" -le "$END_INDEX" ]; then
    echo "=== Sleeping ${DELAY_MINUTES} minutes before next batch ==="
    sleep "$((DELAY_MINUTES * 60))"
  fi
done

echo "=== All requested batches finished ==="
