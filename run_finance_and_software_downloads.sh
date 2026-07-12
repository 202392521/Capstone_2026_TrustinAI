#!/usr/bin/env bash
set -u

ROOT="/Users/qichenzhao/Documents/Codex/2026-06-26/hi"
SCRIPT="$ROOT/outputs/download_industry_top_commented_ai_threads.py"
LOG_DIR="$ROOT/outputs/overnight_download_logs"
mkdir -p "$LOG_DIR"

run_industry() {
  local industry="$1"
  local output_dir="$2"
  local log_name="$3"
  local log_path="$LOG_DIR/$log_name"

  {
    echo "=== $(date) starting industry: $industry ==="
    python3 -u "$SCRIPT" \
      --industry-keyword "$industry" \
      --output-dir "$output_dir" \
      --large-subreddit-count 5 \
      --large-top-posts 30 \
      --large-max-comments-per-post 1000 \
      --large-rest-every-posts 10 \
      --large-rest-minutes 10 \
      --rest-min-active-ai-months 10 \
      --rest-top-posts 10 \
      --rest-max-comments-per-post 300 \
      --subreddit-delay-minutes 1 \
      --pause 0.5
    status=$?
    echo "=== $(date) finished industry: $industry status=$status ==="
    return $status
  } >> "$log_path" 2>&1
}

run_industry \
  "accountant" \
  "$ROOT/outputs/finance_top_commented_ai_threads" \
  "finance_accountant_download.log"

finance_status=$?
if [ "$finance_status" -ne 0 ]; then
  echo "Finance/accountant download failed with status $finance_status. Not starting software engineering." >> "$LOG_DIR/master_download.log"
  exit "$finance_status"
fi

run_industry \
  "software engineer" \
  "$ROOT/outputs/software_engineering_top_commented_ai_threads" \
  "software_engineering_download.log"
