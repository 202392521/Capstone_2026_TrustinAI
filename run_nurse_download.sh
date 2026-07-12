#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="/Users/qichenzhao/Documents/Codex/2026-06-26/hi/outputs"
LOG_DIR="${OUTPUT_ROOT}/overnight_download_logs"
mkdir -p "${LOG_DIR}"

python3 -u "${OUTPUT_ROOT}/download_industry_top_commented_ai_threads.py" \
  --industry-keyword "nurse" \
  --targets "${OUTPUT_ROOT}/nurse_active_ai_months_ge_10_targets.csv" \
  --output-dir "${OUTPUT_ROOT}/nurse_top_commented_ai_threads" \
  --start-month 2023-03 \
  --end-month 2026-01 \
  --large-subreddit-count 5 \
  --large-top-posts 30 \
  --large-max-comments-per-post 1000 \
  --large-rest-every-posts 10 \
  --large-rest-minutes 10 \
  --rest-min-active-ai-months 10 \
  --rest-top-posts 10 \
  --rest-max-comments-per-post 300 \
  --subreddit-delay-minutes 1 \
  --max-candidate-posts-per-query 2000 \
  --pause 0.5 \
  --retries 3 \
  --timeout 90
