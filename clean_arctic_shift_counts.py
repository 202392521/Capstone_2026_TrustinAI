#!/usr/bin/env python3
"""
Clean Arctic Shift raw output into two analysis-friendly CSV files:
1) monthly rows with only monthly denominators and rates
2) subreddit-level summary rows
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence


COUNT_COLUMNS = [
    "chatgpt_count",
    "ai_count",
    "generative_ai_count",
    "any_keyword_count",
    "total_posts_month",
]


def parse_int(value: str) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def rate(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return ""
    return f"{numerator / denominator:.8f}"


def total_available(total: int, any_keyword: int) -> bool:
    return total > 0 and total >= any_keyword


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    missing = [col for col in ["subreddit", "month", *COUNT_COLUMNS] if col not in rows[0]]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")
    return rows


def write_monthly(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subreddit",
        "month",
        "chatgpt_count",
        "ai_count",
        "generative_ai_count",
        "any_keyword_count",
        "total_posts_month",
        "total_posts_available",
        "any_keyword_share_of_month_posts",
        "chatgpt_share_of_month_posts",
        "ai_share_of_month_posts",
        "generative_ai_share_of_month_posts",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            total = parse_int(row.get("total_posts_month", "0"))
            chatgpt = parse_int(row.get("chatgpt_count", "0"))
            ai = parse_int(row.get("ai_count", "0"))
            generative_ai = parse_int(row.get("generative_ai_count", "0"))
            any_keyword = parse_int(row.get("any_keyword_count", "0"))
            has_total = total_available(total, any_keyword)
            writer.writerow(
                {
                    "subreddit": row["subreddit"],
                    "month": row["month"],
                    "chatgpt_count": chatgpt,
                    "ai_count": ai,
                    "generative_ai_count": generative_ai,
                    "any_keyword_count": any_keyword,
                    "total_posts_month": total if has_total else "",
                    "total_posts_available": 1 if has_total else 0,
                    "any_keyword_share_of_month_posts": rate(any_keyword, total) if has_total else "",
                    "chatgpt_share_of_month_posts": rate(chatgpt, total) if has_total else "",
                    "ai_share_of_month_posts": rate(ai, total) if has_total else "",
                    "generative_ai_share_of_month_posts": rate(generative_ai, total) if has_total else "",
                }
            )


def write_summary(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, Dict[str, int]] = {}
    month_counts: Dict[str, set] = {}
    for row in rows:
        subreddit = row["subreddit"]
        bucket = grouped.setdefault(
            subreddit,
            {
                "chatgpt_posts": 0,
                "ai_posts": 0,
                "generative_ai_posts": 0,
                "any_keyword_posts": 0,
                "total_posts_period": 0,
            },
        )
        month_counts.setdefault(subreddit, set()).add(row["month"])
        bucket["chatgpt_posts"] += parse_int(row.get("chatgpt_count", "0"))
        bucket["ai_posts"] += parse_int(row.get("ai_count", "0"))
        bucket["generative_ai_posts"] += parse_int(row.get("generative_ai_count", "0"))
        bucket["any_keyword_posts"] += parse_int(row.get("any_keyword_count", "0"))
        bucket["total_posts_period"] += parse_int(row.get("total_posts_month", "0"))

    fieldnames = [
        "subreddit",
        "months_observed",
        "chatgpt_posts",
        "ai_posts",
        "generative_ai_posts",
        "any_keyword_posts",
        "total_posts_period",
        "total_posts_available",
        "any_keyword_share_of_period_posts",
        "chatgpt_share_of_period_posts",
        "ai_share_of_period_posts",
        "generative_ai_share_of_period_posts",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for subreddit in sorted(grouped):
            bucket = grouped[subreddit]
            total = bucket["total_posts_period"]
            has_total = total_available(total, bucket["any_keyword_posts"])
            writer.writerow(
                {
                    "subreddit": subreddit,
                    "months_observed": len(month_counts[subreddit]),
                    "chatgpt_posts": bucket["chatgpt_posts"],
                    "ai_posts": bucket["ai_posts"],
                    "generative_ai_posts": bucket["generative_ai_posts"],
                    "any_keyword_posts": bucket["any_keyword_posts"],
                    "total_posts_period": total if has_total else "",
                    "total_posts_available": 1 if has_total else 0,
                    "any_keyword_share_of_period_posts": rate(bucket["any_keyword_posts"], total) if has_total else "",
                    "chatgpt_share_of_period_posts": rate(bucket["chatgpt_posts"], total) if has_total else "",
                    "ai_share_of_period_posts": rate(bucket["ai_posts"], total) if has_total else "",
                    "generative_ai_share_of_period_posts": rate(bucket["generative_ai_posts"], total) if has_total else "",
                }
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean Arctic Shift raw counts into monthly and summary CSVs.")
    parser.add_argument("--input", required=True, help="Raw Arctic Shift output CSV.")
    parser.add_argument("--monthly-output", required=True, help="Clean monthly output CSV.")
    parser.add_argument("--summary-output", required=True, help="Subreddit-level summary output CSV.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows = read_rows(Path(args.input).expanduser())
    write_monthly(rows, Path(args.monthly_output).expanduser())
    write_summary(rows, Path(args.summary_output).expanduser())
    print(f"Wrote {args.monthly_output}")
    print(f"Wrote {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
