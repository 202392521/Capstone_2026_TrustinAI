#!/usr/bin/env python3
"""
Rank screened SIC-relevant subreddits by observed AI discussion.

Input is the monthly raw CSV produced by arctic_shift_subreddit_counts.py.
Optional failure log is used only as a data-quality flag.
"""

# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


COUNT_COLS = ["chatgpt_count", "ai_count", "generative_ai_count", "any_keyword_count", "total_posts_month"]


def parse_int(value: object) -> int:
    if value is None or value == "":
        return 0
    return int(float(str(value)))


def parse_failures(path: Optional[Path]) -> Dict[str, Set[Tuple[str, str]]]:
    failures: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    if not path or not path.exists() or path.stat().st_size == 0:
        return failures

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subreddit = (row.get("subreddit") or "").strip()
            month = (row.get("month") or "").strip()
            metric = (row.get("metric") or "").strip()
            if subreddit and month and metric:
                failures[subreddit.lower()].add((month, metric))
    return failures


def load_monthly_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    missing = [col for col in ["subreddit", "month", *COUNT_COLS] if col not in rows[0]]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")
    return rows


def quality_label(total_available: bool, failure_count: int) -> str:
    if failure_count and not total_available:
        return "partial_counts_no_total"
    if failure_count:
        return "partial_counts"
    if not total_available:
        return "complete_counts_no_total"
    return "complete"


def aggregate(rows: Sequence[Dict[str, str]], failures: Dict[str, Set[Tuple[str, str]]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("subreddit") or "").strip()].append(row)

    output: List[Dict[str, object]] = []
    for subreddit, sub_rows in grouped.items():
        if not subreddit:
            continue
        sub_rows = sorted(sub_rows, key=lambda r: r["month"])
        totals = {col: sum(parse_int(r.get(col)) for r in sub_rows) for col in COUNT_COLS}
        ai_rows = [r for r in sub_rows if parse_int(r.get("any_keyword_count")) > 0]
        chatgpt_rows = [r for r in sub_rows if parse_int(r.get("chatgpt_count")) > 0]
        genai_rows = [r for r in sub_rows if parse_int(r.get("generative_ai_count")) > 0]

        peak_row = max(sub_rows, key=lambda r: parse_int(r.get("any_keyword_count")))
        total_available = totals["total_posts_month"] >= totals["any_keyword_count"] and totals["total_posts_month"] > 0
        failure_count = len(failures.get(subreddit.lower(), set()))
        total_posts_period = totals["total_posts_month"] if total_available else ""
        any_per_1000 = (
            f"{totals['any_keyword_count'] / totals['total_posts_month'] * 1000:.4f}"
            if total_available
            else ""
        )

        output.append(
            {
                "subreddit": subreddit,
                "has_ai_discussion": 1 if totals["any_keyword_count"] > 0 else 0,
                "months_observed": len({r["month"] for r in sub_rows}),
                "active_ai_months": len(ai_rows),
                "any_keyword_posts": totals["any_keyword_count"],
                "chatgpt_posts": totals["chatgpt_count"],
                "ai_posts": totals["ai_count"],
                "generative_ai_posts": totals["generative_ai_count"],
                "first_ai_month": ai_rows[0]["month"] if ai_rows else "",
                "last_ai_month": ai_rows[-1]["month"] if ai_rows else "",
                "first_chatgpt_month": chatgpt_rows[0]["month"] if chatgpt_rows else "",
                "first_generative_ai_month": genai_rows[0]["month"] if genai_rows else "",
                "peak_ai_month": peak_row["month"] if totals["any_keyword_count"] > 0 else "",
                "peak_ai_posts": parse_int(peak_row.get("any_keyword_count")) if totals["any_keyword_count"] > 0 else 0,
                "total_posts_period": total_posts_period,
                "ai_posts_per_1000_total_posts": any_per_1000,
                "failure_events": failure_count,
                "data_quality": quality_label(total_available, failure_count),
            }
        )

    return sorted(
        output,
        key=lambda r: (
            int(r["has_ai_discussion"]),
            int(r["any_keyword_posts"]),
            int(r["active_ai_months"]),
            float(r["ai_posts_per_1000_total_posts"] or 0),
        ),
        reverse=True,
    )


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "subreddit",
        "has_ai_discussion",
        "months_observed",
        "active_ai_months",
        "any_keyword_posts",
        "chatgpt_posts",
        "ai_posts",
        "generative_ai_posts",
        "first_ai_month",
        "last_ai_month",
        "first_chatgpt_month",
        "first_generative_ai_month",
        "peak_ai_month",
        "peak_ai_posts",
        "total_posts_period",
        "ai_posts_per_1000_total_posts",
        "failure_events",
        "data_quality",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank SIC-screened subreddits by AI discussion.")
    parser.add_argument("--input", required=True, help="Raw monthly AS counts CSV.")
    parser.add_argument("--failures", help="Optional failures CSV from arctic_shift_subreddit_counts.py.")
    parser.add_argument("--output", required=True, help="Ranked output CSV.")
    parser.add_argument("--active-output", help="Optional output CSV containing only subreddits with AI discussion.")
    parser.add_argument("--inactive-output", help="Optional output CSV containing only subreddits with zero AI discussion.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows = load_monthly_rows(Path(args.input).expanduser())
    failures = parse_failures(Path(args.failures).expanduser() if args.failures else None)
    ranked = aggregate(rows, failures)
    write_csv(Path(args.output).expanduser(), ranked)

    active = [row for row in ranked if int(row["has_ai_discussion"]) == 1]
    inactive = [row for row in ranked if int(row["has_ai_discussion"]) == 0]
    if args.active_output:
        write_csv(Path(args.active_output).expanduser(), active)
    if args.inactive_output:
        write_csv(Path(args.inactive_output).expanduser(), inactive)

    print(f"Subreddits observed: {len(ranked)}")
    print(f"With AI discussion: {len(active)}")
    print(f"Without observed AI discussion: {len(inactive)}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
