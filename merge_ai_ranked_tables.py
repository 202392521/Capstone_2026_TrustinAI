#!/usr/bin/env python3
"""Merge multiple ai_ranked-style CSV files into one de-duplicated table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence


BASE_FIELDS = [
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

QUALITY_SCORE = {
    "complete": 4,
    "complete_counts_no_total": 3,
    "partial_counts": 2,
    "partial_counts_no_total": 1,
}


def parse_int(value: object) -> int:
    if value is None or value == "":
        return 0
    return int(float(str(value)))


def parse_float(value: object) -> float:
    if value is None or value == "":
        return 0.0
    return float(str(value))


def row_score(row: Dict[str, str]) -> tuple:
    return (
        QUALITY_SCORE.get(row.get("data_quality", ""), 0),
        parse_int(row.get("months_observed")),
        parse_int(row.get("any_keyword_posts")),
        parse_int(row.get("active_ai_months")),
        parse_float(row.get("ai_posts_per_1000_total_posts")),
    )


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def merge(inputs: Sequence[Path]) -> List[Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}
    sources: Dict[str, List[str]] = {}

    for path in inputs:
        rows = read_rows(path)
        for row in rows:
            subreddit = (row.get("subreddit") or "").strip()
            if not subreddit:
                continue
            key = subreddit.lower()
            sources.setdefault(key, [])
            if path.name not in sources[key]:
                sources[key].append(path.name)
            if key not in merged or row_score(row) > row_score(merged[key]):
                merged[key] = row

    out_rows = []
    for key, row in merged.items():
        cleaned = {field: row.get(field, "") for field in BASE_FIELDS}
        cleaned["source_files"] = ";".join(sources.get(key, []))
        out_rows.append(cleaned)

    return sorted(
        out_rows,
        key=lambda r: (
            parse_int(r["has_ai_discussion"]),
            parse_int(r["any_keyword_posts"]),
            parse_int(r["active_ai_months"]),
            parse_float(r["ai_posts_per_1000_total_posts"]),
        ),
        reverse=True,
    )


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    fieldnames = [*BASE_FIELDS, "source_files"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge ai_ranked CSV files by subreddit.")
    parser.add_argument("--input", nargs="+", required=True, help="Input ai_ranked/with_ai_discussion CSV files.")
    parser.add_argument("--output", required=True, help="Merged output CSV.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows = merge([Path(p).expanduser() for p in args.input])
    write_csv(Path(args.output).expanduser(), rows)
    active = sum(parse_int(row["has_ai_discussion"]) for row in rows)
    print(f"Merged subreddits: {len(rows)}")
    print(f"With AI discussion: {active}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
