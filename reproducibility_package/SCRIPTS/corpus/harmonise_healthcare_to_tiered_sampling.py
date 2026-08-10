#!/usr/bin/env python3
"""Harmonise healthcare comments to the finance/software tiered sampling rule.

The original healthcare collection attempted top 30 AI-relevant posts for all
eligible subreddits. This script retrospectively applies the same analytical
inclusion rule used for finance/accountant and software engineering:

- first 5 ranked subreddits: keep top 30 most-commented AI candidate posts
- remaining active targets: keep top 10 most-commented AI candidate posts

It does not call any API. It only filters the already downloaded local CSVs.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


OUTPUT_DIR = Path("outputs")
SOURCE_DIR = OUTPUT_DIR / "healthcare_top_commented_ai_threads"
TARGET_DIR = OUTPUT_DIR / "healthcare_top_commented_ai_threads_harmonised_tiered"

COMMENTS_IN = SOURCE_DIR / "healthcare_top_commented_ai_comments_uk_flagged.csv"
POSTS_IN = SOURCE_DIR / "healthcare_top_commented_ai_posts.csv"
TARGETS_IN = OUTPUT_DIR / "healthcare_active_ai_months_ge_10_targets.csv"

COMMENTS_OUT = TARGET_DIR / "healthcare_top_commented_ai_comments_uk_flagged_harmonised_tiered.csv"
POSTS_OUT = TARGET_DIR / "healthcare_top_commented_ai_posts_harmonised_tiered.csv"
UK_STRONG_OUT = TARGET_DIR / "healthcare_top_commented_ai_comments_uk_strong_only_harmonised_tiered.csv"
SUMMARY_OUT = TARGET_DIR / "healthcare_harmonised_tiered_summary.csv"
SUBREDDIT_SUMMARY_OUT = TARGET_DIR / "healthcare_harmonised_tiered_subreddit_summary.csv"


def read_rows(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, rows: Iterable[Dict[str, str]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value or "").strip()))
    except ValueError:
        return default


def is_uk_strong(row: Dict[str, str]) -> bool:
    return as_int(row.get("uk_strong_relevant")) == 1


def keep_limit(row: Dict[str, str]) -> int:
    rank = as_int(row.get("industry_rank_by_total_posts"))
    return 30 if rank and rank <= 5 else 10


def keep_row(row: Dict[str, str]) -> bool:
    post_rank = as_int(row.get("candidate_rank_by_num_comments"))
    return post_rank > 0 and post_rank <= keep_limit(row)


def policy_for(row: Dict[str, str]) -> str:
    rank = as_int(row.get("industry_rank_by_total_posts"))
    return "large_top30" if rank and rank <= 5 else "rest_top10"


def main() -> int:
    comment_fields, comments = read_rows(COMMENTS_IN)
    post_fields, posts = read_rows(POSTS_IN)
    _, targets = read_rows(TARGETS_IN)

    kept_comments = []
    for row in comments:
        if keep_row(row):
            out = dict(row)
            out["download_policy"] = policy_for(row)
            kept_comments.append(out)

    kept_post_ids = {(row.get("subreddit", "").lower(), row.get("post_id", "")) for row in kept_comments}
    kept_posts = []
    for row in posts:
        key = (row.get("subreddit", "").lower(), row.get("post_id", ""))
        if key in kept_post_ids or keep_row(row):
            out = dict(row)
            out["download_policy"] = policy_for(row)
            kept_posts.append(out)

    kept_uk_strong = [row for row in kept_comments if is_uk_strong(row)]

    write_rows(COMMENTS_OUT, kept_comments, comment_fields)
    write_rows(POSTS_OUT, kept_posts, post_fields)
    write_rows(UK_STRONG_OUT, kept_uk_strong, comment_fields)

    source_subreddits = {row.get("subreddit", "").lower() for row in comments if row.get("subreddit", "")}
    kept_subreddits = {row.get("subreddit", "").lower() for row in kept_comments if row.get("subreddit", "")}
    source_posts = {(row.get("subreddit", "").lower(), row.get("post_id", "")) for row in comments if row.get("post_id", "")}
    kept_posts_from_comments = {
        (row.get("subreddit", "").lower(), row.get("post_id", "")) for row in kept_comments if row.get("post_id", "")
    }

    policy_counts = Counter(row.get("download_policy", "") for row in kept_comments)
    summary_rows = [
        {
            "metric": "source_comments",
            "value": str(len(comments)),
            "notes": str(COMMENTS_IN),
        },
        {
            "metric": "harmonised_comments",
            "value": str(len(kept_comments)),
            "notes": "Top 5 subreddits keep top 30 posts; remaining subreddits keep top 10 posts.",
        },
        {
            "metric": "source_unique_subreddits_with_comments",
            "value": str(len(source_subreddits)),
            "notes": "",
        },
        {
            "metric": "harmonised_unique_subreddits_with_comments",
            "value": str(len(kept_subreddits)),
            "notes": "",
        },
        {
            "metric": "source_unique_posts_with_comments",
            "value": str(len(source_posts)),
            "notes": "",
        },
        {
            "metric": "harmonised_unique_posts_with_comments",
            "value": str(len(kept_posts_from_comments)),
            "notes": "",
        },
        {
            "metric": "uk_strong_comments",
            "value": str(len(kept_uk_strong)),
            "notes": "uk_strong_relevant == 1 after harmonisation.",
        },
        {
            "metric": "large_top30_comments",
            "value": str(policy_counts.get("large_top30", 0)),
            "notes": "",
        },
        {
            "metric": "rest_top10_comments",
            "value": str(policy_counts.get("rest_top10", 0)),
            "notes": "",
        },
        {
            "metric": "eligible_targets",
            "value": str(len(targets)),
            "notes": str(TARGETS_IN),
        },
    ]
    write_rows(SUMMARY_OUT, summary_rows, ["metric", "value", "notes"])

    by_subreddit: Dict[str, Dict[str, str]] = {}
    for row in kept_comments:
        sub = row.get("subreddit", "")
        key = sub.lower()
        if key not in by_subreddit:
            by_subreddit[key] = {
                "subreddit": sub,
                "industry_rank_by_total_posts": row.get("industry_rank_by_total_posts", ""),
                "download_policy": row.get("download_policy", ""),
                "target_active_ai_months": row.get("target_active_ai_months", ""),
                "comments": "0",
                "uk_strong_comments": "0",
                "unique_posts": "0",
            }
    post_sets: Dict[str, set[str]] = {key: set() for key in by_subreddit}
    for row in kept_comments:
        key = row.get("subreddit", "").lower()
        by_subreddit[key]["comments"] = str(as_int(by_subreddit[key]["comments"]) + 1)
        if is_uk_strong(row):
            by_subreddit[key]["uk_strong_comments"] = str(as_int(by_subreddit[key]["uk_strong_comments"]) + 1)
        if row.get("post_id"):
            post_sets[key].add(row["post_id"])
    for key, posts_for_sub in post_sets.items():
        by_subreddit[key]["unique_posts"] = str(len(posts_for_sub))
    subreddit_rows = sorted(
        by_subreddit.values(),
        key=lambda row: (as_int(row.get("industry_rank_by_total_posts"), 9999), row.get("subreddit", "").lower()),
    )
    write_rows(
        SUBREDDIT_SUMMARY_OUT,
        subreddit_rows,
        [
            "subreddit",
            "industry_rank_by_total_posts",
            "download_policy",
            "target_active_ai_months",
            "unique_posts",
            "comments",
            "uk_strong_comments",
        ],
    )

    print(f"Source comments: {len(comments)}")
    print(f"Harmonised comments: {len(kept_comments)}")
    print(f"UK strong comments: {len(kept_uk_strong)}")
    print(f"Output directory: {TARGET_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
