#!/usr/bin/env python3
"""Retrospectively harmonise law comments to a tiered sampling rule.

Law was originally collected with a broader top-30 policy. This script keeps:
- top 5 law subreddits by any_keyword_posts: top 30 AI candidate posts
- remaining law subreddits: top 10 AI candidate posts

No API calls are made.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


OUT = Path("outputs")
SOURCE_DIR = OUT / "law_top_commented_ai_threads"
TARGET_DIR = OUT / "law_top_commented_ai_threads_harmonised_tiered"

COMMENTS_IN = SOURCE_DIR / "law_top_commented_ai_comments_uk_flagged.csv"
POSTS_IN = SOURCE_DIR / "law_top_commented_ai_posts.csv"
TARGETS_IN = SOURCE_DIR / "law_selected_active_ai_months_ge_10_subreddits.csv"

COMMENTS_OUT = TARGET_DIR / "law_top_commented_ai_comments_uk_flagged_harmonised_tiered.csv"
POSTS_OUT = TARGET_DIR / "law_top_commented_ai_posts_harmonised_tiered.csv"
UK_STRONG_OUT = TARGET_DIR / "law_top_commented_ai_comments_uk_strong_only_harmonised_tiered.csv"
SUMMARY_OUT = TARGET_DIR / "law_harmonised_tiered_summary.csv"
SUBREDDIT_SUMMARY_OUT = TARGET_DIR / "law_harmonised_tiered_subreddit_summary.csv"


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


def main() -> int:
    comment_fields, comments = read_rows(COMMENTS_IN)
    post_fields, posts = read_rows(POSTS_IN)
    _, targets = read_rows(TARGETS_IN)

    ranked_targets = sorted(
        targets,
        key=lambda row: (-as_int(row.get("any_keyword_posts")), -as_int(row.get("active_ai_months")), row.get("subreddit", "").lower()),
    )
    rank_by_subreddit = {row.get("subreddit", "").lower(): idx for idx, row in enumerate(ranked_targets, start=1)}
    target_by_subreddit = {row.get("subreddit", "").lower(): row for row in ranked_targets}

    def tier_for(subreddit: str) -> str:
        return "large_top30" if rank_by_subreddit.get(subreddit.lower(), 9999) <= 5 else "rest_top10"

    def limit_for(subreddit: str) -> int:
        return 30 if tier_for(subreddit) == "large_top30" else 10

    def add_meta(row: Dict[str, str]) -> Dict[str, str]:
        out = dict(row)
        sub_key = out.get("subreddit", "").lower()
        target = target_by_subreddit.get(sub_key, {})
        out["industry_rank_by_total_posts"] = str(rank_by_subreddit.get(sub_key, ""))
        out["download_policy"] = tier_for(out.get("subreddit", ""))
        out["target_active_ai_months"] = target.get("active_ai_months", "")
        out["target_total_posts_period"] = target.get("total_posts_period", "")
        return out

    final_comment_fields = list(comment_fields)
    for field in ["industry_rank_by_total_posts", "download_policy", "target_active_ai_months", "target_total_posts_period"]:
        if field not in final_comment_fields:
            final_comment_fields.append(field)
    final_post_fields = list(post_fields)
    for field in ["industry_rank_by_total_posts", "download_policy", "target_active_ai_months", "target_total_posts_period"]:
        if field not in final_post_fields:
            final_post_fields.append(field)

    kept_comments = []
    for row in comments:
        rank = as_int(row.get("candidate_rank_by_num_comments"))
        if rank and rank <= limit_for(row.get("subreddit", "")):
            kept_comments.append(add_meta(row))

    kept_post_ids = {(row.get("subreddit", "").lower(), row.get("post_id", "")) for row in kept_comments}
    kept_posts = []
    for row in posts:
        key = (row.get("subreddit", "").lower(), row.get("post_id", ""))
        rank = as_int(row.get("candidate_rank_by_num_comments"))
        if key in kept_post_ids or (rank and rank <= limit_for(row.get("subreddit", ""))):
            kept_posts.append(add_meta(row))

    uk_rows = [row for row in kept_comments if is_uk_strong(row)]
    write_rows(COMMENTS_OUT, kept_comments, final_comment_fields)
    write_rows(POSTS_OUT, kept_posts, final_post_fields)
    write_rows(UK_STRONG_OUT, uk_rows, final_comment_fields)

    source_posts = {(row.get("subreddit", "").lower(), row.get("post_id", "")) for row in comments if row.get("post_id")}
    kept_posts_from_comments = {
        (row.get("subreddit", "").lower(), row.get("post_id", "")) for row in kept_comments if row.get("post_id")
    }
    policy_counts = Counter(row.get("download_policy", "") for row in kept_comments)
    summary_rows = [
        {"metric": "source_comments", "value": str(len(comments)), "notes": str(COMMENTS_IN)},
        {"metric": "harmonised_comments", "value": str(len(kept_comments)), "notes": "Top 5 by any_keyword_posts keep top 30 posts; remaining subreddits keep top 10 posts."},
        {"metric": "source_unique_posts_with_comments", "value": str(len(source_posts)), "notes": ""},
        {"metric": "harmonised_unique_posts_with_comments", "value": str(len(kept_posts_from_comments)), "notes": ""},
        {"metric": "uk_strong_comments", "value": str(len(uk_rows)), "notes": "uk_strong_relevant == 1 after harmonisation."},
        {"metric": "large_top30_comments", "value": str(policy_counts.get("large_top30", 0)), "notes": ""},
        {"metric": "rest_top10_comments", "value": str(policy_counts.get("rest_top10", 0)), "notes": ""},
        {"metric": "eligible_targets", "value": str(len(targets)), "notes": str(TARGETS_IN)},
    ]
    write_rows(SUMMARY_OUT, summary_rows, ["metric", "value", "notes"])

    by_sub: Dict[str, Dict[str, str]] = {}
    post_sets: Dict[str, set[str]] = defaultdict(set)
    for row in kept_comments:
        key = row.get("subreddit", "").lower()
        if key not in by_sub:
            by_sub[key] = {
                "subreddit": row.get("subreddit", ""),
                "industry_rank_by_total_posts": row.get("industry_rank_by_total_posts", ""),
                "download_policy": row.get("download_policy", ""),
                "target_active_ai_months": row.get("target_active_ai_months", ""),
                "comments": "0",
                "uk_strong_comments": "0",
                "unique_posts": "0",
            }
        by_sub[key]["comments"] = str(as_int(by_sub[key]["comments"]) + 1)
        if is_uk_strong(row):
            by_sub[key]["uk_strong_comments"] = str(as_int(by_sub[key]["uk_strong_comments"]) + 1)
        if row.get("post_id"):
            post_sets[key].add(row["post_id"])
    for key, post_ids in post_sets.items():
        by_sub[key]["unique_posts"] = str(len(post_ids))
    write_rows(
        SUBREDDIT_SUMMARY_OUT,
        sorted(by_sub.values(), key=lambda row: (as_int(row.get("industry_rank_by_total_posts"), 9999), row.get("subreddit", "").lower())),
        ["subreddit", "industry_rank_by_total_posts", "download_policy", "target_active_ai_months", "unique_posts", "comments", "uk_strong_comments"],
    )

    print(f"Source comments: {len(comments)}")
    print(f"Harmonised comments: {len(kept_comments)}")
    print(f"UK strong comments: {len(uk_rows)}")
    print(f"Output directory: {TARGET_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
