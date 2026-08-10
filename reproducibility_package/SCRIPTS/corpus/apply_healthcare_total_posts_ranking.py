#!/usr/bin/env python3
"""Apply total-post based healthcare tier ranking if Arctic Shift counts succeed.

This script does not call any API. It reads the total-post retry output and
rebuilds the harmonised healthcare files using:

- top 5 subreddits by total_posts_period_retry: top 30 posts, max 1000 comments/post
- all remaining eligible subreddits: top 10 posts, max 300 comments/post
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


OUT = Path("outputs")
SOURCE_DIR = OUT / "healthcare_top_commented_ai_threads"
TARGET_DIR = OUT / "healthcare_top_commented_ai_threads_harmonised_total_posts_ranked"

COUNT_IN = OUT / "healthcare_total_posts_period_retry_as.csv"
COMMENTS_IN = SOURCE_DIR / "healthcare_top_commented_ai_comments_uk_flagged.csv"
POSTS_IN = SOURCE_DIR / "healthcare_top_commented_ai_posts.csv"

COMMENTS_OUT = TARGET_DIR / "healthcare_top_commented_ai_comments_uk_flagged_total_posts_ranked.csv"
POSTS_OUT = TARGET_DIR / "healthcare_top_commented_ai_posts_total_posts_ranked.csv"
UK_STRONG_OUT = TARGET_DIR / "healthcare_top_commented_ai_comments_uk_strong_only_total_posts_ranked.csv"
TARGETS_OUT = TARGET_DIR / "healthcare_selected_targets_total_posts_ranked.csv"
SUMMARY_OUT = TARGET_DIR / "healthcare_total_posts_ranked_summary.csv"
SUBREDDIT_SUMMARY_OUT = TARGET_DIR / "healthcare_total_posts_ranked_subreddit_summary.csv"


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


def post_key(row: Dict[str, str]) -> Tuple[str, str]:
    return row.get("subreddit", "").lower(), row.get("post_id", "")


def build_ranked_targets(count_rows: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    good: List[Dict[str, str]] = []
    bad: List[Dict[str, str]] = []
    for row in count_rows:
        status = str(row.get("total_posts_count_status") or "").strip()
        total = as_int(row.get("total_posts_period_retry"), -1)
        if status in {"ok", "ok_after_monthly_fallback"} and total >= 0:
            good.append(row)
        else:
            bad.append(row)

    ranked = sorted(
        good,
        key=lambda row: (
            -as_int(row.get("total_posts_period_retry")),
            -as_int(row.get("any_keyword_posts")),
            -as_int(row.get("active_ai_months")),
            row.get("subreddit", "").lower(),
        ),
    )
    for index, row in enumerate(ranked, start=1):
        row["industry_rank_by_total_posts"] = str(index)
        row["download_policy"] = "large_top30" if index <= 5 else "rest_top10"
        row["target_total_posts_period"] = row.get("total_posts_period_retry", "")
    return ranked, bad


def keep_post(row: Dict[str, str], rank_by_subreddit: Dict[str, int]) -> bool:
    subreddit = row.get("subreddit", "").lower()
    rank = rank_by_subreddit.get(subreddit)
    if not rank:
        return False
    post_rank = as_int(row.get("candidate_rank_by_num_comments"))
    limit = 30 if rank <= 5 else 10
    return 0 < post_rank <= limit


def enforce_comment_caps(comments: List[Dict[str, str]], rank_by_subreddit: Dict[str, int]) -> List[Dict[str, str]]:
    kept: List[Dict[str, str]] = []
    seen_per_post: Counter[Tuple[str, str]] = Counter()
    for row in comments:
        key = post_key(row)
        rank = rank_by_subreddit.get(key[0])
        if not rank:
            continue
        cap = 1000 if rank <= 5 else 300
        if seen_per_post[key] >= cap:
            continue
        seen_per_post[key] += 1
        kept.append(row)
    return kept


def add_sampling_metadata(row: Dict[str, str], ranked_targets: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    subreddit = row.get("subreddit", "").lower()
    target = ranked_targets[subreddit]
    out = dict(row)
    out["industry_rank_by_total_posts"] = target.get("industry_rank_by_total_posts", "")
    out["download_policy"] = target.get("download_policy", "")
    out["target_active_ai_months"] = target.get("active_ai_months", out.get("target_active_ai_months", ""))
    out["target_total_posts_period"] = target.get("total_posts_period_retry", "")
    return out


def main() -> int:
    count_fields, count_rows = read_rows(COUNT_IN)
    comment_fields, comments = read_rows(COMMENTS_IN)
    post_fields, posts = read_rows(POSTS_IN)

    ranked_targets, bad_targets = build_ranked_targets(count_rows)
    if bad_targets:
        print("Cannot safely apply total-post ranking yet.")
        print(f"Counted targets: {len(ranked_targets)}")
        print(f"Targets without successful total-post counts: {len(bad_targets)}")
        print("Rerun the count script with --resume before applying this script.")
        return 2
    if len(ranked_targets) < 5:
        print("Cannot apply total-post ranking: fewer than five successful target counts.")
        return 2

    ranked_by_sub = {row.get("subreddit", "").lower(): row for row in ranked_targets}
    rank_by_sub = {key: as_int(row.get("industry_rank_by_total_posts")) for key, row in ranked_by_sub.items()}

    selected_posts = [
        add_sampling_metadata(row, ranked_by_sub) for row in posts if keep_post(row, rank_by_sub)
    ]
    selected_post_keys = {post_key(row) for row in selected_posts}
    eligible_comments = [
        add_sampling_metadata(row, ranked_by_sub)
        for row in comments
        if post_key(row) in selected_post_keys
    ]
    kept_comments = enforce_comment_caps(eligible_comments, rank_by_sub)
    kept_post_keys = {post_key(row) for row in kept_comments}
    kept_posts = [row for row in selected_posts if post_key(row) in kept_post_keys]
    uk_strong = [row for row in kept_comments if is_uk_strong(row)]

    new_comment_fields = list(dict.fromkeys([*comment_fields, "download_policy", "target_total_posts_period"]))
    new_post_fields = list(dict.fromkeys([*post_fields, "download_policy", "target_total_posts_period"]))
    target_fields = list(
        dict.fromkeys(
            [
                *count_fields,
                "industry_rank_by_total_posts",
                "download_policy",
                "target_total_posts_period",
            ]
        )
    )
    write_rows(TARGETS_OUT, ranked_targets, target_fields)
    write_rows(COMMENTS_OUT, kept_comments, new_comment_fields)
    write_rows(POSTS_OUT, kept_posts, new_post_fields)
    write_rows(UK_STRONG_OUT, uk_strong, new_comment_fields)

    by_sub: Dict[str, Dict[str, str]] = {}
    post_sets: Dict[str, set[str]] = defaultdict(set)
    max_comments_by_post: Counter[Tuple[str, str]] = Counter()
    for row in kept_comments:
        sub = row.get("subreddit", "")
        key = sub.lower()
        target = ranked_by_sub[key]
        if key not in by_sub:
            by_sub[key] = {
                "subreddit": sub,
                "industry_rank_by_total_posts": target.get("industry_rank_by_total_posts", ""),
                "download_policy": target.get("download_policy", ""),
                "total_posts_period": target.get("total_posts_period_retry", ""),
                "active_ai_months": target.get("active_ai_months", ""),
                "any_keyword_posts": target.get("any_keyword_posts", ""),
                "unique_posts": "0",
                "comments": "0",
                "uk_strong_comments": "0",
                "maximum_comments_in_one_post": "0",
            }
        by_sub[key]["comments"] = str(as_int(by_sub[key]["comments"]) + 1)
        if is_uk_strong(row):
            by_sub[key]["uk_strong_comments"] = str(as_int(by_sub[key]["uk_strong_comments"]) + 1)
        post_sets[key].add(row.get("post_id", ""))
        max_comments_by_post[post_key(row)] += 1
    for key, values in by_sub.items():
        values["unique_posts"] = str(len(post_sets[key]))
        post_comment_counts = [count for (subreddit, _), count in max_comments_by_post.items() if subreddit == key]
        values["maximum_comments_in_one_post"] = str(max(post_comment_counts, default=0))

    subreddit_summary = sorted(
        by_sub.values(),
        key=lambda row: as_int(row.get("industry_rank_by_total_posts"), 999999),
    )
    write_rows(
        SUBREDDIT_SUMMARY_OUT,
        subreddit_summary,
        [
            "subreddit",
            "industry_rank_by_total_posts",
            "download_policy",
            "total_posts_period",
            "active_ai_months",
            "any_keyword_posts",
            "unique_posts",
            "comments",
            "uk_strong_comments",
            "maximum_comments_in_one_post",
        ],
    )

    top5 = [row.get("subreddit", "") for row in ranked_targets[:5]]
    summary_rows = [
        {"metric": "rank_metric", "value": "total_posts_period_retry", "notes": str(COUNT_IN)},
        {"metric": "eligible_targets_ranked", "value": str(len(ranked_targets)), "notes": ""},
        {"metric": "top5_subreddits", "value": ";".join(top5), "notes": ""},
        {"metric": "comments", "value": str(len(kept_comments)), "notes": ""},
        {"metric": "posts_with_comments", "value": str(len(kept_post_keys)), "notes": ""},
        {"metric": "uk_strong_comments", "value": str(len(uk_strong)), "notes": ""},
    ]
    write_rows(SUMMARY_OUT, summary_rows, ["metric", "value", "notes"])

    print(f"Applied healthcare total-post ranking.")
    print(f"Top 5: {'; '.join(top5)}")
    print(f"Comments: {len(kept_comments)}")
    print(f"Posts with comments: {len(kept_post_keys)}")
    print(f"Output directory: {TARGET_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
