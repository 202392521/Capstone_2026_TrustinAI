#!/usr/bin/env python3
"""Audit subreddit sampling tiers, retained posts, and comment caps."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


OUT = Path("outputs")
MASTER = OUT / "four_industry_ai_comments_master_harmonised_tiered_v2.csv"
AUDIT_OUT = OUT / "sampling_rule_audit_harmonised_tiered_v2.csv"
RANKING_OUT = OUT / "top5_ranking_sensitivity_total_posts_vs_ai_posts_v2.csv"
KEYWORD_OUT = OUT / "candidate_keyword_terms_by_industry_audit_v2.csv"

TARGETS = {
    "finance": OUT / "finance_top_commented_ai_threads/accountant_selected_targets.csv",
    "software_engineering": OUT / "software_engineering_top_commented_ai_threads/software_engineer_selected_targets.csv",
    "law": OUT / "law_top_commented_ai_threads/law_selected_active_ai_months_ge_10_subreddits.csv",
    "healthcare": OUT / "healthcare_active_ai_months_ge_10_targets.csv",
}


def read_rows(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, rows: Iterable[Dict[str, str]], fields: Sequence[str]) -> None:
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


def infer_tier(row: Dict[str, str]) -> str:
    policy = str(row.get("download_policy") or "").strip()
    if policy in {"large_top30", "rest_top10"}:
        return policy
    rank = as_int(row.get("industry_rank_by_total_posts"))
    if rank:
        return "large_top30" if rank <= 5 else "rest_top10"
    return "unassigned_or_legacy"


def main() -> int:
    _, comments = read_rows(MASTER)

    by_sub: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    by_post: Counter[Tuple[str, str, str]] = Counter()
    post_rank: Dict[Tuple[str, str, str], int] = {}
    for row in comments:
        industry = row.get("source_dataset") or row.get("industry_keyword", "")
        subreddit = row.get("subreddit", "")
        by_sub[(industry, subreddit)].append(row)
        post_key = (industry, subreddit, row.get("post_id", ""))
        by_post[post_key] += 1
        post_rank[post_key] = max(post_rank.get(post_key, 0), as_int(row.get("candidate_rank_by_num_comments")))

    audit_rows: List[Dict[str, str]] = []
    for (industry, subreddit), rows in sorted(by_sub.items()):
        tier = infer_tier(rows[0])
        max_posts = 30 if tier == "large_top30" else 10 if tier == "rest_top10" else 0
        max_comments = 1000 if tier == "large_top30" else 300 if tier == "rest_top10" else 0
        post_keys = sorted({(industry, subreddit, row.get("post_id", "")) for row in rows})
        retained_posts = len(post_keys)
        max_comments_one_post = max((by_post[key] for key in post_keys), default=0)
        posts_exceeding_tier_limit = max(0, retained_posts - max_posts) if max_posts else ""
        comments_exceeding_per_post_cap = (
            sum(1 for key in post_keys if by_post[key] > max_comments) if max_comments else ""
        )
        max_candidate_rank = max((post_rank.get(key, 0) for key in post_keys), default=0)
        audit_rows.append(
            {
                "industry": industry,
                "subreddit": subreddit,
                "tier": tier,
                "number_of_retained_posts": str(retained_posts),
                "max_candidate_rank_retained": str(max_candidate_rank),
                "maximum_comments_in_one_post": str(max_comments_one_post),
                "posts_exceeding_tier_limit": str(posts_exceeding_tier_limit),
                "comments_exceeding_per_post_cap": str(comments_exceeding_per_post_cap),
                "expected_post_cap": str(max_posts) if max_posts else "",
                "expected_comment_cap": str(max_comments) if max_comments else "",
                "total_comments": str(len(rows)),
            }
        )

    write_rows(
        AUDIT_OUT,
        audit_rows,
        [
            "industry",
            "subreddit",
            "tier",
            "number_of_retained_posts",
            "max_candidate_rank_retained",
            "maximum_comments_in_one_post",
            "posts_exceeding_tier_limit",
            "comments_exceeding_per_post_cap",
            "expected_post_cap",
            "expected_comment_cap",
            "total_comments",
        ],
    )

    ranking_rows: List[Dict[str, str]] = []
    for industry, path in TARGETS.items():
        _, targets = read_rows(path)
        mixed = sorted(
            targets,
            key=lambda row: (
                -as_int(row.get("total_posts_period")),
                -as_int(row.get("any_keyword_posts")),
                -as_int(row.get("active_ai_months")),
                row.get("subreddit", "").lower(),
            ),
        )[:5]
        ai = sorted(
            targets,
            key=lambda row: (
                -as_int(row.get("any_keyword_posts")),
                -as_int(row.get("active_ai_months")),
                row.get("subreddit", "").lower(),
            ),
        )[:5]
        mixed_names = [row.get("subreddit", "") for row in mixed]
        ai_names = [row.get("subreddit", "") for row in ai]
        ranking_rows.append(
            {
                "industry": industry,
                "mixed_total_posts_top5": ";".join(mixed_names),
                "ai_posts_top5": ";".join(ai_names),
                "same_top5_set": str({name.lower() for name in mixed_names} == {name.lower() for name in ai_names}),
                "only_in_mixed": ";".join(sorted({name for name in mixed_names if name.lower() not in {x.lower() for x in ai_names}})),
                "only_in_ai_posts": ";".join(sorted({name for name in ai_names if name.lower() not in {x.lower() for x in mixed_names}})),
            }
        )
    write_rows(
        RANKING_OUT,
        ranking_rows,
        ["industry", "mixed_total_posts_top5", "ai_posts_top5", "same_top5_set", "only_in_mixed", "only_in_ai_posts"],
    )

    keyword_rows: List[Dict[str, str]] = []
    counts: Dict[Tuple[str, str], int] = Counter()
    for row in comments:
        industry = row.get("source_dataset") or row.get("industry_keyword", "")
        for term in str(row.get("candidate_query_terms") or "").split(";"):
            term = term.strip()
            if term:
                counts[(industry, term)] += 1
    for (industry, term), count in sorted(counts.items()):
        keyword_rows.append({"industry": industry, "candidate_query_term": term, "comment_rows": str(count)})
    write_rows(KEYWORD_OUT, keyword_rows, ["industry", "candidate_query_term", "comment_rows"])

    print(f"Audit rows: {len(audit_rows)} -> {AUDIT_OUT}")
    print(f"Ranking sensitivity -> {RANKING_OUT}")
    print(f"Keyword audit -> {KEYWORD_OUT}")
    problem_posts = [r for r in audit_rows if str(r["posts_exceeding_tier_limit"]) not in {"", "0"}]
    problem_comments = [r for r in audit_rows if str(r["comments_exceeding_per_post_cap"]) not in {"", "0"}]
    legacy = [r for r in audit_rows if r["tier"] == "unassigned_or_legacy"]
    print(f"Rows with post-count tier violations: {len(problem_posts)}")
    print(f"Rows with per-post comment cap violations: {len(problem_comments)}")
    print(f"Legacy/unassigned tier rows: {len(legacy)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
