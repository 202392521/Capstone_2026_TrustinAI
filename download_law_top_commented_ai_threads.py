#!/usr/bin/env python3
"""
Download top-commented AI-related law threads and AI-relevant comments.

Selection:
- Source subreddits from final_correct_industry_active_ai_subreddits.csv
- Keep industry_keyword == lawyer
- Keep active_ai_months >= --min-active-ai-months
- For each subreddit, search expanded AI terms across 2023-03..2026-01
- Rank candidate posts by num_comments, take top N
- Download comments for those posts
- Output only comments whose own body contains expanded AI terms, with context
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import download_ai_posts_comments_for_bertopic as base  # noqa: E402


EXPANDED_AI_QUERIES = [
    "ChatGPT",
    "AI",
    '"generative AI"',
    "copilot",
    "Gemini",
    "Claude",
    "Codex",
    "LLM",
    '"large language model"',
    '"large language models"',
]

EXPANDED_AI_REGEX = re.compile(
    r"\bchat\s*gpt\b|\bchatgpt\b|\bgenerative\s+ai\b|\bgenai\b|\bai\b|"
    r"\bartificial intelligence\b|\bllm(s)?\b|\blarge language model(s)?\b|"
    r"\bgpt[- ]?4(o)?\b|\bgpt[- ]?3(\.5)?\b|\bcopilot\b|\bclaude\b|\bgemini\b|"
    r"\bcodex\b|\bmachine learning\b|\bml\b",
    re.IGNORECASE,
)


# Make the imported row builder use the expanded AI definition too.
base.AI_COMMENT_REGEX = EXPANDED_AI_REGEX


POST_FIELDS = [
    "preliminary_sic_section",
    "industry_keyword",
    "subreddit",
    "post_id",
    "post_month",
    "post_created_utc",
    "post_title",
    "post_selftext",
    "post_permalink",
    "post_score",
    "post_num_comments",
    "post_uk_signal",
    "post_industry_signal",
    "comments_downloaded",
    "candidate_rank_by_num_comments",
    "candidate_query_terms",
]

COMMENT_FIELDS = [
    "preliminary_sic_section",
    "industry_keyword",
    "subreddit",
    "post_id",
    "comment_id",
    "comment_parent_id",
    "post_month",
    "comment_month",
    "post_title",
    "post_context_excerpt",
    "parent_comment_body",
    "previous_nearby_comment_body",
    "comment_body",
    "next_nearby_comment_body",
    "comment_author_flair_text",
    "comment_score",
    "comment_has_ai_term",
    "comment_ai_terms_found",
    "post_uk_signal",
    "comment_uk_signal",
    "likely_uk_relevant",
    "post_industry_signal",
    "comment_industry_signal",
    "likely_industry_participant",
    "stance_context_text",
    "topic_model_text",
    "candidate_rank_by_num_comments",
    "candidate_query_terms",
]


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value or "").strip()))
    except ValueError:
        return default


def load_law_targets(path: Path, min_active_ai_months: int) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    targets = [
        row
        for row in rows
        if base.clean_text(row.get("industry_keyword")).lower() == "lawyer"
        and as_int(row.get("active_ai_months")) >= min_active_ai_months
        and base.clean_text(row.get("subreddit"))
    ]
    targets.sort(key=lambda row: (-as_int(row.get("active_ai_months")), row.get("subreddit", "").lower()))
    return targets


def slice_rows(rows: Sequence[Dict[str, str]], start_index: int, limit: int) -> List[Dict[str, str]]:
    if start_index < 1:
        raise ValueError("--start-index must be 1 or greater.")
    start = start_index - 1
    if limit <= 0:
        return list(rows[start:])
    return list(rows[start : start + limit])


def existing_ids(path: Path, id_column: str) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return {
            str(row.get(id_column) or "").strip()
            for row in csv.DictReader(f)
            if str(row.get(id_column) or "").strip()
        }


def fetch_candidate_posts(
    base_url: str,
    subreddit: str,
    after: int,
    before: int,
    max_candidates: int,
    retries: int,
    pause: float,
    timeout: int,
) -> Dict[str, Dict[str, Any]]:
    posts: Dict[str, Dict[str, Any]] = {}
    query_terms_by_post: Dict[str, set[str]] = {}
    for query in EXPANDED_AI_QUERIES:
        print(f"  candidate post query: {query}", file=sys.stderr)
        cursor = after
        query_count = 0
        while cursor < before and query_count < max_candidates:
            limit = min(100, max_candidates - query_count)
            try:
                payload = base.api_get_json(
                    base_url,
                    "/api/posts/search",
                    {
                        "subreddit": subreddit,
                        "query": query,
                        "after": cursor,
                        "before": before,
                        "sort": "asc",
                        "limit": limit,
                    },
                    retries=retries,
                    pause=pause,
                    timeout=timeout,
                )
            except RuntimeError as exc:
                print(f"    warning: post query failed for r/{subreddit} {query}: {exc}", file=sys.stderr)
                break
            data = payload.get("data") or []
            if not isinstance(data, list) or not data:
                break
            max_created = cursor
            for post in data:
                if not isinstance(post, dict):
                    continue
                post_id = str(post.get("id") or "").strip()
                if not post_id:
                    continue
                posts[post_id] = post
                query_terms_by_post.setdefault(post_id, set()).add(query.strip('"'))
                query_count += 1
                try:
                    max_created = max(max_created, int(float(post.get("created_utc") or cursor)))
                except (TypeError, ValueError):
                    pass
            if len(data) < limit:
                break
            if max_created < cursor:
                break
            cursor = max_created + 1
            time.sleep(pause)
        time.sleep(pause)

    for post_id, post in posts.items():
        post["_candidate_query_terms"] = ";".join(sorted(query_terms_by_post.get(post_id, set())))
    return posts


def top_commented_posts(posts: Dict[str, Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    ranked = sorted(
        posts.values(),
        key=lambda post: (
            -as_int(post.get("num_comments")),
            -as_int(post.get("score")),
            -as_int(post.get("created_utc")),
            str(post.get("id") or ""),
        ),
    )
    return ranked[:limit]


def add_candidate_metadata(row: Dict[str, Any], post: Dict[str, Any], rank: int) -> Dict[str, Any]:
    out = dict(row)
    out["candidate_rank_by_num_comments"] = rank
    out["candidate_query_terms"] = post.get("_candidate_query_terms", "")
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download top-commented AI law threads and AI comments.")
    parser.add_argument(
        "--targets",
        default="/Users/qichenzhao/Documents/Codex/2026-06-26/hi/outputs/final_correct_industry_active_ai_subreddits.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/qichenzhao/Documents/Codex/2026-06-26/hi/outputs/law_top_commented_ai_threads",
    )
    parser.add_argument("--start-month", default="2023-03")
    parser.add_argument("--end-month", default="2026-01")
    parser.add_argument("--min-active-ai-months", type=int, default=10)
    parser.add_argument("--top-posts-per-subreddit", type=int, default=30)
    parser.add_argument("--max-candidate-posts-per-query", type=int, default=2000)
    parser.add_argument("--max-comments-per-post", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--subreddit-limit", type=int, default=1, help="Default 1 for safer runs. Use 0 for all remaining.")
    parser.add_argument("--subreddit-delay-minutes", type=float, default=10)
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--base-url", default=base.AS_BASE_URL)
    parser.add_argument("--include-bots", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    posts_path = output_dir / "law_top_commented_ai_posts.csv"
    comments_path = output_dir / "law_top_commented_ai_comments.csv"
    jsonl_path = output_dir / "law_top_commented_ai_comments.jsonl"
    selected_path = output_dir / "law_selected_active_ai_months_ge_10_subreddits.csv"

    all_targets = load_law_targets(Path(args.targets).expanduser(), args.min_active_ai_months)
    selected_targets = slice_rows(all_targets, args.start_index, args.subreddit_limit)
    after = base.utc_timestamp(args.start_month)
    before = base.utc_timestamp(base.month_after(args.end_month))

    base.write_csv(selected_path, all_targets, list(all_targets[0].keys()) if all_targets else [])

    seen_posts = existing_ids(posts_path, "post_id")
    seen_comments = existing_ids(comments_path, "comment_id")
    print(f"Law targets active_ai_months >= {args.min_active_ai_months}: {len(all_targets)}", file=sys.stderr)
    print(
        f"Selected subreddit slice: {args.start_index}-{args.start_index + len(selected_targets) - 1} of {len(all_targets)}",
        file=sys.stderr,
    )
    print(f"Existing posts: {len(seen_posts)}", file=sys.stderr)
    print(f"Existing comments: {len(seen_comments)}", file=sys.stderr)

    total_posts = 0
    total_comments = 0
    for idx, target in enumerate(selected_targets, start=1):
        subreddit = target.get("subreddit", "")
        print(f"[{idx}/{len(selected_targets)}] r/{subreddit}", file=sys.stderr)
        candidates = fetch_candidate_posts(
            args.base_url,
            subreddit,
            after,
            before,
            args.max_candidate_posts_per_query,
            args.retries,
            args.pause,
            args.timeout,
        )
        chosen_posts = top_commented_posts(candidates, args.top_posts_per_subreddit)
        print(
            f"  candidate posts: {len(candidates)}; selected top-commented posts: {len(chosen_posts)}",
            file=sys.stderr,
        )

        subreddit_post_rows: List[Dict[str, Any]] = []
        subreddit_comment_rows: List[Dict[str, Any]] = []
        for post_rank, post in enumerate(chosen_posts, start=1):
            post_id = str(post.get("id") or "")
            if post_id in seen_posts:
                print(f"    skipping already-downloaded post rank {post_rank}: {post_id}", file=sys.stderr)
                continue
            print(
                f"    comments for rank {post_rank}/{len(chosen_posts)} post {post_id} "
                f"({as_int(post.get('num_comments'))} comments)",
                file=sys.stderr,
            )
            comments = base.fetch_comments_for_post(
                args.base_url,
                post_id,
                args.max_comments_per_post,
                args.retries,
                args.pause,
                args.timeout,
                exclude_bots=not args.include_bots,
            )
            post_row, comment_rows = base.build_rows_for_post(
                target,
                post,
                comments,
                context_chars=900,
                surrounding_context_chars=700,
                require_comment_ai=True,
            )
            post_row = add_candidate_metadata(post_row, post, post_rank)
            subreddit_post_rows.append(post_row)
            seen_posts.add(post_id)

            for comment_row in comment_rows:
                comment_id = str(comment_row.get("comment_id") or "").strip()
                if not comment_id or comment_id in seen_comments:
                    continue
                subreddit_comment_rows.append(add_candidate_metadata(comment_row, post, post_rank))
                seen_comments.add(comment_id)
            time.sleep(args.pause)

        base.append_csv(posts_path, subreddit_post_rows, POST_FIELDS)
        base.append_csv(comments_path, subreddit_comment_rows, COMMENT_FIELDS)
        base.append_jsonl(jsonl_path, subreddit_comment_rows)
        total_posts += len(subreddit_post_rows)
        total_comments += len(subreddit_comment_rows)
        print(
            f"  saved r/{subreddit}: {len(subreddit_post_rows)} posts, {len(subreddit_comment_rows)} AI comments",
            file=sys.stderr,
        )
        if args.subreddit_delay_minutes > 0 and idx < len(selected_targets):
            print(
                f"  sleeping {args.subreddit_delay_minutes} minutes before next subreddit",
                file=sys.stderr,
            )
            time.sleep(args.subreddit_delay_minutes * 60)

    print(f"Posts written this run: {total_posts}")
    print(f"AI comments written this run: {total_comments}")
    print(f"Selected targets: {selected_path}")
    print(f"Posts CSV: {posts_path}")
    print(f"Comments CSV: {comments_path}")
    print(f"Comments JSONL: {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
