#!/usr/bin/env python3
"""
Generic downloader for top-commented AI-related threads by industry.

Policy:
- Sort an industry's active AI subreddits by total_posts_period, then AI volume.
- First N large subreddits: top 30 AI candidate posts, rest every 10 posts.
- Remaining subreddits: only active_ai_months >= 10, top 10 AI candidate posts.
- Keep only comments whose own body contains expanded AI terms.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import download_ai_posts_comments_for_bertopic as base  # noqa: E402
import download_law_top_commented_ai_threads as shared  # noqa: E402


def slugify(value: str) -> str:
    return "_".join(base.clean_text(value).lower().replace("/", " ").split())


def as_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value or "").strip()
        if not text:
            return default
        return int(float(text))
    except ValueError:
        return default


def load_industry_targets(path: Path, industry_keyword: str, min_active_ai_months_for_rest: int) -> List[Dict[str, Any]]:
    industry = base.clean_text(industry_keyword).lower()
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    targets: List[Dict[str, Any]] = [
        dict(row)
        for row in rows
        if base.clean_text(row.get("industry_keyword")).lower() == industry and base.clean_text(row.get("subreddit"))
    ]
    targets.sort(
        key=lambda row: (
            -as_int(row.get("total_posts_period")),
            -as_int(row.get("any_keyword_posts")),
            -as_int(row.get("active_ai_months")),
            base.clean_text(row.get("subreddit")).lower(),
        )
    )
    for index, row in enumerate(targets, start=1):
        row["_industry_rank_by_total_posts"] = index
        row["_download_policy"] = "large_top30" if index <= 5 else "rest_top10"
        row["_eligible_after_policy"] = 1 if index <= 5 or as_int(row.get("active_ai_months")) >= min_active_ai_months_for_rest else 0
    return [row for row in targets if int(row["_eligible_after_policy"]) == 1]


def slice_rows(rows: Sequence[Dict[str, Any]], start_index: int, limit: int) -> List[Dict[str, Any]]:
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


def existing_summary_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def upsert_summary_row(rows: List[Dict[str, Any]], new_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    key = (str(new_row.get("industry_keyword", "")), str(new_row.get("subreddit", "")))
    updated = False
    output: List[Dict[str, Any]] = []
    for row in rows:
        row_key = (str(row.get("industry_keyword", "")), str(row.get("subreddit", "")))
        if row_key == key:
            output.append(new_row)
            updated = True
        else:
            output.append(row)
    if not updated:
        output.append(new_row)
    return output


def add_policy_metadata(row: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["industry_rank_by_total_posts"] = target.get("_industry_rank_by_total_posts", "")
    out["download_policy"] = target.get("_download_policy", "")
    out["target_active_ai_months"] = target.get("active_ai_months", "")
    out["target_total_posts_period"] = target.get("total_posts_period", "")
    return out


POST_FIELDS = shared.POST_FIELDS + [
    "industry_rank_by_total_posts",
    "download_policy",
    "target_active_ai_months",
    "target_total_posts_period",
]

COMMENT_FIELDS = shared.COMMENT_FIELDS + [
    "industry_rank_by_total_posts",
    "download_policy",
    "target_active_ai_months",
    "target_total_posts_period",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download top-commented AI threads/comments for one industry.")
    parser.add_argument("--industry-keyword", required=True, help='Example: "accountant" or "software engineer"')
    parser.add_argument(
        "--targets",
        default="/Users/qichenzhao/Documents/Codex/2026-06-26/hi/outputs/final_correct_industry_active_ai_subreddits.csv",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-month", default="2023-03")
    parser.add_argument("--end-month", default="2026-01")
    parser.add_argument("--large-subreddit-count", type=int, default=5)
    parser.add_argument("--large-top-posts", type=int, default=30)
    parser.add_argument("--large-max-comments-per-post", type=int, default=1000)
    parser.add_argument("--large-rest-every-posts", type=int, default=10)
    parser.add_argument("--large-rest-minutes", type=float, default=10)
    parser.add_argument("--rest-min-active-ai-months", type=int, default=10)
    parser.add_argument("--rest-top-posts", type=int, default=10)
    parser.add_argument("--rest-max-comments-per-post", type=int, default=300)
    parser.add_argument("--subreddit-delay-minutes", type=float, default=1)
    parser.add_argument("--max-candidate-posts-per-query", type=int, default=2000)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--subreddit-limit", type=int, default=0, help="Default 0 = all selected targets.")
    parser.add_argument("--pause", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--base-url", default=base.AS_BASE_URL)
    parser.add_argument("--include-bots", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    industry_slug = slugify(args.industry_keyword)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    posts_path = output_dir / f"{industry_slug}_top_commented_ai_posts.csv"
    comments_path = output_dir / f"{industry_slug}_top_commented_ai_comments.csv"
    jsonl_path = output_dir / f"{industry_slug}_top_commented_ai_comments.jsonl"
    selected_path = output_dir / f"{industry_slug}_selected_targets.csv"
    log_summary_path = output_dir / f"{industry_slug}_run_summary.csv"

    all_targets = load_industry_targets(
        Path(args.targets).expanduser(),
        args.industry_keyword,
        args.rest_min_active_ai_months,
    )
    # Allow large_sbreddit_count to differ from the default 5 without re-sorting.
    for target in all_targets:
        rank = as_int(target.get("_industry_rank_by_total_posts"))
        target["_download_policy"] = "large_top30" if rank <= args.large_subreddit_count else "rest_top10"
    selected_targets = slice_rows(all_targets, args.start_index, args.subreddit_limit)
    after = base.utc_timestamp(args.start_month)
    before = base.utc_timestamp(base.month_after(args.end_month))

    target_fields = [field for field in all_targets[0].keys() if not field.startswith("_")] if all_targets else []
    for field in ["industry_rank_by_total_posts", "download_policy"]:
        if field not in target_fields:
            target_fields.append(field)
    selected_export_rows = []
    for row in all_targets:
        out = {k: v for k, v in row.items() if not k.startswith("_")}
        out["industry_rank_by_total_posts"] = row.get("_industry_rank_by_total_posts", "")
        out["download_policy"] = row.get("_download_policy", "")
        selected_export_rows.append(out)
    base.write_csv(selected_path, selected_export_rows, target_fields)

    seen_posts = existing_ids(posts_path, "post_id")
    seen_comments = existing_ids(comments_path, "comment_id")
    run_rows: List[Dict[str, Any]] = existing_summary_rows(log_summary_path)

    print(f"Industry: {args.industry_keyword}", file=sys.stderr)
    print(f"Eligible targets: {len(all_targets)}", file=sys.stderr)
    print(
        f"Selected target slice: {args.start_index}-{args.start_index + len(selected_targets) - 1} of {len(all_targets)}",
        file=sys.stderr,
    )
    print(f"Existing posts: {len(seen_posts)}", file=sys.stderr)
    print(f"Existing comments: {len(seen_comments)}", file=sys.stderr)

    total_posts = 0
    total_comments = 0
    for idx, target in enumerate(selected_targets, start=1):
        subreddit = target.get("subreddit", "")
        policy = target.get("_download_policy", "")
        is_large = policy == "large_top30"
        top_posts_limit = args.large_top_posts if is_large else args.rest_top_posts
        max_comments = args.large_max_comments_per_post if is_large else args.rest_max_comments_per_post
        print(
            f"[{idx}/{len(selected_targets)}] r/{subreddit} policy={policy} top_posts={top_posts_limit} max_comments={max_comments}",
            file=sys.stderr,
        )
        candidates = shared.fetch_candidate_posts(
            args.base_url,
            subreddit,
            after,
            before,
            args.max_candidate_posts_per_query,
            args.retries,
            args.pause,
            args.timeout,
        )
        chosen_posts = shared.top_commented_posts(candidates, top_posts_limit)
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
                max_comments,
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
            post_row = add_policy_metadata(shared.add_candidate_metadata(post_row, post, post_rank), target)
            subreddit_post_rows.append(post_row)
            seen_posts.add(post_id)

            for comment_row in comment_rows:
                comment_id = str(comment_row.get("comment_id") or "").strip()
                if not comment_id or comment_id in seen_comments:
                    continue
                subreddit_comment_rows.append(
                    add_policy_metadata(shared.add_candidate_metadata(comment_row, post, post_rank), target)
                )
                seen_comments.add(comment_id)
            time.sleep(args.pause)

            if (
                is_large
                and args.large_rest_every_posts > 0
                and post_rank % args.large_rest_every_posts == 0
                and post_rank < len(chosen_posts)
            ):
                print(
                    f"    processed {post_rank} large-subreddit posts; sleeping {args.large_rest_minutes} minutes",
                    file=sys.stderr,
                )
                time.sleep(args.large_rest_minutes * 60)

        base.append_csv(posts_path, subreddit_post_rows, POST_FIELDS)
        base.append_csv(comments_path, subreddit_comment_rows, COMMENT_FIELDS)
        base.append_jsonl(jsonl_path, subreddit_comment_rows)
        total_posts += len(subreddit_post_rows)
        total_comments += len(subreddit_comment_rows)
        run_rows = upsert_summary_row(
            run_rows,
            {
                "industry_keyword": args.industry_keyword,
                "subreddit": subreddit,
                "download_policy": policy,
                "posts_written": len(subreddit_post_rows),
                "ai_comments_written": len(subreddit_comment_rows),
                "candidate_posts": len(candidates),
                "selected_posts": len(chosen_posts),
            },
        )
        base.write_csv(log_summary_path, run_rows, list(run_rows[0].keys()))
        print(
            f"  saved r/{subreddit}: {len(subreddit_post_rows)} posts, {len(subreddit_comment_rows)} AI comments",
            file=sys.stderr,
        )
        if args.subreddit_delay_minutes > 0 and idx < len(selected_targets):
            print(f"  sleeping {args.subreddit_delay_minutes} minutes before next subreddit", file=sys.stderr)
            time.sleep(args.subreddit_delay_minutes * 60)

    print(f"Posts written this run: {total_posts}")
    print(f"AI comments written this run: {total_comments}")
    print(f"Selected targets: {selected_path}")
    print(f"Posts CSV: {posts_path}")
    print(f"Comments CSV: {comments_path}")
    print(f"Comments JSONL: {jsonl_path}")
    print(f"Run summary: {log_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
