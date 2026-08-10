#!/usr/bin/env python3
"""
Score healthcare candidate subreddits for AI discussion using Arctic Shift search.

This script avoids the AS aggregate endpoint, which has been unreliable. It uses
/api/posts/search over the full research period, then aggregates posts locally
by month. Output is compatible with the project's active_ai_months logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_INPUT = "outputs/healthcare_doctor_reddit_search_candidate_subreddits.csv"
DEFAULT_OUTPUT = "outputs/healthcare_doctor_ai_ranked.csv"
DEFAULT_MONTHLY_OUTPUT = "outputs/healthcare_doctor_ai_monthly_counts.csv"
DEFAULT_ACTIVE_OUTPUT = "outputs/healthcare_doctor_active_ai_months_ge_10.csv"
AS_BASE_URL = "https://arctic-shift.photon-reddit.com"

QUERY_SPECS = [
    ("chatgpt", "ChatGPT"),
    ("ai", "AI"),
    ("generative_ai", '"generative AI"'),
    ("artificial_intelligence", '"artificial intelligence"'),
]

RANKED_FIELDS = [
    "subreddit",
    "has_ai_discussion",
    "months_observed",
    "active_ai_months",
    "any_keyword_posts",
    "chatgpt_posts",
    "ai_posts",
    "generative_ai_posts",
    "artificial_intelligence_posts",
    "first_ai_month",
    "last_ai_month",
    "first_chatgpt_month",
    "first_generative_ai_month",
    "first_artificial_intelligence_month",
    "peak_ai_month",
    "peak_ai_posts",
    "failure_events",
    "data_quality",
    "candidate_keywords",
    "candidate_queries",
    "best_search_rank",
    "candidate_search_ranks",
    "candidate_keyword_candidate_counts",
    "candidate_discovery_warnings",
    "subreddit_url",
]

MONTHLY_FIELDS = [
    "subreddit",
    "month",
    "chatgpt_count",
    "ai_count",
    "generative_ai_count",
    "artificial_intelligence_count",
    "any_keyword_count",
]


def parse_month(month: str) -> datetime:
    return datetime.strptime(month + "-01", "%Y-%m-%d").replace(tzinfo=timezone.utc)


def month_after(month: str) -> str:
    dt = parse_month(month)
    if dt.month == 12:
        return dt.replace(year=dt.year + 1, month=1).strftime("%Y-%m")
    return dt.replace(month=dt.month + 1).strftime("%Y-%m")


def unix_month(month: str) -> int:
    return int(parse_month(month).timestamp())


def month_range(start_month: str, end_month: str) -> List[str]:
    months: List[str] = []
    current = start_month
    while current <= end_month:
        months.append(current)
        current = month_after(current)
    return months


def post_id(post: Dict[str, Any]) -> str:
    return str(post.get("id") or "").replace("t3_", "").strip()


def post_created(post: Dict[str, Any]) -> int:
    for key in ["created_utc", "created"]:
        try:
            return int(float(post.get(key)))
        except (TypeError, ValueError):
            pass
    return 0


def post_month(post: Dict[str, Any]) -> str:
    created = post_created(post)
    if not created:
        return ""
    return datetime.fromtimestamp(created, tz=timezone.utc).strftime("%Y-%m")


def as_int(value: Any, default: int = 999999) -> int:
    try:
        text = str(value or "").strip()
        return int(float(text)) if text else default
    except (TypeError, ValueError):
        return default


def clean_subreddit(value: Any) -> str:
    text = str(value or "").strip()
    text = text.replace("https://www.reddit.com/r/", "").replace("https://reddit.com/r/", "")
    text = text.removeprefix("/r/").removeprefix("r/")
    return text.strip("/ ")


def split_values(value: Any) -> List[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_existing_by_subreddit(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    rows = read_rows(path)
    return {
        clean_subreddit(row.get("subreddit", "")).lower(): row
        for row in rows
        if clean_subreddit(row.get("subreddit", ""))
    }


def read_existing_monthly(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    if not path.exists():
        return {}
    rows = read_rows(path)
    return {
        (clean_subreddit(row.get("subreddit", "")).lower(), str(row.get("month", ""))): row
        for row in rows
        if clean_subreddit(row.get("subreddit", "")) and row.get("month")
    }


def aggregate_candidates(rows: Sequence[Dict[str, str]], max_search_rank: int) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        subreddit = clean_subreddit(row.get("subreddit", ""))
        if not subreddit:
            continue
        ranks = [as_int(rank) for rank in split_values(row.get("search_rank"))]
        best_rank = min(ranks) if ranks else 999999
        if max_search_rank > 0 and best_rank > max_search_rank:
            continue
        key = subreddit.lower()
        if key not in grouped:
            grouped[key] = {
                "subreddit": subreddit,
                "subreddit_url": f"https://www.reddit.com/r/{subreddit}/",
                "candidate_keywords": set(),
                "candidate_queries": set(),
                "candidate_search_ranks": [],
                "candidate_keyword_candidate_counts": set(),
                "candidate_discovery_warnings": set(),
                "best_search_rank": best_rank,
            }
        item = grouped[key]
        item["candidate_keywords"].update(split_values(row.get("keyword")))
        item["candidate_queries"].update(split_values(row.get("query")))
        item["candidate_search_ranks"].extend(str(rank) for rank in ranks if rank != 999999)
        item["candidate_keyword_candidate_counts"].update(split_values(row.get("keyword_candidate_count")))
        item["candidate_discovery_warnings"].update(split_values(row.get("discovery_warning")))
        item["best_search_rank"] = min(item["best_search_rank"], best_rank)

    candidates = list(grouped.values())
    for candidate in candidates:
        for key in [
            "candidate_keywords",
            "candidate_queries",
            "candidate_keyword_candidate_counts",
            "candidate_discovery_warnings",
        ]:
            candidate[key] = ";".join(sorted(candidate[key]))
        candidate["candidate_search_ranks"] = ";".join(candidate["candidate_search_ranks"])
    candidates.sort(key=lambda row: (row["best_search_rank"], row["subreddit"].lower()))
    return candidates


def slice_rows(rows: Sequence[Dict[str, Any]], start_index: int, limit: int) -> List[Dict[str, Any]]:
    if start_index < 1:
        raise ValueError("--start-index must be 1 or greater")
    start = start_index - 1
    if limit <= 0:
        return list(rows[start:])
    return list(rows[start : start + limit])


def api_get_json(base_url: str, path: str, params: Dict[str, Any], retries: int, pause: float, timeout: int) -> Dict[str, Any]:
    url = base_url.rstrip("/") + path + "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    last_error: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126 Safari/537.36",
                    "Accept": "application/json,text/plain,*/*",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except BaseException as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= retries:
                break
            wait = pause * (2 ** (attempt - 1))
            print(f"    AS request failed ({exc}); retrying in {wait:.1f}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"AS request failed after {retries} retries: {last_error}\nURL: {url}")


def fetch_query_posts(
    base_url: str,
    subreddit: str,
    query: str,
    after: int,
    before: int,
    max_posts: int,
    retries: int,
    pause: float,
    timeout: int,
) -> List[Dict[str, Any]]:
    posts: Dict[str, Dict[str, Any]] = {}
    cursor = after
    while cursor < before and len(posts) < max_posts:
        limit = min(100, max_posts - len(posts))
        payload = api_get_json(
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
            retries,
            pause,
            timeout,
        )
        data = payload.get("data") or []
        if not isinstance(data, list) or not data:
            break
        max_created = cursor
        for post in data:
            if not isinstance(post, dict):
                continue
            pid = post_id(post)
            if pid:
                posts[pid] = post
            max_created = max(max_created, post_created(post))
        if len(data) < limit or max_created <= cursor:
            break
        cursor = max_created + 1
        time.sleep(pause)
    return sorted(posts.values(), key=post_created)


def score_candidate(
    candidate: Dict[str, Any],
    months: Sequence[str],
    after: int,
    before: int,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    subreddit = candidate["subreddit"]
    query_counts: Dict[str, Dict[str, int]] = {
        query_key: {month: 0 for month in months}
        for query_key, _query in QUERY_SPECS
    }
    any_by_month: Dict[str, set[str]] = {month: set() for month in months}
    failures: List[str] = []

    for query_key, query in QUERY_SPECS:
        print(f"  AS query: {query}", file=sys.stderr)
        try:
            posts = fetch_query_posts(
                args.base_url,
                subreddit,
                query,
                after,
                before,
                args.max_posts_per_query,
                args.retries,
                args.pause,
                args.timeout,
            )
        except RuntimeError as exc:
            failures.append(f"{query}: {str(exc).splitlines()[0]}")
            print(f"    warning: {query} failed for r/{subreddit}: {exc}", file=sys.stderr)
            posts = []
        for post in posts:
            month = post_month(post)
            pid = post_id(post)
            if month in query_counts[query_key] and pid:
                query_counts[query_key][month] += 1
                any_by_month[month].add(pid)
        time.sleep(args.pause)

    any_counts = {month: len(any_by_month[month]) for month in months}
    active_months = [month for month in months if any_counts[month] > 0]
    peak_month, peak_count = ("", 0)
    if any_counts:
        peak_month, peak_count = max(any_counts.items(), key=lambda item: (item[1], item[0]))
        if peak_count == 0:
            peak_month = ""

    def first_month(query_key: str) -> str:
        return next((month for month in months if query_counts[query_key][month] > 0), "")

    ranked_row = {
        "subreddit": subreddit,
        "has_ai_discussion": "1" if active_months else "0",
        "months_observed": str(len(months)),
        "active_ai_months": str(len(active_months)),
        "any_keyword_posts": str(sum(any_counts.values())),
        "chatgpt_posts": str(sum(query_counts["chatgpt"].values())),
        "ai_posts": str(sum(query_counts["ai"].values())),
        "generative_ai_posts": str(sum(query_counts["generative_ai"].values())),
        "artificial_intelligence_posts": str(sum(query_counts["artificial_intelligence"].values())),
        "first_ai_month": active_months[0] if active_months else "",
        "last_ai_month": active_months[-1] if active_months else "",
        "first_chatgpt_month": first_month("chatgpt"),
        "first_generative_ai_month": first_month("generative_ai"),
        "first_artificial_intelligence_month": first_month("artificial_intelligence"),
        "peak_ai_month": peak_month,
        "peak_ai_posts": str(peak_count),
        "failure_events": " || ".join(failures),
        "data_quality": "as_search_local_monthly_counts" if not failures else "as_search_local_monthly_counts_with_failures",
        "candidate_keywords": candidate.get("candidate_keywords", ""),
        "candidate_queries": candidate.get("candidate_queries", ""),
        "best_search_rank": str(candidate.get("best_search_rank", "")),
        "candidate_search_ranks": candidate.get("candidate_search_ranks", ""),
        "candidate_keyword_candidate_counts": candidate.get("candidate_keyword_candidate_counts", ""),
        "candidate_discovery_warnings": candidate.get("candidate_discovery_warnings", ""),
        "subreddit_url": candidate.get("subreddit_url", ""),
    }

    monthly_rows: List[Dict[str, Any]] = []
    for month in months:
        monthly_rows.append(
            {
                "subreddit": subreddit,
                "month": month,
                "chatgpt_count": query_counts["chatgpt"][month],
                "ai_count": query_counts["ai"][month],
                "generative_ai_count": query_counts["generative_ai"][month],
                "artificial_intelligence_count": query_counts["artificial_intelligence"][month],
                "any_keyword_count": any_counts[month],
            }
        )
    return ranked_row, monthly_rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score healthcare Reddit-search candidates for AI active months with AS search.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--monthly-output", default=DEFAULT_MONTHLY_OUTPUT)
    parser.add_argument("--active-output", default=DEFAULT_ACTIVE_OUTPUT)
    parser.add_argument("--start-month", default="2023-03")
    parser.add_argument("--end-month", default="2026-01")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-search-rank", type=int, default=30)
    parser.add_argument("--max-posts-per-query", type=int, default=1000)
    parser.add_argument("--pause", type=float, default=5)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--subreddit-delay-minutes", type=float, default=1)
    parser.add_argument("--active-month-threshold", type=int, default=10)
    parser.add_argument("--base-url", default=AS_BASE_URL)
    parser.add_argument("--resume", action="store_true", help="Keep existing rows and skip already scored subreddits.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_rows = read_rows(Path(args.input).expanduser())
    candidates = aggregate_candidates(input_rows, args.max_search_rank)
    selected = slice_rows(candidates, args.start_index, args.limit)
    months = month_range(args.start_month, args.end_month)
    after = unix_month(args.start_month)
    before = unix_month(month_after(args.end_month))

    print(f"Input rows: {len(input_rows)}")
    print(f"Unique candidates after rank filter: {len(candidates)}")
    print(f"Selected candidates: {args.start_index}-{args.start_index + len(selected) - 1}")
    print(f"Months: {months[0]} to {months[-1]} ({len(months)})")

    output_path = Path(args.output).expanduser()
    monthly_output_path = Path(args.monthly_output).expanduser()
    active_output_path = Path(args.active_output).expanduser()
    existing_ranked = read_existing_by_subreddit(output_path) if args.resume else {}
    existing_monthly = read_existing_monthly(monthly_output_path) if args.resume else {}

    ranked_by_subreddit: Dict[str, Dict[str, Any]] = dict(existing_ranked)
    monthly_by_key: Dict[Tuple[str, str], Dict[str, Any]] = dict(existing_monthly)
    if args.resume and existing_ranked:
        print(f"Resuming: found {len(existing_ranked)} already scored subreddits")

    for index, candidate in enumerate(selected, start=args.start_index):
        subreddit_key = clean_subreddit(candidate["subreddit"]).lower()
        if args.resume and subreddit_key in ranked_by_subreddit:
            print(f"[{index}/{len(candidates)}] r/{candidate['subreddit']} already scored; skipping", file=sys.stderr)
            continue
        print(f"[{index}/{len(candidates)}] r/{candidate['subreddit']} keywords={candidate.get('candidate_keywords','')}", file=sys.stderr)
        ranked_row, candidate_monthly_rows = score_candidate(candidate, months, after, before, args)
        ranked_by_subreddit[subreddit_key] = ranked_row
        for monthly_row in candidate_monthly_rows:
            monthly_by_key[(subreddit_key, str(monthly_row.get("month", "")))] = monthly_row
        print(
            f"  active_ai_months={ranked_row['active_ai_months']} posts={ranked_row['any_keyword_posts']}",
            file=sys.stderr,
        )
        if args.subreddit_delay_minutes > 0 and index < args.start_index + len(selected) - 1:
            time.sleep(args.subreddit_delay_minutes * 60)

    ranked_rows = list(ranked_by_subreddit.values())
    monthly_rows = list(monthly_by_key.values())
    ranked_rows.sort(key=lambda row: (int(row["active_ai_months"]), int(row["any_keyword_posts"])), reverse=True)
    monthly_rows.sort(key=lambda row: (str(row.get("subreddit", "")).lower(), str(row.get("month", ""))))
    active_rows = [
        row for row in ranked_rows
        if int(row.get("active_ai_months") or 0) >= args.active_month_threshold
    ]

    write_csv(output_path, ranked_rows, RANKED_FIELDS)
    write_csv(monthly_output_path, monthly_rows, MONTHLY_FIELDS)
    write_csv(active_output_path, active_rows, RANKED_FIELDS)
    print(f"Ranked rows: {len(ranked_rows)}")
    print(f"Active >= {args.active_month_threshold}: {len(active_rows)}")
    print(f"Output: {args.output}")
    print(f"Monthly output: {args.monthly_output}")
    print(f"Active output: {args.active_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
