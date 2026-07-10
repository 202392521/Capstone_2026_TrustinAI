#!/usr/bin/env python3
"""
Filter the SIC screening sheet to software-engineer-related subreddits,
count AI discussion for those subreddits with Arctic Shift, then rank them.

This is a narrow wrapper around:
  - arctic_shift_subreddit_counts.py
  - rank_ai_discussion_subreddits.py

Default outputs are written under:
  outputs/software_engineer_ai/
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence


WORKSPACE = Path("/Users/qichenzhao/Documents/Codex/2026-06-26/hi")
DEFAULT_INPUT = Path("/Users/qichenzhao/Documents/subreddit_screening.numbers")
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs" / "software_engineer_ai"
DEFAULT_GLOBAL_RANKED = WORKSPACE / "outputs" / "sic_screened_subreddits_ai_ranked.csv"
ARCTIC_SCRIPT = WORKSPACE / "outputs" / "arctic_shift_subreddit_counts.py"
RANK_SCRIPT = WORKSPACE / "outputs" / "rank_ai_discussion_subreddits.py"


YES_VALUES = {"yes", "y", "true", "1", "include", "included", "selected", "keep"}


def normalize_keyword(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def keyword_slug(value: str) -> str:
    return normalize_keyword(value).replace(" ", "_")


def export_numbers_to_csv(numbers_path: Path) -> Path:
    out_dir = Path(tempfile.mkdtemp(prefix="numbers_export_"))
    out_file = out_dir / f"{numbers_path.stem}.csv"
    script = f'''
set inputPath to POSIX file "{numbers_path}"
set outputPath to POSIX file "{out_file}"
tell application "Numbers"
    activate
    set docRef to open inputPath
    export docRef to outputPath as CSV
    close docRef saving no
end tell
'''
    try:
        subprocess.run(["osascript", "-e", script], check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            "Could not export .numbers automatically. Export it to CSV from Numbers and rerun with --input exported.csv. "
            f"Details: {detail}"
        ) from exc
    csv_files = sorted(out_dir.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"Numbers export completed but no CSV was created in {out_dir}")
    return csv_files[0]


def read_rows(path: Path) -> List[Dict[str, str]]:
    if path.suffix.lower() == ".numbers":
        print(f"Exporting Numbers file to temporary CSV: {path}", file=sys.stderr)
        path = export_numbers_to_csv(path)
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def filter_software_engineer_rows(rows: Sequence[Dict[str, str]], keyword: str) -> List[Dict[str, str]]:
    wanted = normalize_keyword(keyword)
    filtered: List[Dict[str, str]] = []
    seen = set()
    for row in rows:
        if normalize_keyword(row.get("keyword", "")) != wanted:
            continue
        if row.get("final_decision", "").strip().lower() not in YES_VALUES:
            continue
        subreddit = row.get("subreddit", "").strip()
        if not subreddit:
            continue
        key = subreddit.lower()
        if key in seen:
            continue
        seen.add(key)
        filtered.append(row)
    return filtered


def write_filtered_csv(rows: Sequence[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No software-engineer rows with final_decision=yes were found.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote filtered subreddit list: {path}", file=sys.stderr)
    print(f"Filtered subreddits: {len(rows)}", file=sys.stderr)


def run_command(cmd: Sequence[str]) -> None:
    print("Running:", " ".join(cmd), file=sys.stderr)
    subprocess.run(list(cmd), check=True)


def run_counts_for_batches(args: argparse.Namespace, filtered_csv: Path, raw_output: Path) -> None:
    # Batch at the wrapper level so each chunk can pause between API bursts.
    total = count_filtered_rows(filtered_csv)
    start = args.start_index
    while start <= total:
        batch_end = min(start + args.batch_size - 1, total)
        print(f"=== software engineer batch {start}-{batch_end} of {total} ===", file=sys.stderr)
        run_command(
            [
                "python3",
                str(ARCTIC_SCRIPT),
                "--input",
                str(filtered_csv),
                "--yes-column",
                "final_decision",
                "--subreddit-column",
                "subreddit",
                "--start-month",
                args.start_month,
                "--end-month",
                args.end_month,
                "--start-index",
                str(start),
                "--batch-size",
                str(args.batch_size),
                "--output",
                str(raw_output),
                "--resume",
                "--retries",
                str(args.retries),
                "--monthly-fallback-retries",
                str(args.monthly_fallback_retries),
                "--max-month-failures",
                str(args.max_month_failures),
                "--timeout",
                str(args.timeout),
                "--pause",
                str(args.pause),
            ]
        )
        start += args.batch_size
        if start <= total and args.delay_minutes > 0:
            print(f"Sleeping {args.delay_minutes} minutes before next batch", file=sys.stderr)
            time.sleep(args.delay_minutes * 60)


def count_filtered_rows(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return sum(1 for _ in csv.DictReader(f))


def run_ranker(raw_output: Path, ranked: Path, active: Path, inactive: Path) -> None:
    failures = raw_output.with_name(f"{raw_output.stem}_failures.csv")
    cmd = [
        "python3",
        str(RANK_SCRIPT),
        "--input",
        str(raw_output),
        "--output",
        str(ranked),
        "--active-output",
        str(active),
        "--inactive-output",
        str(inactive),
    ]
    if failures.exists():
        cmd.extend(["--failures", str(failures)])
    run_command(cmd)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_rows_like(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def merge_active_into_global(global_ranked: Path, active: Path, combined: Path) -> None:
    global_rows = read_csv_rows(global_ranked)
    active_rows = read_csv_rows(active)
    if not global_rows:
        return
    fieldnames = list(global_rows[0].keys())
    merged: Dict[str, Dict[str, str]] = {}
    for row in global_rows:
        subreddit = (row.get("subreddit") or "").strip().lower()
        if subreddit:
            merged[subreddit] = row
    for row in active_rows:
        subreddit = (row.get("subreddit") or "").strip().lower()
        if subreddit:
            merged[subreddit] = row

    def sort_key(row: Dict[str, str]) -> tuple:
        return (
            int(float(row.get("has_ai_discussion") or 0)),
            int(float(row.get("any_keyword_posts") or 0)),
            int(float(row.get("active_ai_months") or 0)),
            float(row.get("ai_posts_per_1000_total_posts") or 0),
        )

    rows = sorted(merged.values(), key=sort_key, reverse=True)
    write_rows_like(combined, rows, fieldnames)
    print(f"Wrote combined ranked output: {combined}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI discussion ranking for software-engineer screened subreddits.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="subreddit_screening CSV or Numbers file.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--keyword", default="software engineer", help="Keyword value to filter. Default: software engineer.")
    parser.add_argument("--start-month", default="2023-03")
    parser.add_argument("--end-month", default="2026-01")
    parser.add_argument("--start-index", type=int, default=1, help="1-based row index within filtered software-engineer list.")
    parser.add_argument("--batch-size", type=int, default=5, help="Subreddits per batch. Default: 5.")
    parser.add_argument("--delay-minutes", type=float, default=10, help="Pause between batches. Default: 10.")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--monthly-fallback-retries", type=int, default=1)
    parser.add_argument("--max-month-failures", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--pause", type=float, default=0.5)
    parser.add_argument("--filter-only", action="store_true", help="Only write filtered CSV; do not call Arctic Shift.")
    parser.add_argument("--rank-only", action="store_true", help="Only rerun ranking from existing raw counts.")
    parser.add_argument("--global-ranked", default=str(DEFAULT_GLOBAL_RANKED), help="Existing global ai_ranked CSV to merge into.")
    parser.add_argument(
        "--combined-output",
        help="Combined ranked CSV. Default: <output-dir>/sic_screened_subreddits_ai_ranked_plus_software_engineer.csv",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = keyword_slug(args.keyword)
    filtered_csv = output_dir / f"{slug}_screened_subreddits.csv"
    raw_output = output_dir / f"{slug}_ai_counts_as.csv"
    ranked = output_dir / f"{slug}_ai_ranked.csv"
    active = output_dir / f"{slug}_with_ai_discussion.csv"
    inactive = output_dir / f"{slug}_without_ai_discussion.csv"

    if not args.rank_only:
        rows = read_rows(Path(args.input).expanduser())
        filtered = filter_software_engineer_rows(rows, args.keyword)
        write_filtered_csv(filtered, filtered_csv)
        if args.filter_only:
            return 0
        run_counts_for_batches(args, filtered_csv, raw_output)

    if not raw_output.exists() or raw_output.stat().st_size == 0:
        raise ValueError(f"Raw counts file does not exist yet: {raw_output}")
    run_ranker(raw_output, ranked, active, inactive)
    combined = Path(args.combined_output).expanduser() if args.combined_output else output_dir / "sic_screened_subreddits_ai_ranked_plus_software_engineer.csv"
    merge_active_into_global(Path(args.global_ranked).expanduser(), active, combined)
    print(f"Wrote ranked output: {ranked}")
    print(f"Wrote active output: {active}")
    print(f"Wrote inactive output: {inactive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
