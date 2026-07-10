#!/usr/bin/env python3
"""Check whether each screened industry has at least one active AI subreddit."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence


YES_VALUES = {"yes", "y", "true", "1", "include", "included", "selected", "keep"}


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
    subprocess.run(["osascript", "-e", script], check=True, text=True, capture_output=True)
    csv_files = sorted(out_dir.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"Numbers export completed but no CSV was created in {out_dir}")
    return csv_files[0]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if path.suffix.lower() == ".numbers":
        path = export_numbers_to_csv(path)
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def is_valid_industry_keyword(keyword: str) -> bool:
    value = keyword.strip().lower()
    if not value:
        return False
    if value.endswith(".py") or ".py " in value or value.startswith("python "):
        return False
    if re.search(r"\.(sh|py|js|ts|r|do)(\s|$)", value):
        return False
    return True


def load_ranked(path: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv(path)
    return {(row.get("subreddit") or "").strip().lower(): row for row in rows if row.get("subreddit")}


def summarize(
    screening_rows: Sequence[Dict[str, str]],
    ranked: Dict[str, Dict[str, str]],
    group_col: str,
) -> List[Dict[str, str]]:
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in screening_rows:
        keyword = row.get("keyword", "")
        if not is_valid_industry_keyword(keyword):
            continue
        if row.get("final_decision", "").strip().lower() not in YES_VALUES:
            continue
        group = (row.get(group_col) or "").strip()
        if not group:
            continue
        groups[group].append(row)

    output: List[Dict[str, str]] = []
    for group, rows in sorted(groups.items()):
        all_subs = set()
        observed = set()
        active = set()
        missing = set()
        for row in rows:
            subreddit = (row.get("subreddit") or "").strip()
            if not subreddit:
                continue
            key = subreddit.lower()
            all_subs.add(key)
            if key not in ranked:
                missing.add(key)
                continue
            observed.add(key)
            rank_row = ranked[key]
            has_ai = int(float(rank_row.get("has_ai_discussion") or 0)) == 1
            any_posts = int(float(rank_row.get("any_keyword_posts") or 0))
            if has_ai and any_posts > 0:
                active.add(key)

        top_active = sorted(
            active,
            key=lambda sub: int(float(ranked[sub].get("any_keyword_posts") or 0)),
            reverse=True,
        )[:5]
        output.append(
            {
                "group": group,
                "screened_yes_subreddits": str(len(all_subs)),
                "observed_in_complete_ranked": str(len(observed)),
                "active_ai_subreddits": str(len(active)),
                "missing_from_ranked": str(len(missing)),
                "has_at_least_one_active": "yes" if active else "no",
                "top_active_subreddits": ";".join(ranked[sub].get("subreddit", sub) for sub in top_active),
            }
        )
    return output


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "group",
        "screened_yes_subreddits",
        "observed_in_complete_ranked",
        "active_ai_subreddits",
        "missing_from_ranked",
        "has_at_least_one_active",
        "top_active_subreddits",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check active AI subreddit coverage by industry.")
    parser.add_argument("--screening", required=True, help="subreddit_screening CSV or Numbers file.")
    parser.add_argument("--ranked", required=True, help="Complete ai_ranked CSV.")
    parser.add_argument("--keyword-output", required=True, help="Coverage output grouped by keyword.")
    parser.add_argument("--sic-output", required=True, help="Coverage output grouped by preliminary_sic_section.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    screening_rows = read_csv(Path(args.screening).expanduser())
    ranked = load_ranked(Path(args.ranked).expanduser())
    keyword_rows = summarize(screening_rows, ranked, "keyword")
    sic_rows = summarize(screening_rows, ranked, "preliminary_sic_section")
    write_csv(Path(args.keyword_output).expanduser(), keyword_rows)
    write_csv(Path(args.sic_output).expanduser(), sic_rows)
    print(f"Wrote {args.keyword_output}")
    print(f"Wrote {args.sic_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
