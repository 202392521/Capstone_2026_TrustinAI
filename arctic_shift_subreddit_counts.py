#!/usr/bin/env python3
"""
Count monthly Reddit submissions mentioning ChatGPT / AI / generative AI for
subreddits marked "yes" in a screening CSV or Numbers file, using Arctic Shift.

Example:
  python3 arctic_shift_subreddit_counts.py \
    --input /Users/qichenzhao/Documents/subreddit_screening.numbers \
    --yes-column final_decision \
    --subreddit-column subreddit \
    --output /Users/qichenzhao/Documents/Codex/2026-06-26/hi/outputs/subreddit_chatgpt_ai_counts_as.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


AS_BASE_URL = "https://arctic-shift.photon-reddit.com"
START_MONTH = "2023-03"
END_MONTH = "2026-01"

KEYWORD_QUERIES = {
    "chatgpt_count": "ChatGPT",
    "ai_count": "AI",
    "generative_ai_count": '"generative AI"',
    "any_keyword_count": '(ChatGPT OR AI OR "generative AI")',
}

YES_VALUES = {"yes", "y", "true", "1", "include", "included", "selected", "keep"}
NO_VALUES = {"no", "n", "false", "0", "exclude", "excluded", "skip", "maybe", "unclear"}


@dataclass(frozen=True)
class MonthWindow:
    label: str
    start_utc: int
    end_utc: int


class ArcticShiftRequestError(RuntimeError):
    def __init__(self, message: str, url: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.url = url
        self.status = status


class FailureLogger:
    fieldnames = ["subreddit", "month", "metric", "reason", "status"]

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._needs_header = not self.path.exists() or self.path.stat().st_size == 0

    def log(self, subreddit: str, month: str, metric: str, reason: str, status: Optional[int]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if self._needs_header:
                writer.writeheader()
                self._needs_header = False
            writer.writerow(
                {
                    "subreddit": subreddit,
                    "month": month,
                    "metric": metric,
                    "reason": reason,
                    "status": status if status is not None else "",
                }
            )


def utc_timestamp(year: int, month: int, day: int = 1) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())


def next_month(year: int, month: int) -> Tuple[int, int]:
    if month == 12:
        return year + 1, 1
    return year, month + 1


def month_windows(start_month: str, end_month: str) -> List[MonthWindow]:
    start_year, start_mon = map(int, start_month.split("-"))
    end_year, end_mon = map(int, end_month.split("-"))
    windows: List[MonthWindow] = []
    year, mon = start_year, start_mon
    while (year, mon) <= (end_year, end_mon):
        ny, nm = next_month(year, mon)
        windows.append(
            MonthWindow(
                label=f"{year:04d}-{mon:02d}",
                start_utc=utc_timestamp(year, mon),
                end_utc=utc_timestamp(ny, nm),
            )
        )
        year, mon = ny, nm
    return windows


def normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def normalize_subreddit(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^https?://(?:www\.)?reddit\.com/r/", "", value, flags=re.I)
    value = re.sub(r"^/?r/", "", value, flags=re.I)
    return value.strip("/ ")


def is_yes(value: Any) -> bool:
    return str(value).strip().lower() in YES_VALUES


def is_yes_no_like(values: Iterable[str]) -> bool:
    seen = {str(v).strip().lower() for v in values if str(v).strip()}
    if not seen:
        return False
    return bool(seen & YES_VALUES) and seen.issubset(YES_VALUES | NO_VALUES)


def export_numbers_to_csv(numbers_path: Path) -> Path:
    """Use macOS Numbers to export the document as CSV."""
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
    except FileNotFoundError as exc:
        raise RuntimeError("Cannot export .numbers: osascript is not available.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            "Cannot export .numbers automatically. Open it in Numbers and export as CSV, "
            f"then rerun with --input exported.csv. Details: {detail}"
        ) from exc

    csv_files = sorted(out_dir.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"Numbers export completed but no CSV was created in {out_dir}")
    return csv_files[0]


def read_screening_rows(input_path: Path) -> List[Dict[str, str]]:
    if input_path.suffix.lower() == ".numbers":
        print(f"Exporting Numbers file to temporary CSV: {input_path}", file=sys.stderr)
        input_path = export_numbers_to_csv(input_path)

    if input_path.suffix.lower() != ".csv":
        raise ValueError("Please provide a .csv file, or a .numbers file on macOS with Numbers installed.")

    with input_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [{k: (v or "").strip() for k, v in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"No data rows found in {input_path}")
    return rows


def choose_column(headers: Sequence[str], requested: Optional[str], candidates: Sequence[str], label: str) -> str:
    if requested:
        for header in headers:
            if normalize_header(header) == normalize_header(requested):
                return header
        raise ValueError(f"Could not find requested {label} column: {requested}")

    normalized = {normalize_header(h): h for h in headers}
    for candidate in candidates:
        if normalize_header(candidate) in normalized:
            return normalized[normalize_header(candidate)]
    raise ValueError(f"Could not auto-detect {label} column. Available columns: {', '.join(headers)}")


def choose_yes_column(headers: Sequence[str], rows: Sequence[Dict[str, str]], requested: Optional[str]) -> str:
    if requested:
        for header in headers:
            if normalize_header(header) == normalize_header(requested):
                return header
        raise ValueError(f"Could not find requested yes column: {requested}")

    yes_no_columns = [h for h in headers if is_yes_no_like(row.get(h, "") for row in rows)]
    for explicit_default in ("final_decision", "include", "included", "selected"):
        for header in yes_no_columns:
            if normalize_header(header) == explicit_default:
                return header
    if len(yes_no_columns) == 1:
        return yes_no_columns[0]
    raise ValueError(
        "Multiple possible yes/no columns found. Please rerun with --yes-column. "
        f"Candidates: {', '.join(yes_no_columns) if yes_no_columns else 'none'}"
    )


def load_yes_subreddits(
    input_path: Path,
    subreddit_column: Optional[str],
    yes_column: Optional[str],
    start_index: int,
    batch_size: Optional[int],
    limit: Optional[int],
) -> List[str]:
    rows = read_screening_rows(input_path)
    headers = list(rows[0].keys())
    subreddit_col = choose_column(
        headers,
        subreddit_column,
        candidates=("subreddit", "subreddits", "subreddit_name", "community", "name"),
        label="subreddit",
    )
    yes_col = choose_yes_column(headers, rows, yes_column)

    subreddits: List[str] = []
    seen = set()
    for row in rows:
        if not is_yes(row.get(yes_col, "")):
            continue
        subreddit = normalize_subreddit(row.get(subreddit_col, ""))
        if not subreddit:
            continue
        key = subreddit.lower()
        if key in seen:
            continue
        seen.add(key)
        subreddits.append(subreddit)

    print(f"Using subreddit column: {subreddit_col}", file=sys.stderr)
    print(f"Using yes column: {yes_col}", file=sys.stderr)
    print(f"Loaded {len(subreddits)} yes-labeled subreddits before batching", file=sys.stderr)

    if start_index < 1:
        raise ValueError("--start-index must be 1 or greater.")
    start = start_index - 1
    end = len(subreddits) if batch_size is None else start + batch_size
    if limit is not None:
        end = min(end, start + limit)
    selected = subreddits[start:end]
    print(f"Selected subreddits {start + 1}-{min(end, len(subreddits))} of {len(subreddits)}", file=sys.stderr)
    print(f"Loaded {len(selected)} yes-labeled subreddits for this run", file=sys.stderr)
    return selected


def api_get_json(
    base_url: str,
    path: str,
    params: Dict[str, Any],
    retries: int,
    timeout: int,
    pause: float,
) -> Dict[str, Any]:
    encoded = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    url = f"{base_url.rstrip('/')}{path}?{encoded}"
    last_error: Optional[Exception] = None

    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "codex-arctic-shift-counts/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if attempt < retries - 1:
                wait = pause * (2 ** attempt)
                print(f"    request failed (HTTP Error {exc.code}: {exc.reason}); retrying in {wait:.1f}s", file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"    request failed (HTTP Error {exc.code}: {exc.reason}); no retries left", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - retry HTTP/network/API transient failures.
            last_error = exc
            if attempt < retries - 1:
                wait = pause * (2 ** attempt)
                print(f"    request failed ({exc}); retrying in {wait:.1f}s", file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"    request failed ({exc}); no retries left", file=sys.stderr)

    status = last_error.code if isinstance(last_error, urllib.error.HTTPError) else None
    raise ArcticShiftRequestError(
        f"Arctic Shift request failed after {retries} retries: {last_error}\nURL: {url}",
        url=url,
        status=status,
    )


def month_from_as_bucket(value: Any, windows: Sequence[MonthWindow]) -> Optional[str]:
    """Map Arctic Shift bucket labels to YYYY-MM.

    Monthly aggregate buckets may appear as the previous day 22:00/23:00 UTC due
    to local time bucket boundaries. Adding 12 hours places them safely in the
    intended month.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)) or str(value).isdigit():
        dt = datetime.fromtimestamp(int(value), tz=timezone.utc)
    else:
        raw = str(value).strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)

    shifted = dt + timedelta(hours=12)
    label = f"{shifted.year:04d}-{shifted.month:02d}"
    valid = {w.label for w in windows}
    return label if label in valid else None


def parse_count(value: Any) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def fetch_keyword_monthly_counts(
    base_url: str,
    subreddit: str,
    query: str,
    windows: Sequence[MonthWindow],
    metric: str,
    retries: int,
    monthly_fallback_retries: int,
    max_month_failures: int,
    timeout: int,
    pause: float,
    failure_logger: FailureLogger,
) -> Dict[str, int]:
    counts = {w.label: 0 for w in windows}
    try:
        payload = api_get_json(
            base_url,
            "/api/posts/search/aggregate",
            {
                "aggregate": "created_utc",
                "frequency": "month",
                "subreddit": subreddit,
                "query": query,
                "after": windows[0].start_utc,
                "before": windows[-1].end_utc,
            },
            retries=retries,
            timeout=timeout,
            pause=pause,
        )
    except ArcticShiftRequestError as exc:
        print(f"    full-range aggregate failed; falling back to month-by-month ({exc.status})", file=sys.stderr)
        failure_logger.log(subreddit, "ALL", metric, "full_range_aggregate_failed", exc.status)
        return fetch_keyword_counts_month_by_month(
            base_url,
            subreddit,
            query,
            windows,
            metric,
            monthly_fallback_retries,
            max_month_failures,
            timeout,
            pause,
            failure_logger,
        )

    data = payload.get("data") or []
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected aggregate response for r/{subreddit}: {payload}")

    for item in data:
        if not isinstance(item, dict):
            continue
        label = month_from_as_bucket(item.get("created_utc") or item.get("date"), windows)
        if label:
            counts[label] = parse_count(item.get("count") or item.get("value"))
    return counts


def fetch_keyword_counts_month_by_month(
    base_url: str,
    subreddit: str,
    query: str,
    windows: Sequence[MonthWindow],
    metric: str,
    retries: int,
    max_month_failures: int,
    timeout: int,
    pause: float,
    failure_logger: FailureLogger,
) -> Dict[str, int]:
    counts = {w.label: 0 for w in windows}
    failed_months = 0
    for window in windows:
        if max_month_failures and failed_months >= max_month_failures:
            print(
                f"    too many failed months for r/{subreddit}; skipping remaining months for this keyword",
                file=sys.stderr,
            )
            for remaining in windows[windows.index(window):]:
                failure_logger.log(subreddit, remaining.label, metric, "skipped_after_max_month_failures", None)
            break
        try:
            payload = api_get_json(
                base_url,
                "/api/posts/search/aggregate",
                {
                    "aggregate": "created_utc",
                    "frequency": "month",
                    "subreddit": subreddit,
                    "query": query,
                    "after": window.start_utc,
                    "before": window.end_utc,
                },
                retries=retries,
                timeout=timeout,
                pause=pause,
            )
        except ArcticShiftRequestError as exc:
            failed_months += 1
            print(f"    warning: {window.label} failed for r/{subreddit}; writing 0 ({exc.status})", file=sys.stderr)
            failure_logger.log(subreddit, window.label, metric, "month_aggregate_failed", exc.status)
            continue

        data = payload.get("data") or []
        if not isinstance(data, list):
            failed_months += 1
            print(f"    warning: unexpected {window.label} response for r/{subreddit}; writing 0", file=sys.stderr)
            failure_logger.log(subreddit, window.label, metric, "unexpected_month_response", None)
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            label = month_from_as_bucket(item.get("created_utc") or item.get("date"), [window])
            if label == window.label:
                counts[window.label] = parse_count(item.get("count") or item.get("value"))
                break
        failed_months = 0
        time.sleep(pause)
    return counts


def fetch_total_post_monthly_counts(
    base_url: str,
    subreddit: str,
    windows: Sequence[MonthWindow],
    retries: int,
    timeout: int,
    pause: float,
) -> Dict[str, int]:
    counts = {w.label: 0 for w in windows}
    try:
        payload = api_get_json(
            base_url,
            "/api/time_series",
            {
                "key": f"r/{subreddit}/posts/count",
                "precision": "month",
                "after": windows[0].start_utc,
                "before": windows[-1].end_utc,
            },
            retries=retries,
            timeout=timeout,
            pause=pause,
        )
    except ArcticShiftRequestError as exc:
        print(f"    time_series failed; falling back to month-by-month ({exc.status})", file=sys.stderr)
        return fetch_total_posts_month_by_month(base_url, subreddit, windows, retries, timeout, pause)

    data = payload.get("data") or []
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected time_series response for r/{subreddit}: {payload}")

    for item in data:
        if not isinstance(item, dict):
            continue
        label = month_from_as_bucket(item.get("date") or item.get("created_utc"), windows)
        if label:
            counts[label] = parse_count(item.get("value") or item.get("count"))
    return counts


def fetch_total_posts_month_by_month(
    base_url: str,
    subreddit: str,
    windows: Sequence[MonthWindow],
    retries: int,
    timeout: int,
    pause: float,
) -> Dict[str, int]:
    counts = {w.label: 0 for w in windows}
    for window in windows:
        try:
            payload = api_get_json(
                base_url,
                "/api/time_series",
                {
                    "key": f"r/{subreddit}/posts/count",
                    "precision": "month",
                    "after": window.start_utc,
                    "before": window.end_utc,
                },
                retries=retries,
                timeout=timeout,
                pause=pause,
            )
        except ArcticShiftRequestError as exc:
            print(f"    warning: total posts {window.label} failed for r/{subreddit}; writing 0 ({exc.status})", file=sys.stderr)
            continue

        data = payload.get("data") or []
        if not isinstance(data, list):
            print(f"    warning: unexpected total-post response for r/{subreddit}; writing 0", file=sys.stderr)
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            label = month_from_as_bucket(item.get("date") or item.get("created_utc"), [window])
            if label == window.label:
                counts[window.label] = parse_count(item.get("value") or item.get("count"))
                break
        time.sleep(pause)
    return counts


def completed_subreddits_from_output(output_path: Path, windows: Sequence[MonthWindow]) -> set:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()

    expected_months = {w.label for w in windows}
    seen_months: Dict[str, set] = {}
    with output_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subreddit = (row.get("subreddit") or "").strip()
            month = (row.get("month") or "").strip()
            if not subreddit or month not in expected_months:
                continue
            seen_months.setdefault(subreddit.lower(), set()).add(month)

    return {subreddit for subreddit, months in seen_months.items() if months == expected_months}


def write_results(
    output_path: Path,
    subreddits: Sequence[str],
    windows: Sequence[MonthWindow],
    base_url: str,
    retries: int,
    monthly_fallback_retries: int,
    max_month_failures: int,
    timeout: int,
    pause: float,
    resume: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_col = f"total_posts_{windows[0].label.replace('-', '_')}_to_{windows[-1].label.replace('-', '_')}"
    failure_path = output_path.with_name(f"{output_path.stem}_failures.csv")
    failure_logger = FailureLogger(failure_path)
    print(f"Failure log: {failure_path}", file=sys.stderr)
    fieldnames = [
        "subreddit",
        "month",
        "chatgpt_count",
        "ai_count",
        "generative_ai_count",
        "any_keyword_count",
        "total_posts_month",
        total_col,
    ]

    completed = completed_subreddits_from_output(output_path, windows) if resume else set()
    if completed:
        print(f"Resuming: found {len(completed)} complete subreddits in existing output", file=sys.stderr)

    mode = "a" if resume and output_path.exists() and output_path.stat().st_size > 0 else "w"
    with output_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()

        for index, subreddit in enumerate(subreddits, start=1):
            if subreddit.lower() in completed:
                print(f"[{index}/{len(subreddits)}] r/{subreddit} already complete; skipping", file=sys.stderr)
                continue
            print(f"[{index}/{len(subreddits)}] r/{subreddit}", file=sys.stderr)
            keyword_results: Dict[str, Dict[str, int]] = {}
            for column, query in KEYWORD_QUERIES.items():
                print(f"  keyword aggregate: {column}", file=sys.stderr)
                keyword_results[column] = fetch_keyword_monthly_counts(
                    base_url,
                    subreddit,
                    query,
                    windows,
                    column,
                    retries,
                    monthly_fallback_retries,
                    max_month_failures,
                    timeout,
                    pause,
                    failure_logger,
                )
                time.sleep(pause)

            print("  total posts time_series", file=sys.stderr)
            total_monthly = fetch_total_post_monthly_counts(base_url, subreddit, windows, retries, timeout, pause)
            total_period = sum(total_monthly.values())

            for window in windows:
                writer.writerow(
                    {
                        "subreddit": subreddit,
                        "month": window.label,
                        "chatgpt_count": keyword_results["chatgpt_count"].get(window.label, 0),
                        "ai_count": keyword_results["ai_count"].get(window.label, 0),
                        "generative_ai_count": keyword_results["generative_ai_count"].get(window.label, 0),
                        "any_keyword_count": keyword_results["any_keyword_count"].get(window.label, 0),
                        "total_posts_month": total_monthly.get(window.label, 0),
                        total_col: total_period,
                    }
                )
            f.flush()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count monthly Reddit submission mentions using Arctic Shift API."
    )
    parser.add_argument("--input", required=True, help="Screening CSV, or .numbers on macOS with Numbers installed.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--yes-column", help="Column whose value must be yes/include/true/1.")
    parser.add_argument("--subreddit-column", help="Column containing subreddit names.")
    parser.add_argument("--start-month", default=START_MONTH, help="Inclusive YYYY-MM start month. Default: 2023-03.")
    parser.add_argument("--end-month", default=END_MONTH, help="Inclusive YYYY-MM end month. Default: 2026-01.")
    parser.add_argument("--base-url", default=AS_BASE_URL, help=f"Arctic Shift base URL. Default: {AS_BASE_URL}")
    parser.add_argument("--retries", type=int, default=4, help="Retries per API request. Default: 4.")
    parser.add_argument(
        "--monthly-fallback-retries",
        type=int,
        default=1,
        help="Retries for month-by-month fallback requests after a full-range 422. Default: 1.",
    )
    parser.add_argument(
        "--max-month-failures",
        type=int,
        default=3,
        help="Stop month-by-month fallback for a keyword after this many consecutive failed months. Default: 3.",
    )
    parser.add_argument("--timeout", type=int, default=90, help="Request timeout in seconds. Default: 90.")
    parser.add_argument("--pause", type=float, default=0.5, help="Seconds to pause between requests. Default: 0.5.")
    parser.add_argument("--limit-subreddits", type=int, help="Only process first N yes-labeled subreddits for testing.")
    parser.add_argument("--start-index", type=int, default=1, help="1-based index of first yes-labeled subreddit to process.")
    parser.add_argument("--batch-size", type=int, help="Only process this many subreddits from --start-index.")
    parser.add_argument("--resume", action="store_true", help="Append to existing output and skip complete subreddits.")
    parser.add_argument("--dry-run", action="store_true", help="Load input and print subreddits without API calls.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    subreddits = load_yes_subreddits(
        Path(args.input).expanduser(),
        args.subreddit_column,
        args.yes_column,
        args.start_index,
        args.batch_size,
        args.limit_subreddits,
    )
    if not subreddits:
        raise ValueError("No yes-labeled subreddits found.")

    if args.dry_run:
        print("First subreddits:", ", ".join(subreddits[:20]), file=sys.stderr)
        return 0

    windows = month_windows(args.start_month, args.end_month)
    write_results(
        Path(args.output).expanduser(),
        subreddits,
        windows,
        args.base_url,
        args.retries,
        args.monthly_fallback_retries,
        args.max_month_failures,
        args.timeout,
        args.pause,
        args.resume,
    )
    print(f"Done. Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
