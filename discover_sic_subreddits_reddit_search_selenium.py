#!/usr/bin/env python3
"""
Discover candidate subreddits for all ONS SIC sections using Reddit community search.

This script extends the user's original Selenium workflow:
  keyword -> reddit community search page -> scroll -> extract /r/ links from HTML.

It does NOT use PullPush or Arctic Shift. This makes it useful as the broad
preliminary subreddit discovery stage before manual screening and AS scoring.

This code only serves to preliminary research, which aims to adapt a general view of SIC-related industry subreddit availability 
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from urllib.parse import quote

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchWindowException, WebDriverException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


DEFAULT_KEYWORDS = (
    "/Users/qichenzhao/Documents/Codex/2026-06-26/hi/outputs/"
    "ons_sic_occupation_keywords_expanded.csv"
)
DEFAULT_OUTPUT = (
    "/Users/qichenzhao/Documents/Codex/2026-06-26/hi/outputs/"
    "sic_reddit_search_candidate_subreddits.csv"
)

BASE_URL = "https://www.reddit.com/search/?q={query}&type=communities"

OUTPUT_COLUMNS = [
    "discovery_timestamp",
    "sic_section_code",
    "sic_section_name",
    "keyword",
    "query",
    "preliminary_sic_section",
    "subreddit",
    "subreddit_url",
    "raw_text",
    "source_url",
    "search_rank",
    "keyword_candidate_count",
    "discovery_warning",
    "keep",
    "manual_sic_section",
    "screening_status",
    "notes",
]


def normalize_subreddit(value: str) -> str:
    value = value.strip()
    value = value.replace("https://www.reddit.com/r/", "")
    value = value.replace("https://reddit.com/r/", "")
    value = value.replace("/r/", "")
    value = re.sub(r"^r/", "", value, flags=re.I)
    return value.strip("/ ")


def sic_label(row: Dict[str, str]) -> str:
    code = (row.get("sic_section_code") or "").strip()
    name = (row.get("sic_section_name") or "").strip()
    if code and name:
        short_name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
        return f"{code}_{short_name}"
    return row.get("preliminary_sic_section", "").strip()


def load_keywords(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No keyword rows found in {path}")

    # Supports both the new SIC keyword file and the user's original two-column file.
    if "query" not in rows[0] and "keyword" in rows[0]:
        for row in rows:
            row["query"] = row.get("keyword", "")
            row["occupation_keyword"] = row.get("keyword", "")
    elif "occupation_keyword" not in rows[0] and "keyword" in rows[0]:
        for row in rows:
            row["occupation_keyword"] = row.get("keyword", "")

    missing = [col for col in ["occupation_keyword", "query"] if col not in rows[0]]
    if missing:
        raise ValueError(f"Keyword file missing required columns: {', '.join(missing)}")
    return rows


def slice_rows(rows: Sequence[Dict[str, str]], start_index: int, limit: int) -> List[Dict[str, str]]:
    if start_index < 1:
        raise ValueError("--start-index must be 1 or greater.")
    start = start_index - 1
    if limit <= 0:
        return list(rows[start:])
    return list(rows[start : start + limit])


def start_driver(headless: bool) -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    else:
        options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def scroll_page(driver: webdriver.Chrome, n_scrolls: int, scroll_pause: float) -> None:
    for _ in range(n_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause)


def extract_communities_from_html(
    html: str,
    keyword_row: Dict[str, str],
    source_url: str,
) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    rows: List[Dict[str, str]] = []
    seen = set()
    timestamp = datetime.now().isoformat(timespec="seconds")

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/r/" not in href:
            continue

        try:
            subreddit = href.split("/r/")[1].split("/")[0].strip()
        except IndexError:
            continue

        subreddit = normalize_subreddit(subreddit)
        if not subreddit:
            continue
        subreddit_key = subreddit.lower()
        if subreddit_key in seen:
            continue
        seen.add(subreddit_key)

        section_code = (keyword_row.get("sic_section_code") or "").strip()
        section_name = (keyword_row.get("sic_section_name") or "").strip()
        occupation = (keyword_row.get("occupation_keyword") or keyword_row.get("keyword") or "").strip()
        query = (keyword_row.get("query") or occupation).strip()
        preliminary_sic = sic_label(keyword_row)

        rows.append(
            {
                "discovery_timestamp": timestamp,
                "sic_section_code": section_code,
                "sic_section_name": section_name,
                "keyword": occupation,
                "query": query,
                "preliminary_sic_section": preliminary_sic,
                "subreddit": subreddit,
                "subreddit_url": f"https://www.reddit.com/r/{subreddit}/",
                "raw_text": link.get_text(" ", strip=True),
                "source_url": source_url,
                "search_rank": str(len(rows) + 1),
                "keyword_candidate_count": "",
                "discovery_warning": "",
                "keep": "",
                "manual_sic_section": "",
                "screening_status": "pending_review",
                "notes": "",
            }
        )
    return rows


def discover_for_keyword(
    driver: webdriver.Chrome,
    keyword_row: Dict[str, str],
    n_scrolls: int,
    initial_wait: float,
    scroll_pause: float,
) -> List[Dict[str, str]]:
    occupation = (keyword_row.get("occupation_keyword") or keyword_row.get("keyword") or "").strip()
    query = (keyword_row.get("query") or occupation).strip()
    section = (keyword_row.get("sic_section_code") or keyword_row.get("preliminary_sic_section") or "").strip()
    url = BASE_URL.format(query=quote(query))

    print("\n" + "=" * 80)
    print(f"Section: {section}")
    print(f"Occupation keyword: {occupation}")
    print(f"Query: {query}")
    print(f"Opening: {url}")

    driver.get(url)
    time.sleep(initial_wait)
    scroll_page(driver, n_scrolls=n_scrolls, scroll_pause=scroll_pause)

    rows = extract_communities_from_html(driver.page_source, keyword_row, url)
    warning = ""
    if not rows:
        warning = "no_visible_communities_or_page_load_issue"
        print(f"Warning: {warning}")
    elif len(rows) >= 60:
        warning = "hit_visible_result_cap_possible"
        print(f"Warning: {warning}")

    for row in rows:
        row["keyword_candidate_count"] = str(len(rows))
        row["discovery_warning"] = warning

    print(f"Found {len(rows)} candidate communities.")
    return rows


def safe_quit_driver(driver: Optional[webdriver.Chrome]) -> None:
    if driver is None:
        return
    try:
        driver.quit()
    except WebDriverException:
        pass


def browser_error_message(error: BaseException) -> str:
    message = str(error).splitlines()[0] if str(error) else error.__class__.__name__
    return f"{error.__class__.__name__}: {message}"


def read_existing_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def merge_rows(existing_rows: Sequence[Dict[str, str]], new_rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    unique: Dict[str, Dict[str, str]] = {}

    def add_or_merge(row: Dict[str, str]) -> None:
        subreddit = normalize_subreddit(row.get("subreddit", ""))
        if not subreddit:
            return
        key = subreddit.lower()
        row = {field: row.get(field, "") for field in OUTPUT_COLUMNS}
        row["subreddit"] = subreddit
        row["subreddit_url"] = f"https://www.reddit.com/r/{subreddit}/"

        if key not in unique:
            unique[key] = row
            return

        old = unique[key]
        for field in [
            "keyword",
            "query",
            "preliminary_sic_section",
            "sic_section_code",
            "sic_section_name",
            "search_rank",
            "keyword_candidate_count",
            "discovery_warning",
        ]:
            old_values = [v for v in old.get(field, "").split(";") if v]
            new_values = [v for v in row.get(field, "").split(";") if v]
            for value in new_values:
                if value and value not in old_values:
                    old_values.append(value)
            old[field] = ";".join(old_values)

        if row.get("raw_text") and row["raw_text"] not in old.get("raw_text", ""):
            old["raw_text"] = (old.get("raw_text", "") + " || " + row["raw_text"]).strip(" |")
        if row.get("source_url") and row["source_url"] not in old.get("source_url", ""):
            old["source_url"] = (old.get("source_url", "") + " || " + row["source_url"]).strip(" |")

    for row in existing_rows:
        add_or_merge(row)
    for row in new_rows:
        add_or_merge(row)

    final_rows = list(unique.values())
    final_rows.sort(key=lambda row: (row.get("preliminary_sic_section", ""), row["subreddit"].lower()))
    return final_rows


def write_rows(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OUTPUT_COLUMNS})


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover SIC candidate subreddits from Reddit community search.")
    parser.add_argument("--keywords", default=DEFAULT_KEYWORDS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="0 means run all rows from start-index.")
    parser.add_argument("--scrolls", type=int, default=5)
    parser.add_argument("--initial-wait", type=float, default=6)
    parser.add_argument("--scroll-pause", type=float, default=2)
    parser.add_argument("--keyword-pause", type=float, default=3)
    parser.add_argument("--browser-retries", type=int, default=2)
    parser.add_argument("--browser-restart-wait", type=float, default=10)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing output and overwrite from this run.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    keyword_rows = load_keywords(Path(args.keywords).expanduser())
    selected_rows = slice_rows(keyword_rows, args.start_index, args.limit)
    output_path = Path(args.output).expanduser()

    print(f"Loaded keyword rows: {len(keyword_rows)}")
    print(f"Selected rows: {args.start_index}-{args.start_index + len(selected_rows) - 1}")
    print(f"Output file: {output_path}")

    all_new_rows: List[Dict[str, str]] = []
    driver: Optional[webdriver.Chrome] = start_driver(headless=args.headless)
    try:
        for row in selected_rows:
            discovered: List[Dict[str, str]] = []
            for attempt in range(1, args.browser_retries + 2):
                try:
                    if driver is None:
                        print("Starting a fresh Chrome session...")
                        driver = start_driver(headless=args.headless)
                    discovered = discover_for_keyword(
                        driver,
                        row,
                        n_scrolls=args.scrolls,
                        initial_wait=args.initial_wait,
                        scroll_pause=args.scroll_pause,
                    )
                    break
                except (NoSuchWindowException, WebDriverException) as error:
                    print(f"Browser failed on this keyword: {browser_error_message(error)}")
                    safe_quit_driver(driver)
                    driver = None
                    if attempt > args.browser_retries:
                        keyword = row.get("occupation_keyword") or row.get("keyword") or row.get("query") or ""
                        print(f"Warning: giving up on keyword after {attempt} attempts: {keyword}")
                        break
                    print(
                        f"Restarting Chrome in {args.browser_restart_wait:.1f}s "
                        f"(attempt {attempt + 1}/{args.browser_retries + 1})..."
                    )
                    time.sleep(args.browser_restart_wait)
            all_new_rows.extend(discovered)
            time.sleep(args.keyword_pause)
    finally:
        safe_quit_driver(driver)

    existing_rows = [] if args.fresh else read_existing_rows(output_path)
    final_rows = merge_rows(existing_rows, all_new_rows)
    write_rows(output_path, final_rows)

    print("\nDiscovery finished.")
    print(f"New rows discovered this run: {len(all_new_rows)}")
    print(f"Total unique candidate subreddits: {len(final_rows)}")
    print(f"Saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
