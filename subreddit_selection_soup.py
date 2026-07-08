import csv
import time
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RAW_DIR / "candidate_subreddits.csv"

KEYWORDS = [
    ("teacher", "P_Education"),
    ("nurse", "Q_Health_Social_Work"),
    ("software engineer", "J_Information_Communication"),
    ("accountant", "K_Financial_Insurance"),
    ("lawyer", "M_Professional_Scientific_Technical"),
]

BASE_URL = "https://www.reddit.com/search/?q={query}&type=communities"

OUTPUT_COLUMNS = [
    "keyword",
    "preliminary_sic_section",
    "subreddit",
    "subreddit_url",
    "raw_text",
    "keep",
    "manual_sic_section",
    "screening_status",
    "notes",
]


def start_driver() -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def scroll_page(driver: webdriver.Chrome, n_scrolls: int = 5) -> None:
    for _ in range(n_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)


def extract_communities_from_html(html: str, keyword: str, sic: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    seen = set()

    for link in soup.find_all("a", href=True):
        href = link["href"]

        if "/r/" not in href:
            continue

        try:
            subreddit = href.split("/r/")[1].split("/")[0].strip()
        except IndexError:
            continue

        if not subreddit:
            continue

        subreddit_key = subreddit.lower()

        if subreddit_key in seen:
            continue

        seen.add(subreddit_key)

        rows.append({
            "keyword": keyword,
            "preliminary_sic_section": sic,
            "subreddit": subreddit,
            "subreddit_url": f"https://www.reddit.com/r/{subreddit}/",
            "raw_text": link.get_text(" ", strip=True),
            "keep": "",
            "manual_sic_section": "",
            "screening_status": "pending_review",
            "notes": "",
        })

    return rows


def discover_for_keyword(driver: webdriver.Chrome, keyword: str, sic: str) -> List[Dict[str, str]]:
    url = BASE_URL.format(query=quote(keyword))
    print(f"\nOpening: {url}")

    driver.get(url)
    time.sleep(6)

    scroll_page(driver, n_scrolls=5)

    rows = extract_communities_from_html(driver.page_source, keyword, sic)
    print(f"Found {len(rows)} candidate communities for keyword: {keyword}")

    return rows


def save_rows(rows: List[Dict[str, str]]) -> None:
    unique = {}
    for row in rows:
        key = row["subreddit"].lower()
        if key not in unique:
            unique[key] = row
        else:
            old_keywords = unique[key]["keyword"]
            new_keyword = row["keyword"]
            if new_keyword not in old_keywords.split(";"):
                unique[key]["keyword"] = old_keywords + ";" + new_keyword

    final_rows = list(unique.values())
    final_rows.sort(key=lambda x: (x["keyword"], x["subreddit"].lower()))

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"\nSaved {len(final_rows)} candidate subreddits to:")
    print(OUTPUT_FILE)


def main() -> None:
    all_rows = []

    driver = start_driver()

    try:
        for keyword, sic in KEYWORDS:
            rows = discover_for_keyword(driver, keyword, sic)
            all_rows.extend(rows)
            time.sleep(3)
    finally:
        driver.quit()

    if not all_rows:
        print("No communities found.")
        return

    save_rows(all_rows)


if __name__ == "__main__":
    main()