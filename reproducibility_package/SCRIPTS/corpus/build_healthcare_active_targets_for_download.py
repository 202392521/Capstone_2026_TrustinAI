#!/usr/bin/env python3
"""Build healthcare download targets from active AI month scoring output."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List


PACKAGE_DIR = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PACKAGE_DIR / "OUTPUTS"

INPUT = OUTPUTS_DIR / "healthcare_doctor_active_ai_months_ge_10_cleaned_full.csv"
OUTPUT = OUTPUTS_DIR / "healthcare_active_ai_months_ge_10_targets.csv"

FIELDS = [
    "preliminary_sic_section",
    "industry_keyword",
    "subreddit",
    "active_ai_discussion",
    "months_observed",
    "active_ai_months",
    "any_keyword_posts",
    "chatgpt_posts",
    "ai_posts",
    "generative_ai_posts",
    "artificial_intelligence_posts",
    "first_ai_month",
    "last_ai_month",
    "peak_ai_month",
    "peak_ai_posts",
    "total_posts_period",
    "ai_posts_per_1000_total_posts",
    "data_quality",
    "source_files",
]


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def as_int(value: str) -> int:
    try:
        text = str(value or "").strip()
        return int(float(text)) if text else 0
    except ValueError:
        return 0


def write_rows(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows = read_rows(INPUT)
    output_rows: List[Dict[str, str]] = []
    for row in rows:
        output_rows.append(
            {
                "preliminary_sic_section": "Q_Health_Social_Work",
                "industry_keyword": "healthcare",
                "subreddit": row.get("subreddit", ""),
                "active_ai_discussion": row.get("has_ai_discussion", ""),
                "months_observed": row.get("months_observed", ""),
                "active_ai_months": row.get("active_ai_months", ""),
                "any_keyword_posts": row.get("any_keyword_posts", ""),
                "chatgpt_posts": row.get("chatgpt_posts", ""),
                "ai_posts": row.get("ai_posts", ""),
                "generative_ai_posts": row.get("generative_ai_posts", ""),
                "artificial_intelligence_posts": row.get("artificial_intelligence_posts", ""),
                "first_ai_month": row.get("first_ai_month", ""),
                "last_ai_month": row.get("last_ai_month", ""),
                "peak_ai_month": row.get("peak_ai_month", ""),
                "peak_ai_posts": row.get("peak_ai_posts", ""),
                "total_posts_period": row.get("total_posts_period", ""),
                "ai_posts_per_1000_total_posts": row.get("ai_posts_per_1000_total_posts", ""),
                "data_quality": row.get("data_quality", ""),
                "source_files": INPUT.name,
            }
        )
    output_rows.sort(
        key=lambda item: (
            -as_int(item.get("active_ai_months", "")),
            -as_int(item.get("any_keyword_posts", "")),
            item.get("subreddit", "").lower(),
        )
    )
    write_rows(OUTPUT, output_rows)
    print(f"Wrote {len(output_rows)} targets: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
