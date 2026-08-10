#!/usr/bin/env python3
"""Merge four industry corpora with healthcare ranked by total posts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


OUTPUT_DIR = Path("outputs")

INPUTS: List[Tuple[str, Path]] = [
    ("finance", OUTPUT_DIR / "finance_top_commented_ai_threads/accountant_top_commented_ai_comments_uk_flagged.csv"),
    (
        "law",
        OUTPUT_DIR
        / "law_top_commented_ai_threads_harmonised_tiered"
        / "law_top_commented_ai_comments_uk_flagged_harmonised_tiered.csv",
    ),
    (
        "software_engineering",
        OUTPUT_DIR / "software_engineering_top_commented_ai_threads/software_engineer_top_commented_ai_comments_uk_flagged.csv",
    ),
    (
        "healthcare",
        OUTPUT_DIR
        / "healthcare_top_commented_ai_threads_harmonised_total_posts_ranked"
        / "healthcare_top_commented_ai_comments_uk_flagged_total_posts_ranked.csv",
    ),
]

OUTPUT = OUTPUT_DIR / "four_industry_ai_comments_master_harmonised_total_posts_ranked_v3.csv"
SUMMARY = OUTPUT_DIR / "four_industry_ai_comments_master_harmonised_total_posts_ranked_v3_summary.csv"
UK_STRONG_OUTPUT = OUTPUT_DIR / "four_industry_ai_comments_master_harmonised_total_posts_ranked_v3_uk_strong_only.csv"

PREFERRED_FIELD_ORDER = [
    "source_dataset",
    "source_file",
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
    "uk_strong_relevant",
    "uk_strong_reason",
    "uk_signal_terms",
    "foreign_signal_terms",
    "post_industry_signal",
    "comment_industry_signal",
    "likely_industry_participant",
    "stance_context_text",
    "topic_model_text",
    "candidate_rank_by_num_comments",
    "candidate_query_terms",
    "industry_rank_by_total_posts",
    "download_policy",
    "target_active_ai_months",
    "target_total_posts_period",
]


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


def is_uk_strong(row: Dict[str, str]) -> bool:
    try:
        return int(float(str(row.get("uk_strong_relevant", "") or 0))) == 1
    except ValueError:
        return False


def main() -> int:
    all_rows: List[Dict[str, str]] = []
    seen_fields: List[str] = []
    summaries: List[Dict[str, str]] = []

    for source_dataset, path in INPUTS:
        fields, rows = read_rows(path)
        for field in fields:
            if field not in seen_fields:
                seen_fields.append(field)
        enriched = []
        for row in rows:
            out = dict(row)
            out["source_dataset"] = source_dataset
            out["source_file"] = path.name
            enriched.append(out)
        all_rows.extend(enriched)
        uk_count = sum(1 for row in enriched if is_uk_strong(row))
        summaries.append(
            {
                "source_dataset": source_dataset,
                "source_file": path.name,
                "rows": str(len(enriched)),
                "uk_strong_relevant_1": str(uk_count),
                "uk_strong_relevant_0": str(len(enriched) - uk_count),
                "unique_subreddits": str(len({row.get("subreddit", "") for row in enriched if row.get("subreddit", "")})),
                "unique_posts": str(len({row.get("post_id", "") for row in enriched if row.get("post_id", "")})),
            }
        )

    fields_out = list(PREFERRED_FIELD_ORDER)
    for field in seen_fields:
        if field not in fields_out:
            fields_out.append(field)

    all_rows.sort(
        key=lambda row: (
            row.get("source_dataset", ""),
            row.get("subreddit", "").lower(),
            row.get("post_month", ""),
            row.get("post_id", ""),
            row.get("comment_id", ""),
        )
    )
    write_rows(OUTPUT, all_rows, fields_out)
    write_rows(UK_STRONG_OUTPUT, [row for row in all_rows if is_uk_strong(row)], fields_out)

    total_uk = sum(1 for row in all_rows if is_uk_strong(row))
    summaries.append(
        {
            "source_dataset": "TOTAL",
            "source_file": "",
            "rows": str(len(all_rows)),
            "uk_strong_relevant_1": str(total_uk),
            "uk_strong_relevant_0": str(len(all_rows) - total_uk),
            "unique_subreddits": str(len({row.get("subreddit", "") for row in all_rows if row.get("subreddit", "")})),
            "unique_posts": str(len({row.get("post_id", "") for row in all_rows if row.get("post_id", "")})),
        }
    )
    write_rows(
        SUMMARY,
        summaries,
        [
            "source_dataset",
            "source_file",
            "rows",
            "uk_strong_relevant_1",
            "uk_strong_relevant_0",
            "unique_subreddits",
            "unique_posts",
        ],
    )

    print(f"Wrote {len(all_rows)} rows: {OUTPUT}")
    print(f"Wrote summary: {SUMMARY}")
    print(f"Wrote UK strong subset: {UK_STRONG_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
