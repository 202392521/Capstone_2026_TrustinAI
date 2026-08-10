#!/usr/bin/env python3
"""Remove conservative Law LLM = Master of Laws false positives.

The project uses LLM as an AI keyword, but law discussions also use LLM to mean
Master of Laws. This script keeps real large-language-model comments and removes
only rows where the target/main text is clearly about legal education, degrees,
bar admission, or Master of Laws programmes, with no AI-technical cues.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUTS = Path("outputs")
DEFAULT_INPUT = OUTPUTS / "four_industry_ai_comments_master_harmonised_total_posts_ranked_v3.csv"
DEFAULT_OUTPUT = OUTPUTS / "four_industry_ai_comments_master_harmonised_total_posts_ranked_v4_law_llm_degree_cleaned.csv"
DEFAULT_REMOVED = OUTPUTS / "law_master_of_laws_llm_false_positives_removed_v4.csv"
DEFAULT_SUMMARY = OUTPUTS / "law_master_of_laws_llm_false_positive_filter_summary_v4.csv"


MASTER_OF_LAWS_PATTERN = re.compile(
    r"\b("
    r"master of laws|masters? degree in law|law masters|"
    r"llm degree|llm program|llm programme|llm programs?|llm programmes?|"
    r"us llm|u\.s\. llm|foreign llm|llm students?|llm graduates?|"
    r"llm and passing|llm and pass|llm to pass|llm/pass|"
    r"llm\s+from|with an llm|have an llm|has an llm|having an llm|"
    r"getting an llm|get an llm|got an llm|do an llm|did an llm|"
    r"need an llm|"
    r"llm in [a-z ]{0,30}(law|nyc|new york|london|paris|tax)|"
    r"llbs?|ny bar|new york bar|california bar|bar directly|"
    r"graduate llb|qualify as a solicitor|senior status|"
    r"cambridge law|glasgow|law ba"
    r")\b",
    re.IGNORECASE,
)

AI_TECH_PATTERN = re.compile(
    r"\b("
    r"chatgpt|chat gpt|generative ai|artificial intelligence|"
    r"claude|gemini|copilot|large language model|language model|"
    r"foundation model|trained|training data|neural|model output|"
    r"generated|generates|prompt|prompts|tokens?|hallucinat|"
    r"copyright|open source|algorithm|code|coding|software|benchmark|"
    r"api|anthropic|openai|gpt[- ]?\d|gpt4|gpt5"
    r")\b",
    re.IGNORECASE,
)

CONTEXT_COLUMNS = [
    "post_title",
    "post_context_excerpt",
    "parent_comment_body",
    "previous_nearby_comment_body",
    "comment_body",
    "next_nearby_comment_body",
]

MAIN_COLUMNS = ["post_title", "post_context_excerpt", "comment_body"]


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def join_columns(row: pd.Series, columns: list[str]) -> str:
    return " ".join(clean_text(row.get(column, "")) for column in columns)


def is_law_row(row: pd.Series) -> bool:
    source = clean_text(row.get("source_dataset", "")).lower()
    keyword = clean_text(row.get("industry_keyword", "")).lower()
    return source == "law" or keyword in {"law", "lawyer", "legal"}


def is_master_of_laws_false_positive(row: pd.Series) -> bool:
    if not is_law_row(row):
        return False

    main_text = join_columns(row, MAIN_COLUMNS)
    comment_text = clean_text(row.get("comment_body", ""))
    all_context = join_columns(row, CONTEXT_COLUMNS)

    if not re.search(r"\bllms?\b", all_context, flags=re.IGNORECASE):
        return False

    main_has_master_cue = bool(MASTER_OF_LAWS_PATTERN.search(main_text))
    comment_has_master_cue = bool(MASTER_OF_LAWS_PATTERN.search(comment_text))
    main_has_ai_cue = bool(AI_TECH_PATTERN.search(main_text))
    comment_has_ai_cue = bool(AI_TECH_PATTERN.search(comment_text))

    if main_has_master_cue and not main_has_ai_cue:
        return True
    if comment_has_master_cue and not comment_has_ai_cue:
        return True
    return False


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--removed-output", default=str(DEFAULT_REMOVED))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    removed_path = Path(args.removed_output).expanduser()
    summary_path = Path(args.summary_output).expanduser()

    df = pd.read_csv(input_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    mask = df.apply(is_master_of_laws_false_positive, axis=1)
    removed = df.loc[mask].copy()
    kept = df.loc[~mask].copy()

    removed["false_positive_reason"] = (
        "law_llm_master_of_laws_degree_or_bar_admission_context_without_ai_technical_cue"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(output_path, index=False)
    removed.to_csv(removed_path, index=False)

    summary = [
        {"metric": "input_file", "value": str(input_path)},
        {"metric": "output_file", "value": str(output_path)},
        {"metric": "removed_file", "value": str(removed_path)},
        {"metric": "input_rows", "value": len(df)},
        {"metric": "output_rows", "value": len(kept)},
        {"metric": "removed_rows", "value": len(removed)},
        {
            "metric": "law_rows_input",
            "value": int(df.apply(is_law_row, axis=1).sum()),
        },
        {
            "metric": "law_rows_output",
            "value": int(kept.apply(is_law_row, axis=1).sum()),
        },
        {
            "metric": "filter_policy",
            "value": (
                "Conservative removal of Law rows where LLM is used as Master of Laws "
                "in degree/bar/admissions context and the target/main text lacks AI-technical cues."
            ),
        },
    ]
    write_summary(summary_path, summary)

    print(f"Input rows: {len(df)}")
    print(f"Removed Law Master-of-Laws LLM false positives: {len(removed)}")
    print(f"Output rows: {len(kept)}")
    print(f"Cleaned master: {output_path}")
    print(f"Removed audit: {removed_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
