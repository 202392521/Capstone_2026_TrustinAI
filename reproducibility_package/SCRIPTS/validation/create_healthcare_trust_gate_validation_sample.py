#!/usr/bin/env python3
"""Create a healthcare validation sample for frozen Prompt V2 trust gates.

The output deliberately separates:
1. a pure random sample for estimating real-world performance/prevalence;
2. an enriched suspected-trust sample for observing error types.

Do not pool the two samples for prevalence estimates.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PACKAGE_DIR / "OUTPUTS"
PROMPT_FILE = PACKAGE_DIR / "PROMPTS" / "FROZEN_PROMPT_V2.txt"

DEFAULT_INPUT = OUTPUTS_DIR / "healthcare_top_commented_ai_threads" / "healthcare_top_commented_ai_comments.csv"
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "healthcare_trust_gate_validation_v2"
FROZEN_PROMPT_FILE = PROMPT_FILE

TRUST_PATTERNS: list[tuple[str, str]] = [
    ("trust", r"\btrust(?:ed|ing|worthy|worthiness)?\b"),
    ("rely", r"\brely\b|\brelies\b|\brelied\b|\breliance\b|\brelying\b"),
    ("depend", r"\bdepend(?:s|ed|ing|able|ability)?\b"),
    ("reliable", r"\breliable\b|\breliability\b|\bunreliable\b"),
    ("accurate", r"\baccurate\b|\baccuracy\b|\binaccurate\b|\binaccuracy\b"),
    ("correct", r"\bcorrect\b|\bincorrect\b|\bwrong\b|\bright answer\b"),
    ("hallucination", r"\bhallucinat\w*\b"),
    ("fabrication", r"\bfabricat\w*\b|\bmade[- ]?up\b|\binvent(?:s|ed|ing)?\b"),
    ("verify", r"\bverify\b|\bverified\b|\bverification\b|\bvalidate\b|\bvalidation\b"),
    ("check", r"\bcheck\b|\bchecked\b|\bchecking\b|\bdouble[- ]?check\b|\bfact[- ]?check\b"),
    ("review", r"\breview\b|\breviewed\b|\breviewer\b|\bhuman review\b"),
    ("oversight", r"\boversight\b|\bsupervision\b|\bsupervise\b|\bhuman in the loop\b"),
    ("liability", r"\bliability\b|\bliable\b|\bmalpractice\b|\bnegligence\b"),
    ("accountability", r"\baccountab\w*\b|\bresponsib\w*\b"),
    ("confidentiality", r"\bconfidential\w*\b|\bprivacy\b|\bprivate data\b|\bdata security\b"),
    ("ethics", r"\bethic\w*\b|\bbias(?:ed)?\b|\bfairness\b"),
    ("safety", r"\bsafe\b|\bunsafe\b|\bsafety\b|\brisk(?:y)?\b"),
]

OUTPUT_COLUMNS = [
    "sample_id",
    "sample_group",
    "sample_use_note",
    "source_file",
    "original_row_index_1_based",
    "industry",
    "subreddit",
    "post_id",
    "comment_id",
    "comment_parent_id",
    "post_month",
    "comment_month",
    "post_title",
    "parent_comment",
    "comment_body",
    "stance_context_text",
    "comment_ai_terms_found",
    "trust_prefilter_score",
    "trust_prefilter_terms",
    "human_has_substantive_trust_content",
    "human_has_explicit_trust_boundary",
    "human_trust_note",
    "gpt5_v2_has_substantive_trust_content",
    "gpt5_v2_has_explicit_trust_boundary",
    "gpt5_v2_human_trust_construct",
    "gpt5_v2_human_trust_boundary",
    "gpt5_v2_annotation_confidence",
]


def clean_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def score_trust_prefilter(row: pd.Series) -> tuple[int, str]:
    text = " ".join(
        clean_text(row.get(col, ""))
        for col in [
            "comment_body",
            "parent_comment_body",
            "post_title",
            "post_context_excerpt",
            "previous_nearby_comment_body",
            "next_nearby_comment_body",
        ]
    )
    found: list[str] = []
    for name, pattern in TRUST_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(name)
    return len(found), ";".join(found)


def make_row(row: pd.Series, sample_group: str, sample_number: int, source_file: Path) -> dict[str, Any]:
    note = (
        "pure_random_sample_for_real_world_gate_performance_and_prevalence_estimation"
        if sample_group == "random_prevalence"
        else "keyword_enriched_suspected_trust_sample_for_error_analysis_only_not_prevalence"
    )
    return {
        "sample_id": f"healthcare_{sample_group}_{sample_number:03d}",
        "sample_group": sample_group,
        "sample_use_note": note,
        "source_file": str(source_file),
        "original_row_index_1_based": row.get("_original_row_index_1_based", ""),
        "industry": "healthcare",
        "subreddit": row.get("subreddit", ""),
        "post_id": row.get("post_id", ""),
        "comment_id": row.get("comment_id", ""),
        "comment_parent_id": row.get("comment_parent_id", ""),
        "post_month": row.get("post_month", ""),
        "comment_month": row.get("comment_month", ""),
        "post_title": row.get("post_title", ""),
        "parent_comment": row.get("parent_comment_body", ""),
        "comment_body": row.get("comment_body", ""),
        "stance_context_text": row.get("stance_context_text", ""),
        "comment_ai_terms_found": row.get("comment_ai_terms_found", ""),
        "trust_prefilter_score": row.get("trust_prefilter_score", 0),
        "trust_prefilter_terms": row.get("trust_prefilter_terms", ""),
        "human_has_substantive_trust_content": "",
        "human_has_explicit_trust_boundary": "",
        "human_trust_note": "",
        "gpt5_v2_has_substantive_trust_content": "",
        "gpt5_v2_has_explicit_trust_boundary": "",
        "gpt5_v2_human_trust_construct": "",
        "gpt5_v2_human_trust_boundary": "",
        "gpt5_v2_annotation_confidence": "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create healthcare trust-gate validation samples.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--random-n", type=int, default=50)
    parser.add_argument("--enriched-n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=445)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df = df[df["comment_body"].map(lambda x: bool(clean_text(x)))].copy()
    df["_original_row_index_1_based"] = df.index + 2
    if "comment_id" in df.columns:
        df = df.drop_duplicates(subset=["comment_id"], keep="first").copy()

    scores = df.apply(score_trust_prefilter, axis=1, result_type="expand")
    df["trust_prefilter_score"] = scores[0].astype(int)
    df["trust_prefilter_terms"] = scores[1]

    random_sample = df.sample(n=min(args.random_n, len(df)), random_state=args.seed).copy()
    random_ids = set(random_sample["comment_id"].astype(str))

    enriched_pool = df[(df["trust_prefilter_score"] > 0) & (~df["comment_id"].astype(str).isin(random_ids))].copy()
    enriched_pool = enriched_pool.sort_values(
        ["trust_prefilter_score", "comment_score"],
        ascending=[False, False],
        kind="mergesort",
    )
    enriched_sample = enriched_pool.head(args.enriched_n).copy()

    random_rows = [make_row(row, "random_prevalence", i + 1, input_path) for i, (_, row) in enumerate(random_sample.iterrows())]
    enriched_rows = [make_row(row, "trust_enriched_error_analysis", i + 1, input_path) for i, (_, row) in enumerate(enriched_sample.iterrows())]

    random_out = output_dir / "healthcare_random_50_for_trust_gate_validation.csv"
    enriched_out = output_dir / "healthcare_trust_enriched_20_for_error_analysis.csv"
    combined_out = output_dir / "healthcare_trust_gate_validation_sample_70_do_not_pool_prevalence.csv"

    pd.DataFrame(random_rows, columns=OUTPUT_COLUMNS).to_csv(random_out, index=False)
    pd.DataFrame(enriched_rows, columns=OUTPUT_COLUMNS).to_csv(enriched_out, index=False)
    pd.DataFrame(random_rows + enriched_rows, columns=OUTPUT_COLUMNS).to_csv(combined_out, index=False)

    summary = pd.DataFrame(
        [
            {"metric": "created_at", "value": datetime.now().isoformat(timespec="seconds")},
            {"metric": "frozen_prompt", "value": "Prompt V2"},
            {"metric": "frozen_prompt_file", "value": str(FROZEN_PROMPT_FILE)},
            {"metric": "source_file", "value": str(input_path)},
            {"metric": "unique_healthcare_comments_available", "value": len(df)},
            {"metric": "random_sample_rows", "value": len(random_rows)},
            {"metric": "enriched_sample_rows", "value": len(enriched_rows)},
            {"metric": "enriched_candidate_pool_rows", "value": len(enriched_pool)},
            {"metric": "random_seed", "value": args.seed},
        ]
    )
    summary.to_csv(output_dir / "healthcare_trust_gate_validation_sample_summary.csv", index=False)

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Freeze Prompt V2 as the best validated trust-gate prompt and create a healthcare audit sample.",
        "source_file": str(input_path),
        "frozen_prompt": "Prompt V2",
        "frozen_prompt_file": str(FROZEN_PROMPT_FILE),
        "random_n": args.random_n,
        "enriched_n": args.enriched_n,
        "seed": args.seed,
        "trust_prefilter_patterns": [{"name": name, "regex": pattern} for name, pattern in TRUST_PATTERNS],
        "prevalence_warning": "Use only the random_prevalence sample to estimate real-world prevalence. The enriched sample is for error analysis only.",
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = """# Healthcare Trust Gate Validation Sample, Prompt V2 Frozen

Prompt V2 is frozen as the best validated prompt for the main trust-gate task.

This directory contains two deliberately separate samples:

- `healthcare_random_50_for_trust_gate_validation.csv`: pure random sample. Use this to estimate real-world Prompt V2 gate performance and, with caution, prevalence.
- `healthcare_trust_enriched_20_for_error_analysis.csv`: keyword-enriched suspected trust sample. Use this to inspect likely false positives, false negatives, and boundary cases.
- `healthcare_trust_gate_validation_sample_70_do_not_pool_prevalence.csv`: convenience file for annotation. Do not pool the two groups for prevalence estimates.

Important reporting rule:

The random sample and enriched sample answer different questions. The random sample estimates typical performance in the healthcare corpus. The enriched sample intentionally over-represents likely trust content and should only be used for error analysis.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Healthcare comments available: {len(df)}")
    print(f"Random sample: {len(random_rows)}")
    print(f"Enriched suspected-trust sample: {len(enriched_rows)}")
    print(f"Enriched candidate pool: {len(enriched_pool)}")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
