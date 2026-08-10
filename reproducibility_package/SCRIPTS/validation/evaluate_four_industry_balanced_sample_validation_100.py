#!/usr/bin/env python3
"""Rebuild the 100-row validation set entirely inside the balanced 2,000.

Composition:
- 75 existing blinded human annotations (25 finance, law, software engineering);
- 2 existing healthcare annotations that overlap the balanced 2,000; and
- 23 newly sampled and blindly human-coded healthcare comments.

This script is offline. It never calls an API and never changes human or GPT
labels. The displaced 23 healthcare holdout rows are retained separately as a
supplementary robustness sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import evaluate_four_industry_full_schema_validation_100 as metrics


ROOT = Path(os.environ.get("REPRO_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUTPUTS = ROOT / "outputs"

OLD_MERGED = (
    OUTPUTS
    / "four_industry_full_schema_validation_100_performance_2026-07-30"
    / "four_industry_full_schema_validation_100_human_gpt_merged.csv"
)
BALANCED_2000 = (
    OUTPUTS
    / "balanced_2000_final_stance_analysis_2026-08-01"
    / "balanced_2000_stance_analysis.csv"
)
NEW_HEALTHCARE_23 = (
    OUTPUTS
    / "healthcare_balanced_validation_23_2026-08-03"
    / "healthcare_balanced_validation_23_annotated.csv"
)
GPT_2000 = (
    OUTPUTS
    / "frozen_prompt_v2_stratified_sample_2000_2026-07-24"
    / "all_comments_v2_annotated.csv"
)
OUTDIR = OUTPUTS / "four_industry_balanced_sample_validation_100_performance_2026-08-04"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def prepare_new_healthcare() -> pd.DataFrame:
    human = read_csv(NEW_HEALTHCARE_23)
    gpt = read_csv(GPT_2000)
    gpt_columns = [
        "comment_id",
        "model",
        "api_status",
        "parse_status",
        "retry_count",
        "latency_seconds",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "gpt_has_substantive_trust_content",
        "gpt_has_explicit_trust_boundary",
        "gpt_human_ai_relevance",
        "gpt_human_attitude",
        "gpt_human_attitude_target",
        "gpt_human_use",
        "gpt_human_capability_assessment",
        "gpt_human_trust_construct",
        "gpt_human_trust_boundary",
        "gpt_human_evidence",
        "gpt_annotation_confidence",
        "gpt_brief_reason",
        "gpt_evidence_quote",
    ]
    pred = gpt[[c for c in gpt_columns if c in gpt.columns]].drop_duplicates("comment_id")
    out = human.merge(pred, on="comment_id", how="left", validate="one_to_one")
    out["sample_source"] = "new_blinded_balanced_healthcare_23"
    return out


def build_validation_set() -> tuple[pd.DataFrame, pd.DataFrame]:
    old = read_csv(OLD_MERGED)
    balanced_ids = set(read_csv(BALANCED_2000)["comment_id"])

    non_health = old[old["industry"].ne("healthcare")].copy()
    old_health = old[old["industry"].eq("healthcare")].copy()
    health_overlap = old_health[old_health["comment_id"].isin(balanced_ids)].copy()
    supplementary = old_health[~old_health["comment_id"].isin(balanced_ids)].copy()

    if len(non_health) != 75:
        raise ValueError(f"Expected 75 existing non-healthcare rows; found {len(non_health)}")
    if len(health_overlap) != 2:
        raise ValueError(f"Expected 2 existing healthcare rows in balanced 2,000; found {len(health_overlap)}")
    if len(supplementary) != 23:
        raise ValueError(f"Expected 23 displaced healthcare holdout rows; found {len(supplementary)}")

    non_health["sample_source"] = "existing_non_healthcare_validation_75"
    health_overlap["sample_source"] = "existing_balanced_healthcare_overlap_2"
    new_health = prepare_new_healthcare()

    combined = pd.concat([non_health, health_overlap, new_health], ignore_index=True, sort=False)
    combined = metrics.harmonise(combined)
    combined = combined.sort_values(["industry", "validation_id", "comment_id"]).reset_index(drop=True)

    if len(combined) != 100 or combined["comment_id"].nunique() != 100:
        raise ValueError("Revised validation must contain exactly 100 unique comments")
    counts = combined["industry"].value_counts().to_dict()
    expected = {"finance": 25, "healthcare": 25, "law": 25, "software_engineering": 25}
    if counts != expected:
        raise ValueError(f"Industry counts differ from expected: {counts}")
    outside = set(combined["comment_id"]) - balanced_ids
    if outside:
        raise ValueError(f"Validation contains {len(outside)} comments outside balanced 2,000")

    return combined, supplementary


def evaluate_fields(df: pd.DataFrame, outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cm_dir = outdir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    per_classes = []
    cms = []
    disagreements = []
    for field in metrics.FIELDS:
        summary, per_class, cm_long, disagreement = metrics.evaluate_field(df, field)
        summaries.append(summary)
        if not per_class.empty:
            per_classes.append(per_class)
        if not cm_long.empty:
            cms.append(cm_long)
            cm_long.pivot(index="human_label", columns="gpt_label", values="count").fillna(0).astype(int).to_csv(
                cm_dir / f"{field}.csv"
            )
        if not disagreement.empty:
            disagreements.append(disagreement)
    return (
        pd.DataFrame(summaries),
        pd.concat(per_classes, ignore_index=True),
        pd.concat(cms, ignore_index=True),
        pd.concat(disagreements, ignore_index=True),
    )


def evaluate_gates(df: pd.DataFrame, outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cm_dir = outdir / "binary_gate_confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cm_rows = []
    baselines = []
    specs = [
        ("substantive_trust_content", "human_has_substantive_trust_content", "gpt_has_substantive_trust_content"),
        ("explicit_trust_boundary", "human_has_explicit_trust_boundary", "gpt_has_explicit_trust_boundary"),
    ]
    for gate_name, human_col, gpt_col in specs:
        row, cm, baseline = metrics.evaluate_gate(df, gate_name, human_col, gpt_col)
        rows.append(row)
        cm.to_csv(cm_dir / f"{gate_name}.csv")
        long = cm.reset_index(names="human_label").melt(
            id_vars="human_label", var_name="gpt_label", value_name="count"
        )
        long.insert(0, "gate", gate_name)
        cm_rows.append(long)
        baselines.append(baseline)
    return pd.DataFrame(rows), pd.concat(cm_rows, ignore_index=True), pd.concat(baselines, ignore_index=True)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    revised, supplementary = build_validation_set()

    revised_path = OUTDIR / "four_industry_balanced_sample_validation_100_human_gpt_merged.csv"
    revised.to_csv(revised_path, index=False)
    supplementary.to_csv(OUTDIR / "supplementary_displaced_healthcare_holdout_23.csv", index=False)

    completeness = metrics.validate_completeness(revised)
    completeness.to_csv(OUTDIR / "validation_100_completeness_check.csv", index=False)
    revised["industry"].value_counts().rename_axis("industry").reset_index(name="n").to_csv(
        OUTDIR / "validation_100_industry_counts.csv", index=False
    )
    revised["sample_source"].value_counts().rename_axis("sample_source").reset_index(name="n").to_csv(
        OUTDIR / "validation_100_sample_sources.csv", index=False
    )

    field_summary, per_class, cm_long, disagreements = evaluate_fields(revised, OUTDIR)
    field_summary.to_csv(OUTDIR / "validation_100_six_field_metrics_summary.csv", index=False)
    per_class.to_csv(OUTDIR / "validation_100_six_field_per_class_metrics.csv", index=False)
    cm_long.to_csv(OUTDIR / "validation_100_six_field_confusion_matrices_long.csv", index=False)
    disagreements.to_csv(OUTDIR / "validation_100_six_field_disagreements.csv", index=False)

    gate_summary, gate_cm_long, gate_baselines = evaluate_gates(revised, OUTDIR)
    gate_summary.to_csv(OUTDIR / "validation_100_binary_gate_metrics_summary.csv", index=False)
    gate_cm_long.to_csv(OUTDIR / "validation_100_binary_gate_confusion_matrices_long.csv", index=False)
    gate_baselines.to_csv(OUTDIR / "validation_100_binary_gate_baseline_comparison.csv", index=False)

    by_industry = metrics.evaluate_gates_by_industry(revised)
    by_industry.to_csv(OUTDIR / "validation_100_binary_gate_metrics_by_industry.csv", index=False)
    bootstrap = metrics.stratified_bootstrap_gate_ci(
        revised, "human_has_explicit_trust_boundary", "gpt_has_explicit_trust_boundary"
    )
    bootstrap.to_csv(OUTDIR / "validation_100_explicit_boundary_bootstrap_ci.csv", index=False)

    accuracy_rows = []
    for industry, sub in revised.groupby("industry", sort=True):
        for field in metrics.FIELDS:
            human = sub[field].map(metrics.clean_label)
            pred = sub[f"gpt_{field}"].map(metrics.clean_label)
            valid = human.ne("") & pred.ne("")
            accuracy_rows.append(
                {
                    "industry": industry,
                    "field": field,
                    "n_valid": int(valid.sum()),
                    "matches_n": int((human[valid] == pred[valid]).sum()),
                    "accuracy": round(float((human[valid] == pred[valid]).mean()), 4) if valid.any() else "",
                }
            )
    pd.DataFrame(accuracy_rows).to_csv(OUTDIR / "validation_100_six_field_accuracy_by_industry.csv", index=False)

    old_fields = read_csv(
        OLD_MERGED.parent / "four_industry_full_schema_validation_100_metrics_summary.csv"
    )[["field", "accuracy", "macro_f1", "weighted_f1"]].rename(
        columns={"accuracy": "old_accuracy", "macro_f1": "old_macro_f1", "weighted_f1": "old_weighted_f1"}
    )
    comparison = old_fields.merge(
        field_summary[["field", "accuracy", "macro_f1", "weighted_f1"]], on="field", how="outer"
    ).rename(columns={"accuracy": "revised_accuracy", "macro_f1": "revised_macro_f1", "weighted_f1": "revised_weighted_f1"})
    comparison.to_csv(OUTDIR / "old_vs_revised_six_field_metrics.csv", index=False)

    with pd.ExcelWriter(OUTDIR / "four_industry_balanced_sample_validation_100_workbook.xlsx") as writer:
        field_summary.to_excel(writer, sheet_name="field_metrics", index=False)
        per_class.to_excel(writer, sheet_name="per_class_metrics", index=False)
        gate_summary.to_excel(writer, sheet_name="gate_metrics", index=False)
        gate_baselines.to_excel(writer, sheet_name="gate_baselines", index=False)
        by_industry.to_excel(writer, sheet_name="gate_by_industry", index=False)
        bootstrap.to_excel(writer, sheet_name="boundary_bootstrap_ci", index=False)
        comparison.to_excel(writer, sheet_name="old_vs_revised", index=False)

    manifest = {
        "run_type": "offline_recalculation_no_api_calls",
        "old_validation_file": str(OLD_MERGED),
        "balanced_2000_file": str(BALANCED_2000),
        "new_blinded_healthcare_23_file": str(NEW_HEALTHCARE_23),
        "gpt_prediction_source": str(GPT_2000),
        "composition": {
            "existing_non_healthcare": 75,
            "existing_balanced_healthcare_overlap": 2,
            "new_blinded_balanced_healthcare": 23,
        },
        "n_rows": len(revised),
        "unique_comment_ids": int(revised["comment_id"].nunique()),
        "industry_counts": revised["industry"].value_counts().sort_index().to_dict(),
        "output_dir": str(OUTDIR),
    }
    (OUTDIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Validation rows: {len(revised)}; unique IDs: {revised['comment_id'].nunique()}")
    print(revised["industry"].value_counts().sort_index().to_string())
    print("\nSix-field metrics:")
    print(field_summary[["field", "n_valid", "accuracy", "macro_f1", "weighted_f1"]].to_string(index=False))
    print("\nBinary gate metrics:")
    print(gate_summary[["gate", "true_positive", "true_negative", "false_positive", "false_negative", "accuracy", "specificity", "balanced_accuracy", "f1", "matthews_correlation_coefficient"]].to_string(index=False))
    print(f"\nOutput directory: {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
