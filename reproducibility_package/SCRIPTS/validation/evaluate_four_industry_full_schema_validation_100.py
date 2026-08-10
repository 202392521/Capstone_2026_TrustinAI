#!/usr/bin/env python3
"""Create and evaluate a four-industry 100-row validation set.

The validation set combines:
- the existing 75-row non-healthcare full-schema validation set
  (25 finance, 25 law, 25 software engineering), and
- a fixed-seed 25-row subsample from the locked 50-row healthcare holdout.

This is an offline audit. No API calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


ROOT = Path(os.environ.get("REPRO_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUTPUTS = ROOT / "outputs"

NON_HEALTH_HUMAN = (
    OUTPUTS
    / "capability_trust_construct_transferability_audit_75_2026-07-29"
    / "non_healthcare_full_schema_validation_75_annotated.csv"
)
NON_HEALTH_GPT = OUTPUTS / "frozen_prompt_v2_stratified_sample_2000_2026-07-24" / "all_comments_v2_annotated.csv"
HEALTH_MERGED = (
    OUTPUTS
    / "healthcare_trust_gate_validation_v2"
    / "prompt_v2_holdout50_eval_2026-07-23"
    / "health_holdout_50_prompt_v2_vs_human_merged.csv"
)
OUTDIR = OUTPUTS / "four_industry_full_schema_validation_100_performance_2026-07-30"

RANDOM_SEED = 20260730
HEALTHCARE_N = 25
BOOTSTRAP_ITERATIONS = 5000

FIELDS = [
    "human_attitude",
    "human_attitude_target",
    "human_use",
    "human_capability_assessment",
    "human_trust_construct",
    "human_trust_boundary",
]

CONTEXT_COLUMNS = [
    "validation_id",
    "sample_source",
    "industry",
    "subreddit",
    "post_id",
    "comment_id",
    "post_month",
    "comment_month",
    "UK_status",
    "post_title",
    "parent_context",
    "comment_text",
    "human_note",
]

TRUST_CONSTRUCT_POSITIVE = {
    "perceived_trustworthiness",
    "attitudinal_trust_willingness_to_rely",
    "trusting_behaviour",
    "multiple_trust_constructs",
}


def clean_label(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def sorted_labels(*series: Iterable[str]) -> list[str]:
    values: set[str] = set()
    for s in series:
        values.update(clean_label(v) for v in s)
    return sorted(v for v in values if v)


def make_gate_from_construct(value: object) -> str:
    label = clean_label(value)
    if not label:
        return ""
    return "yes" if label in TRUST_CONSTRUCT_POSITIVE else "no"


def make_gate_from_boundary(value: object) -> str:
    label = clean_label(value)
    if not label:
        return ""
    return "no" if label == "no_trust_boundary_discussed" else "yes"


def bool_gate(value: object) -> str:
    label = clean_label(value).lower()
    if label in {"yes", "true", "1"}:
        return "yes"
    if label in {"no", "false", "0"}:
        return "no"
    return ""


def load_non_healthcare() -> pd.DataFrame:
    human = pd.read_csv(NON_HEALTH_HUMAN, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    gpt = pd.read_csv(NON_HEALTH_GPT, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    gpt_cols = [
        "comment_id",
        "gpt_human_attitude",
        "gpt_human_attitude_target",
        "gpt_human_use",
        "gpt_human_capability_assessment",
        "gpt_human_trust_construct",
        "gpt_human_trust_boundary",
        "gpt_human_evidence",
        "gpt_has_substantive_trust_content",
        "gpt_has_explicit_trust_boundary",
        "gpt_annotation_confidence",
        "gpt_brief_reason",
        "gpt_evidence_quote",
    ]
    gpt_keep = gpt[[c for c in gpt_cols if c in gpt.columns]].drop_duplicates("comment_id")
    merged = human.merge(gpt_keep, on="comment_id", how="left", validate="one_to_one")
    merged["sample_source"] = "non_healthcare_full_schema_validation_75"
    merged["parent_context"] = merged.get("parent_context", "")
    merged["comment_text"] = merged.get("comment_text", "")
    return merged


def load_healthcare_sample() -> pd.DataFrame:
    df = pd.read_csv(HEALTH_MERGED, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df = df.sample(n=HEALTHCARE_N, random_state=RANDOM_SEED).sort_values("comment_id").reset_index(drop=True)
    df["validation_id"] = [f"healthcare_holdout25_{i:03d}" for i in range(1, len(df) + 1)]
    df["sample_source"] = "healthcare_locked_holdout50_fixed_seed_25"
    df["parent_context"] = df.get("parent_comment", "")
    df["comment_text"] = df.get("comment_body", "")
    df["human_note"] = df.get("human_trust_note", "")
    df["UK_status"] = df.get("UK_status", "")
    return df


def harmonise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CONTEXT_COLUMNS + FIELDS:
        if col not in out.columns:
            out[col] = ""
    for field in FIELDS:
        pred_col = f"gpt_{field}"
        if pred_col not in out.columns:
            out[pred_col] = ""
    if "human_has_substantive_trust_content" not in out.columns:
        out["human_has_substantive_trust_content"] = out["human_trust_construct"].map(make_gate_from_construct)
    else:
        out["human_has_substantive_trust_content"] = out["human_has_substantive_trust_content"].map(bool_gate)
        missing = out["human_has_substantive_trust_content"].eq("")
        out.loc[missing, "human_has_substantive_trust_content"] = out.loc[missing, "human_trust_construct"].map(
            make_gate_from_construct
        )
    if "human_has_explicit_trust_boundary" not in out.columns:
        out["human_has_explicit_trust_boundary"] = out["human_trust_boundary"].map(make_gate_from_boundary)
    else:
        out["human_has_explicit_trust_boundary"] = out["human_has_explicit_trust_boundary"].map(bool_gate)
        missing = out["human_has_explicit_trust_boundary"].eq("")
        out.loc[missing, "human_has_explicit_trust_boundary"] = out.loc[missing, "human_trust_boundary"].map(
            make_gate_from_boundary
        )
    if "gpt_has_substantive_trust_content" not in out.columns:
        out["gpt_has_substantive_trust_content"] = out["gpt_human_trust_construct"].map(make_gate_from_construct)
    else:
        out["gpt_has_substantive_trust_content"] = out["gpt_has_substantive_trust_content"].map(bool_gate)
        missing = out["gpt_has_substantive_trust_content"].eq("")
        out.loc[missing, "gpt_has_substantive_trust_content"] = out.loc[missing, "gpt_human_trust_construct"].map(
            make_gate_from_construct
        )
    if "gpt_has_explicit_trust_boundary" not in out.columns:
        out["gpt_has_explicit_trust_boundary"] = out["gpt_human_trust_boundary"].map(make_gate_from_boundary)
    else:
        out["gpt_has_explicit_trust_boundary"] = out["gpt_has_explicit_trust_boundary"].map(bool_gate)
        missing = out["gpt_has_explicit_trust_boundary"].eq("")
        out.loc[missing, "gpt_has_explicit_trust_boundary"] = out.loc[missing, "gpt_human_trust_boundary"].map(
            make_gate_from_boundary
        )
    return out


def validate_completeness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in FIELDS:
        missing = df[field].map(clean_label).eq("")
        rows.append(
            {
                "field": field,
                "status": "complete" if not missing.any() else "has_missing_values",
                "missing_n": int(missing.sum()),
                "non_missing_n": int((~missing).sum()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_field(df: pd.DataFrame, field: str) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_col = f"gpt_{field}"
    cols = [field, pred_col] + [c for c in CONTEXT_COLUMNS if c in df.columns] + [
        c for c in ["gpt_annotation_confidence", "gpt_brief_reason", "gpt_evidence_quote"] if c in df.columns
    ]
    subset = df[cols].copy()
    subset[field] = subset[field].map(clean_label)
    subset[pred_col] = subset[pred_col].map(clean_label)
    comparable = subset[(subset[field] != "") & (subset[pred_col] != "")].copy()
    y_true = comparable[field].tolist()
    y_pred = comparable[pred_col].tolist()
    labels = sorted_labels(y_true, y_pred)
    if not labels:
        empty = pd.DataFrame()
        return {
            "field": field,
            "status": "no_valid_pairs",
            "n_total_rows": len(df),
            "n_valid": 0,
            "matches_n": 0,
            "mismatches_n": 0,
            "accuracy": "",
            "macro_f1": "",
            "weighted_f1": "",
            "n_human_classes": 0,
            "n_gpt_classes": 0,
            "min_human_class_n": "",
            "classes_with_human_n_lt_5": "",
        }, empty, empty, empty

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame(
        {
            "field": field,
            "class": labels,
            "human_sample_n": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)
    cm.index.name = "human_label"
    cm.columns.name = "gpt_label"
    cm_long = cm.reset_index().melt(id_vars="human_label", var_name="gpt_label", value_name="count")
    cm_long.insert(0, "field", field)
    disagreements = comparable[comparable[field] != comparable[pred_col]].copy()
    disagreements.insert(0, "field", field)
    disagreements = disagreements.rename(columns={field: "human_label", pred_col: "gpt_label"})
    human_counts = comparable[field].value_counts()
    low_n = human_counts[human_counts < 5].index.tolist()
    matches = int((comparable[field] == comparable[pred_col]).sum())
    return {
        "field": field,
        "status": "evaluated",
        "n_total_rows": len(df),
        "n_valid": len(comparable),
        "matches_n": matches,
        "mismatches_n": int(len(comparable) - matches),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0), 4),
        "n_human_classes": comparable[field].nunique(),
        "n_gpt_classes": comparable[pred_col].nunique(),
        "min_human_class_n": int(human_counts.min()) if len(human_counts) else "",
        "classes_with_human_n_lt_5": ";".join(low_n),
    }, per_class, cm_long, disagreements


def binary_metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    n = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn) - (fp * fn)) / (denom**0.5) if denom else 0.0
    return {
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2 if n else 0.0,
        "negative_predictive_value": npv,
        "f1": f1,
        "matthews_correlation_coefficient": mcc,
    }


def evaluate_gate(
    df: pd.DataFrame,
    gate_name: str,
    human_col: str,
    gpt_col: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    comparable = df[[human_col, gpt_col]].copy()
    comparable[human_col] = comparable[human_col].map(clean_label)
    comparable[gpt_col] = comparable[gpt_col].map(clean_label)
    comparable = comparable[(comparable[human_col] != "") & (comparable[gpt_col] != "")]
    y_true = comparable[human_col]
    y_pred = comparable[gpt_col]
    labels = ["yes", "no"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])
    actual = binary_metrics_from_counts(tp, tn, fp, fn)
    positive_n = int((y_true == "yes").sum())
    negative_n = int((y_true == "no").sum())
    always_positive = binary_metrics_from_counts(positive_n, 0, negative_n, 0)
    always_negative = binary_metrics_from_counts(0, negative_n, 0, positive_n)
    positive_prevalence = float((y_true == "yes").mean()) if len(y_true) else 0.0
    predicted_positive_rate = float((y_pred == "yes").mean()) if len(y_pred) else 0.0
    row = {
        "gate": gate_name,
        "n": int(len(comparable)),
        "gold_positive_count": positive_n,
        "gold_negative_count": negative_n,
        "predicted_positive_count": int((y_pred == "yes").sum()),
        "predicted_negative_count": int((y_pred == "no").sum()),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "positive_class_prevalence": round(positive_prevalence, 4),
        "predicted_positive_rate": round(predicted_positive_rate, 4),
        **{k: round(v, 4) for k, v in actual.items()},
        "always_positive_accuracy": round(always_positive["accuracy"], 4),
        "always_positive_f1": round(always_positive["f1"], 4),
        "always_positive_specificity": round(always_positive["specificity"], 4),
        "always_positive_balanced_accuracy": round(always_positive["balanced_accuracy"], 4),
        "always_positive_mcc": round(always_positive["matthews_correlation_coefficient"], 4),
        "always_negative_accuracy": round(always_negative["accuracy"], 4),
        "always_negative_f1": round(always_negative["f1"], 4),
        "always_negative_specificity": round(always_negative["specificity"], 4),
        "always_negative_balanced_accuracy": round(always_negative["balanced_accuracy"], 4),
        "always_negative_mcc": round(always_negative["matthews_correlation_coefficient"], 4),
    }
    cm_df = pd.DataFrame(cm, index=["human_yes", "human_no"], columns=["gpt_yes", "gpt_no"])
    baseline_rows = []
    for baseline_name, counts, metrics, pred_rate in [
        (
            "actual_model",
            {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn},
            actual,
            predicted_positive_rate,
        ),
        (
            "always_positive",
            {"true_positive": positive_n, "true_negative": 0, "false_positive": negative_n, "false_negative": 0},
            always_positive,
            1.0,
        ),
        (
            "always_negative",
            {"true_positive": 0, "true_negative": negative_n, "false_positive": 0, "false_negative": positive_n},
            always_negative,
            0.0,
        ),
    ]:
        baseline_rows.append(
            {
                "gate": gate_name,
                "model_or_baseline": baseline_name,
                "n": int(len(comparable)),
                "positive_class_prevalence": round(positive_prevalence, 4),
                "predicted_positive_rate": round(pred_rate, 4),
                **counts,
                **{k: round(v, 4) for k, v in metrics.items()},
            }
        )
    return row, cm_df, pd.DataFrame(baseline_rows)


def evaluate_gates_by_industry(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    gate_specs = [
        ("substantive_trust_content", "human_has_substantive_trust_content", "gpt_has_substantive_trust_content"),
        ("explicit_trust_boundary", "human_has_explicit_trust_boundary", "gpt_has_explicit_trust_boundary"),
    ]
    for industry, sub in df.groupby("industry", sort=True):
        for gate_name, human_col, gpt_col in gate_specs:
            row, _, _ = evaluate_gate(sub, gate_name, human_col, gpt_col)
            row = {"industry": industry, **row}
            rows.append(row)
    return pd.DataFrame(rows)


def stratified_bootstrap_gate_ci(
    df: pd.DataFrame,
    human_col: str,
    gpt_col: str,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    comparable = df[["industry", human_col, gpt_col]].copy()
    comparable[human_col] = comparable[human_col].map(clean_label)
    comparable[gpt_col] = comparable[gpt_col].map(clean_label)
    comparable = comparable[(comparable[human_col] != "") & (comparable[gpt_col] != "")].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    strata = {industry: sub.index.to_numpy() for industry, sub in comparable.groupby("industry", sort=True)}
    metrics = [
        "precision",
        "recall",
        "f1",
        "specificity",
        "balanced_accuracy",
        "matthews_correlation_coefficient",
    ]
    samples: dict[str, list[float]] = {metric: [] for metric in metrics}

    for _ in range(iterations):
        sampled_indices = []
        for indices in strata.values():
            sampled_indices.extend(rng.choice(indices, size=len(indices), replace=True).tolist())
        sample = comparable.loc[sampled_indices]
        y_true = sample[human_col]
        y_pred = sample[gpt_col]
        cm = confusion_matrix(y_true, y_pred, labels=["yes", "no"])
        tp = int(cm[0, 0])
        fn = int(cm[0, 1])
        fp = int(cm[1, 0])
        tn = int(cm[1, 1])
        metric_values = binary_metrics_from_counts(tp, tn, fp, fn)
        for metric in metrics:
            samples[metric].append(metric_values[metric])

    rows = []
    for metric, values in samples.items():
        arr = np.array(values, dtype=float)
        rows.append(
            {
                "gate": "explicit_trust_boundary",
                "metric": metric,
                "bootstrap_iterations": iterations,
                "resampling": "stratified_by_industry",
                "estimate_mean": round(float(np.mean(arr)), 4),
                "ci_95_lower": round(float(np.quantile(arr, 0.025)), 4),
                "ci_95_upper": round(float(np.quantile(arr, 0.975)), 4),
            }
        )
    return pd.DataFrame(rows)


def summarise_boundary_error_pattern(industry_gate_summary: pd.DataFrame) -> pd.DataFrame:
    boundary = industry_gate_summary[industry_gate_summary["gate"] == "explicit_trust_boundary"].copy()
    rows = []
    for _, row in boundary.iterrows():
        fp = int(row["false_positive"])
        fn = int(row["false_negative"])
        total_errors = fp + fn
        rows.append(
            {
                "industry": row["industry"],
                "n": row["n"],
                "gold_positive_count": row["gold_positive_count"],
                "gold_negative_count": row["gold_negative_count"],
                "false_positive": fp,
                "false_negative": fn,
                "total_errors": total_errors,
                "false_negative_share_of_errors": round(fn / total_errors, 4) if total_errors else 0.0,
                "specificity": row["specificity"],
                "balanced_accuracy": row["balanced_accuracy"],
                "matthews_correlation_coefficient": row["matthews_correlation_coefficient"],
                "interpretation": (
                    "No boundary-gate errors in this 25-row stratum."
                    if total_errors == 0
                    else "Errors are mainly false negatives."
                    if fn > fp
                    else "Errors are mainly false positives."
                    if fp > fn
                    else "False positives and false negatives are balanced."
                ),
            }
        )
    return pd.DataFrame(rows)


def compare_non_healthcare_and_healthcare_boundary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, sub in [
        ("finance_law_software_previous_75", df[df["industry"].ne("healthcare")]),
        ("healthcare_added_25", df[df["industry"].eq("healthcare")]),
        ("four_industry_total_100", df),
    ]:
        row, _, _ = evaluate_gate(
            sub,
            "explicit_trust_boundary",
            "human_has_explicit_trust_boundary",
            "gpt_has_explicit_trust_boundary",
        )
        rows.append({"comparison_group": group_name, **row})
    return pd.DataFrame(rows)


def write_readme(
    out_dir: Path,
    field_summary: pd.DataFrame,
    gate_summary: pd.DataFrame,
    industry_gate_summary: pd.DataFrame,
    boundary_bootstrap_ci: pd.DataFrame,
    boundary_error_pattern: pd.DataFrame,
) -> None:
    substantive = gate_summary[gate_summary["gate"] == "substantive_trust_content"].iloc[0].to_dict()
    boundary = gate_summary[gate_summary["gate"] == "explicit_trust_boundary"].iloc[0].to_dict()
    lines = [
        "# Four-industry full-schema validation 100 audit",
        "",
        "This offline audit combines 25 human-coded comments from each of the four industries: finance, healthcare, law and software engineering. No API calls were made.",
        "",
        "## Headline conclusion",
        "",
        "This balanced-by-industry validation set is more methodologically consistent than the earlier 75-row non-healthcare-only audit. The substantive-trust gate should still be interpreted cautiously: it remains heavily positive-biased and has weak negative-case discrimination. The explicit-trust-boundary gate remains the stronger binary validation result because it identifies both positive and negative cases.",
        "",
        "Fine-grained trust construct and trust-boundary subtype labels remain insufficiently reliable for standalone quantitative claims; use them only as exploratory descriptors or for qualitative error analysis unless further validated.",
        "",
        "## Binary gate summary",
        "",
        f"- Substantive trust content: TP={substantive['true_positive']}, TN={substantive['true_negative']}, FP={substantive['false_positive']}, FN={substantive['false_negative']}, accuracy={substantive['accuracy']}, specificity={substantive['specificity']}, balanced accuracy={substantive['balanced_accuracy']}, MCC={substantive['matthews_correlation_coefficient']}, F1={substantive['f1']}.",
        f"- Explicit trust boundary: TP={boundary['true_positive']}, TN={boundary['true_negative']}, FP={boundary['false_positive']}, FN={boundary['false_negative']}, accuracy={boundary['accuracy']}, specificity={boundary['specificity']}, balanced accuracy={boundary['balanced_accuracy']}, MCC={boundary['matthews_correlation_coefficient']}, F1={boundary['f1']}.",
        "",
        "## Industry-stratified binary gate evaluation",
        "",
        "The two binary gates are also evaluated separately within each industry. Each stratum contains 25 comments, so these rows should be read as error-pattern diagnostics rather than precise industry-level performance estimates.",
        "",
    ]
    boundary_by_industry = industry_gate_summary[
        industry_gate_summary["gate"] == "explicit_trust_boundary"
    ].sort_values("industry")
    for _, row in boundary_by_industry.iterrows():
        lines.append(
            f"- {row['industry']}: explicit-boundary TP={row['true_positive']}, TN={row['true_negative']}, FP={row['false_positive']}, FN={row['false_negative']}, specificity={row['specificity']}, balanced accuracy={row['balanced_accuracy']}, MCC={row['matthews_correlation_coefficient']}."
        )
    lines.extend(
        [
            "",
            "Boundary error-pattern interpretation:",
            "",
        ]
    )
    max_error = boundary_error_pattern.sort_values(["total_errors", "false_negative"], ascending=False).iloc[0]
    for _, row in boundary_error_pattern.sort_values("industry").iterrows():
        lines.append(
            f"- {row['industry']}: total errors={row['total_errors']} (FP={row['false_positive']}, FN={row['false_negative']}); {row['interpretation']}"
        )
    lines.extend(
        [
            "",
            f"The largest number of explicit-boundary errors in the 100-row validation set occurs in {max_error['industry']} (n={max_error['total_errors']}). This identifies where the decline from the earlier non-healthcare-only validation is most concentrated.",
            "",
            "## Stratified bootstrap confidence intervals",
            "",
            "The explicit-trust-boundary gate has 95% bootstrap confidence intervals computed by resampling within industry strata.",
            "",
        ]
    )
    for _, row in boundary_bootstrap_ci.iterrows():
        lines.append(
            f"- {row['metric']}: mean={row['estimate_mean']}, 95% CI [{row['ci_95_lower']}, {row['ci_95_upper']}]."
        )
    lines.extend(
        [
            "",
            "## Field-level summary",
            "",
        ]
    )
    for _, row in field_summary.iterrows():
        lines.append(
            f"- {row['field']}: n_valid={row['n_valid']}, accuracy={row['accuracy']}, macro-F1={row['macro_f1']}, weighted-F1={row['weighted_f1']}."
        )
    lines.extend(
        [
            "",
            "## Important notes",
            "",
            "- The healthcare rows are a fixed-seed subsample of 25 from the locked 50-row healthcare holdout.",
            "- `human_evidence` is excluded because it was used only as a reference note during earlier coding and is not part of the final validation schema.",
            "- This validation set checks transferability and error patterns; it should not be used as a prevalence estimate.",
            "- Bootstrap confidence intervals are based on 5,000 resamples stratified by industry.",
        ]
    )
    (out_dir / "README_four_industry_full_schema_validation_100.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cm_dir = OUTDIR / "confusion_matrices"
    gate_cm_dir = OUTDIR / "binary_gate_confusion_matrices"
    cm_dir.mkdir(exist_ok=True)
    gate_cm_dir.mkdir(exist_ok=True)

    non_health = load_non_healthcare()
    health = load_healthcare_sample()
    merged = pd.concat([non_health, health], ignore_index=True, sort=False)
    merged = harmonise(merged)
    merged = merged.sort_values(["industry", "validation_id", "comment_id"]).reset_index(drop=True)
    merged.to_csv(OUTDIR / "four_industry_full_schema_validation_100_human_gpt_merged.csv", index=False)

    validate_completeness(merged).to_csv(OUTDIR / "four_industry_full_schema_validation_100_completeness_check.csv", index=False)
    merged["industry"].value_counts().rename_axis("industry").reset_index(name="n").to_csv(
        OUTDIR / "four_industry_full_schema_validation_100_industry_counts.csv", index=False
    )

    summaries: list[dict[str, object]] = []
    per_classes: list[pd.DataFrame] = []
    cms: list[pd.DataFrame] = []
    disagreements: list[pd.DataFrame] = []
    for field in FIELDS:
        summary, per_class, cm_long, disagreement = evaluate_field(merged, field)
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

    field_summary = pd.DataFrame(summaries)
    field_summary.to_csv(OUTDIR / "four_industry_full_schema_validation_100_metrics_summary.csv", index=False)
    pd.concat(per_classes, ignore_index=True).to_csv(
        OUTDIR / "four_industry_full_schema_validation_100_per_class_metrics.csv", index=False
    )
    pd.concat(cms, ignore_index=True).to_csv(
        OUTDIR / "four_industry_full_schema_validation_100_confusion_matrices_long.csv", index=False
    )
    pd.concat(disagreements, ignore_index=True).to_csv(
        OUTDIR / "four_industry_full_schema_validation_100_disagreements.csv", index=False
    )

    gate_rows: list[dict[str, object]] = []
    gate_cm_longs: list[pd.DataFrame] = []
    gate_baseline_rows: list[pd.DataFrame] = []
    for gate_name, human_col, gpt_col in [
        ("substantive_trust_content", "human_has_substantive_trust_content", "gpt_has_substantive_trust_content"),
        ("explicit_trust_boundary", "human_has_explicit_trust_boundary", "gpt_has_explicit_trust_boundary"),
    ]:
        gate_row, gate_cm, gate_baselines = evaluate_gate(merged, gate_name, human_col, gpt_col)
        gate_rows.append(gate_row)
        gate_cm.to_csv(gate_cm_dir / f"{gate_name}.csv")
        gate_cm_long = gate_cm.reset_index(names="human_label").melt(
            id_vars="human_label", var_name="gpt_label", value_name="count"
        )
        gate_cm_long.insert(0, "gate", gate_name)
        gate_cm_longs.append(gate_cm_long)
        gate_baseline_rows.append(gate_baselines)

    gate_summary = pd.DataFrame(gate_rows)
    gate_cm_long_df = pd.concat(gate_cm_longs, ignore_index=True)
    gate_baseline_df = pd.concat(gate_baseline_rows, ignore_index=True)
    gate_summary.to_csv(OUTDIR / "four_industry_full_schema_validation_100_gate_metrics_summary.csv", index=False)
    gate_cm_long_df.to_csv(OUTDIR / "four_industry_full_schema_validation_100_binary_gate_confusion_matrices_long.csv", index=False)
    gate_baseline_df.to_csv(OUTDIR / "four_industry_full_schema_validation_100_binary_gate_baseline_comparison.csv", index=False)
    industry_gate_summary = evaluate_gates_by_industry(merged)
    industry_gate_summary.to_csv(
        OUTDIR / "four_industry_full_schema_validation_100_gate_metrics_by_industry.csv", index=False
    )
    boundary_bootstrap_ci = stratified_bootstrap_gate_ci(
        merged,
        "human_has_explicit_trust_boundary",
        "gpt_has_explicit_trust_boundary",
    )
    boundary_bootstrap_ci.to_csv(
        OUTDIR / "four_industry_full_schema_validation_100_explicit_boundary_bootstrap_ci.csv", index=False
    )
    boundary_error_pattern = summarise_boundary_error_pattern(industry_gate_summary)
    boundary_error_pattern.to_csv(
        OUTDIR / "four_industry_full_schema_validation_100_explicit_boundary_error_pattern_by_industry.csv",
        index=False,
    )
    boundary_previous_vs_healthcare = compare_non_healthcare_and_healthcare_boundary(merged)
    boundary_previous_vs_healthcare.to_csv(
        OUTDIR / "four_industry_full_schema_validation_100_explicit_boundary_previous75_vs_healthcare25.csv",
        index=False,
    )

    by_industry = []
    for industry, sub in merged.groupby("industry"):
        for field in [f for f in FIELDS if f in sub.columns and f"gpt_{f}" in sub.columns]:
            h = sub[field].map(clean_label)
            p = sub[f"gpt_{field}"].map(clean_label)
            mask = h.ne("") & p.ne("")
            by_industry.append(
                {
                    "industry": industry,
                    "field": field,
                    "n_valid": int(mask.sum()),
                    "matches_n": int((h[mask] == p[mask]).sum()),
                    "accuracy": round(float((h[mask] == p[mask]).mean()), 4) if mask.any() else "",
                }
            )
    pd.DataFrame(by_industry).to_csv(OUTDIR / "four_industry_full_schema_validation_100_accuracy_by_industry.csv", index=False)

    with pd.ExcelWriter(OUTDIR / "four_industry_full_schema_validation_100_workbook.xlsx") as writer:
        field_summary.to_excel(writer, sheet_name="field_metrics", index=False)
        pd.concat(per_classes, ignore_index=True).to_excel(writer, sheet_name="per_class_metrics", index=False)
        gate_summary.to_excel(writer, sheet_name="gate_metrics", index=False)
        gate_baseline_df.to_excel(writer, sheet_name="gate_baselines", index=False)
        gate_cm_long_df.to_excel(writer, sheet_name="gate_confusion_long", index=False)
        industry_gate_summary.to_excel(writer, sheet_name="gate_by_industry", index=False)
        boundary_bootstrap_ci.to_excel(writer, sheet_name="boundary_bootstrap_ci", index=False)
        boundary_error_pattern.to_excel(writer, sheet_name="boundary_error_pattern", index=False)
        boundary_previous_vs_healthcare.to_excel(writer, sheet_name="prev75_vs_healthcare25", index=False)

    write_readme(
        OUTDIR,
        field_summary,
        gate_summary,
        industry_gate_summary,
        boundary_bootstrap_ci,
        boundary_error_pattern,
    )
    (OUTDIR / "run_manifest.json").write_text(
        json.dumps(
            {
                "non_healthcare_human_file": str(NON_HEALTH_HUMAN),
                "non_healthcare_gpt_file": str(NON_HEALTH_GPT),
                "healthcare_merged_file": str(HEALTH_MERGED),
                "healthcare_n": HEALTHCARE_N,
                "healthcare_random_seed": RANDOM_SEED,
                "output_dir": str(OUTDIR),
                "n_rows": len(merged),
                "industry_counts": merged["industry"].value_counts().to_dict(),
                "note": "Offline four-industry validation audit. No API calls were made.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Validation rows: {len(merged)}")
    print(merged["industry"].value_counts().sort_index().to_string())
    print("\nGate metrics:")
    print(gate_summary.to_string(index=False))
    print("\nOutput dir:", OUTDIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
