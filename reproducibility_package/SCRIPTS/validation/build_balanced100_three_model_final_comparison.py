#!/usr/bin/env python3
"""Build the final fair three-model comparison on the balanced 100 set.

This script is evaluation-only. It never imports or calls the OpenAI SDK and
uses only predictions already saved in local checkpoint JSONL files.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from gpt56_annotation_common import (
    LABEL_FIELDS,
    LABEL_OPTIONS,
    derive_boundary_gate,
    derive_substantive_gate,
    iter_jsonl,
)


HERE = Path(__file__).resolve().parent
FINAL_100 = (
    HERE
    / "four_industry_balanced_sample_validation_100_performance_2026-08-04"
    / "four_industry_balanced_sample_validation_100_human_gpt_merged.csv"
)

OLD_CHECKPOINTS = {
    "gpt56": HERE
    / "gpt56_cost_controlled_annotation_pipeline_2026-07-31"
    / "validation_stage_1_single_none"
    / "checkpoints.jsonl",
    "gpt51": HERE
    / "gpt51_cost_controlled_annotation_pipeline_2026-07-31"
    / "validation_stage_1_single_none"
    / "checkpoints.jsonl",
}

NEW_HEALTHCARE23_CHECKPOINTS = {
    "gpt56": HERE
    / "balanced_100_other_models_healthcare23_blind_2026-08-09"
    / "gpt56"
    / "validation_stage_1_single_none"
    / "checkpoints.jsonl",
    "gpt51": HERE
    / "balanced_100_other_models_healthcare23_blind_2026-08-09"
    / "gpt51"
    / "validation_stage_1_single_none"
    / "checkpoints.jsonl",
}

BLIND_INPUT_23 = (
    HERE
    / "balanced_100_other_models_healthcare23_blind_2026-08-09"
    / "healthcare23_MODEL_INPUT_ONLY.csv"
)

BALANCED_2000 = (
    HERE
    / "balanced_2000_final_stance_analysis_2026-08-01"
    / "balanced_2000_stance_analysis.csv"
)

OUT = HERE / "balanced_100_three_model_final_comparison_2026-08-09"

MODEL_COLUMNS = {
    "gpt5mini": "GPT-5 mini",
    "gpt56": "GPT-5.6",
    "gpt51": "GPT-5.1",
}

MODEL_COLORS = {
    "GPT-5 mini": "#2f718e",
    "GPT-5.6": "#78aec4",
    "GPT-5.1": "#c4dce7",
}

DISPLAY_FIELDS = {
    "attitude": "Attitude",
    "attitude_target": "Attitude target",
    "use": "AI use",
    "capability_assessment": "Capability assessment",
    "trust_construct": "Trust construct",
    "trust_boundary": "Fine-grained trust boundary",
}

DISPLAY_GATES = {
    "substantive_trust_content": "Substantive trust content",
    "explicit_trust_boundary": "Explicit trust boundary",
}


def clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def load_checkpoint_records(path: Path, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in iter_jsonl(path):
        if clean(record.get("completion_status")) != "completed":
            continue
        comment_id = clean(record.get("stable_comment_id"))
        labels = record.get("parsed_labels") or {}
        row: dict[str, Any] = {
            "comment_id": comment_id,
            "checkpoint_source": source,
            "returned_model_id": clean(record.get("returned_model_id")),
            "prompt_sha256": clean(record.get("prompt_hash")),
            "schema_sha256": clean(record.get("schema_hash")),
            "retry_count": int(record.get("retry_count") or 0),
        }
        for field in LABEL_FIELDS:
            row[field] = clean(labels.get(field))
        rows.append(row)
    return rows


def combine_model_checkpoints(model: str, final_ids: set[str]) -> pd.DataFrame:
    rows = load_checkpoint_records(OLD_CHECKPOINTS[model], "original_validation_run")
    rows += load_checkpoint_records(
        NEW_HEALTHCARE23_CHECKPOINTS[model], "healthcare23_blind_completion"
    )
    frame = pd.DataFrame(rows)
    frame = frame.loc[frame["comment_id"].isin(final_ids)].copy()

    duplicated = frame.loc[frame["comment_id"].duplicated(keep=False)]
    if not duplicated.empty:
        compare_columns = list(LABEL_FIELDS) + [
            "returned_model_id",
            "prompt_sha256",
            "schema_sha256",
        ]
        for comment_id, group in duplicated.groupby("comment_id"):
            if any(group[column].nunique(dropna=False) != 1 for column in compare_columns):
                raise ValueError(f"Conflicting duplicate predictions for {model}: {comment_id}")
        frame = frame.drop_duplicates("comment_id", keep="last")

    if len(frame) != 100 or frame["comment_id"].nunique() != 100:
        missing = sorted(final_ids - set(frame["comment_id"]))
        raise ValueError(
            f"{model} does not cover the final balanced 100: n={len(frame)}, missing={missing}"
        )

    for field in LABEL_FIELDS:
        invalid = sorted(set(frame[field]) - set(LABEL_OPTIONS[field]))
        if invalid or frame[field].eq("").any():
            raise ValueError(f"Invalid {model} labels for {field}: {invalid}")

    renamed = {
        field: f"{model}_{field}" for field in LABEL_FIELDS
    } | {
        "checkpoint_source": f"{model}_checkpoint_source",
        "returned_model_id": f"{model}_returned_model_id",
        "prompt_sha256": f"{model}_prompt_sha256",
        "schema_sha256": f"{model}_schema_sha256",
        "retry_count": f"{model}_retry_count",
    }
    return frame.rename(columns=renamed)


def load_joined() -> pd.DataFrame:
    gold = pd.read_csv(FINAL_100, dtype=str, keep_default_na=False)
    if len(gold) != 100 or gold["comment_id"].nunique() != 100:
        raise ValueError("Final balanced validation file must contain 100 unique comments")
    industry_counts = gold["industry"].value_counts().to_dict()
    if len(industry_counts) != 4 or set(industry_counts.values()) != {25}:
        raise ValueError(f"Expected 25 comments per industry, found {industry_counts}")

    for field in LABEL_FIELDS:
        gold[f"gpt5mini_{field}"] = gold[f"gpt_human_{field}"].map(clean)
        for column in (f"human_{field}", f"gpt5mini_{field}"):
            invalid = sorted(set(gold[column]) - set(LABEL_OPTIONS[field]))
            if invalid or gold[column].eq("").any():
                raise ValueError(f"Invalid or missing labels in {column}: {invalid}")

    final_ids = set(gold["comment_id"].map(clean))
    joined = gold.merge(
        combine_model_checkpoints("gpt56", final_ids), on="comment_id", validate="one_to_one"
    )
    joined = joined.merge(
        combine_model_checkpoints("gpt51", final_ids), on="comment_id", validate="one_to_one"
    )
    if len(joined) != 100 or joined["comment_id"].nunique() != 100:
        raise ValueError("Final merge changed the balanced validation sample")
    return joined.sort_values(["industry", "comment_id"], kind="stable").reset_index(drop=True)


def multiclass_metrics(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    confusion: list[dict[str, Any]] = []
    for field in LABEL_FIELDS:
        labels = LABEL_OPTIONS[field]
        truth = joined[f"human_{field}"]
        for model_key, model_name in MODEL_COLUMNS.items():
            pred = joined[f"{model_key}_{field}"]
            macro = precision_recall_fscore_support(
                truth, pred, average="macro", zero_division=0
            )
            weighted = precision_recall_fscore_support(
                truth, pred, average="weighted", zero_division=0
            )
            summary.append(
                {
                    "field": field,
                    "field_display": DISPLAY_FIELDS[field],
                    "model": model_name,
                    "N": len(truth),
                    "accuracy": accuracy_score(truth, pred),
                    "macro_precision": macro[0],
                    "macro_recall": macro[1],
                    "macro_f1": macro[2],
                    "weighted_f1": weighted[2],
                    "exact_agreement_n": int(truth.eq(pred).sum()),
                }
            )
            precision, recall, f1, support = precision_recall_fscore_support(
                truth, pred, labels=labels, average=None, zero_division=0
            )
            matrix = confusion_matrix(truth, pred, labels=labels)
            for row_idx, label in enumerate(labels):
                per_class.append(
                    {
                        "field": field,
                        "field_display": DISPLAY_FIELDS[field],
                        "model": model_name,
                        "class": label,
                        "human_n": int(support[row_idx]),
                        "predicted_n": int(pred.eq(label).sum()),
                        "precision": precision[row_idx],
                        "recall": recall[row_idx],
                        "f1": f1[row_idx],
                    }
                )
                for col_idx, predicted in enumerate(labels):
                    confusion.append(
                        {
                            "field": field,
                            "field_display": DISPLAY_FIELDS[field],
                            "model": model_name,
                            "human_label": label,
                            "predicted_label": predicted,
                            "n": int(matrix[row_idx, col_idx]),
                        }
                    )
    return pd.DataFrame(summary), pd.DataFrame(per_class), pd.DataFrame(confusion)


def gate_series(joined: pd.DataFrame, who: str, gate: str) -> pd.Series:
    if gate == "substantive_trust_content":
        field, derive = "trust_construct", derive_substantive_gate
    else:
        field, derive = "trust_boundary", derive_boundary_gate
    source = f"human_{field}" if who == "human" else f"{who}_{field}"
    # The established binary evaluation treats the one human 'unclear' label
    # as not-positive. Model predictions contain no unclear gate labels.
    return joined[source].map(lambda value: "yes" if derive(clean(value)) == "yes" else "no")


def binary_metric_row(truth: pd.Series, pred: pd.Series) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=["no", "yes"]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "N": len(truth),
        "gold_positive_n": int(truth.eq("yes").sum()),
        "gold_negative_n": int(truth.eq("no").sum()),
        "predicted_positive_n": int(pred.eq("yes").sum()),
        "predicted_negative_n": int(pred.eq("no").sum()),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "accuracy": accuracy_score(truth, pred),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy_score(truth, pred),
        "NPV": npv,
        "F1": f1,
        "MCC": matthews_corrcoef(truth, pred),
        "gold_positive_prevalence": truth.eq("yes").mean(),
        "predicted_positive_prevalence": pred.eq("yes").mean(),
    }


def binary_metrics(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled: list[dict[str, Any]] = []
    by_industry: list[dict[str, Any]] = []
    for gate, gate_display in DISPLAY_GATES.items():
        truth = gate_series(joined, "human", gate)
        for model_key, model_name in MODEL_COLUMNS.items():
            pred = gate_series(joined, model_key, gate)
            pooled.append(
                {
                    "gate": gate,
                    "gate_display": gate_display,
                    "model": model_name,
                    **binary_metric_row(truth, pred),
                }
            )
            for industry, group in joined.groupby("industry", sort=True):
                group_truth = gate_series(group, "human", gate)
                group_pred = gate_series(group, model_key, gate)
                by_industry.append(
                    {
                        "gate": gate,
                        "gate_display": gate_display,
                        "model": model_name,
                        "industry": industry,
                        **binary_metric_row(group_truth, group_pred),
                    }
                )
    return pd.DataFrame(pooled), pd.DataFrame(by_industry)


def prediction_distributions(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field in LABEL_FIELDS:
        for model_key, model_name in MODEL_COLUMNS.items():
            counts = joined[f"{model_key}_{field}"].value_counts()
            for label in LABEL_OPTIONS[field]:
                rows.append(
                    {
                        "field": field,
                        "field_display": DISPLAY_FIELDS[field],
                        "model": model_name,
                        "label": label,
                        "predicted_n": int(counts.get(label, 0)),
                        "predicted_share": float(counts.get(label, 0) / len(joined)),
                    }
                )
    return pd.DataFrame(rows)


def paired_outputs(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    tasks: list[tuple[str, str]] = [("multiclass", field) for field in LABEL_FIELDS]
    tasks += [("binary_gate", gate) for gate in DISPLAY_GATES]
    metadata = [
        "validation_id",
        "comment_id",
        "industry",
        "subreddit",
        "post_id",
        "post_title",
        "parent_context",
        "comment_text",
        "human_note",
    ]

    for model_a, model_b in combinations(MODEL_COLUMNS, 2):
        for task_type, field in tasks:
            if task_type == "multiclass":
                truth = joined[f"human_{field}"]
                pred_a = joined[f"{model_a}_{field}"]
                pred_b = joined[f"{model_b}_{field}"]
            else:
                truth = gate_series(joined, "human", field)
                pred_a = gate_series(joined, model_a, field)
                pred_b = gate_series(joined, model_b, field)
            a_correct = pred_a.eq(truth)
            b_correct = pred_b.eq(truth)
            summary_rows.append(
                {
                    "task_type": task_type,
                    "field_or_gate": field,
                    "model_a": MODEL_COLUMNS[model_a],
                    "model_b": MODEL_COLUMNS[model_b],
                    "both_correct": int((a_correct & b_correct).sum()),
                    "model_a_only_correct": int((a_correct & ~b_correct).sum()),
                    "model_b_only_correct": int((~a_correct & b_correct).sum()),
                    "both_wrong": int((~a_correct & ~b_correct).sum()),
                    "prediction_agreement": pred_a.eq(pred_b).mean(),
                }
            )
            masks = {
                "model_a_correct_model_b_wrong": a_correct & ~b_correct,
                "model_b_correct_model_a_wrong": ~a_correct & b_correct,
            }
            for case_type, mask in masks.items():
                for idx in joined.index[mask]:
                    row = {column: joined.at[idx, column] for column in metadata}
                    row.update(
                        {
                            "task_type": task_type,
                            "field_or_gate": field,
                            "comparison": f"{MODEL_COLUMNS[model_a]} vs {MODEL_COLUMNS[model_b]}",
                            "case_type": case_type,
                            "human_label": truth.at[idx],
                            "model_a": MODEL_COLUMNS[model_a],
                            "model_a_prediction": pred_a.at[idx],
                            "model_b": MODEL_COLUMNS[model_b],
                            "model_b_prediction": pred_b.at[idx],
                        }
                    )
                    case_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(case_rows)


def audit(joined: pd.DataFrame) -> dict[str, Any]:
    blind = pd.read_csv(BLIND_INPUT_23, dtype=str, keep_default_na=False)
    expected_blind_columns = ["comment_id", "parent_context", "target_comment"]
    model_hashes: dict[str, Any] = {}
    for model in ("gpt56", "gpt51"):
        model_hashes[model] = {
            "returned_model_ids": sorted(joined[f"{model}_returned_model_id"].unique()),
            "prompt_hashes": sorted(joined[f"{model}_prompt_sha256"].unique()),
            "schema_hashes": sorted(joined[f"{model}_schema_sha256"].unique()),
            "retry_count_total": int(pd.to_numeric(joined[f"{model}_retry_count"]).sum()),
            "original_run_n": int(
                joined[f"{model}_checkpoint_source"].eq("original_validation_run").sum()
            ),
            "healthcare23_completion_n": int(
                joined[f"{model}_checkpoint_source"].eq("healthcare23_blind_completion").sum()
            ),
        }

    mini_metadata = pd.read_csv(
        BALANCED_2000,
        usecols=["comment_id", "model_identifier", "prompt_sha256", "schema_sha256"],
        dtype=str,
        keep_default_na=False,
    )
    mini_metadata = mini_metadata.loc[mini_metadata["comment_id"].isin(joined["comment_id"])]

    checks = {
        "final_rows_equal_100": len(joined) == 100,
        "comment_ids_unique": joined["comment_id"].nunique() == 100,
        "exactly_25_per_industry": set(joined["industry"].value_counts()) == {25},
        "gpt56_complete_100": joined["gpt56_attitude"].ne("").sum() == 100,
        "gpt51_complete_100": joined["gpt51_attitude"].ne("").sum() == 100,
        "new_blind_input_has_only_whitelisted_fields": list(blind.columns)
        == expected_blind_columns,
        "new_blind_input_has_23_unique_rows": len(blind) == 23
        and blind["comment_id"].nunique() == 23,
        "new_blind_input_excludes_human_gold_notes": not any(
            any(fragment in column.lower() for fragment in ("human", "gold", "note", "gpt"))
            for column in blind.columns
        ),
        "gpt56_and_gpt51_prompt_hash_match": model_hashes["gpt56"]["prompt_hashes"]
        == model_hashes["gpt51"]["prompt_hashes"],
        "gpt56_and_gpt51_schema_hash_match": model_hashes["gpt56"]["schema_hashes"]
        == model_hashes["gpt51"]["schema_hashes"],
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "industry_counts": joined["industry"].value_counts().sort_index().to_dict(),
        "new_healthcare23_blind_input": str(BLIND_INPUT_23),
        "new_healthcare23_input_columns": list(blind.columns),
        "model_run_metadata": model_hashes,
        "gpt5mini_metadata": {
            "model_identifiers": sorted(mini_metadata["model_identifier"].unique()),
            "prompt_hashes": sorted(mini_metadata["prompt_sha256"].unique()),
            "schema_hashes": sorted(mini_metadata["schema_sha256"].unique()),
            "note": (
                "GPT-5 mini used the same Frozen Prompt V2 but an earlier richer response "
                "schema. Evaluation uses the six label fields shared by all three models."
            ),
        },
    }


def render_binary_table(frame: pd.DataFrame, output_stem: Path) -> None:
    gate_order = ["substantive_trust_content", "explicit_trust_boundary"]
    model_order = list(MODEL_COLUMNS.values())
    ordered = frame.assign(
        gate_order=frame["gate"].map({value: idx for idx, value in enumerate(gate_order)}),
        model_order=frame["model"].map({value: idx for idx, value in enumerate(model_order)}),
    ).sort_values(["gate_order", "model_order"])
    shown = pd.DataFrame(
        {
            "Gate / model": ordered.apply(
                lambda row: f"{'Substantive' if row['gate'] == 'substantive_trust_content' else 'Explicit'} / "
                f"{row['model'].replace('GPT-', '')}",
                axis=1,
            ),
            "Precision": ordered["precision"].map(lambda x: f"{x:.3f}"),
            "Recall": ordered["recall"].map(lambda x: f"{x:.3f}"),
            "Specificity": ordered["specificity"].map(lambda x: f"{x:.3f}"),
            "Balanced acc.": ordered["balanced_accuracy"].map(lambda x: f"{x:.3f}"),
            "F1": ordered["F1"].map(lambda x: f"{x:.3f}"),
            "MCC": ordered["MCC"].map(lambda x: f"{x:.3f}"),
        }
    )

    fig, ax = plt.subplots(figsize=(11.8, 4.9))
    ax.axis("off")
    table = ax.table(
        cellText=shown.values,
        colLabels=shown.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.25, 0.12, 0.12, 0.13, 0.15, 0.10, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.75)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d7e2e6")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#d9edf4")
            cell.set_text_props(weight="bold", color="#183945")
        else:
            model_name = ordered.iloc[row - 1]["model"]
            base = MODEL_COLORS[model_name]
            cell.set_facecolor(base if col == 0 else "#f4f9fb")
            if col == 0:
                cell.set_text_props(weight="bold", color="white" if model_name != "GPT-5.1" else "#183945")
            if row == 4:
                cell.set_linewidth(1.4)
                cell.set_edgecolor("#afc5ce")
    ax.text(
        0,
        1.03,
        "Binary gates: fair three-model comparison on the balanced 100-comment validation set",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color="#183945",
        ha="left",
    )
    ax.text(
        0,
        -0.03,
        "N = 100 (25 comments per industry). All models are evaluated on identical comments; the 23 added healthcare comments were blind to human labels and notes.",
        transform=ax.transAxes,
        fontsize=8.8,
        color="#526b75",
        ha="left",
    )
    plt.tight_layout(rect=(0.02, 0.08, 0.98, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_multiclass_figure(summary: pd.DataFrame, output_stem: Path) -> None:
    order = [DISPLAY_FIELDS[field] for field in LABEL_FIELDS]
    fig, ax = plt.subplots(figsize=(11.8, 6.6))
    sns.barplot(
        data=summary,
        y="field_display",
        x="macro_f1",
        hue="model",
        order=order,
        hue_order=list(MODEL_COLUMNS.values()),
        palette=MODEL_COLORS,
        ax=ax,
    )
    ax.set_xlim(0, 0.72)
    ax.set_xlabel("Macro-F1")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#dfe8eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(title="", frameon=False, ncol=3, loc="lower right")
    ax.set_title(
        "Six-field annotation performance on the balanced 100-comment validation set",
        loc="left",
        fontsize=14,
        fontweight="bold",
        color="#183945",
        pad=14,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8.5, color="#304b55")
    fig.text(
        0.01,
        0.01,
        "Macro-F1 weights each allowed class equally; classes with no correct predictions receive F1 = 0.",
        fontsize=8.8,
        color="#526b75",
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_confusion_pdf(joined: pd.DataFrame, output: Path) -> None:
    with PdfPages(output) as pdf:
        for field in LABEL_FIELDS:
            labels = LABEL_OPTIONS[field]
            fig, axes = plt.subplots(1, 3, figsize=(20, 7.2), constrained_layout=True)
            truth = joined[f"human_{field}"]
            for ax, (model_key, model_name) in zip(axes, MODEL_COLUMNS.items()):
                matrix = confusion_matrix(truth, joined[f"{model_key}_{field}"], labels=labels)
                sns.heatmap(
                    matrix,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    square=True,
                    xticklabels=labels,
                    yticklabels=labels,
                    linewidths=0.5,
                    linecolor="white",
                    ax=ax,
                )
                ax.set_title(model_name, fontweight="bold")
                ax.set_xlabel("Predicted label")
                ax.set_ylabel("Human label")
                ax.tick_params(axis="x", rotation=55, labelsize=8)
                ax.tick_params(axis="y", rotation=0, labelsize=8)
            fig.suptitle(
                f"Confusion matrices: {DISPLAY_FIELDS[field]}",
                fontsize=16,
                fontweight="bold",
            )
            pdf.savefig(fig, bbox_inches="tight", facecolor="white")
            plt.close(fig)


def write_readme(
    audit_data: dict[str, Any],
    multiclass: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    best_multiclass = (
        multiclass.sort_values(["field", "macro_f1"], ascending=[True, False])
        .groupby("field", as_index=False)
        .first()
    )
    best_gates = (
        gates.sort_values(["gate", "balanced_accuracy"], ascending=[True, False])
        .groupby("gate", as_index=False)
        .first()
    )
    lines = [
        "# Final balanced-100 three-model comparison",
        "",
        "Created 2026-08-09 without any API calls.",
        "",
        "## Validation design",
        "",
        "- 100 unique comments: 25 each from finance, healthcare, law and software engineering.",
        "- GPT-5 mini predictions come from the frozen balanced analytical sample.",
        "- GPT-5.6 and GPT-5.1 reuse 77 existing predictions and add the 23 previously missing healthcare predictions.",
        "- The 23-row model input contained only `comment_id`, `parent_context` and `target_comment`.",
        "- Human labels, gold labels and notes were absent from the model input.",
        "- All evaluations use the same six-label codebook and the same Frozen Prompt V2.",
        "- GPT-5 mini used an earlier richer response schema; GPT-5.6 and GPT-5.1 used the same six-field schema. Only the six shared labels are compared.",
        "",
        f"Audit verdict: **{audit_data['verdict']}**.",
        "",
        "## Best macro-F1 by field",
        "",
    ]
    for _, row in best_multiclass.iterrows():
        lines.append(f"- {row['field_display']}: {row['model']} ({row['macro_f1']:.3f})")
    lines += ["", "## Best balanced accuracy by binary gate", ""]
    for _, row in best_gates.iterrows():
        lines.append(
            f"- {row['gate_display']}: {row['model']} "
            f"(balanced accuracy {row['balanced_accuracy']:.3f}; F1 {row['F1']:.3f}; MCC {row['MCC']:.3f})"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "These are validation estimates from a stratified 100-comment sample. Per-industry estimates use only 25 comments and should be treated as transfer diagnostics rather than precise performance estimates.",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    joined = load_joined()
    audit_data = audit(joined)
    if audit_data["verdict"] != "PASS":
        raise ValueError(f"Coverage/blinding audit failed: {audit_data}")

    multiclass, per_class, confusion = multiclass_metrics(joined)
    gates, gates_by_industry = binary_metrics(joined)
    distributions = prediction_distributions(joined)
    paired_summary, paired_cases = paired_outputs(joined)

    joined.to_csv(OUT / "balanced100_human_and_three_model_predictions.csv", index=False)
    multiclass.to_csv(OUT / "three_model_six_field_summary.csv", index=False)
    per_class.to_csv(OUT / "three_model_per_class_metrics.csv", index=False)
    confusion.to_csv(OUT / "three_model_confusion_matrices_long.csv", index=False)
    distributions.to_csv(OUT / "three_model_prediction_distributions.csv", index=False)
    gates.to_csv(OUT / "three_model_binary_gate_metrics.csv", index=False)
    gates_by_industry.to_csv(OUT / "three_model_binary_gate_metrics_by_industry.csv", index=False)
    paired_summary.to_csv(OUT / "three_model_paired_correctness_summary.csv", index=False)
    paired_cases.to_csv(OUT / "three_model_paired_disagreement_cases.csv", index=False)
    paired_cases.loc[
        paired_cases["comparison"].eq("GPT-5 mini vs GPT-5.6")
    ].to_csv(OUT / "gpt5mini_vs_gpt56_paired_cases.csv", index=False)
    (OUT / "coverage_and_blinding_audit.json").write_text(
        json.dumps(audit_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with pd.ExcelWriter(OUT / "balanced100_three_model_final_comparison.xlsx", engine="openpyxl") as writer:
        multiclass.to_excel(writer, sheet_name="six_field_summary", index=False)
        per_class.to_excel(writer, sheet_name="per_class_metrics", index=False)
        gates.to_excel(writer, sheet_name="binary_gates", index=False)
        gates_by_industry.to_excel(writer, sheet_name="gates_by_industry", index=False)
        distributions.to_excel(writer, sheet_name="prediction_counts", index=False)
        paired_summary.to_excel(writer, sheet_name="paired_summary", index=False)
        paired_cases.to_excel(writer, sheet_name="paired_cases", index=False)
        confusion.to_excel(writer, sheet_name="confusion_long", index=False)

    render_binary_table(gates, OUT / "FIG_balanced100_three_model_binary_gates")
    render_multiclass_figure(multiclass, OUT / "FIG_balanced100_three_model_six_field_macro_f1")
    render_confusion_pdf(joined, OUT / "three_model_confusion_matrix_heatmaps.pdf")
    write_readme(audit_data, multiclass, gates)

    print("Audit:", audit_data["verdict"])
    print("Industry counts:", joined["industry"].value_counts().sort_index().to_dict())
    print("\nSix-field macro-F1:")
    print(
        multiclass.pivot(index="field_display", columns="model", values="macro_f1")
        .round(4)
        .to_string()
    )
    print("\nBinary gates:")
    print(
        gates[
            [
                "gate_display",
                "model",
                "precision",
                "recall",
                "specificity",
                "balanced_accuracy",
                "F1",
                "MCC",
                "TP",
                "TN",
                "FP",
                "FN",
            ]
        ].round(4).to_string(index=False)
    )
    print("\nOutputs:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
