#!/usr/bin/env python3
"""Combine and visualise post-lock UMAP sensitivity results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "bertopic_umap_sensitivity_ALL_FINAL_MODELS_2026-08-09"
POOLED = (
    ROOT
    / "cross_industry_pooled_bertopic_FINAL_2026-07-27"
    / "audits"
    / "umap_parameter_sensitivity_2026-08-09"
    / "umap_sensitivity_metrics.csv"
)
INDUSTRY = OUTPUT / "umap_sensitivity_all_industry_models.csv"
BOUNDARY = (
    ROOT
    / "explicit_boundary_bertopic_FINAL_2026-08-03"
    / "audits"
    / "umap_parameter_sensitivity_2026-08-09"
    / "umap_sensitivity_metrics.csv"
)


MODEL_ORDER = [
    "Pooled corpus",
    "Finance",
    "Healthcare",
    "Law",
    "Software engineering / IT",
    "Explicit-boundary subset",
]
CONFIG_ORDER = [
    "nn5_md0",
    "nn10_md0",
    "nn15_md0_reproduction",
    "nn30_md0",
    "nn50_md0",
    "nn15_md01",
    "nn15_md05",
]
CONFIG_LABELS = {
    "nn5_md0": "5 / 0",
    "nn10_md0": "10 / 0",
    "nn15_md0_reproduction": "15 / 0\n(locked)",
    "nn30_md0": "30 / 0",
    "nn50_md0": "50 / 0",
    "nn15_md01": "15 / 0.1",
    "nn15_md05": "15 / 0.5",
}
MODEL_LABELS = {
    "pooled": "Pooled corpus",
    "finance": "Finance",
    "healthcare": "Healthcare",
    "law": "Law",
    "software_engineering": "Software engineering / IT",
    "explicit_boundary": "Explicit-boundary subset",
}


def load_results() -> pd.DataFrame:
    pooled = pd.read_csv(POOLED)
    pooled["model"] = "pooled"
    pooled["locked_substantive_topics"] = 18
    industry = pd.read_csv(INDUSTRY).rename(columns={"industry": "model"})
    boundary = pd.read_csv(BOUNDARY)
    combined = pd.concat([pooled, industry, boundary], ignore_index=True, sort=False)
    combined["model_label"] = combined["model"].map(MODEL_LABELS)
    combined["config_label"] = combined["config"].map(CONFIG_LABELS)
    return combined


def plot_heatmap(combined: pd.DataFrame) -> None:
    values = (
        combined.pivot(index="model_label", columns="config", values="weighted_best_topic_jaccard")
        .reindex(index=MODEL_ORDER, columns=CONFIG_ORDER)
    )
    topic_counts = (
        combined.pivot(index="model_label", columns="config", values="candidate_substantive_topics")
        .reindex(index=MODEL_ORDER, columns=CONFIG_ORDER)
    )
    cmap = LinearSegmentedColormap.from_list(
        "paper_blue",
        ["#f3f8fb", "#d8e9f3", "#a9cfe3", "#6fa8ca", "#2d6f98"],
    )
    fig, ax = plt.subplots(figsize=(11.8, 6.4))
    image = ax.imshow(values.to_numpy(), vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(CONFIG_ORDER)), [CONFIG_LABELS[item] for item in CONFIG_ORDER])
    ax.set_yticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER)
    ax.set_xlabel("UMAP n_neighbors / min_dist")
    ax.set_ylabel("")
    ax.tick_params(axis="both", length=0, labelsize=10)
    for row in range(len(MODEL_ORDER)):
        for column in range(len(CONFIG_ORDER)):
            value = values.iloc[row, column]
            topics = int(topic_counts.iloc[row, column])
            color = "white" if value >= 0.66 else "#243746"
            ax.text(
                column,
                row,
                f"{value:.2f}\n({topics} topics)",
                ha="center",
                va="center",
                fontsize=9,
                color=color,
                fontweight="semibold" if CONFIG_ORDER[column] == "nn15_md0_reproduction" else "normal",
            )
    for column in range(len(CONFIG_ORDER) + 1):
        ax.axvline(column - 0.5, color="white", linewidth=2)
    for row in range(len(MODEL_ORDER) + 1):
        ax.axhline(row - 0.5, color="white", linewidth=2)
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.07, pad=0.18, aspect=40)
    colorbar.set_label("Weighted best-topic document Jaccard overlap with locked assignments", fontsize=10)
    colorbar.outline.set_visible(False)
    fig.text(
        0.5,
        0.015,
        "Cells show weighted topic-overlap Jaccard; parentheses show substantive-topic count. "
        "All embeddings and HDBSCAN settings are frozen within each model.",
        ha="center",
        fontsize=9,
        color="#526474",
    )
    fig.subplots_adjust(left=0.23, right=0.98, top=0.97, bottom=0.26)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / f"figure_umap_parameter_sensitivity_topic_recovery.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    combined = load_results()
    combined.to_csv(OUTPUT / "umap_sensitivity_ALL_locked_bertopic_models.csv", index=False)
    concise_columns = [
        "model_label",
        "config",
        "n_neighbors",
        "min_dist",
        "candidate_substantive_topics",
        "candidate_outlier_rate",
        "ari_all",
        "ari_common_nonout",
        "weighted_best_topic_jaccard",
        "weighted_reference_topic_recall",
        "reference_topics_jaccard_ge_050",
        "locked_substantive_topics",
    ]
    combined[concise_columns].to_csv(OUTPUT / "umap_sensitivity_concise_report.csv", index=False)
    plot_heatmap(combined)
    print(f"Wrote combined audit outputs to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
