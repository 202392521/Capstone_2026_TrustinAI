#!/usr/bin/env python3
"""Create the final within-industry explicit-boundary meta-theme heatmap.

The figure reuses the frozen explicit-boundary BERTopic assignments and the
researcher-verified topic-to-meta-theme mapping. It does not refit BERTopic.
Each industry column is normalized by that industry's comments assigned to
the six substantive meta-themes, so every column sums to 100%.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


BASE_DIR = Path(
    "outputs/"
    "explicit_boundary_bertopic_FINAL_2026-08-03"
)
ASSIGNMENTS_CSV = BASE_DIR / "explicit_boundary_document_topic_assignments.csv"
MAPPING_CSV = BASE_DIR / "explicit_boundary_original_topic_to_reporting_theme_mapping.csv"

OUTPUT_STEM = BASE_DIR / "FIG5_explicit_boundary_meta_themes_by_industry_with_overall_share_no_title"

INDUSTRY_ORDER = ["finance", "healthcare", "law", "software_engineering"]
INDUSTRY_LABELS = {
    "finance": "Finance",
    "healthcare": "Healthcare",
    "law": "Law",
    "software_engineering": "Software / IT",
}

META_THEME_ORDER = [
    "Verification and epistemic reliability",
    "Task-specific augmentation and occupational limits",
    "Human oversight and professional accountability",
    "Authorship, detection and institutional governance",
    "Privacy, confidentiality and regulatory safeguards",
    "Relational and human-judgement boundaries",
]
META_THEME_LABELS = {
    "Verification and epistemic reliability": "Verification & epistemic reliability",
    "Task-specific augmentation and occupational limits": "Task-specific augmentation",
    "Human oversight and professional accountability": "Human oversight & accountability",
    "Authorship, detection and institutional governance": "Authorship / detection / governance",
    "Privacy, confidentiality and regulatory safeguards": "Privacy / confidentiality",
    "Relational and human-judgement boundaries": "Relational / human judgement",
}


def load_and_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    assignments = pd.read_csv(ASSIGNMENTS_CSV)
    mapping = pd.read_csv(MAPPING_CSV)

    required_assignment_columns = {"comment_id", "industry", "boundary_topic"}
    required_mapping_columns = {"source_topic_id", "final_meta_theme", "reporting_decision"}
    missing_assignments = required_assignment_columns - set(assignments.columns)
    missing_mapping = required_mapping_columns - set(mapping.columns)
    if missing_assignments:
        raise ValueError(f"Missing assignment columns: {sorted(missing_assignments)}")
    if missing_mapping:
        raise ValueError(f"Missing mapping columns: {sorted(missing_mapping)}")

    if assignments["comment_id"].duplicated().any():
        raise ValueError("Duplicate comment_id values found in frozen topic assignments")

    merged = assignments.merge(
        mapping[["source_topic_id", "final_meta_theme", "reporting_decision"]],
        left_on="boundary_topic",
        right_on="source_topic_id",
        how="left",
        validate="many_to_one",
    )
    if merged["source_topic_id"].isna().any():
        missing_topics = sorted(merged.loc[merged["source_topic_id"].isna(), "boundary_topic"].unique())
        raise ValueError(f"Unmapped boundary topics: {missing_topics}")

    substantive = merged.loc[merged["final_meta_theme"].notna()].copy()
    unexpected_industries = sorted(set(substantive["industry"]) - set(INDUSTRY_ORDER))
    if unexpected_industries:
        raise ValueError(f"Unexpected industries: {unexpected_industries}")
    unexpected_themes = sorted(set(substantive["final_meta_theme"]) - set(META_THEME_ORDER))
    if unexpected_themes:
        raise ValueError(f"Unexpected meta-themes: {unexpected_themes}")

    counts = pd.crosstab(substantive["final_meta_theme"], substantive["industry"])
    counts = counts.reindex(index=META_THEME_ORDER, columns=INDUSTRY_ORDER, fill_value=0)
    denominators = counts.sum(axis=0)
    percentages = counts.div(denominators, axis=1) * 100

    if len(substantive) != 646:
        raise ValueError(f"Expected 646 substantive assignments, found {len(substantive)}")
    if not np.allclose(percentages.sum(axis=0).to_numpy(), 100.0):
        raise ValueError("Within-industry percentages do not sum to 100%")

    audit = pd.DataFrame(
        {
            "industry": INDUSTRY_ORDER,
            "display_label": [INDUSTRY_LABELS[value] for value in INDUSTRY_ORDER],
            "all_modelled_explicit_boundary_comments": [
                int((merged["industry"] == value).sum()) for value in INDUSTRY_ORDER
            ],
            "topic_minus_1_unassigned_comments": [
                int(((merged["industry"] == value) & (merged["boundary_topic"] == -1)).sum())
                for value in INDUSTRY_ORDER
            ],
            "substantive_boundary_assignments_denominator": [
                int(denominators[value]) for value in INDUSTRY_ORDER
            ],
            "column_percentage_sum": [float(percentages[value].sum()) for value in INDUSTRY_ORDER],
        }
    )
    return counts, percentages, denominators, audit


def save_tables(
    counts: pd.DataFrame,
    percentages: pd.DataFrame,
    denominators: pd.Series,
    audit: pd.DataFrame,
) -> None:
    counts_out = counts.copy()
    counts_out.index = [META_THEME_LABELS[value] for value in counts_out.index]
    counts_out.columns = [INDUSTRY_LABELS[value] for value in counts_out.columns]
    counts_out.index.name = "meta_theme"
    counts_out.to_csv(BASE_DIR / "explicit_boundary_meta_theme_by_industry_counts.csv")

    pct_out = percentages.copy()
    pct_out.index = [META_THEME_LABELS[value] for value in pct_out.index]
    pct_out.columns = [INDUSTRY_LABELS[value] for value in pct_out.columns]
    pct_out.index.name = "meta_theme"
    pct_out.to_csv(BASE_DIR / "explicit_boundary_meta_theme_by_industry_percent.csv", float_format="%.6f")

    long_rows = []
    for theme in META_THEME_ORDER:
        for industry in INDUSTRY_ORDER:
            long_rows.append(
                {
                    "meta_theme": META_THEME_LABELS[theme],
                    "industry": INDUSTRY_LABELS[industry],
                    "n_comments": int(counts.loc[theme, industry]),
                    "industry_denominator": int(denominators[industry]),
                    "within_industry_percent": float(percentages.loc[theme, industry]),
                }
            )
    pd.DataFrame(long_rows).to_csv(
        BASE_DIR / "explicit_boundary_meta_theme_by_industry_long.csv", index=False
    )
    audit.to_csv(BASE_DIR / "explicit_boundary_meta_theme_by_industry_denominator_audit.csv", index=False)


def plot_heatmap(counts: pd.DataFrame, percentages: pd.DataFrame, denominators: pd.Series) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "boundary_blues",
        ["#F5F9FC", "#DCECF6", "#B8D8EA", "#82B6D3", "#4E8FB8", "#235E86"],
    )

    values = percentages.to_numpy()
    fig, ax = plt.subplots(figsize=(12.2, 7.1))
    image = ax.imshow(values, cmap=cmap, vmin=0, vmax=60, aspect="auto")

    x_labels = [
        f"{INDUSTRY_LABELS[industry]}\n(n={int(denominators[industry])})"
        for industry in INDUSTRY_ORDER
    ]
    overall_denominator = int(counts.to_numpy().sum())
    overall_counts = counts.sum(axis=1)
    y_labels = [
        (
            f"{META_THEME_LABELS[theme]}\n"
            f"Overall {100 * int(overall_counts.loc[theme]) / overall_denominator:.1f}% "
            f"(n={int(overall_counts.loc[theme])})"
        )
        for theme in META_THEME_ORDER
    ]
    ax.set_xticks(np.arange(len(INDUSTRY_ORDER)), labels=x_labels)
    ax.set_yticks(np.arange(len(META_THEME_ORDER)), labels=y_labels)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=12)
    ax.tick_params(axis="y", length=0, pad=10)

    for label in ax.get_xticklabels():
        label.set_fontsize(11.5)
        label.set_fontweight("semibold")
        label.set_color("#243746")
    for label in ax.get_yticklabels():
        label.set_fontsize(10.5)
        label.set_color("#243746")
        label.set_linespacing(1.22)

    # White cell boundaries improve scanability without turning the plot into a spreadsheet.
    ax.set_xticks(np.arange(-0.5, len(INDUSTRY_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(META_THEME_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            count = int(counts.iloc[row, col])
            text_color = "white" if value >= 32 else "#213541"
            ax.text(
                col,
                row,
                f"{value:.1f}%\n(n={count})",
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="semibold" if value >= 20 else "normal",
                color=text_color,
                linespacing=1.25,
            )

    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.055, pad=0.12, aspect=38)
    colorbar.set_label(
        "Within-industry share of substantive boundary assignments (%)",
        fontsize=10.5,
        color="#334A58",
        labelpad=8,
    )
    colorbar.ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
    colorbar.ax.tick_params(labelsize=9.5, colors="#4C626F", length=0)
    colorbar.outline.set_visible(False)

    fig.text(
        0.5,
        0.012,
        f"Row labels show each meta-theme's overall share of the {overall_denominator} substantive-theme "
        "assignments. Columns are normalized within industry and sum to 100%; cells show percentage and "
        "count. Topic -1 is excluded.",
        ha="center",
        va="bottom",
        fontsize=9.3,
        color="#5B6D78",
    )
    fig.subplots_adjust(left=0.39, right=0.985, top=0.83, bottom=0.21)

    for suffix, kwargs in {
        ".png": {"dpi": 320},
        ".pdf": {},
        ".svg": {},
    }.items():
        fig.savefig(f"{OUTPUT_STEM}{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def main() -> None:
    counts, percentages, denominators, audit = load_and_audit()
    save_tables(counts, percentages, denominators, audit)
    plot_heatmap(counts, percentages, denominators)

    print(f"Substantive assignments: {int(denominators.sum())}")
    print("Industry denominators:")
    for industry in INDUSTRY_ORDER:
        print(f"  {INDUSTRY_LABELS[industry]}: {int(denominators[industry])}")
    print(f"PNG: {OUTPUT_STEM}.png")
    print(f"PDF: {OUTPUT_STEM}.pdf")
    print(f"SVG: {OUTPUT_STEM}.svg")


if __name__ == "__main__":
    main()
