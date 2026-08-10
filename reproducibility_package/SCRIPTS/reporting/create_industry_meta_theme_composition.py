#!/usr/bin/env python3
"""Create Industry x meta-theme composition outputs for substantive trust discourse.

This script uses the locked 1,204 substantive-trust comments and the already
joined frozen pooled meta-theme assignments. It does not refit BERTopic and does
not change any labels. It only creates reporting-level composition tables and a
4 x 6 heatmap for RQ3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter


BASE = Path("outputs")
STANCE_DIR = BASE / "stance_analysis_FINAL_2026-07-28"
INPUT = STANCE_DIR / "analysis_ready_data/stance_analysis_ready_substantive_trust_comments.csv"
OUT = STANCE_DIR / "topic_heatmap_and_model_reports"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"

INDUSTRY_ORDER = ["finance", "healthcare", "law", "software_engineering"]
INDUSTRY_LABELS = {
    "finance": "Finance",
    "healthcare": "Healthcare",
    "law": "Law",
    "software_engineering": "Software engineering",
}

REPORTING_META_THEME_ORDER = [
    "AI in professional work and occupational change",
    "Industry-specific professional applications",
    "Reliability, accountability and human oversight",
    "Education, training and assessment",
    "AI capabilities and conceptual boundaries",
    "Relational work and limits of substitution",
]

THEME_CONSOLIDATION = {
    "AI in professional work and occupational change": "AI in professional work and occupational change",
    "Industry-specific professional applications": "Industry-specific professional applications",
    "Reliability, accountability and human oversight": "Reliability, accountability and human oversight",
    "Education, training and assessment": "Education, training and assessment",
    "AI capabilities and conceptual boundaries": "AI capabilities and conceptual boundaries",
    "Relational work and limits of substitution": "Relational work and limits of substitution",
    # Minor substantive meta-themes are folded into the closest reporting-level
    # meta-theme so the dissertation figure can use a stable six-theme scheme.
    "Augmentation and productivity": "Industry-specific professional applications",
    "AI tool ecosystem and adoption": "Industry-specific professional applications",
    "Privacy, confidentiality and governance": "Reliability, accountability and human oversight",
    "Reliability and human verification": "Reliability, accountability and human oversight",
    "Responsibility, liability and governance": "Reliability, accountability and human oversight",
    "Employment processes and occupational gatekeeping": "AI in professional work and occupational change",
    "Professional identity and boundaries of expertise": "AI in professional work and occupational change",
    "Market hype and organisational governance": "AI in professional work and occupational change",
}

EXCLUDED_META_THEMES = {
    "Not a substantive analytical theme",
    "Unassigned / no pooled topic",
}


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT)
    df["industry"] = df["industry"].astype(str).str.strip()
    df["meta_theme"] = df["meta_theme"].fillna("Unassigned / no pooled topic").astype(str).str.strip()
    df["reporting_meta_theme"] = df["meta_theme"].map(THEME_CONSOLIDATION)
    df["meta_theme_composition_status"] = np.where(
        df["meta_theme"].isin(EXCLUDED_META_THEMES),
        "excluded_non_substantive_or_unassigned",
        np.where(df["reporting_meta_theme"].isna(), "unmapped_review_required", "included_in_six_theme_composition"),
    )
    return df


def write_audit_tables(df: pd.DataFrame) -> None:
    audit = (
        df.groupby(["meta_theme", "reporting_meta_theme", "meta_theme_composition_status"], dropna=False)
        .agg(
            comments=("comment_id", "count"),
            unique_posts=("post_id", "nunique"),
            unique_industries=("industry", "nunique"),
        )
        .reset_index()
        .sort_values(["meta_theme_composition_status", "comments"], ascending=[True, False])
    )
    audit.to_csv(TABLES / "industry_meta_theme_composition_consolidation_audit.csv", index=False)

    denominator = (
        df.groupby(["industry", "meta_theme_composition_status"])
        .size()
        .rename("comments")
        .reset_index()
        .pivot_table(index="industry", columns="meta_theme_composition_status", values="comments", fill_value=0)
        .reindex(INDUSTRY_ORDER, fill_value=0)
    )
    denominator["total_substantive_trust_comments"] = denominator.sum(axis=1)
    denominator["included_in_six_theme_share"] = (
        denominator.get("included_in_six_theme_composition", 0) / denominator["total_substantive_trust_comments"]
    )
    denominator.to_csv(TABLES / "industry_meta_theme_composition_denominator_audit.csv")


def build_composition_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    included = df[df["meta_theme_composition_status"] == "included_in_six_theme_composition"].copy()
    counts = (
        included.pivot_table(
            index="industry",
            columns="reporting_meta_theme",
            values="comment_id",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(index=INDUSTRY_ORDER, columns=REPORTING_META_THEME_ORDER, fill_value=0)
        .astype(int)
    )
    percents = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
    counts.to_csv(TABLES / "industry_by_reporting_meta_theme_counts_4x6.csv")
    percents.round(3).to_csv(TABLES / "industry_by_reporting_meta_theme_percent_4x6.csv")

    top3_rows = []
    for industry in INDUSTRY_ORDER:
        row_counts = counts.loc[industry]
        row_pct = percents.loc[industry]
        for rank, theme in enumerate(row_counts.sort_values(ascending=False).head(3).index, start=1):
            top3_rows.append(
                {
                    "industry": industry,
                    "industry_label": INDUSTRY_LABELS[industry],
                    "rank": rank,
                    "reporting_meta_theme": theme,
                    "comments": int(row_counts[theme]),
                    "percent_within_included_six_theme_comments": round(float(row_pct[theme]), 3),
                    "included_six_theme_denominator": int(row_counts.sum()),
                }
            )
    pd.DataFrame(top3_rows).to_csv(TABLES / "top3_reporting_meta_themes_by_industry.csv", index=False)
    return counts, percents


def plot_heatmap(counts: pd.DataFrame, percents: pd.DataFrame) -> None:
    values = percents.to_numpy()
    cmap = LinearSegmentedColormap.from_list("industry_theme_heat", ["#F7F7F7", "#BFD8D5", "#2F6F73"])

    fig, ax = plt.subplots(figsize=(12.4, 4.9))
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=max(50, np.nanmax(values)))

    column_labels = [
        "AI in professional work\nand occupational change",
        "Industry-specific\nprofessional applications",
        "Reliability, accountability\nand human oversight",
        "Education, training\nand assessment",
        "AI capabilities and\nconceptual boundaries",
        "Relational work and\nlimits of substitution",
    ]
    row_labels = [
        f"{INDUSTRY_LABELS[industry]}\n(n={int(counts.loc[industry].sum())})"
        for industry in INDUSTRY_ORDER
    ]

    ax.set_xticks(np.arange(len(REPORTING_META_THEME_ORDER)), column_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(INDUSTRY_ORDER)), row_labels)
    ax.set_title("Meta-theme composition of substantive trust discourse by industry", pad=14)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            text_color = "#111111"
            ax.text(j, i, f"{val:.0f}%\n({int(counts.iloc[i, j])})", ha="center", va="center", fontsize=8.5, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.025)
    cbar.set_label("Within-industry share")
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure5_industry_by_meta_theme_composition_heatmap.png", dpi=300)
    fig.savefig(FIGURES / "figure5_industry_by_meta_theme_composition_heatmap.pdf")
    plt.close(fig)


def append_caption_and_readme(df: pd.DataFrame, counts: pd.DataFrame) -> None:
    caption_path = OUT / "figure_captions_topic_heatmap_and_models.md"
    caption_text = caption_path.read_text(encoding="utf-8") if caption_path.exists() else "# Figure Captions\n"
    addition = """

## Figure 5. Meta-theme composition of substantive trust discourse by industry

Cells show the percentage of each industry's substantive-trust comments assigned to each of six reporting-level meta-themes. Percentages are row-wise and sum to 100% within each industry after excluding pooled outlier/unassigned categories from this specific composition figure. Minor substantive meta-themes were folded into the closest reporting-level theme; the consolidation and denominator audit are reported in the accompanying CSV tables. Counts are shown in parentheses.
"""
    if "Figure 5. Meta-theme composition" not in caption_text:
        caption_path.write_text(caption_text.rstrip() + addition, encoding="utf-8")

    readme_path = OUT / "README_topic_heatmap_and_model_reports.md"
    readme_text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# Topic Heatmap and Model Reports\n"
    included = int((df["meta_theme_composition_status"] == "included_in_six_theme_composition").sum())
    excluded = int((df["meta_theme_composition_status"] != "included_in_six_theme_composition").sum())
    addition = f"""

## Industry x meta-theme composition

`figure5_industry_by_meta_theme_composition_heatmap` reports a 4 x 6 heatmap for RQ3: the composition of substantive trust discourse by industry.

The figure uses {included} comments mapped to six reporting-level meta-themes. It excludes {excluded} comments assigned to pooled non-substantive or unassigned categories from this specific composition denominator; these rows remain in the audit tables and in the underlying 1,204-row stance-analysis dataset.

Rows are industry-normalised: each industry row sums to 100% across the six reporting-level meta-themes.

Key tables:

- `industry_by_reporting_meta_theme_counts_4x6.csv`
- `industry_by_reporting_meta_theme_percent_4x6.csv`
- `top3_reporting_meta_themes_by_industry.csv`
- `industry_meta_theme_composition_consolidation_audit.csv`
- `industry_meta_theme_composition_denominator_audit.csv`
"""
    if "Industry x meta-theme composition" not in readme_text:
        readme_path.write_text(readme_text.rstrip() + addition, encoding="utf-8")


def main() -> int:
    ensure_dirs()
    df = load_data()
    write_audit_tables(df)
    counts, percents = build_composition_tables(df)
    plot_heatmap(counts, percents)
    append_caption_and_readme(df, counts)

    print(f"Input substantive-trust comments: {len(df)}")
    print(f"Included in 4x6 composition: {(df['meta_theme_composition_status'] == 'included_in_six_theme_composition').sum()}")
    print(f"Excluded from 4x6 denominator: {(df['meta_theme_composition_status'] != 'included_in_six_theme_composition').sum()}")
    print("Top 3 reporting meta-themes by industry:")
    print(pd.read_csv(TABLES / "top3_reporting_meta_themes_by_industry.csv").to_string(index=False))
    print(f"Figure: {FIGURES / 'figure5_industry_by_meta_theme_composition_heatmap.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
