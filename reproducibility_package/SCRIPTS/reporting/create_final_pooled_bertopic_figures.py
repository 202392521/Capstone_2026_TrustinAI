#!/usr/bin/env python3
"""Create final dissertation-ready figures for pooled BERTopic meta-theme analysis."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("REPRO_TEMP_DIR", "/tmp")) / "mplconfig_topicviz"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FINAL_DIR = Path(
    "outputs/"
    "cross_industry_pooled_bertopic_FINAL_2026-07-27"
)
FIG_DIR = FINAL_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

INDUSTRIES = ["finance", "law", "software_engineering", "healthcare"]
INDUSTRY_LABELS = {
    "finance": "Finance",
    "law": "Law",
    "software_engineering": "Software engineering",
    "healthcare": "Healthcare",
}

STATUS_LABELS = {
    "shared_broad_partly_noisy": "Shared broad / partly noisy",
    "shared_mixed": "Shared / mixed",
    "shared_software_concentrated": "Shared, software-concentrated",
    "software_concentrated": "Software-concentrated",
    "industry_specific": "Industry-specific",
}

STATUS_COLORS = {
    "shared_broad_partly_noisy": "#8172b3",
    "shared_mixed": "#4c72b0",
    "shared_software_concentrated": "#55a868",
    "software_concentrated": "#64b5cd",
    "industry_specific": "#c44e52",
}

INDUSTRY_COLORS = {
    "finance": "#4c72b0",
    "law": "#dd8452",
    "software_engineering": "#55a868",
    "healthcare": "#c44e52",
}


def wrap_label(text: str, width: int = 36) -> str:
    words = str(text).split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if sum(len(w) for w in current) + len(current) + len(word) > width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def short_topic_label(row: pd.Series, width: int = 34) -> str:
    topic = int(row["pooled_topic"])
    label = str(row["final_label"])
    if topic == -1:
        label = "Outliers / unassigned"
    return f"T{topic}: {wrap_label(label, width)}"


def savefig(name: str) -> None:
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping = pd.read_csv(FINAL_DIR / "cross_industry_pooled_topic_meta_mapping_FINAL.csv")
    counts = pd.read_csv(FINAL_DIR / "final_model/pooled_topic_by_industry_counts.csv")
    shares = pd.read_csv(FINAL_DIR / "final_model/pooled_topic_by_industry_within_group_share.csv")
    topic0_belongs = pd.read_csv(
        FINAL_DIR / "audits/topic0_manual_audit/topic0_audit_summary/topic0_belongs_summary.csv"
    )
    topic0_categories = pd.read_csv(
        FINAL_DIR / "audits/topic0_manual_audit/topic0_audit_summary/topic0_yes_partly_category_summary.csv"
    )
    return mapping, counts, shares, topic0_belongs, topic0_categories


def figure_1_topic_size(mapping: pd.DataFrame) -> None:
    plot_df = mapping[mapping["pooled_topic"] >= 0].copy()
    plot_df = plot_df.sort_values("count", ascending=True)
    colors = [STATUS_COLORS.get(s, "#999999") for s in plot_df["cross_industry_status"]]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(range(len(plot_df)), plot_df["count"], color=colors, edgecolor="none")
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels([short_topic_label(r, 38) for _, r in plot_df.iterrows()], fontsize=8)
    ax.set_xlabel("Number of comments assigned to topic")
    ax.set_title("Pooled BERTopic topic size distribution")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(row["count"] + max(plot_df["count"]) * 0.006, i, f"{int(row['count']):,}", va="center", fontsize=7)

    handles = []
    seen = set()
    for status in plot_df["cross_industry_status"]:
        if status in seen:
            continue
        seen.add(status)
        handles.append(plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=STATUS_COLORS.get(status, "#999999"), markersize=8, label=STATUS_LABELS.get(status, status)))
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    savefig("fig1_pooled_topic_size_distribution")


def figure_2_topic_industry_distribution(mapping: pd.DataFrame, counts: pd.DataFrame) -> None:
    plot_topics = mapping[mapping["pooled_topic"] >= 0].sort_values("count", ascending=True)
    mat = counts[counts["topic"].isin(plot_topics["pooled_topic"])].set_index("topic").loc[plot_topics["pooled_topic"], INDUSTRIES]

    fig, ax = plt.subplots(figsize=(12, 10.5))
    left = np.zeros(len(mat))
    y = np.arange(len(mat))
    for industry in INDUSTRIES:
        vals = mat[industry].to_numpy()
        ax.barh(y, vals, left=left, color=INDUSTRY_COLORS[industry], label=INDUSTRY_LABELS[industry], edgecolor="none")
        left += vals
    ax.set_yticks(y)
    labels = []
    meta = mapping.set_index("pooled_topic")
    for topic in mat.index:
        row = meta.loc[topic]
        labels.append(f"T{topic}: {wrap_label(row['final_label'], 38)}")
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Number of comments")
    ax.set_title("Topic × industry distribution in the pooled model")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    savefig("fig2_topic_by_industry_distribution")


def figure_3_industry_enrichment(mapping: pd.DataFrame, shares: pd.DataFrame) -> None:
    plot_topics = mapping[mapping["pooled_topic"] >= 0].copy()
    # Use only substantive non-outlier topics for baseline distribution.
    topic_sizes = plot_topics.set_index("pooled_topic")["count"]
    baseline = topic_sizes / topic_sizes.sum()
    share_df = shares[shares["topic"].isin(plot_topics["pooled_topic"])].set_index("topic")[INDUSTRIES]
    share_df = share_df.loc[plot_topics["pooled_topic"]]
    enrich = share_df.div(baseline, axis=0).replace([np.inf, -np.inf], np.nan)
    log2_enrich = np.log2(enrich)
    log2_enrich = log2_enrich.clip(-3, 3)

    fig, ax = plt.subplots(figsize=(8.5, 8.6))
    im = ax.imshow(log2_enrich.T.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(np.arange(len(log2_enrich.index)))
    ax.set_xticklabels([f"T{int(t)}" for t in log2_enrich.index], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(INDUSTRIES)))
    ax.set_yticklabels([INDUSTRY_LABELS[i] for i in INDUSTRIES], fontsize=9)
    ax.set_title("Industry enrichment by pooled topic")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("log2(within-industry topic share / overall topic share)")
    for y in range(len(INDUSTRIES)):
        for x, topic in enumerate(log2_enrich.index):
            val = log2_enrich.iloc[x, y]
            if pd.notna(val) and abs(val) >= 1.0:
                ax.text(x, y, f"{val:.1f}", ha="center", va="center", fontsize=6, color="black")
    savefig("fig3_industry_topic_enrichment_heatmap")


def figure_4_topic0_audit(topic0_belongs: pd.DataFrame, topic0_categories: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.2),
        gridspec_kw={"width_ratios": [0.9, 1.7], "wspace": 0.42},
    )

    belongs_order = ["Yes", "Partly / ambiguous", "No"]
    belongs_df = topic0_belongs.set_index("belongs_to_proposed_theme").reindex(belongs_order).reset_index()
    belongs_colors = {"Yes": "#55a868", "Partly / ambiguous": "#dd8452", "No": "#c44e52"}
    axes[0].bar(belongs_df["belongs_to_proposed_theme"], belongs_df["n"], color=[belongs_colors[x] for x in belongs_df["belongs_to_proposed_theme"]])
    axes[0].set_title("Belongs to proposed Topic 0 theme")
    axes[0].set_ylabel("Audited comments")
    axes[0].tick_params(axis="x", rotation=25)
    for i, row in belongs_df.iterrows():
        axes[0].text(i, row["n"] + 1, f"{int(row['n'])}\n{row['percent_of_audited_sample']:.1f}%", ha="center", fontsize=8)
    axes[0].set_ylim(0, max(belongs_df["n"]) * 1.22)

    cat_df = topic0_categories.sort_values("n", ascending=True)
    axes[1].barh(range(len(cat_df)), cat_df["n"], color="#4c72b0")
    axes[1].set_yticks(range(len(cat_df)))
    axes[1].set_yticklabels([wrap_label(x, 30) for x in cat_df["primary_semantic_category"]], fontsize=8)
    axes[1].set_title("Primary semantic categories among Yes / Partly")
    axes[1].set_xlabel("Audited Yes / Partly comments")
    for i, (_, row) in enumerate(cat_df.iterrows()):
        axes[1].text(row["n"] + 0.4, i, f"{int(row['n'])} ({row['percent_among_yes_and_partly']:.1f}%)", va="center", fontsize=8)
    axes[1].set_xlim(0, max(cat_df["n"]) * 1.25)
    for ax in axes:
        ax.grid(axis="y" if ax is axes[0] else "x", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Manual audit of pooled Topic 0 stratified sample", y=1.02)
    savefig("fig4_topic0_manual_audit")


def figure_5_topic_similarity(mapping: pd.DataFrame, shares: pd.DataFrame) -> None:
    # A compact similarity figure based on industry distribution profiles. This is a diagnostic
    # figure, not a replacement for semantic interpretation.
    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import pdist
    except Exception:
        return

    plot_topics = mapping[mapping["pooled_topic"] >= 0].copy()
    share_df = shares[shares["topic"].isin(plot_topics["pooled_topic"])].set_index("topic")[INDUSTRIES]
    share_df = share_df.loc[plot_topics["pooled_topic"]]
    # Row-normalize industry profiles.
    profile = share_df.div(share_df.sum(axis=1), axis=0).fillna(0)
    distances = pdist(profile.to_numpy(), metric="cosine")
    Z = linkage(distances, method="average")

    labels = [f"T{int(t)}" for t in profile.index]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=45)
    ax.set_title("Topic clustering by industry-composition profile")
    ax.set_ylabel("Cosine distance")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    savefig("fig5_topic_industry_profile_clustering")


def write_captions(mapping: pd.DataFrame) -> None:
    captions = """# Final Pooled BERTopic Figures: Captions and Result Interpretations

These figures are for the locked final pooled BERTopic package. Percentages for the Topic 0 audit refer only to the manually audited stratified sample, not to the exact prevalence in all Topic 0 comments.

## Figure 1. Pooled BERTopic topic size distribution

**Research question answered:** Which pooled topics dominate the cross-industry corpus, and how uneven is the topic structure?

**Caption:** Distribution of comments across the 18 substantive pooled BERTopic topics. Bars are coloured by the final interpretive status assigned in the meta-mapping table. Topic 0 is visibly the largest topic and is therefore reported as a broad, partially noisy theme after manual audit rather than as a narrow topic.

**Result interpretation:** The pooled model is highly uneven. Topic 0 contains 12,438 comments, far larger than any other topic, while several later topics are small and tool- or industry-specific. This supports interpreting the pooled model as a mixture of broad shared concerns and more localised occupational themes.

## Figure 2. Topic by industry distribution

**Research question answered:** Which topics are genuinely shared across industries, and which are concentrated in one occupational domain?

**Caption:** Stacked distribution of industry membership within each pooled topic. Topic labels include the dominant industry and its share. This figure shows that not all pooled topics are cross-industry: some are clearly healthcare-, law-, finance-, or software-specific.

**Result interpretation:** Topic 1 is overwhelmingly healthcare-focused, Topic 4 law-focused, Topic 6 and Topic 14 finance-focused, and several coding-tool topics are software-focused. Topic 0 and Topic 2 are broader, although still software-influenced. The cross-industry result is therefore a two-level structure: shared concerns plus industry-specific AI problems.

## Figure 3. Industry enrichment by pooled topic

**Research question answered:** Which topics are over- or under-represented within each industry relative to their overall pooled frequency?

**Caption:** Heatmap of log2 enrichment scores comparing each topic's share within an industry to its overall share across substantive non-outlier topics. Positive values indicate over-representation in that industry; negative values indicate under-representation.

**Result interpretation:** Healthcare is enriched in patient care, therapy and HIPAA/privacy themes. Law is enriched in paralegal/litigation and contract/proprietary-information themes. Finance is enriched in tax/accounting and audit themes. Software engineering is enriched in coding tools, vibe coding, model-version debates and conceptual model-capability discussions.

## Figure 4. Topic 0 manual audit

**Research question answered:** Is the very large Topic 0 interpretable enough to retain, and what does it mainly contain?

**Caption:** Results of the manual audit of 100 Topic 0 comments, sampled with 25 comments per industry and at most three comments per post. The left panel shows whether comments belonged to the proposed broad theme; the right panel shows primary semantic categories among comments coded Yes or Partly / ambiguous.

**Result interpretation:** Topic 0 is retainable but not clean. In the audited stratified sample, 58% were coded Yes, 16% Partly / ambiguous, and 26% No. Among Yes/Partly comments, job displacement and professional change was the largest subcategory, followed by automation/productivity and capability/limitations. Topic 0 should be described as a broad partially noisy professional-work theme.

## Figure 5. Topic clustering by industry-composition profile

**Research question answered:** Do topics cluster by their industry composition, supporting the shared-versus-industry-specific interpretation?

**Caption:** Hierarchical clustering of pooled topics using their industry-composition profiles. The figure is a diagnostic view of topic composition rather than a semantic similarity model.

**Result interpretation:** Topics with strongly similar industry profiles cluster together, reinforcing the conclusion that the pooled model contains both shared themes and industry-specific theme families. This figure should be used only as a supplementary diagnostic if space allows.
"""
    (FIG_DIR / "figure_captions_and_interpretation.md").write_text(captions, encoding="utf-8")


def main() -> None:
    mapping, counts, shares, topic0_belongs, topic0_categories = load_data()
    figure_1_topic_size(mapping)
    figure_2_topic_industry_distribution(mapping, counts)
    figure_3_industry_enrichment(mapping, shares)
    figure_4_topic0_audit(topic0_belongs, topic0_categories)
    figure_5_topic_similarity(mapping, shares)
    write_captions(mapping)
    manifest = sorted(str(p.relative_to(FIG_DIR)) for p in FIG_DIR.iterdir() if p.is_file())
    (FIG_DIR / "FIGURE_MANIFEST.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print("Figures written to:", FIG_DIR)
    for item in manifest:
        print(item)


if __name__ == "__main__":
    main()
