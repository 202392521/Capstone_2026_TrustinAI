#!/usr/bin/env python3
"""Generate dissertation-ready visualizations from frozen analysis tables."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("outputs")
OUT = ROOT / "dissertation_visualizations_2026-07-31"

STANCE_DIR = ROOT / "stance_analysis_frozen_prompt_v2_stratified_sample_2000_2026-07-28"
LEX_DIR = ROOT / "explicit_boundary_lexical_analysis_stratified2000_2026-07-31"
POOLED_DIR = ROOT / "cross_industry_pooled_bertopic_FINAL_2026-07-27"
PRED_DIR = ROOT / "frozen_prompt_v2_stratified_sample_2000_2026-07-24"


INDUSTRY_ORDER = ["finance", "healthcare", "law", "software_engineering"]
INDUSTRY_LABELS = {
    "finance": "Finance",
    "healthcare": "Healthcare",
    "law": "Law",
    "software_engineering": "Software engineering",
}
ATT_ORDER = ["positive", "mixed_or_ambivalent", "neutral_descriptive", "negative"]
ATT_LABELS = {
    "positive": "Positive",
    "mixed_or_ambivalent": "Mixed / ambivalent",
    "neutral_descriptive": "Neutral / descriptive",
    "negative": "Negative",
}
ATT_COLORS = {
    "positive": "#72b7c2",
    "mixed_or_ambivalent": "#e8bd5a",
    "neutral_descriptive": "#aeb9c4",
    "negative": "#c96f62",
}
SERIES = {
    "teal": "#4f9699",
    "teal_light": "#a8d2d1",
    "gold": "#e8bd5a",
    "rose": "#c96f62",
    "grey": "#aeb9c4",
    "green": "#8bbf88",
    "blue": "#8eb5d6",
}


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def nice_industry(x: str) -> str:
    return INDUSTRY_LABELS.get(x, str(x).replace("_", " ").title())


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4a4a4a",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "text.color": "#222222",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.titleweight": "regular",
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.8,
        }
    )


def fig_trust_summary() -> None:
    df = pd.read_csv(STANCE_DIR / "01_industry_substantive_trust_and_boundary_summary.csv")
    df["industry_label"] = df["industry"].map(nice_industry)
    df = df.set_index("industry").loc[INDUSTRY_ORDER].reset_index()
    x = np.arange(len(df))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.bar(
        x - width / 2,
        df["substantive_trust_rate"],
        width,
        color=SERIES["teal"],
        label="Substantive trust-related",
    )
    ax.bar(
        x + width / 2,
        df["boundary_rate_among_substantive_trust"],
        width,
        color=SERIES["gold"],
        label="Explicit boundary among trust-related",
    )
    ax.set_ylim(0, 0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(df["industry_label"])
    ax.set_ylabel("Share of comments")
    ax.set_title("Trust-related discourse and explicit trust boundaries by industry")
    ax.yaxis.set_major_formatter(lambda v, pos: f"{v:.0%}")
    ax.grid(axis="y")
    for i, row in df.iterrows():
        ax.text(i - width / 2, row["substantive_trust_rate"] + 0.018, pct(row["substantive_trust_rate"]), ha="center", fontsize=9)
        ax.text(i + width / 2, row["boundary_rate_among_substantive_trust"] + 0.018, pct(row["boundary_rate_among_substantive_trust"]), ha="center", fontsize=9)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    savefig(fig, "fig01_trust_discourse_and_boundary_by_industry")
    df.to_csv(OUT / "fig01_source_trust_discourse_and_boundary_by_industry.csv", index=False)


def fig_stance_composition() -> None:
    df = pd.read_csv(STANCE_DIR / "02_attitude_distribution_among_substantive_trust.csv")
    pivot = df.pivot(index="industry", columns="label", values="rate").reindex(INDUSTRY_ORDER).fillna(0)
    counts = df.pivot(index="industry", columns="label", values="n").reindex(INDUSTRY_ORDER).fillna(0)
    denominators = df.groupby("industry")["substantive_trust_denominator"].max().reindex(INDUSTRY_ORDER)
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    for att in ATT_ORDER:
        vals = pivot.get(att, pd.Series(0, index=pivot.index)).to_numpy()
        ax.bar(x, vals, bottom=bottom, color=ATT_COLORS[att], label=ATT_LABELS[att], width=0.72)
        for i, v in enumerate(vals):
            if v >= 0.075:
                ax.text(i, bottom[i] + v / 2, pct(v), ha="center", va="center", fontsize=9, color="#222222")
        bottom += vals
    labels = [f"{nice_industry(ind)}\nn={int(denominators.loc[ind])}" for ind in pivot.index]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of substantive trust-related comments")
    ax.set_title("Attitudinal stance among substantive trust-related comments by industry")
    ax.yaxis.set_major_formatter(lambda v, pos: f"{v:.0%}")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
    fig.tight_layout()
    savefig(fig, "fig02_stance_composition_substantive_trust_by_industry")
    pivot.to_csv(OUT / "fig02_source_stance_rates.csv")
    counts.to_csv(OUT / "fig02_source_stance_counts.csv")


def fig_trust_vs_other_stance() -> None:
    all_df = pd.read_csv(PRED_DIR / "all_comments_v2_annotated.csv", low_memory=False)
    all_df = all_df[all_df["industry"].isin(INDUSTRY_ORDER)].copy()
    all_df["trust_group"] = np.where(
        all_df["gpt_has_substantive_trust_content"].astype(str).str.lower().eq("yes"),
        "Substantive trust-related",
        "Other AI-related",
    )
    all_df["attitude"] = all_df["gpt_human_attitude"].astype(str).str.strip().str.lower()
    all_df = all_df[all_df["attitude"].isin(ATT_ORDER)].copy()
    grouped = all_df.groupby(["industry", "trust_group", "attitude"]).size().rename("n").reset_index()
    denom = grouped.groupby(["industry", "trust_group"])["n"].sum().rename("denominator").reset_index()
    grouped = grouped.merge(denom, on=["industry", "trust_group"], how="left")
    grouped["rate"] = grouped["n"] / grouped["denominator"]
    fig, ax = plt.subplots(figsize=(12.2, 6.3))
    groups = []
    xlabels = []
    x_positions = []
    for gi, industry in enumerate(INDUSTRY_ORDER):
        for tg in ["Substantive trust-related", "Other AI-related"]:
            groups.append((industry, tg))
            x_positions.append(gi * 2.45 + (0 if tg.startswith("Substantive") else 0.9))
            xlabels.append("Substantive\ntrust-related" if tg.startswith("Substantive") else "Other\nAI-related")
    x = np.array(x_positions)
    bottom = np.zeros(len(groups))
    for att in ATT_ORDER:
        vals = []
        for industry, tg in groups:
            sub = grouped[(grouped["industry"] == industry) & (grouped["trust_group"] == tg) & (grouped["attitude"] == att)]
            vals.append(float(sub["rate"].iloc[0]) if not sub.empty else 0)
        vals = np.array(vals)
        ax.bar(x, vals, bottom=bottom, color=ATT_COLORS[att], label=ATT_LABELS[att], width=0.72)
        for i, v in enumerate(vals):
            if v >= 0.10:
                ax.text(i, bottom[i] + v / 2, pct(v), ha="center", va="center", fontsize=8.3)
        bottom += vals
    for boundary in [1.68, 4.13, 6.58]:
        ax.axvline(boundary, color="#dddddd", lw=1)
    for gi, industry in enumerate(INDUSTRY_ORDER):
        center = gi * 2.45 + 0.45
        ax.text(center, -0.16, nice_industry(industry), ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.75, x_positions[-1] + 0.75)
    ax.set_ylabel("Share of comments in group")
    ax.set_title("Attitudinal composition of substantive trust-related and other AI-related comments, by industry")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8.8)
    ax.yaxis.set_major_formatter(lambda v, pos: f"{v:.0%}")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4)
    fig.tight_layout()
    savefig(fig, "fig03_stance_composition_trust_related_vs_other_ai")
    grouped.to_csv(OUT / "fig03_source_stance_trust_vs_other_ai.csv", index=False)


def fig_boundary_by_stance() -> None:
    tp = pd.read_csv(STANCE_DIR / "trust_positive_corpus_for_stance_analysis.csv", low_memory=False)
    tp["attitude"] = tp["gpt_human_attitude"].astype(str).str.strip().str.lower()
    tp = tp[tp["attitude"].isin(ATT_ORDER)].copy()
    tp["boundary_yes"] = tp["gpt_has_explicit_trust_boundary"].astype(str).str.lower().eq("yes")
    rows = []
    for att, sub in tp.groupby("attitude"):
        n = len(sub)
        yes = int(sub["boundary_yes"].sum())
        p = yes / n if n else 0
        se = (p * (1 - p) / n) ** 0.5 if n else 0
        rows.append({"attitude": att, "yes": yes, "n": n, "rate": p, "ci_low": max(0, p - 1.96 * se), "ci_high": min(1, p + 1.96 * se)})
    df = pd.DataFrame(rows).sort_values("rate", ascending=False)
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.errorbar(
        df["rate"],
        y,
        xerr=[df["rate"] - df["ci_low"], df["ci_high"] - df["rate"]],
        fmt="o",
        color="#2f7f83",
        ecolor="#93bec0",
        elinewidth=2,
        capsize=4,
        markersize=8,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([ATT_LABELS[a] for a in df["attitude"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Comments containing an explicit trust boundary (%)")
    ax.set_title("Percentage of substantive-trust comments containing an explicit trust boundary,\nby attitudinal stance")
    ax.xaxis.set_major_formatter(lambda v, pos: f"{v:.0%}")
    ax.grid(axis="x")
    for i, row in enumerate(df.itertuples()):
        x = min(row.rate + 0.035, 0.96)
        ax.text(x, i - 0.16, f"{row.rate:.1%} ({row.yes}/{row.n})", va="center", fontsize=9)
    fig.tight_layout()
    savefig(fig, "fig04_explicit_boundary_by_attitudinal_stance")
    df.to_csv(OUT / "fig04_source_explicit_boundary_by_stance.csv", index=False)


def fig_meta_theme_heatmap() -> None:
    tp = pd.read_csv(STANCE_DIR / "trust_positive_corpus_for_stance_analysis.csv", low_memory=False)
    docs = pd.read_csv(POOLED_DIR / "final_model/pooled_document_topic_assignments.csv", usecols=["comment_id", "pooled_topic"], low_memory=False)
    mapping = pd.read_csv(POOLED_DIR / "cross_industry_pooled_topic_meta_mapping_FINAL.csv")
    mapping = mapping[["pooled_topic", "broader_meta_theme", "cross_industry_status"]]
    joined = tp.merge(docs, on="comment_id", how="left").merge(mapping, on="pooled_topic", how="left")
    joined["broader_meta_theme"] = joined["broader_meta_theme"].fillna("Missing / unmapped")
    joined = joined[joined["broader_meta_theme"].ne("Not a substantive analytical theme")].copy()
    counts = joined.groupby(["industry", "broader_meta_theme"]).size().rename("n").reset_index()
    totals = counts.groupby("industry")["n"].sum().rename("denominator").reset_index()
    counts = counts.merge(totals, on="industry", how="left")
    counts["rate"] = counts["n"] / counts["denominator"]
    theme_order = counts.groupby("broader_meta_theme")["n"].sum().sort_values(ascending=False).index.tolist()[:8]
    heat = (
        counts[counts["broader_meta_theme"].isin(theme_order)]
        .pivot(index="industry", columns="broader_meta_theme", values="rate")
        .reindex(INDUSTRY_ORDER)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(13.5, 5.1))
    im = ax.imshow(heat.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=max(0.01, heat.values.max()))
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels([nice_industry(x) for x in heat.index])
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(["\n".join(textwrap.wrap(c, 17)) for c in heat.columns], rotation=25, ha="right", fontsize=8.3)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j]
            if v >= 0.04:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8.5, color="#111111")
    ax.set_title("Meta-theme composition of substantive trust-related discourse by industry")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Share within industry")
    fig.tight_layout()
    savefig(fig, "fig05_industry_meta_theme_composition_heatmap")
    counts.to_csv(OUT / "fig05_source_industry_meta_theme_composition.csv", index=False)


def fig_boundary_keyness() -> None:
    key = pd.read_csv(LEX_DIR / "keyness_boundary_positive_vs_negative_1gram.csv")
    top = key.sort_values("log_odds_z", ascending=False).head(16).sort_values("log_odds_z")
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.barh(top["term"], top["log_odds_z"], color=SERIES["green"])
    ax.set_xlabel("Weighted log-odds z-score")
    ax.set_title("Terms disproportionately associated with explicit trust-boundary comments")
    ax.grid(axis="x")
    fig.tight_layout()
    savefig(fig, "fig06_explicit_boundary_positive_keyness_terms")
    top.to_csv(OUT / "fig06_source_explicit_boundary_keyness_terms.csv", index=False)


def fig_pooled_topic_industry_composition() -> None:
    topic_counts = pd.read_csv(POOLED_DIR / "final_model/pooled_topic_by_industry_counts.csv")
    mapping = pd.read_csv(POOLED_DIR / "cross_industry_pooled_topic_meta_mapping_FINAL.csv")
    topic_col = "topic" if "topic" in topic_counts.columns else "pooled_topic"
    mapping_topic_col = "pooled_topic"
    merged = topic_counts.merge(mapping[["pooled_topic", "final_label", "count", "cross_industry_status"]], left_on=topic_col, right_on=mapping_topic_col, how="left")
    merged = merged[merged[topic_col].ne(-1)].copy()
    merged = merged.sort_values("count", ascending=False).head(10)
    ind_cols = [c for c in INDUSTRY_ORDER if c in merged.columns]
    shares = merged[ind_cols].div(merged[ind_cols].sum(axis=1), axis=0).fillna(0)
    labels = [f"T{int(t)}: " + "\n".join(textwrap.wrap(str(label), 24)) for t, label in zip(merged[topic_col], merged["final_label"])]
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    y = np.arange(len(merged))
    left = np.zeros(len(merged))
    colors = [SERIES["teal"], SERIES["gold"], SERIES["rose"], SERIES["blue"]]
    for col, color in zip(ind_cols, colors):
        vals = shares[col].to_numpy()
        ax.barh(y, vals, left=left, label=nice_industry(col), color=color)
        for i, v in enumerate(vals):
            if v >= 0.12:
                ax.text(left[i] + v / 2, i, pct(v), ha="center", va="center", fontsize=8)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Industry share within topic")
    ax.set_title("Industry composition of the ten largest pooled BERTopic themes")
    ax.xaxis.set_major_formatter(lambda v, pos: f"{v:.0%}")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4)
    fig.tight_layout()
    savefig(fig, "fig07_largest_pooled_topics_industry_composition")
    merged.to_csv(OUT / "fig07_source_largest_pooled_topics_industry_composition.csv", index=False)


def write_readme() -> None:
    text = f"""# Dissertation Visualizations

Generated on 2026-07-31 from frozen analysis outputs.

## Figures

1. `fig01_trust_discourse_and_boundary_by_industry`: compares substantive trust-related discourse and explicit boundary rates across industries.
2. `fig02_stance_composition_substantive_trust_by_industry`: 100% stacked stance composition among substantive trust-related comments.
3. `fig03_stance_composition_trust_related_vs_other_ai`: compares stance in substantive trust-related vs other AI-related comments.
4. `fig04_explicit_boundary_by_attitudinal_stance`: shows how often each stance category contains an explicit trust boundary.
5. `fig05_industry_meta_theme_composition_heatmap`: shows which pooled meta-themes dominate each industry's substantive trust discourse.
6. `fig06_explicit_boundary_positive_keyness_terms`: lexical terms disproportionately associated with explicit boundary-positive comments.
7. `fig07_largest_pooled_topics_industry_composition`: industry composition of the largest pooled BERTopic themes.

Each figure is saved as `.png` and `.pdf`, with source CSV tables in the same folder.
"""
    (OUT / "README_visualizations.md").write_text(text, encoding="utf-8")


def main() -> int:
    ensure_out()
    set_style()
    fig_trust_summary()
    fig_stance_composition()
    fig_trust_vs_other_stance()
    fig_boundary_by_stance()
    fig_meta_theme_heatmap()
    fig_boundary_keyness()
    fig_pooled_topic_industry_composition()
    write_readme()
    manifest = {
        "output_dir": str(OUT),
        "figures": sorted([p.name for p in OUT.glob("*.png")]),
    }
    (OUT / "visualization_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated {len(manifest['figures'])} PNG figures in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
