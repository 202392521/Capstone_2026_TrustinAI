#!/usr/bin/env python3
"""Create publication-ready BERTopic figures without UK/non-UK splitting.

The script uses only the locked human-reviewed industry workbooks and the
locked pooled BERTopic outputs. It does not refit any topic model.
"""

from __future__ import annotations

import colorsys
import json
import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("REPRO_TEMP_DIR", "/tmp")) / "mplconfig_bertopic_no_uk"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(os.environ.get("REPRO_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUTPUTS = ROOT / "outputs"
OUTDIR = OUTPUTS / "bertopic_visualizations_no_uk_FINAL_2026-08-04"
BY_DIR = OUTDIR / "bertopics_by_industry"
ACROSS_DIR = OUTDIR / "bertopics_across_industry"
TABLE_DIR = OUTDIR / "source_tables"

POOLED_DIR = OUTPUTS / "cross_industry_pooled_bertopic_FINAL_2026-07-27"

INDUSTRY_FILES = {
    "finance": Path(
        "MAPPINGS/manual_interpretation_workbooks/"
        "finance_topic_manual_interpretation.xlsx"
    ),
    "healthcare": Path(
        "MAPPINGS/manual_interpretation_workbooks/"
        "healthcare_topic_manual_interpretation.xlsx"
    ),
    "law": Path(
        "MAPPINGS/manual_interpretation_workbooks/"
        "law_topic_manual_interpretation.xlsx"
    ),
    "software_engineering": Path(
        "MAPPINGS/manual_interpretation_workbooks/"
        "software_engineering_topic_manual_interpretation.xlsx"
    ),
}

INDUSTRY_SHEETS = {
    "finance": "Finance Topic Review",
    "healthcare": "Healthcare Topic Review",
    "law": "Law Topic Review",
    "software_engineering": "Software Topic Review",
}

INDUSTRY_LABELS = {
    "finance": "Finance and accounting",
    "healthcare": "Healthcare",
    "law": "Law",
    "software_engineering": "Software engineering and IT",
}

INDUSTRY_SHORT = {
    "finance": "Finance",
    "healthcare": "Healthcare",
    "law": "Law",
    "software_engineering": "Software engineering",
}

INDUSTRY_COLORS = {
    "finance": "#4F8FA8",
    "healthcare": "#78A989",
    "law": "#D48770",
    "software_engineering": "#9A8CB3",
}

STATUS_COLORS = {
    "shared_broad_partly_noisy": "#7896B5",
    "shared_mixed": "#69A79A",
    "shared_software_concentrated": "#9A8CB3",
    "software_concentrated": "#B09DC0",
    "industry_specific": "#D48770",
}

STATUS_LABELS = {
    "shared_broad_partly_noisy": "Shared broad / partly noisy",
    "shared_mixed": "Shared / mixed",
    "shared_software_concentrated": "Shared, software-concentrated",
    "software_concentrated": "Software-concentrated",
    "industry_specific": "Industry-specific",
}

TOP_N_INDUSTRY = 8

DISPLAY_LABEL_OVERRIDES = {
    "Automation of accounting tasks and the future of the profession":
        "Accounting automation and the future of the profession",
    "AI therapy, emotional support and the value of human relationships":
        "AI therapy, emotional support and human relationships",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": "#4B5563",
            "axes.linewidth": 0.8,
            "axes.titleweight": "medium",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "text.color": "#1F2937",
            "axes.labelcolor": "#374151",
        }
    )


def wrap(text: object, width: int) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def lighten(color: str, amount: float) -> str:
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = 1 - amount * (1 - l)
    return mcolors.to_hex(colorsys.hls_to_rgb(h, l, s))


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_industry_workbook(industry: str) -> pd.DataFrame:
    df = pd.read_excel(
        INDUSTRY_FILES[industry],
        sheet_name=INDUSTRY_SHEETS[industry],
        header=8,
    )
    required = [
        "Topic ID",
        "Count",
        "Researcher-assigned label",
        "Decision",
        "Merge target topic",
        "Primary meta-theme",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{industry} workbook missing columns: {missing}")
    df = df[required].copy()
    df["Topic ID"] = pd.to_numeric(df["Topic ID"], errors="coerce")
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce")
    df = df.dropna(subset=["Topic ID", "Count"]).copy()
    df["Topic ID"] = df["Topic ID"].astype(int)
    df["Count"] = df["Count"].astype(int)
    return df


def reporting_topics(industry: str) -> tuple[pd.DataFrame, dict[str, int | float]]:
    raw = load_industry_workbook(industry)
    stats = {
        "modelled_comments": int(raw["Count"].sum()),
        "outlier_comments": int(raw.loc[raw["Topic ID"].eq(-1), "Count"].sum()),
    }
    kept = raw[raw["Decision"].isin(["Keep", "Merge"])].copy()
    labels = raw.set_index("Topic ID")["Researcher-assigned label"].to_dict()
    meta = raw.set_index("Topic ID")["Primary meta-theme"].to_dict()
    kept["reporting_topic"] = np.where(
        kept["Decision"].eq("Merge"),
        pd.to_numeric(kept["Merge target topic"], errors="coerce"),
        kept["Topic ID"],
    )
    if kept["reporting_topic"].isna().any():
        raise ValueError(f"{industry} has a Merge row without a valid target")
    kept["reporting_topic"] = kept["reporting_topic"].astype(int)
    grouped = kept.groupby("reporting_topic", as_index=False)["Count"].sum()
    grouped["label"] = grouped["reporting_topic"].map(labels)
    grouped["meta_theme"] = grouped["reporting_topic"].map(meta)
    grouped = grouped.sort_values("Count", ascending=False).reset_index(drop=True)
    grouped["share_reporting_comments"] = grouped["Count"] / grouped["Count"].sum()
    stats["reporting_comments"] = int(grouped["Count"].sum())
    stats["reporting_topics"] = int(len(grouped))
    stats["outlier_rate"] = stats["outlier_comments"] / stats["modelled_comments"]
    return grouped, stats


def build_industry_source_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    summaries = []
    for industry in INDUSTRY_FILES:
        topics, stats = reporting_topics(industry)
        topics.insert(0, "industry", industry)
        frames.append(topics)
        summaries.append({"industry": industry, **stats})
    all_topics = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    return all_topics, summary


def figure_industry_topics(
    all_topics: pd.DataFrame,
    summary: pd.DataFrame,
    no_title: bool = False,
) -> None:
    order = ["finance", "healthcare", "law", "software_engineering"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)

    for ax, industry in zip(axes.flat, order):
        data = all_topics[all_topics["industry"].eq(industry)].copy()
        top = data.head(TOP_N_INDUSTRY).copy()
        top["label"] = top["label"].replace(DISPLAY_LABEL_OVERRIDES)
        top = top.sort_values("share_reporting_comments", ascending=True).reset_index(drop=True)
        remainder_n = int(data.iloc[TOP_N_INDUSTRY:]["Count"].sum())
        if remainder_n:
            remainder = pd.DataFrame(
                {
                    "industry": [industry],
                    "reporting_topic": [-99],
                    "Count": [remainder_n],
                    "label": [f"Remaining retained topics ({len(data) - TOP_N_INDUSTRY})"],
                    "meta_theme": ["Other"],
                    "share_reporting_comments": [remainder_n / data["Count"].sum()],
                }
            )
            # barh draws index 0 at the bottom. Prepending the aggregate keeps it
            # out of the substantive-topic ranking and fixes it at the panel base.
            top = pd.concat([remainder, top], ignore_index=True)

        n = len(top)
        base = INDUSTRY_COLORS[industry]
        shades = [lighten(base, 0.48 + 0.48 * (i / max(n - 1, 1))) for i in range(n)]
        shades = [
            "#C9CDD2" if label.startswith("Remaining retained") else color
            for label, color in zip(top["label"], shades)
        ]

        y = np.arange(n)
        ax.barh(y, top["share_reporting_comments"] * 100, color=shades, edgecolor="white", linewidth=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([wrap(x, 39) for x in top["label"]], fontsize=8.2)
        ax.set_xlim(0, 45)
        ax.set_xticks(np.arange(0, 46, 5))
        ax.set_xlabel("Share of retained reporting-topic comments (%)")
        info = summary[summary["industry"].eq(industry)].iloc[0]
        ax.set_title(
            f"{INDUSTRY_LABELS[industry]}\n"
            f"{int(info['reporting_topics'])} retained topics | "
            f"n = {int(info['reporting_comments']):,} retained-topic comments",
            pad=12,
        )
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        for i, value in enumerate(top["share_reporting_comments"] * 100):
            ax.text(value + 0.55, i, f"{value:.1f}%", va="center", fontsize=8.2, color="#374151")

    if not no_title:
        fig.suptitle("Largest manually reviewed BERTopic themes by industry", fontsize=17, fontweight="medium")
    fig.text(
        0.5,
        0.012,
        "Percentages are calculated among comments assigned to retained reporting topics. "
        "Topic -1 and manually excluded incoherent or out-of-scope topics are omitted. "
        "Remaining retained topics are grouped for display. Comments containing fewer than "
        "30 cleaned characters were excluded before modelling.",
        ha="center",
        fontsize=8.7,
        color="#4B5563",
        wrap=True,
    )
    # Reserve a dedicated lower margin so the caption never collides with the
    # lower-panel x-axis titles in PNG/PDF exports.
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.065, 1.0, 0.97 if no_title else 0.92))
    suffix = "_no_title" if no_title else ""
    save_figure(fig, BY_DIR, f"fig1_largest_manually_reviewed_topics_by_industry{suffix}")


def load_pooled() -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = pd.read_csv(POOLED_DIR / "cross_industry_pooled_topic_meta_mapping_FINAL.csv")
    counts = pd.read_csv(POOLED_DIR / "final_model/pooled_topic_by_industry_counts.csv")
    mapping = mapping[mapping["pooled_topic"].ge(0)].copy()
    counts = counts[counts["topic"].ge(0)].copy()
    return mapping, counts


def pooled_label(row: pd.Series, width: int = 42) -> str:
    return f"T{int(row['pooled_topic'])}: {wrap(row['final_label'], width)}"


def figure_pooled_sizes(mapping: pd.DataFrame) -> None:
    data = mapping.sort_values("count", ascending=True).copy()
    colors = [STATUS_COLORS.get(x, "#AEB4BA") for x in data["cross_industry_status"]]
    fig, ax = plt.subplots(figsize=(12.5, 10.5), constrained_layout=True)
    y = np.arange(len(data))
    ax.barh(y, data["count"], color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([pooled_label(row) for _, row in data.iterrows()], fontsize=8.3)
    ax.set_xlabel("Comments assigned to topic")
    ax.set_title("Topic prevalence in the cross-industry pooled BERTopic model", pad=14)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    max_count = data["count"].max()
    for i, value in enumerate(data["count"]):
        ax.text(value + max_count * 0.008, i, f"{int(value):,}", va="center", fontsize=8)
    handles = []
    for status in data["cross_industry_status"].drop_duplicates():
        handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markerfacecolor=STATUS_COLORS.get(status, "#AEB4BA"),
                markeredgecolor="none",
                markersize=9,
                label=STATUS_LABELS.get(status, status.replace("_", " ").title()),
            )
        )
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.2)
    fig.text(
        0.5,
        -0.015,
        "Substantive topics only. Topic -1 (5,478 comments; 15.1% of modelled comments) is excluded from this figure.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    save_figure(fig, ACROSS_DIR, "fig1_pooled_topic_prevalence")


def figure_pooled_industry_composition(mapping: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    order = mapping.sort_values("count", ascending=False)["pooled_topic"].astype(int).tolist()
    industries = ["finance", "healthcare", "law", "software_engineering"]
    matrix = counts.set_index("topic").loc[order, industries]
    pct = matrix.div(matrix.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(13.5, 10.5), constrained_layout=True)
    y = np.arange(len(order))
    left = np.zeros(len(order))
    for industry in industries:
        vals = pct[industry].to_numpy()
        ax.barh(
            y,
            vals,
            left=left,
            color=INDUSTRY_COLORS[industry],
            edgecolor="white",
            linewidth=0.7,
            label=INDUSTRY_SHORT[industry],
        )
        for i, (start, value) in enumerate(zip(left, vals)):
            if value >= 8:
                ax.text(start + value / 2, i, f"{value:.0f}%", ha="center", va="center", fontsize=7.4, color="white")
        left += vals

    meta = mapping.set_index("pooled_topic")
    ax.set_yticks(y)
    ax.set_yticklabels([f"T{topic}: {wrap(meta.loc[topic, 'final_label'], 40)}" for topic in order], fontsize=8.2)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Industry composition within topic (%)")
    # Reserve a dedicated line between the title and plot for the four-industry legend.
    ax.set_title("Industry composition of cross-industry pooled BERTopic themes", pad=46)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.002), frameon=False, fontsize=8.5)
    fig.text(
        0.5,
        -0.015,
        "Each bar totals 100%. Percent labels are shown for segments representing at least 8% of a topic.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    save_figure(fig, ACROSS_DIR, "fig2_pooled_topic_industry_composition_100pct")

    long = pct.reset_index(names="pooled_topic").melt(
        id_vars="pooled_topic", var_name="industry", value_name="percent_within_topic"
    )
    long = long.merge(mapping[["pooled_topic", "final_label", "cross_industry_status"]], on="pooled_topic", how="left")
    return long


def figure_pooled_enrichment(mapping: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    industries = ["finance", "healthcare", "law", "software_engineering"]
    order = mapping.sort_values("count", ascending=False)["pooled_topic"].astype(int).tolist()
    matrix = counts.set_index("topic").loc[order, industries].astype(float)
    within_industry = matrix.div(matrix.sum(axis=0), axis=1)
    overall = matrix.sum(axis=1) / matrix.values.sum()
    log2_enrichment = np.log2(within_industry.div(overall, axis=0)).replace([np.inf, -np.inf], np.nan)
    clipped = log2_enrichment.clip(-3, 3)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "muted_diverging",
        ["#B56F66", "#E7B8A7", "#F5F2EC", "#A9CDBF", "#397E7B"],
    )
    fig, ax = plt.subplots(figsize=(10.8, 11.5), constrained_layout=True)
    im = ax.imshow(clipped.to_numpy(), aspect="auto", cmap=cmap, vmin=-3, vmax=3)
    ax.set_xticks(np.arange(len(industries)))
    ax.set_xticklabels([INDUSTRY_SHORT[x] for x in industries], fontsize=9)
    meta = mapping.set_index("pooled_topic")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([f"T{t}: {wrap(meta.loc[t, 'final_label'], 39)}" for t in order], fontsize=8.1)
    ax.set_title("Relative industry enrichment of pooled BERTopic themes", pad=14)
    for i in range(len(order)):
        for j in range(len(industries)):
            value = log2_enrichment.iloc[i, j]
            if pd.notna(value):
                color = "white" if abs(float(value)) >= 1.7 else "#374151"
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7.2, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log2(observed topic share / expected share)")
    cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    fig.text(
        0.5,
        -0.015,
        "Positive values indicate over-representation within an industry. Colour is capped at +/-3; cell labels show uncapped log2 enrichment.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    save_figure(fig, ACROSS_DIR, "fig3_pooled_topic_industry_enrichment")

    long = log2_enrichment.reset_index(names="pooled_topic").melt(
        id_vars="pooled_topic", var_name="industry", value_name="log2_enrichment"
    )
    long = long.merge(mapping[["pooled_topic", "final_label"]], on="pooled_topic", how="left")
    return long


def write_readme(industry_summary: pd.DataFrame) -> None:
    lines = [
        "# BERTopic visualisations without UK/non-UK splitting",
        "",
        "Status: FINAL figure package generated 2026-08-04.",
        "",
        "These figures deliberately remove the UK-focused/non-UK comparison. They use all eligible comments in the four locked industry-specific BERTopic models and the locked cross-industry pooled BERTopic model.",
        "",
        "No BERTopic model was refitted and no topic label or mapping was changed.",
        "",
        "## BERTopics by industry",
        "",
        "`fig1_largest_verified_topics_by_industry` compares the largest researcher-verified reporting topics within each industry. Topic -1 and topics marked Exclude are omitted. Existing Merge decisions are applied before shares are calculated. Shares use retained reporting-topic comments within each industry as the denominator.",
        "",
        "## BERTopics across industry",
        "",
        "- `fig1_pooled_topic_prevalence`: size of each substantive pooled topic.",
        "- `fig2_pooled_topic_industry_composition_100pct`: industry composition within every pooled topic; each bar sums to 100%.",
        "- `fig3_pooled_topic_industry_enrichment`: log2 observed/expected enrichment, with colour capped at +/-3 and uncapped values printed in cells.",
        "",
        "## Industry-model audit counts",
        "",
    ]
    for _, row in industry_summary.iterrows():
        lines.append(
            f"- {INDUSTRY_LABELS[row['industry']]}: {int(row['modelled_comments']):,} modelled comments; "
            f"{int(row['outlier_comments']):,} Topic -1 comments ({row['outlier_rate']:.1%}); "
            f"{int(row['reporting_topics'])} retained reporting topics after merge/exclude decisions."
        )
    lines.extend(
        [
            "",
            "## Version note",
            "",
            "The final human-reviewed software-engineering workbook corresponds to `min_topic_size=50, min_samples=5` (18,813 modelled comments; Topic -1=7,220). The older master-lock text naming `min35_ms5` is inconsistent with that workbook. This figure package follows the later human-reviewed workbook, which the researcher identified as the final retained interpretation file.",
            "",
            "Historical UK/non-UK visualisations remain archived in their original folders but are not part of this package or the final reporting design.",
        ]
    )
    (OUTDIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    setup_style()
    BY_DIR.mkdir(parents=True, exist_ok=True)
    ACROSS_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    industry_topics, industry_summary = build_industry_source_table()
    industry_topics.to_csv(TABLE_DIR / "industry_reporting_topics_after_manual_decisions.csv", index=False)
    industry_summary.to_csv(TABLE_DIR / "industry_model_summary.csv", index=False)
    figure_industry_topics(industry_topics, industry_summary)

    mapping, counts = load_pooled()
    mapping.to_csv(TABLE_DIR / "pooled_final_topic_mapping_substantive.csv", index=False)
    counts.to_csv(TABLE_DIR / "pooled_topic_by_industry_counts_substantive.csv", index=False)
    figure_pooled_sizes(mapping)
    composition = figure_pooled_industry_composition(mapping, counts)
    composition.to_csv(TABLE_DIR / "pooled_topic_industry_composition_percent.csv", index=False)
    enrichment = figure_pooled_enrichment(mapping, counts)
    enrichment.to_csv(TABLE_DIR / "pooled_topic_industry_log2_enrichment.csv", index=False)

    write_readme(industry_summary)
    manifest = {
        "created": "2026-08-04",
        "uk_split_used": False,
        "models_refitted": False,
        "industry_workbooks": {k: str(v) for k, v in INDUSTRY_FILES.items()},
        "pooled_model_folder": str(POOLED_DIR),
        "figures": sorted(str(p.relative_to(OUTDIR)) for p in OUTDIR.rglob("*.png")),
        "note": "All figures use locked all-comment topic models; no UK/non-UK split is present.",
    }
    (OUTDIR / "FIGURE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Figures written to {OUTDIR}")
    for path in sorted(OUTDIR.rglob("*.png")):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
