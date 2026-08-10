#!/usr/bin/env python3
"""Measure and plot pooled-topic sharedness across industries using JSD."""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize


INDUSTRY_ORDER = [
    "finance",
    "healthcare",
    "law",
    "software_engineering",
]

INDUSTRY_LABELS = {
    "finance": "Finance and accounting",
    "healthcare": "Healthcare",
    "law": "Law",
    "software_engineering": "Software engineering and IT",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assignments",
        type=Path,
        default=base_dir / "final_model" / "pooled_document_topic_assignments.csv",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=base_dir / "cross_industry_pooled_topic_meta_mapping_FINAL.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=base_dir / "figures")
    parser.add_argument("--no-title", action="store_true")
    return parser.parse_args()


def js_divergence_bits(p: np.ndarray, q: np.ndarray) -> float:
    """Return Jensen-Shannon divergence in bits, bounded between 0 and 1."""
    midpoint = 0.5 * (p + q)

    def kl_divergence(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))

    return 0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint)


def calculate_sharedness(assignments: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = {"industry", "pooled_topic"} - set(assignments.columns)
    if missing:
        raise ValueError(f"Assignments are missing required columns: {sorted(missing)}")

    observed_industries = set(assignments["industry"].dropna().unique())
    unexpected = observed_industries - set(INDUSTRY_ORDER)
    if unexpected:
        raise ValueError(f"Unexpected industry labels: {sorted(unexpected)}")

    baseline_counts = assignments["industry"].value_counts().reindex(INDUSTRY_ORDER, fill_value=0)
    baseline = baseline_counts / baseline_counts.sum()

    topic_industry = pd.crosstab(assignments["pooled_topic"], assignments["industry"])
    topic_industry = topic_industry.reindex(columns=INDUSTRY_ORDER, fill_value=0)

    mapping_lookup = mapping.set_index("pooled_topic")
    rows: list[dict[str, object]] = []
    for topic, counts in topic_industry.iterrows():
        topic = int(topic)
        if topic == -1:
            continue
        if topic not in mapping_lookup.index:
            raise ValueError(f"Topic {topic} has assignments but no final mapping row")

        distribution = counts / counts.sum()
        jsd = js_divergence_bits(distribution.to_numpy(float), baseline.to_numpy(float))
        dominant_industry = str(distribution.idxmax())
        map_row = mapping_lookup.loc[topic]
        row: dict[str, object] = {
            "pooled_topic": topic,
            "final_label": str(map_row["final_label"]),
            "broader_meta_theme": str(map_row["broader_meta_theme"]),
            "cross_industry_status": str(map_row["cross_industry_status"]),
            "topic_count": int(counts.sum()),
            "js_divergence_bits": jsd,
            "dominant_industry": dominant_industry,
            "dominant_industry_label": INDUSTRY_LABELS[dominant_industry],
            "dominant_industry_share": float(distribution.max()),
        }
        for industry in INDUSTRY_ORDER:
            row[f"topic_share_{industry}"] = float(distribution[industry])
            row[f"baseline_share_{industry}"] = float(baseline[industry])
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(
        ["js_divergence_bits", "pooled_topic"], ascending=[True, True]
    ).reset_index(drop=True)
    result.insert(0, "sharedness_rank", np.arange(1, len(result) + 1))

    mapped_counts = mapping.loc[mapping["pooled_topic"] != -1, "count"].sum()
    assignment_counts = int((assignments["pooled_topic"] != -1).sum())
    if int(mapped_counts) != assignment_counts:
        raise ValueError(
            f"Substantive-topic count mismatch: mapping={mapped_counts}, assignments={assignment_counts}"
        )
    return result, baseline


def wrap_label(topic: int, label: str, width: int = 49) -> str:
    wrapped = textwrap.wrap(label, width=width, break_long_words=False, break_on_hyphens=False)
    if not wrapped:
        return f"T{topic}"
    wrapped[0] = f"T{topic}  {wrapped[0]}"
    return "\n".join(wrapped)


def make_figure(
    data: pd.DataFrame,
    baseline: pd.Series,
    output_dir: Path,
    no_title: bool = False,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 17,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.3,
        }
    )

    values = data["js_divergence_bits"].to_numpy(float)
    labels = [
        wrap_label(int(row.pooled_topic), str(row.final_label))
        for row in data.itertuples(index=False)
    ]
    y = np.arange(len(data))

    cmap = LinearSegmentedColormap.from_list(
        "sharedness",
        ["#E5EFF7", "#C5DAEA", "#91B9D4", "#568FB6", "#245F89"],
    )
    norm = Normalize(vmin=0.0, vmax=max(0.70, float(values.max())))
    colors = cmap(norm(values))

    fig, ax = plt.subplots(figsize=(15.5, 12.6), constrained_layout=False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bars = ax.barh(y, values, color=colors, height=0.64, edgecolor="none")

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.72)
    ax.set_xticks(np.arange(0, 0.71, 0.1))
    ax.set_xlabel("Jensen–Shannon divergence from pooled-corpus industry baseline (bits)", labelpad=10)
    ax.xaxis.grid(True, color="#DCE2E5", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#7B858B")
    ax.tick_params(axis="y", length=0, pad=8, colors="#28343B")
    ax.tick_params(axis="x", colors="#4E5B61")

    for bar, value in zip(bars, values):
        ax.text(
            min(value + 0.009, 0.704),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=9.2,
            color="#27343A",
            fontweight="semibold",
        )

    if not no_title:
        ax.set_title(
            "Cross-industry distribution of pooled BERTopic themes",
            loc="left",
            pad=36,
            color="#202A30",
            fontweight="semibold",
        )
        ax.text(
            0,
            1.026,
            "Lower divergence indicates a topic whose industry composition more closely matches the pooled corpus.",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            color="#55636A",
        )
    ax.text(
        0,
        1.002,
        "MORE CROSS-INDUSTRY",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#6F9FBE",
        fontweight="bold",
    )
    ax.text(
        1,
        1.002,
        "MORE INDUSTRY-CONCENTRATED",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#245F89",
        fontweight="bold",
    )

    baseline_text = "; ".join(
        f"{INDUSTRY_LABELS[industry]} {baseline[industry] * 100:.1f}%"
        for industry in INDUSTRY_ORDER
    )
    fig.text(
        0.31,
        0.025,
        (
            "Baseline: all pooled-modelled comments (N=36,171), including Topic −1 — "
            f"{baseline_text}. Topic −1 is excluded from the displayed themes."
        ),
        ha="left",
        va="bottom",
        fontsize=9.2,
        color="#59666C",
    )

    fig.subplots_adjust(left=0.31, right=0.965, top=0.94 if no_title else 0.90, bottom=0.09)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_no_title" if no_title else ""
    stem = output_dir / f"fig6_pooled_topic_cross_industry_jsd_sharedness{suffix}"
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_caption(data: pd.DataFrame, baseline: pd.Series, output_dir: Path) -> None:
    closest = data.iloc[0]
    furthest = data.iloc[-1]
    baseline_text = ", ".join(
        f"{INDUSTRY_LABELS[industry]} {baseline[industry] * 100:.1f}%"
        for industry in INDUSTRY_ORDER
    )
    caption = f"""# Figure 6 caption

**Cross-industry distribution of pooled BERTopic themes.** Bars report Jensen–Shannon divergence (base-2 logarithms) between each topic's observed industry distribution and the industry distribution of all pooled-modelled comments. Lower values indicate closer alignment with the pooled-corpus baseline and therefore greater cross-industry balance; higher values indicate stronger industry concentration. The baseline (N=36,171) is {baseline_text}. Topic −1 contributes to this corpus baseline but is excluded from the displayed substantive themes. No threshold is imposed to convert the continuous divergence measure into a binary shared/industry-specific classification.

The closest topic to the pooled baseline is T{int(closest['pooled_topic'])} ({closest['final_label']}; JSD={closest['js_divergence_bits']:.3f}). The most industry-concentrated topic is T{int(furthest['pooled_topic'])} ({furthest['final_label']}; JSD={furthest['js_divergence_bits']:.3f}).
"""
    (output_dir / "fig6_pooled_topic_cross_industry_jsd_sharedness_caption.md").write_text(
        caption, encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    assignments = pd.read_csv(args.assignments)
    mapping = pd.read_csv(args.mapping)
    data, baseline = calculate_sharedness(assignments, mapping)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(
        args.output_dir / "fig6_pooled_topic_cross_industry_jsd_sharedness_data.csv",
        index=False,
    )
    make_figure(data, baseline, args.output_dir, no_title=args.no_title)
    write_caption(data, baseline, args.output_dir)

    print(f"Modelled comments: {len(assignments):,}")
    print("Baseline industry shares:")
    for industry in INDUSTRY_ORDER:
        print(f"  {industry}: {baseline[industry]:.4f}")
    print(f"Substantive topics plotted: {len(data)}")
    print(data[["sharedness_rank", "pooled_topic", "js_divergence_bits", "dominant_industry"]].to_string(index=False))
    print(f"Outputs: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
