#!/usr/bin/env python3
"""Create final stance-analysis tables, figures, and modelling outputs.

This script uses the frozen Prompt V2 stratified annotation sample and links it
to the locked pooled BERTopic/meta-theme outputs. The main denominator is the
set of substantive-trust comments within the 2,000-comment stratified sample.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


BASE = Path("outputs")
STANCE_IN = BASE / "frozen_prompt_v2_stratified_sample_2000_2026-07-24/all_comments_v2_annotated.csv"
POOLED_FINAL = BASE / "cross_industry_pooled_bertopic_FINAL_2026-07-27"
POOLED_ASSIGNMENTS = POOLED_FINAL / "final_model/pooled_document_topic_assignments.csv"
POOLED_MAPPING = POOLED_FINAL / "cross_industry_pooled_topic_meta_mapping_FINAL.csv"
LOCKED_CROSSWALK = POOLED_FINAL / "crosswalk/pooled_to_locked_document_join.csv"

OUT = BASE / "stance_analysis_FINAL_2026-07-28"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
MODELS = OUT / "models"
DATA = OUT / "analysis_ready_data"

INDUSTRY_ORDER = ["finance", "healthcare", "law", "software_engineering"]
INDUSTRY_LABELS = {
    "finance": "Finance",
    "healthcare": "Healthcare",
    "law": "Law",
    "software_engineering": "Software engineering",
}
ATTITUDE_ORDER = ["positive", "mixed_or_ambivalent", "neutral_descriptive", "negative", "unclear"]
DISPLAY_ATTITUDE_ORDER = ["positive", "mixed_or_ambivalent", "neutral_descriptive", "negative"]
ATTITUDE_LABELS = {
    "positive": "Positive",
    "mixed_or_ambivalent": "Mixed / ambivalent",
    "neutral_descriptive": "Neutral / descriptive",
    "negative": "Negative",
    "unclear": "Unclear",
}
ATTITUDE_COLORS = {
    "positive": "#3B8EA5",
    "mixed_or_ambivalent": "#E2A93B",
    "neutral_descriptive": "#8C96A3",
    "negative": "#B75548",
    "unclear": "#C7C7C7",
}


def ensure_dirs() -> None:
    for path in [OUT, TABLES, FIGURES, MODELS, DATA]:
        path.mkdir(parents=True, exist_ok=True)


def norm_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def norm_bool_yes(value: object) -> bool:
    return norm_text(value).lower() == "yes"


def stable_merge_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["industry"].astype(str).str.strip()
        + "||"
        + df["post_id"].astype(str).str.strip()
        + "||"
        + df["comment_id"].astype(str).str.strip()
    )


def load_analysis_data() -> pd.DataFrame:
    stance = pd.read_csv(STANCE_IN)
    stance["industry"] = stance["industry"].map(norm_text)
    stance["post_id"] = stance["post_id"].map(norm_text)
    stance["comment_id"] = stance["comment_id"].map(norm_text)
    stance["merge_key"] = stable_merge_key(stance)

    stance["substantive_trust_positive"] = stance["substantive_trust_content"].map(norm_bool_yes)
    stance["boundary_present"] = stance["trust_boundary"].map(norm_bool_yes)
    stance["attitude"] = stance["gpt_human_attitude"].map(norm_text)
    stance["trust_construct"] = stance["gpt_human_trust_construct"].map(norm_text)
    stance["trust_boundary_type"] = stance["gpt_human_trust_boundary"].map(norm_text)
    stance["attitude_target"] = stance["gpt_human_attitude_target"].map(norm_text)
    stance["capability_assessment"] = stance["gpt_human_capability_assessment"].map(norm_text)
    stance["evidence_type"] = stance["gpt_human_evidence"].map(norm_text)
    stance["UK_status"] = stance.get("UK_status", stance.get("uk_status", "")).map(norm_text)
    stance["country_group"] = stance.get("country_group", stance["UK_status"]).map(norm_text)

    assignments = pd.read_csv(
        POOLED_ASSIGNMENTS,
        usecols=["industry", "post_id", "comment_id", "pooled_topic", "pooled_outlier"],
    )
    assignments["industry"] = assignments["industry"].map(norm_text)
    assignments["post_id"] = assignments["post_id"].map(norm_text)
    assignments["comment_id"] = assignments["comment_id"].map(norm_text)
    assignments["merge_key"] = stable_merge_key(assignments)
    assignments = assignments.drop_duplicates("merge_key")

    mapping = pd.read_csv(
        POOLED_MAPPING,
        usecols=[
            "pooled_topic",
            "final_label",
            "broader_meta_theme",
            "cross_industry_status",
            "dominant_industry",
            "dominant_industry_share_percent",
        ],
    )

    crosswalk = pd.read_csv(
        LOCKED_CROSSWALK,
        usecols=["industry", "post_id", "comment_id", "locked_topic", "manual_label", "manual_decision", "manual_confidence"],
    )
    crosswalk["industry"] = crosswalk["industry"].map(norm_text)
    crosswalk["post_id"] = crosswalk["post_id"].map(norm_text)
    crosswalk["comment_id"] = crosswalk["comment_id"].map(norm_text)
    crosswalk["merge_key"] = stable_merge_key(crosswalk)
    crosswalk = crosswalk.drop_duplicates("merge_key")

    merged = stance.merge(assignments[["merge_key", "pooled_topic", "pooled_outlier"]], on="merge_key", how="left")
    merged = merged.merge(
        mapping,
        on="pooled_topic",
        how="left",
    )
    merged = merged.merge(
        crosswalk[["merge_key", "locked_topic", "manual_label", "manual_decision", "manual_confidence"]],
        on="merge_key",
        how="left",
    )
    merged["meta_theme"] = merged["broader_meta_theme"].fillna("Unassigned / no pooled topic")
    merged["locked_industry_topic"] = merged["manual_label"].fillna("")

    trust = merged[merged["substantive_trust_positive"]].copy()
    selected_cols = [
        "comment_id",
        "post_id",
        "industry",
        "subreddit",
        "UK_status",
        "country_group",
        "post_month",
        "comment_month",
        "post_title",
        "target_comment",
        "comment_body",
        "attitude",
        "boundary_present",
        "trust_construct",
        "trust_boundary_type",
        "attitude_target",
        "capability_assessment",
        "evidence_type",
        "meta_theme",
        "locked_industry_topic",
        "locked_topic",
        "pooled_topic",
        "final_label",
        "cross_industry_status",
        "dominant_industry",
        "gpt_annotation_confidence",
        "gpt_brief_reason",
        "gpt_evidence_quote",
    ]
    keep_cols = [c for c in selected_cols if c in trust.columns]
    trust[keep_cols].to_csv(DATA / "stance_analysis_ready_substantive_trust_comments.csv", index=False)

    missing_topic = trust[trust["pooled_topic"].isna() | trust["locked_industry_topic"].eq("")].copy()
    if not missing_topic.empty:
        missing_cols = [
            "comment_id",
            "industry",
            "subreddit",
            "post_id",
            "post_month",
            "comment_month",
            "pooled_topic",
            "locked_topic",
            "locked_industry_topic",
            "post_title",
            "target_comment",
        ]
        missing_topic[[c for c in missing_cols if c in missing_topic.columns]].assign(
            missing_topic_reason=(
                "Substantive-trust comment did not match the final pooled/locked BERTopic assignment files; "
                "likely excluded from BERTopic modelling during final text cleaning or model input filtering."
            )
        ).to_csv(DATA / "missing_topic_assignment_audit_rows.csv", index=False)

    audit_rows = []
    audit_rows.append({"check": "input_rows", "value": len(stance)})
    audit_rows.append({"check": "substantive_trust_rows", "value": len(trust)})
    audit_rows.append({"check": "duplicate_comment_id_count", "value": int(trust["comment_id"].duplicated().sum())})
    audit_rows.append({"check": "missing_attitude_count", "value": int(trust["attitude"].eq("").sum())})
    audit_rows.append({"check": "missing_pooled_topic_count", "value": int(trust["pooled_topic"].isna().sum())})
    audit_rows.append({"check": "missing_locked_industry_topic_count", "value": int(trust["locked_industry_topic"].eq("").sum())})
    pd.DataFrame(audit_rows).to_csv(DATA / "stance_analysis_ready_data_audit.csv", index=False)

    return trust


def add_pct_columns(counts: pd.DataFrame, group_cols: list[str], count_col: str = "n") -> pd.DataFrame:
    out = counts.copy()
    denom = out.groupby(group_cols)[count_col].transform("sum")
    out["percent"] = np.where(denom > 0, out[count_col] / denom * 100, 0)
    return out


def table_industry_attitude(trust: pd.DataFrame) -> pd.DataFrame:
    counts = trust.groupby(["industry", "attitude"]).size().rename("n").reset_index()
    counts = add_pct_columns(counts, ["industry"])
    counts.to_csv(TABLES / "table1_industry_by_attitude_long.csv", index=False)

    rows = []
    for industry in INDUSTRY_ORDER:
        part = counts[counts["industry"] == industry]
        row = {"industry": INDUSTRY_LABELS[industry], "total_trust_n": int(part["n"].sum())}
        for attitude in DISPLAY_ATTITUDE_ORDER:
            match = part[part["attitude"] == attitude]
            n = int(match["n"].iloc[0]) if not match.empty else 0
            pct = float(match["percent"].iloc[0]) if not match.empty else 0.0
            row[f"{ATTITUDE_LABELS[attitude]} n"] = n
            row[f"{ATTITUDE_LABELS[attitude]} %"] = round(pct, 1)
        rows.append(row)
    wide = pd.DataFrame(rows)
    wide.to_csv(TABLES / "table1_industry_by_attitude_wide.csv", index=False)
    return counts


def table_attitude_boundary(trust: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for attitude in DISPLAY_ATTITUDE_ORDER:
        part = trust[trust["attitude"] == attitude]
        total = len(part)
        present = int(part["boundary_present"].sum())
        absent = total - present
        rows.append(
            {
                "attitude": ATTITUDE_LABELS[attitude],
                "attitude_code": attitude,
                "boundary_present_n": present,
                "boundary_absent_n": absent,
                "total_substantive_trust_n": total,
                "boundary_rate_percent": round(present / total * 100, 1) if total else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "table2_attitude_by_boundary.csv", index=False)
    return out


def cluster_bootstrap_boundary_ci(
    trust: pd.DataFrame,
    attitudes: Iterable[str],
    n_boot: int = 2000,
    seed: int = 445,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    post_ids = trust["post_id"].fillna("").astype(str)
    posts = pd.Index(post_ids.unique())
    per_post = (
        trust.assign(_post_id=post_ids, _boundary=trust["boundary_present"].astype(int))
        .groupby(["_post_id", "attitude"])["_boundary"]
        .agg(boundary_n="sum", total_n="size")
        .reset_index()
    )
    boundary_by_attitude = (
        per_post.pivot_table(index="_post_id", columns="attitude", values="boundary_n", aggfunc="sum", fill_value=0)
        .reindex(index=posts, columns=list(attitudes), fill_value=0)
        .to_numpy()
    )
    total_by_attitude = (
        per_post.pivot_table(index="_post_id", columns="attitude", values="total_n", aggfunc="sum", fill_value=0)
        .reindex(index=posts, columns=list(attitudes), fill_value=0)
        .to_numpy()
    )
    rows = []
    attitudes = list(attitudes)
    n_posts = len(posts)
    for j, attitude in enumerate(attitudes):
        part = trust[trust["attitude"] == attitude]
        rate = float(part["boundary_present"].mean()) if len(part) else np.nan
        estimates = []
        for _ in range(n_boot):
            sampled_idx = rng.integers(0, n_posts, size=n_posts)
            total = total_by_attitude[sampled_idx, j].sum()
            if total:
                estimates.append(float(boundary_by_attitude[sampled_idx, j].sum() / total))
        if estimates:
            lo, hi = np.quantile(estimates, [0.025, 0.975])
        else:
            lo, hi = np.nan, np.nan
        rows.append(
            {
                "attitude_code": attitude,
                "attitude": ATTITUDE_LABELS[attitude],
                "n": len(part),
                "boundary_rate": rate,
                "boundary_rate_percent": round(rate * 100, 1) if not np.isnan(rate) else np.nan,
                "ci_low": lo,
                "ci_high": hi,
                "ci_low_percent": round(lo * 100, 1) if not np.isnan(lo) else np.nan,
                "ci_high_percent": round(hi * 100, 1) if not np.isnan(hi) else np.nan,
                "bootstrap_clusters": len(posts),
                "bootstrap_iterations": n_boot,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "boundary_rate_by_attitude_cluster_bootstrap_ci.csv", index=False)
    return out


def plot_industry_attitude(counts: pd.DataFrame) -> None:
    matrix = (
        counts.pivot_table(index="industry", columns="attitude", values="percent", fill_value=0)
        .reindex(index=INDUSTRY_ORDER, columns=DISPLAY_ATTITUDE_ORDER, fill_value=0)
    )
    totals = counts.groupby("industry")["n"].sum().reindex(INDUSTRY_ORDER)

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    bottom = np.zeros(len(matrix))
    x = np.arange(len(matrix))
    for attitude in DISPLAY_ATTITUDE_ORDER:
        vals = matrix[attitude].to_numpy()
        if vals.sum() == 0 and attitude == "unclear":
            continue
        bars = ax.bar(
            x,
            vals,
            bottom=bottom,
            label=ATTITUDE_LABELS[attitude],
            color=ATTITUDE_COLORS[attitude],
            edgecolor="white",
            linewidth=0.8,
        )
        for i, (bar, val) in enumerate(zip(bars, vals)):
            if val >= 9:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom[i] + val / 2,
                    f"{val:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if attitude in {"negative", "positive"} else "#202020",
                    fontweight="bold",
                )
        bottom += vals

    labels = [f"{INDUSTRY_LABELS[ind]}\nn={int(totals[ind])}" for ind in INDUSTRY_ORDER]
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_ylabel("Share of substantive-trust comments")
    ax.set_title("Attitudinal stance among substantive trust comments by industry", pad=14)
    ax.legend(ncol=4, bbox_to_anchor=(0.5, -0.18), loc="upper center", frameon=False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(FIGURES / "figure1_stance_composition_by_industry.png", dpi=300)
    fig.savefig(FIGURES / "figure1_stance_composition_by_industry.pdf")
    plt.close(fig)


def plot_boundary_rate(ci: pd.DataFrame) -> None:
    plot = ci[ci["n"] > 0].copy()
    table2_path = TABLES / "table2_attitude_by_boundary.csv"
    if table2_path.exists():
        table2 = pd.read_csv(table2_path)
        table2 = table2.set_index("attitude_code")
        plot["explicit_boundary_n"] = plot["attitude_code"].map(table2["boundary_present_n"]).astype(int)
    else:
        plot["explicit_boundary_n"] = np.nan
    plot = plot.sort_values(["boundary_rate", "n"], ascending=[False, False]).reset_index(drop=True)

    y = np.arange(len(plot))
    x = plot["boundary_rate"].to_numpy()
    xerr = np.vstack([x - plot["ci_low"].to_numpy(), plot["ci_high"].to_numpy() - x])

    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    ax.barh(y, np.ones(len(plot)), color="#EEF2F2", height=0.44, edgecolor="none")
    ax.barh(y, x, color="#7FB3B5", height=0.44, edgecolor="none")
    ax.errorbar(
        x,
        y,
        xerr=xerr,
        fmt="o",
        color="#2F6F73",
        ecolor="#8DB7B9",
        elinewidth=2.5,
        capsize=4,
        markersize=8,
    )
    for xi, yi, row in zip(x, y, plot.itertuples(index=False)):
        numerator = int(row.explicit_boundary_n)
        denominator = int(row.n)
        ax.text(
            min(xi + 0.025, 0.985),
            yi,
            f"{numerator}/{denominator} ({row.boundary_rate_percent:.1f}%)",
            va="center",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
    ylabels = [f"{row.attitude}\nn={int(row.n)}" for row in plot.itertuples(index=False)]
    ax.set_yticks(y, ylabels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlabel("Comments containing an explicit trust boundary (%)")
    ax.set_title(
        "Percentage of substantive-trust comments containing an explicit trust boundary,\nby attitudinal stance",
        pad=14,
    )
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure2_boundary_rate_by_stance_cluster_bootstrap.png", dpi=300)
    fig.savefig(FIGURES / "figure2_boundary_rate_by_stance_cluster_bootstrap.pdf")
    plt.close(fig)


def run_supplementary_models(trust: pd.DataFrame) -> None:
    try:
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
    except Exception as exc:  # pragma: no cover
        (MODELS / "model_error.txt").write_text(f"statsmodels unavailable: {exc}\n", encoding="utf-8")
        return

    model_df = trust[trust["attitude"].isin(["positive", "negative", "mixed_or_ambivalent", "neutral_descriptive"])].copy()
    model_df["boundary_int"] = model_df["boundary_present"].astype(int)
    model_df["industry"] = pd.Categorical(model_df["industry"], categories=INDUSTRY_ORDER)
    model_df["attitude"] = pd.Categorical(
        model_df["attitude"],
        categories=["positive", "mixed_or_ambivalent", "neutral_descriptive", "negative"],
    )

    boundary_model = smf.logit("boundary_int ~ C(attitude, Treatment(reference='positive')) + C(industry, Treatment(reference='finance'))", data=model_df)
    boundary_fit = boundary_model.fit(disp=False, cov_type="cluster", cov_kwds={"groups": model_df["post_id"].astype(str)})
    boundary_params = pd.DataFrame(
        {
            "term": boundary_fit.params.index,
            "coef": boundary_fit.params.values,
            "std_err_clustered_by_post": boundary_fit.bse.values,
            "p_value": boundary_fit.pvalues.values,
            "odds_ratio": np.exp(boundary_fit.params.values),
            "ci_low_or": np.exp(boundary_fit.conf_int()[0].values),
            "ci_high_or": np.exp(boundary_fit.conf_int()[1].values),
        }
    )
    boundary_params.to_csv(MODELS / "supplementary_logit_boundary_by_attitude_industry_clustered.csv", index=False)

    pred_rows = []
    for attitude in ["positive", "mixed_or_ambivalent", "neutral_descriptive", "negative"]:
        for industry in INDUSTRY_ORDER:
            pred_rows.append({"attitude": attitude, "industry": industry})
    pred_df = pd.DataFrame(pred_rows)
    pred_df["predicted_boundary_probability"] = boundary_fit.predict(pred_df)
    pred_df.to_csv(MODELS / "supplementary_logit_boundary_predicted_probabilities.csv", index=False)

    try:
        # statsmodels MNLogit has a less convenient formula API for categorical
        # outputs. Coefficients are supplemental only; descriptive probabilities
        # remain the main reported results.
        y = pd.Categorical(model_df["attitude"], categories=["negative", "positive", "mixed_or_ambivalent", "neutral_descriptive"])
        y_codes = y.codes
        x = pd.get_dummies(model_df["industry"], prefix="industry", drop_first=True).astype(float)
        x = sm.add_constant(x, has_constant="add")
        mn = sm.MNLogit(y_codes, x)
        mn_fit = mn.fit(disp=False, maxiter=200, cov_type="cluster", cov_kwds={"groups": model_df["post_id"].astype(str)})
        params = mn_fit.params.copy()
        params.to_csv(MODELS / "supplementary_multinomial_attitude_by_industry_coefficients.csv")
        probs = model_df.groupby(["industry", "attitude"]).size().rename("n").reset_index()
        probs = add_pct_columns(probs, ["industry"])
        probs.to_csv(MODELS / "supplementary_multinomial_attitude_descriptive_probabilities.csv", index=False)
    except Exception as exc:
        (MODELS / "supplementary_multinomial_model_note.txt").write_text(
            f"Multinomial model was not estimated successfully; use descriptive probabilities. Error: {exc}\n",
            encoding="utf-8",
        )


def write_readme(trust: pd.DataFrame, table1: pd.DataFrame, table2: pd.DataFrame) -> None:
    denominator = trust.groupby("industry").size().reindex(INDUSTRY_ORDER).to_dict()
    unclear_n = int((trust["attitude"] == "unclear").sum())
    text = f"""# Final Stance Analysis Outputs

Created on 2026-07-28 from the Frozen Prompt V2 stratified annotation sample.

## Denominator

The main denominator is substantive-trust comments only, not the full Reddit corpus.

- Total substantive-trust comments: {len(trust)}
- Finance: {denominator.get('finance', 0)}
- Healthcare: {denominator.get('healthcare', 0)}
- Law: {denominator.get('law', 0)}
- Software engineering: {denominator.get('software_engineering', 0)}
- Unclear attitude labels within substantive-trust comments: {unclear_n}

No substantive-trust comments were classified as unclear in the analytical sample; the unclear category is therefore omitted from the main tables and figures.

Three substantive-trust comments did not match the final pooled/locked BERTopic topic-assignment files. They are retained in the stance analysis and audited separately in `analysis_ready_data/missing_topic_assignment_audit_rows.csv`; this does not affect the industry x stance or stance x boundary analyses.

## Main figures

- `figures/figure1_stance_composition_by_industry.png`: 100% stacked bar chart of attitude composition within substantive-trust comments for each industry.
- `figures/figure2_boundary_rate_by_stance_cluster_bootstrap.png`: explicit-trust-boundary rate by stance, with 95% confidence intervals generated by resampling posts rather than individual comments.

## Main tables

- `tables/table1_industry_by_attitude_wide.csv`: exact counts and percentages for Industry x attitude.
- `tables/table2_attitude_by_boundary.csv`: exact counts and rates for attitude x boundary.

## Important interpretation note

The 75-comment non-healthcare validation sample supports broad descriptive use of attitude predictions, but neutral/descriptive labels and small industry-level validation differences should be interpreted cautiously. Do not use the validation sample to error-correct the full 1,204-comment stance estimates.

This folder is the locked final stance-analysis output. Do not change model labels or rerun stance proportions unless a data or code error is found.
"""
    (OUT / "README_stance_analysis_FINAL.md").write_text(text, encoding="utf-8")

    captions = """# Figure Captions and Results Notes

## Figure 1. Attitudinal stance among substantive trust comments by industry

Percentages are calculated within substantive-trust comments in each industry, rather than within the full Reddit corpus. This means the figure compares stance composition among comments that already discuss AI trust, reliability, reliance, or acceptable-use boundaries.

## Figure 2. Percentage of substantive-trust comments containing an explicit trust boundary, by attitudinal stance

Categories are ordered by observed boundary prevalence. The left-side `n` reports the total number of substantive-trust comments in each attitudinal stance category. The filled bar shows the percentage of comments in that stance category that contain an explicit trust boundary, and the pale remainder shows comments without an explicit boundary. Labels report the numerator and denominator followed by the percentage. Points and whiskers show 95% confidence intervals from a cluster bootstrap resampling posts, not individual comments, to account for comments being nested within Reddit threads.

Core interpretation: mixed or ambivalent stance should not be read as simple indecision. In this corpus it frequently marks conditional acceptance, where users identify both AI's potential value and the conditions under which reliance is appropriate.
"""
    (OUT / "figure_captions_and_interpretation.md").write_text(captions, encoding="utf-8")


def main() -> int:
    ensure_dirs()
    trust = load_analysis_data()

    table1 = table_industry_attitude(trust)
    table2 = table_attitude_boundary(trust)
    ci = cluster_bootstrap_boundary_ci(
        trust,
        attitudes=["positive", "mixed_or_ambivalent", "neutral_descriptive", "negative", "unclear"],
    )
    plot_industry_attitude(table1)
    plot_boundary_rate(ci)
    run_supplementary_models(trust)
    write_readme(trust, table1, table2)

    print(f"Substantive-trust rows: {len(trust)}")
    print(trust.groupby("industry").size().reindex(INDUSTRY_ORDER).to_string())
    print(f"Outputs: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
