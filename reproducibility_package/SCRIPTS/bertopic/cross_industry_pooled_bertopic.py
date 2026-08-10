#!/usr/bin/env python3
"""Run a pooled cross-industry BERTopic analysis.

This script creates a complementary cross-industry topic layer. It does not
modify or replace the four locked industry-specific BERTopic models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("REPRO_TEMP_DIR", "/tmp")) / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(os.environ.get("REPRO_TEMP_DIR", "/tmp")) / "numba_cache"))

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from bertopic_industry_comment_body_only import (
    DEFAULT_EMBEDDING_MODEL,
    INDUSTRY_IDENTITY_STOPWORDS,
    STOPWORDS,
    clean_text,
    make_representation_model,
    short,
    topic_words,
)


OUTPUTS = Path(os.environ.get("REPRO_OUTPUTS_DIR", "outputs"))
DEFAULT_MASTER = OUTPUTS / "four_industry_ai_comments_master_harmonised_total_posts_ranked_v4_law_llm_degree_cleaned.csv"
DEFAULT_OUTPUT = OUTPUTS / "cross_industry_pooled_bertopic"
RANDOM_SEED = 42
MIN_DOC_CHARS = 30

LOCKED_ASSIGNMENTS = {
    "finance": OUTPUTS
    / "finance_top_commented_ai_threads/bertopic_finance_final_v3_comment_body_only_identity_stopwords/finance_document_topic_assignments.csv",
    "law": OUTPUTS
    / "law_top_commented_ai_threads/bertopic_law_final_v4_comment_body_only_identity_stopwords_min35_llm_degree_cleaned/law_document_topic_assignments.csv",
    "software_engineering": OUTPUTS
    / "software_engineering_top_commented_ai_threads/bertopic_software_engineering_final_v3_comment_body_only_identity_stopwords_sensitivity_min50_ms5/software_engineering_document_topic_assignments.csv",
    "healthcare": OUTPUTS
    / "healthcare_top_commented_ai_threads/bertopic_healthcare_final_v3_comment_body_only_identity_stopwords/healthcare_document_topic_assignments.csv",
}

MANUAL_WORKBOOKS = {
    "finance": Path(
        "MAPPINGS/manual_interpretation_workbooks/finance_topic_manual_interpretation.xlsx"
    ),
    "law": Path(
        "MAPPINGS/manual_interpretation_workbooks/law_topic_manual_interpretation.xlsx"
    ),
    "software_engineering": Path(
        "MAPPINGS/manual_interpretation_workbooks/software_engineering_topic_manual_interpretation.xlsx"
    ),
    "healthcare": Path(
        "MAPPINGS/manual_interpretation_workbooks/healthcare_topic_manual_interpretation.xlsx"
    ),
}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def counter_text(values: list[Any] | pd.Series, limit: int = 12) -> str:
    counter = Counter(clean_text(value) for value in values if clean_text(value))
    return "; ".join(f"{key}={value}" for key, value in counter.most_common(limit))


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()


def normalise_industry(row: pd.Series) -> str:
    values = [
        clean_text(row.get("source_dataset")).lower(),
        clean_text(row.get("industry_keyword")).lower(),
        clean_text(row.get("industry")).lower(),
    ]
    for value in values:
        if value in {"finance", "accountant", "accounting", "finance_accounting"}:
            return "finance"
        if value in {"law", "lawyer", "legal"}:
            return "law"
        if value in {"software_engineering", "software engineer", "software_engineer", "software", "it"}:
            return "software_engineering"
        if value in {"healthcare", "health", "nurse", "doctor"}:
            return "healthcare"
    return ""


def region_group(value: Any) -> str:
    return "UK-focused" if clean_text(value) == "1" else "General / non-UK or geographically unclear"


def pooled_stopwords() -> list[str]:
    words = set(STOPWORDS)
    for industry_words in INDUSTRY_IDENTITY_STOPWORDS.values():
        words.update(industry_words)
    return sorted(words)


def make_umap() -> UMAP:
    return UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=RANDOM_SEED)


def make_hdbscan(min_topic_size: int, min_samples: int) -> HDBSCAN:
    return HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=min_samples,
        metric="euclidean",
        prediction_data=True,
        core_dist_n_jobs=1,
    )


def make_vectorizer(min_df: int, max_df: float, ngram_max: int) -> CountVectorizer:
    return CountVectorizer(
        stop_words=pooled_stopwords(),
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_df=max_df,
    )


def load_master(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    master["_original_row_index"] = np.arange(len(master))
    master["industry"] = master.apply(normalise_industry, axis=1)
    four = master[master["industry"].isin(["finance", "law", "software_engineering", "healthcare"])].copy()
    four["stable_comment_id"] = four["comment_id"].map(clean_text)
    missing_id = four["stable_comment_id"].eq("")
    four.loc[missing_id, "stable_comment_id"] = four.loc[missing_id, "_original_row_index"].map(lambda x: f"row_{x}")
    four["bertopic_text_clean"] = four["comment_body"].map(clean_text)
    four["region_group"] = four["uk_strong_relevant"].map(region_group)
    valid = four[
        four["bertopic_text_clean"].str.len().ge(MIN_DOC_CHARS)
        & four["comment_body"].map(clean_text).astype(bool)
    ].copy()
    blank = four.loc[~four.index.isin(valid.index)].copy()
    exact_dupes = valid.duplicated(
        subset=["stable_comment_id", "post_id", "comment_body", "industry", "subreddit"],
        keep="first",
    )
    valid = valid.loc[~exact_dupes].copy()
    valid["_pooled_pos"] = np.arange(len(valid))
    return master, valid, blank


def export_corpus_audits(master: pd.DataFrame, valid: pd.DataFrame, blank: pd.DataFrame, output_dir: Path) -> None:
    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    valid.to_csv(corpus_dir / "pooled_valid_documents.csv", index=False)
    safe_cols = [
        "_original_row_index",
        "stable_comment_id",
        "comment_id",
        "post_id",
        "industry",
        "subreddit",
        "post_month",
        "uk_strong_relevant",
        "region_group",
    ]
    valid[safe_cols].to_csv(corpus_dir / "pooled_valid_document_index.csv", index=False)

    audit_rows = []
    for industry, group in master[master["industry"].isin(["finance", "law", "software_engineering", "healthcare"])].groupby("industry"):
        valid_group = valid[valid["industry"].eq(industry)]
        blank_group = blank[blank["industry"].eq(industry)]
        audit_rows.append(
            {
                "industry": industry,
                "master_rows": len(group),
                "valid_comment_body_rows": len(valid_group),
                "excluded_blank_or_short_rows": len(blank_group),
                "unique_posts": valid_group["post_id"].map(clean_text).replace("", pd.NA).dropna().nunique(),
                "unique_subreddits": valid_group["subreddit"].map(clean_text).replace("", pd.NA).dropna().nunique(),
                "uk_focused_rows": int(valid_group["uk_strong_relevant"].astype(str).eq("1").sum()),
                "general_non_uk_or_unclear_rows": int((~valid_group["uk_strong_relevant"].astype(str).eq("1")).sum()),
            }
        )
    pd.DataFrame(audit_rows).to_csv(corpus_dir / "pooled_corpus_audit.csv", index=False)

    dupes = valid[valid["stable_comment_id"].duplicated(keep=False)].copy()
    dupes[
        ["stable_comment_id", "comment_id", "post_id", "industry", "subreddit", "comment_body"]
    ].to_csv(corpus_dir / "duplicate_stable_comment_ids_audit.csv", index=False)


def load_or_encode_embeddings(
    docs: list[str], output_dir: Path, embedding_model_path: str, force_reencode: bool = False
) -> tuple[SentenceTransformer, np.ndarray]:
    embeddings_path = output_dir / "corpus" / "pooled_comment_body_embeddings_all_minilm_l6_v2.npy"
    print(f"Loading embedding model: {embedding_model_path}")
    model = SentenceTransformer(embedding_model_path)
    if embeddings_path.exists() and not force_reencode:
        print(f"Loading cached pooled embeddings: {embeddings_path}")
        return model, np.load(embeddings_path)
    print(f"Encoding {len(docs)} pooled documents")
    embeddings = model.encode(docs, show_progress_bar=True, batch_size=64, normalize_embeddings=True)
    np.save(embeddings_path, embeddings)
    return model, embeddings


def representative_rows(
    topic_model: BERTopic,
    topic_id: int,
    topic_rows: dict[int, list[dict[str, Any]]],
    topic_doc_pairs: dict[int, list[tuple[str, dict[str, Any]]]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    pairs = topic_doc_pairs.get(topic_id, [])
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    for rep_doc in topic_model.get_representative_docs(topic_id) or []:
        for doc, row in pairs:
            cid = clean_text(row.get("stable_comment_id"))
            if cid in used:
                continue
            if doc == rep_doc:
                chosen.append(row)
                used.add(cid)
                break
        if len(chosen) >= limit:
            return chosen
    for row in topic_rows.get(topic_id, []):
        cid = clean_text(row.get("stable_comment_id"))
        if cid in used:
            continue
        chosen.append(row)
        used.add(cid)
        if len(chosen) >= limit:
            break
    return chosen


def topic_domination(assigned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for topic_id, group in assigned.groupby("pooled_topic"):
        post_counts = group["post_id"].astype(str).str.strip().value_counts()
        subreddit_counts = group["subreddit"].astype(str).str.strip().value_counts()
        industry_counts = group["industry"].astype(str).str.strip().value_counts()
        n = len(group)
        top3 = int(post_counts.head(3).sum()) if len(post_counts) else 0
        rows.append(
            {
                "topic": int(topic_id),
                "comments": n,
                "unique_posts": int(post_counts.size),
                "unique_subreddits": int(subreddit_counts.size),
                "largest_post_comment_n": int(post_counts.iloc[0]) if len(post_counts) else 0,
                "largest_post_comment_share": round(float(post_counts.iloc[0] / n), 4) if len(post_counts) else 0,
                "top_3_posts_comment_n": top3,
                "top_3_posts_comment_share": round(float(top3 / n), 4) if n else 0,
                "largest_subreddit": subreddit_counts.index[0] if len(subreddit_counts) else "",
                "largest_subreddit_n": int(subreddit_counts.iloc[0]) if len(subreddit_counts) else 0,
                "largest_subreddit_share": round(float(subreddit_counts.iloc[0] / n), 4) if len(subreddit_counts) else 0,
                "largest_industry": industry_counts.index[0] if len(industry_counts) else "",
                "largest_industry_n": int(industry_counts.iloc[0]) if len(industry_counts) else 0,
                "largest_industry_share": round(float(industry_counts.iloc[0] / n), 4) if len(industry_counts) else 0,
                "top_post_ids": "; ".join(f"{idx}={value}" for idx, value in post_counts.head(3).items()),
                "top_subreddits": "; ".join(f"{idx}={value}" for idx, value in subreddit_counts.head(8).items()),
                "industry_distribution": "; ".join(f"{idx}={value}" for idx, value in industry_counts.items()),
            }
        )
    return pd.DataFrame(rows).sort_values(["topic"])


def topic_industry_metrics(assigned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nonout = assigned[assigned["pooled_topic"].ne(-1)].copy()
    industry_total = assigned.groupby("industry").size().rename("industry_modelled_n")
    industry_nonout_total = nonout.groupby("industry").size().rename("industry_nonout_n")

    counts = assigned.pivot_table(index="pooled_topic", columns="industry", values="stable_comment_id", aggfunc="count", fill_value=0)
    counts.index.name = "topic"
    counts = counts.reset_index()

    within_all = counts.copy()
    for col in [c for c in within_all.columns if c != "topic"]:
        denom = float(industry_total.get(col, 0))
        within_all[col] = within_all[col] / denom if denom else 0

    nonout_counts = nonout.pivot_table(index="pooled_topic", columns="industry", values="stable_comment_id", aggfunc="count", fill_value=0)
    nonout_counts.index.name = "topic"
    within_nonout = nonout_counts.reset_index()
    for col in [c for c in within_nonout.columns if c != "topic"]:
        denom = float(industry_nonout_total.get(col, 0))
        within_nonout[col] = within_nonout[col] / denom if denom else 0

    posts = assigned.pivot_table(index="pooled_topic", columns="industry", values="post_id", aggfunc=lambda x: x.astype(str).str.strip().replace("", np.nan).dropna().nunique(), fill_value=0)
    posts.index.name = "topic"
    posts = posts.reset_index()

    subs = assigned.pivot_table(index="pooled_topic", columns="industry", values="subreddit", aggfunc=lambda x: x.astype(str).str.strip().replace("", np.nan).dropna().nunique(), fill_value=0)
    subs.index.name = "topic"
    subs = subs.reset_index()

    return counts, within_all, within_nonout, posts, subs


def industry_spread_metrics(assigned: pd.DataFrame) -> pd.DataFrame:
    nonout = assigned[assigned["pooled_topic"].ne(-1)].copy()
    industry_nonout_total = nonout.groupby("industry").size()
    rows = []
    for topic, group in assigned.groupby("pooled_topic"):
        counts = group["industry"].value_counts()
        p = {}
        for industry in ["finance", "law", "software_engineering", "healthcare"]:
            denom = float(industry_nonout_total.get(industry, 0))
            p[industry] = counts.get(industry, 0) / denom if denom else 0.0
        total_p = sum(p.values())
        if total_p > 0:
            q = {k: v / total_p for k, v in p.items()}
            entropy = -sum(v * math.log(v) for v in q.values() if v > 0) / math.log(4)
        else:
            entropy = 0.0
        rows.append(
            {
                "topic": int(topic),
                "industries_with_at_least_10_comments": int(sum(counts.get(i, 0) >= 10 for i in p)),
                "industries_with_at_least_0_5pct_nonout_prevalence": int(sum(v >= 0.005 for v in p.values())),
                "normalised_industry_spread_entropy": round(entropy, 6),
                **{f"{industry}_nonout_prevalence": round(value, 6) for industry, value in p.items()},
            }
        )
    return pd.DataFrame(rows).sort_values("topic")


def summarize_model(
    topic_model: BERTopic,
    valid: pd.DataFrame,
    docs: list[str],
    topics: list[int] | np.ndarray,
    output_dir: Path,
    prefix: str,
    save_model: bool = False,
) -> dict[str, Any]:
    model_dir = output_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    assigned = valid.copy()
    assigned["pooled_topic"] = [int(x) for x in topics]
    assigned["pooled_outlier"] = assigned["pooled_topic"].eq(-1).astype(int)
    assigned.to_csv(model_dir / f"{prefix}_document_topic_assignments.csv", index=False)

    info = topic_model.get_topic_info()
    info.to_csv(model_dir / f"{prefix}_topic_info.csv", index=False)
    if save_model:
        topic_model.save(model_dir / "pooled_bertopic_model", serialization="pickle")

    rows = assigned.to_dict("records")
    topic_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    topic_doc_pairs: dict[int, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for row, doc, tid in zip(rows, docs, topics):
        topic_id = int(tid)
        topic_rows[topic_id].append(row)
        topic_doc_pairs[topic_id].append((doc, row))

    summary_rows = []
    rep_rows = []
    n_all = len(assigned)
    n_nonout = int(assigned["pooled_topic"].ne(-1).sum())
    for _, info_row in info.iterrows():
        tid = int(info_row["Topic"])
        group = pd.DataFrame(topic_rows.get(tid, []))
        reps = representative_rows(topic_model, tid, topic_rows, topic_doc_pairs, 10)
        count = int(info_row["Count"])
        summary_rows.append(
            {
                "topic": tid,
                "count": count,
                "share_all_modelled_comments": round(count / n_all, 6) if n_all else 0,
                "share_nonout_comments": round(count / n_nonout, 6) if tid != -1 and n_nonout else "",
                "name": info_row.get("Name", ""),
                "representation": info_row.get("Representation", ""),
                "top_words": topic_words(topic_model, tid),
                "representative_comment_ids": "; ".join(clean_text(r.get("stable_comment_id")) for r in reps),
                "representative_comments_short": " || ".join(short(r.get("comment_body"), 220) for r in reps[:5]),
                "unique_posts": group["post_id"].map(clean_text).replace("", pd.NA).dropna().nunique() if len(group) else 0,
                "unique_subreddits": group["subreddit"].map(clean_text).replace("", pd.NA).dropna().nunique() if len(group) else 0,
                "industry_distribution": counter_text(group["industry"], 4) if len(group) else "",
                "subreddit_distribution": counter_text(group["subreddit"], 12) if len(group) else "",
                "post_month_distribution": counter_text(group["post_month"], 12) if len(group) else "",
                "region_distribution": counter_text(group["region_group"], 2) if len(group) else "",
            }
        )
        for rank, rep in enumerate(reps, start=1):
            rep_rows.append(
                {
                    "topic": tid,
                    "rank_in_topic": rank,
                    "stable_comment_id": clean_text(rep.get("stable_comment_id")),
                    "comment_id": clean_text(rep.get("comment_id")),
                    "post_id": clean_text(rep.get("post_id")),
                    "industry": clean_text(rep.get("industry")),
                    "subreddit": clean_text(rep.get("subreddit")),
                    "region_group": clean_text(rep.get("region_group")),
                    "post_month": clean_text(rep.get("post_month")),
                    "comment_body": short(rep.get("comment_body"), 1200),
                }
            )

    pd.DataFrame(summary_rows).to_csv(model_dir / f"{prefix}_topic_summary.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(model_dir / f"{prefix}_representative_comments.csv", index=False)

    domination = topic_domination(assigned)
    domination.to_csv(model_dir / f"{prefix}_topic_domination_audit.csv", index=False)
    flags = domination[
        (domination["largest_post_comment_share"] >= 0.4)
        | (domination["top_3_posts_comment_share"] >= 0.6)
        | (domination["largest_subreddit_share"] >= 0.8)
        | ((domination["topic"].ne(-1)) & (domination["comments"] / max(n_nonout, 1) > 0.30))
    ].copy()
    flags.to_csv(model_dir / f"{prefix}_topic_domination_flags.csv", index=False)

    counts, within_all, within_nonout, post_matrix, subreddit_matrix = topic_industry_metrics(assigned)
    counts.to_csv(model_dir / f"{prefix}_topic_by_industry_counts.csv", index=False)
    within_all.to_csv(model_dir / f"{prefix}_topic_by_industry_within_all_modelled_share.csv", index=False)
    within_nonout.to_csv(model_dir / f"{prefix}_topic_by_industry_within_nonout_share.csv", index=False)
    post_matrix.to_csv(model_dir / f"{prefix}_topic_by_industry_unique_posts.csv", index=False)
    subreddit_matrix.to_csv(model_dir / f"{prefix}_topic_by_industry_unique_subreddits.csv", index=False)
    industry_spread_metrics(assigned).to_csv(model_dir / f"{prefix}_topic_industry_spread_metrics.csv", index=False)

    outliers_by_industry = (
        assigned.assign(is_outlier=assigned["pooled_topic"].eq(-1).astype(int))
        .groupby("industry")
        .agg(total_comments=("stable_comment_id", "count"), outlier_count=("is_outlier", "sum"))
        .reset_index()
    )
    outliers_by_industry["outlier_rate"] = outliers_by_industry["outlier_count"] / outliers_by_industry["total_comments"]
    outliers_by_industry.to_csv(model_dir / f"{prefix}_outlier_rates_by_industry.csv", index=False)

    try:
        tpc = topic_model.topics_per_class(docs, classes=assigned["industry"].tolist())
        tpc.to_csv(model_dir / f"{prefix}_topics_per_industry_representation.csv", index=False)
    except Exception as exc:
        (model_dir / f"{prefix}_topics_per_industry_representation_ERROR.txt").write_text(str(exc), encoding="utf-8")

    largest_substantive = assigned[assigned["pooled_topic"].ne(-1)]["pooled_topic"].value_counts()
    largest_all_share = float(largest_substantive.iloc[0] / n_all) if len(largest_substantive) and n_all else 0.0
    largest_nonout_share = float(largest_substantive.iloc[0] / n_nonout) if len(largest_substantive) and n_nonout else 0.0
    overview = {
        "modelled_documents": n_all,
        "substantive_topics": int(assigned[assigned["pooled_topic"].ne(-1)]["pooled_topic"].nunique()),
        "outlier_count": int(assigned["pooled_topic"].eq(-1).sum()),
        "outlier_rate": round(float(assigned["pooled_topic"].eq(-1).mean()), 6) if n_all else 0,
        "largest_substantive_topic_share_all": round(largest_all_share, 6),
        "largest_substantive_topic_share_nonout": round(largest_nonout_share, 6),
        "domination_flag_count": int(len(flags)),
    }
    pd.DataFrame([overview]).to_csv(model_dir / f"{prefix}_model_overview.csv", index=False)
    return overview


def fit_bertopic(
    docs: list[str],
    embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    min_topic_size: int,
    min_samples: int,
    min_df: int,
    max_df: float,
    ngram_max: int,
    representation_model: str,
) -> tuple[BERTopic, list[int]]:
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=make_vectorizer(min_df, max_df, ngram_max),
        umap_model=make_umap(),
        hdbscan_model=make_hdbscan(min_topic_size, min_samples),
        representation_model=make_representation_model(representation_model),
        min_topic_size=min_topic_size,
        nr_topics=None,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
    return topic_model, [int(x) for x in topics]


def choose_model(comparison: pd.DataFrame) -> int:
    workable = comparison[
        comparison["substantive_topics"].between(20, 60)
        & comparison["largest_substantive_topic_share_nonout"].le(0.30)
    ].copy()
    if workable.empty:
        # Prefer middle granularity if none pass the narrow automatic heuristic.
        return 150 if 150 in set(comparison["min_topic_size"]) else int(comparison.iloc[0]["min_topic_size"])
    if 150 in set(workable["min_topic_size"]):
        return 150
    return int(workable.sort_values(["domination_flag_count", "outlier_rate"]).iloc[0]["min_topic_size"])


def build_balanced_sample(valid: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    capped_parts = []
    for (_, post_id), group in valid.groupby(["industry", "post_id"], dropna=False):
        if len(group) > 50:
            idx = rng.choice(group.index.to_numpy(), size=50, replace=False)
            capped_parts.append(group.loc[idx])
        else:
            capped_parts.append(group)
    capped = pd.concat(capped_parts, ignore_index=False)
    min_n = int(capped.groupby("industry").size().min())
    samples = []
    for industry, group in capped.groupby("industry"):
        samples.append(group.sample(n=min_n, random_state=RANDOM_SEED))
    balanced = pd.concat(samples).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    audit = (
        balanced.groupby("industry")
        .agg(
            sampled_comments=("stable_comment_id", "count"),
            unique_posts=("post_id", pd.Series.nunique),
            unique_subreddits=("subreddit", pd.Series.nunique),
        )
        .reset_index()
    )
    (output_dir / "balanced_sensitivity").mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "balanced_sensitivity/balanced_sample_audit.csv", index=False)
    balanced.to_csv(output_dir / "balanced_sensitivity/balanced_valid_documents.csv", index=False)
    return balanced


def read_manual_labels(industry: str) -> pd.DataFrame:
    path = MANUAL_WORKBOOKS[industry]
    if not path.exists():
        return pd.DataFrame(columns=["locked_topic", "manual_label", "manual_decision", "manual_confidence"])
    raw = pd.read_excel(path, sheet_name=0, header=8)
    cols = {c: str(c).strip() for c in raw.columns}
    raw = raw.rename(columns=cols)
    topic_col = next((c for c in raw.columns if str(c).strip().lower() == "topic id"), None)
    label_col = next((c for c in raw.columns if "researcher-assigned" in str(c).lower()), None)
    decision_col = next((c for c in raw.columns if str(c).strip().lower() == "decision"), None)
    conf_col = next((c for c in raw.columns if str(c).strip().lower() == "confidence"), None)
    if topic_col is None:
        return pd.DataFrame(columns=["locked_topic", "manual_label", "manual_decision", "manual_confidence"])
    out = pd.DataFrame()
    out["locked_topic"] = pd.to_numeric(raw[topic_col], errors="coerce").astype("Int64")
    out["manual_label"] = raw[label_col].map(clean_text) if label_col else ""
    out["manual_decision"] = raw[decision_col].map(clean_text) if decision_col else ""
    out["manual_confidence"] = raw[conf_col].map(clean_text) if conf_col else ""
    out = out.dropna(subset=["locked_topic"]).copy()
    out["locked_topic"] = out["locked_topic"].astype(int)
    return out


def build_crosswalk(final_assigned: pd.DataFrame, output_dir: Path) -> None:
    frames = []
    label_frames = []
    for industry, path in LOCKED_ASSIGNMENTS.items():
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        df["industry"] = industry
        df["stable_comment_id"] = df["comment_id"].map(clean_text)
        df = df[["stable_comment_id", "industry", "bertopic_topic"]].rename(
            columns={"bertopic_topic": "locked_topic"}
        )
        df["locked_topic"] = pd.to_numeric(df["locked_topic"], errors="coerce").astype("Int64")
        frames.append(df)
        labels = read_manual_labels(industry)
        labels["industry"] = industry
        label_frames.append(labels)

    cross_dir = output_dir / "crosswalk"
    cross_dir.mkdir(parents=True, exist_ok=True)
    if not frames:
        (cross_dir / "pooled_to_locked_topic_crosswalk_ERROR.txt").write_text(
            "No locked assignment files found.", encoding="utf-8"
        )
        return
    locked = pd.concat(frames, ignore_index=True)
    labels = pd.concat(label_frames, ignore_index=True) if label_frames else pd.DataFrame()
    joined = final_assigned.merge(locked, on=["stable_comment_id", "industry"], how="left")
    if not labels.empty:
        joined = joined.merge(labels, on=["industry", "locked_topic"], how="left")
    joined.to_csv(cross_dir / "pooled_to_locked_document_join.csv", index=False)

    rows = []
    for (pooled_topic, industry), group in joined.groupby(["pooled_topic", "industry"]):
        total = len(group)
        counts = group["locked_topic"].dropna().astype(int).value_counts()
        top = counts.head(3)
        for rank, (locked_topic, n) in enumerate(top.items(), start=1):
            label_row = labels[(labels["industry"].eq(industry)) & (labels["locked_topic"].eq(int(locked_topic)))] if not labels.empty else pd.DataFrame()
            rows.append(
                {
                    "pooled_topic": int(pooled_topic),
                    "industry": industry,
                    "rank": rank,
                    "locked_topic": int(locked_topic),
                    "overlap_count": int(n),
                    "pooled_topic_industry_count": total,
                    "share_of_pooled_topic_industry": round(float(n / total), 6) if total else 0,
                    "locked_manual_label": label_row["manual_label"].iloc[0] if len(label_row) else "",
                    "locked_manual_decision": label_row["manual_decision"].iloc[0] if len(label_row) else "",
                    "locked_manual_confidence": label_row["manual_confidence"].iloc[0] if len(label_row) else "",
                    "manual_label_missing": not bool(len(label_row)),
                }
            )
    cross = pd.DataFrame(rows)
    cross.to_csv(cross_dir / "pooled_to_locked_topic_crosswalk.csv", index=False)
    summary = cross[cross["rank"].eq(1)].copy()
    summary.to_csv(cross_dir / "pooled_to_locked_topic_crosswalk_summary.csv", index=False)


def write_region_ready(final_assigned: pd.DataFrame, output_dir: Path) -> None:
    region_dir = output_dir / "region_ready"
    region_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "stable_comment_id",
        "comment_id",
        "post_id",
        "industry",
        "subreddit",
        "region_group",
        "pooled_topic",
        "pooled_outlier",
        "post_month",
    ]
    final_assigned[cols].to_csv(region_dir / "pooled_document_topics_with_region_group.csv", index=False)


def write_method_files(output_dir: Path, selected_min_topic_size: int, comparison: pd.DataFrame) -> None:
    readme = f"""# Cross-Industry Pooled BERTopic

This directory contains a complementary pooled BERTopic model fitted across all four final industries.

The four locked industry-specific BERTopic models remain the primary within-industry interpretation layer. This pooled model provides shared topic IDs for cross-industry comparison.

Selected final all-comment model: min_topic_size={selected_min_topic_size}, min_samples=5.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    note = f"""# Method Note

## Why fit a separate pooled model?

The four locked models were fitted separately for detailed within-industry interpretation. A new pooled model is needed because direct comparison requires all comments to share the same topic ID space.

## Why not merge the locked BERTopic models?

The four locked models use corpus-specific granularities and were selected for within-industry interpretability. Merging their fitted topics would not create a clean common clustering solution.

## Input text

The pooled model uses `comment_body` only. It does not concatenate post titles and does not overwrite the v4 master file.

## Topic -1

Topic -1 is retained as the model-designated outlier/unassigned class. It is not reduced or reassigned by `reduce_outliers()`. Substantive interpretation of any residual theme inside -1 should be a later human-review decision.

## Balanced sensitivity

Because software engineering contributes many more comments than the other industries, one balanced sensitivity model is fitted using equal industry sample sizes after capping each post at 50 comments. This is used only to check whether broad theme structure recurs; it is not used for main prevalence estimates.

## UK comparison

No separate UK BERTopic model is fitted. Later UK-focused vs General/non-UK-or-unclear comparison should use the same pooled topic assignments plus `region_group`.

## Candidate grid

{comparison.to_string(index=False)}
"""
    (output_dir / "METHOD_NOTE.md").write_text(note, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", default=str(DEFAULT_MASTER))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=0.8)
    parser.add_argument("--ngram-max", type=int, default=3)
    parser.add_argument("--representation-model", default="keybert_mmr")
    parser.add_argument("--force-reencode", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in [
        "corpus",
        "candidate_models",
        "final_model",
        "balanced_sensitivity",
        "audits",
        "crosswalk",
        "region_ready",
    ]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    print("Pipeline summary")
    print(f"Embedding model: {args.embedding_model}")
    print("Text input: comment_body only")
    print("UMAP: n_neighbors=15, n_components=5, min_dist=0.0, metric=cosine, random_state=42")
    print("HDBSCAN grid: min_topic_size=[100,150,200], min_samples=5, metric=euclidean")
    print(f"Vectorizer: ngram=(1,{args.ngram_max}), min_df={args.min_df}, max_df={args.max_df}")
    print(f"Representation model: {args.representation_model}")
    print("Outliers: retained; no reduce_outliers; no nr_topics=auto")

    master, valid, blank = load_master(Path(args.master).expanduser())
    export_corpus_audits(master, valid, blank, output_dir)
    docs = valid["bertopic_text_clean"].tolist()
    embedding_model, embeddings = load_or_encode_embeddings(docs, output_dir, args.embedding_model, args.force_reencode)

    candidate_rows = []
    candidate_objects: dict[int, tuple[BERTopic, list[int]]] = {}
    for min_topic_size in [100, 150, 200]:
        candidate_dir = output_dir / "candidate_models" / f"min{min_topic_size}_ms5"
        print(f"\nFitting candidate min_topic_size={min_topic_size}, min_samples=5")
        model, topics = fit_bertopic(
            docs,
            embeddings,
            embedding_model,
            min_topic_size,
            5,
            args.min_df,
            args.max_df,
            args.ngram_max,
            args.representation_model,
        )
        overview = summarize_model(model, valid, docs, topics, candidate_dir, f"min{min_topic_size}_ms5")
        overview.update({"min_topic_size": min_topic_size, "min_samples": 5})
        candidate_rows.append(overview)
        candidate_objects[min_topic_size] = (model, topics)

    comparison = pd.DataFrame(candidate_rows)
    comparison = comparison[
        [
            "min_topic_size",
            "min_samples",
            "modelled_documents",
            "substantive_topics",
            "outlier_count",
            "outlier_rate",
            "largest_substantive_topic_share_all",
            "largest_substantive_topic_share_nonout",
            "domination_flag_count",
        ]
    ]
    comparison.to_csv(output_dir / "audits/pooled_model_candidate_comparison.csv", index=False)

    selected = choose_model(comparison)
    print(f"\nSelected pooled model min_topic_size={selected}, min_samples=5")
    selected_model, selected_topics = candidate_objects[selected]
    final_dir = output_dir / "final_model"
    final_overview = summarize_model(
        selected_model,
        valid,
        docs,
        selected_topics,
        final_dir,
        "pooled",
        save_model=True,
    )

    # Copy core final audits into /audits with required names.
    copy_map = {
        final_dir / "pooled_topic_domination_audit.csv": output_dir / "audits/pooled_topic_domination_audit.csv",
        final_dir / "pooled_topic_domination_flags.csv": output_dir / "audits/pooled_topic_post_and_subreddit_audit.csv",
        final_dir / "pooled_outlier_rates_by_industry.csv": output_dir / "audits/pooled_outlier_rates_by_industry.csv",
        final_dir / "pooled_model_overview.csv": output_dir / "pooled_model_overview.csv",
    }
    for src, dst in copy_map.items():
        if src.exists():
            shutil.copy2(src, dst)

    # Required final filenames.
    rename_copy = {
        "pooled_document_topic_assignments.csv": "pooled_document_topic_assignments.csv",
        "pooled_topic_summary.csv": "pooled_topic_summary.csv",
        "pooled_representative_comments.csv": "pooled_representative_comments.csv",
        "pooled_topic_by_industry_counts.csv": "pooled_topic_by_industry_counts.csv",
        "pooled_topic_by_industry_within_nonout_share.csv": "pooled_topic_by_industry_within_group_share.csv",
        "pooled_topics_per_industry_representation.csv": "pooled_topics_per_industry_representation.csv",
        "pooled_topic_industry_spread_metrics.csv": "pooled_topic_industry_spread_metrics.csv",
    }
    for src_name, dst_name in rename_copy.items():
        src = final_dir / src_name
        if src.exists() and src_name != dst_name:
            shutil.copy2(src, final_dir / dst_name)

    final_assigned = pd.read_csv(final_dir / "pooled_document_topic_assignments.csv", dtype=str, keep_default_na=False)
    final_assigned["pooled_topic"] = pd.to_numeric(final_assigned["pooled_topic"], errors="coerce").fillna(-999).astype(int)
    final_assigned["pooled_outlier"] = final_assigned["pooled_topic"].eq(-1).astype(int)
    build_crosswalk(final_assigned, output_dir)
    write_region_ready(final_assigned, output_dir)

    # Balanced sensitivity.
    balanced = build_balanced_sample(valid, output_dir)
    balanced_docs = balanced["bertopic_text_clean"].tolist()
    if "_pooled_pos" in balanced.columns:
        balanced_embeddings = embeddings[balanced["_pooled_pos"].astype(int).to_numpy()]
    else:
        balanced_embeddings = embedding_model.encode(
            balanced_docs, show_progress_bar=True, batch_size=64, normalize_embeddings=True
        )
    b_model, b_topics = fit_bertopic(
        balanced_docs,
        balanced_embeddings,
        embedding_model,
        selected,
        5,
        args.min_df,
        args.max_df,
        args.ngram_max,
        args.representation_model,
    )
    balanced_overview = summarize_model(
        b_model,
        balanced.reset_index(drop=True),
        balanced_docs,
        b_topics,
        output_dir / "balanced_sensitivity",
        "balanced",
        save_model=False,
    )
    (output_dir / "balanced_sensitivity/full_vs_balanced_interpretive_comparison.md").write_text(
        "# Full vs Balanced Sensitivity\n\n"
        "This file records a qualitative check point. Topic IDs are not equivalent between the full and balanced models. "
        "Use the balanced topic summary and representative comments to assess whether high-level themes recur when industries have equal influence.\n\n"
        f"Full model substantive topics: {final_overview['substantive_topics']}\n\n"
        f"Balanced model substantive topics: {balanced_overview['substantive_topics']}\n",
        encoding="utf-8",
    )

    write_method_files(output_dir, selected, comparison)
    run_summary = {
        "master_file": str(Path(args.master).expanduser()),
        "master_rows_total": len(master),
        "valid_pooled_documents": len(valid),
        "selected_min_topic_size": selected,
        "selected_min_samples": 5,
        "final_substantive_topics": final_overview["substantive_topics"],
        "final_outlier_count": final_overview["outlier_count"],
        "final_outlier_rate": final_overview["outlier_rate"],
        "final_largest_substantive_topic_share_nonout": final_overview["largest_substantive_topic_share_nonout"],
        "final_domination_flag_count": final_overview["domination_flag_count"],
        "balanced_substantive_topics": balanced_overview["substantive_topics"],
        "balanced_outlier_rate": balanced_overview["outlier_rate"],
        "embedding_model": args.embedding_model,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    pd.DataFrame([run_summary]).to_csv(output_dir / "pooled_bertopic_run_summary.csv", index=False)
    print(json.dumps(run_summary, indent=2))
    print(f"Wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
