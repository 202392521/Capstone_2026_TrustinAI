#!/usr/bin/env python3
"""Run industry-specific BERTopic models using comment_body only.

This is the formal body-only BERTopic pipeline used after the healthcare
sensitivity check showed that repeated post titles can create thread-driven
topics. Metadata is retained only for interpretation and auditing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
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
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


OUTPUTS = Path(os.environ.get("REPRO_OUTPUTS_DIR", "outputs"))
DEFAULT_INPUT = OUTPUTS / "four_industry_ai_comments_master_harmonised_total_posts_ranked_v3.csv"
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "REPRO_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
START_MONTH = "2023-03"
END_MONTH = "2026-01"

INDUSTRY_ALIASES = {
    "finance": {"finance", "accountant", "accounting", "finance_accounting"},
    "law": {"law", "lawyer", "legal"},
    "software_engineering": {"software_engineering", "software_engineer", "it", "software"},
    "healthcare": {"healthcare", "nurse", "health", "doctor"},
}

STOPWORDS = {
    "ai",
    "chatgpt",
    "chat",
    "gpt",
    "llm",
    "llms",
    "generative",
    "artificial",
    "intelligence",
    "use",
    "using",
    "used",
    "just",
    "like",
    "really",
    "think",
    "know",
    "dont",
    "doesnt",
    "isnt",
    "would",
    "could",
    "should",
    "people",
    "thing",
    "things",
    "reddit",
    "post",
    "comment",
    "amp",
    "work",
    "job",
    "jobs",
    "professional",
}

# These are occupational identity/background terms only. They are removed from
# CountVectorizer/c-TF-IDF topic labels, but the full comment_body remains in the
# sentence-transformer embeddings used for clustering.
INDUSTRY_IDENTITY_STOPWORDS = {
    "finance": {
        "accountant",
        "accountants",
        "accounting",
        "finance",
        "financial",
        "bookkeeper",
        "bookkeepers",
        "bookkeeping",
        "cpa",
        "cpas",
        "acca",
    },
    "law": {
        "law",
        "laws",
        "lawyer",
        "lawyers",
        "legal",
        "attorney",
        "attorneys",
        "solicitor",
        "solicitors",
        "barrister",
        "barristers",
        "counsel",
        "firm",
        "firms",
        "court",
        "courts",
    },
    "software_engineering": {
        "developer",
        "developers",
        "engineer",
        "engineers",
        "engineering",
        "software",
        "programmer",
        "programmers",
        "programming",
        "computer",
        "computers",
        "cs",
        "coder",
        "coders",
        "coding",
        "dev",
        "devs",
        "webdev",
    },
    "healthcare": {
        "healthcare",
        "health",
        "medical",
        "clinical",
        "medicine",
        "med",
        "md",
        "mds",
        "doc",
        "docs",
        "dr",
        "drs",
        "clinician",
        "clinicians",
        "doctor",
        "doctors",
        "physician",
        "physicians",
        "nurse",
        "nurses",
        "nursing",
        "therapist",
        "therapists",
        "radiologist",
        "radiologists",
        "pharmacist",
        "pharmacists",
        "dentist",
        "dentists",
        "hospital",
        "hospitals",
    },
}


def clean_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def short(value: Any, max_chars: int = 700) -> str:
    text = clean_text(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def counter_text(values: list[str] | pd.Series, limit: int = 12) -> str:
    counter = Counter(clean_text(value) for value in values if clean_text(value))
    return "; ".join(f"{key}={value}" for key, value in counter.most_common(limit))


def month_out_of_window(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    return (text != "") & ((text < START_MONTH) | (text > END_MONTH))


def make_representation_model(value: str) -> Any:
    value = value.lower().strip()
    if value in {"none", "off", ""}:
        return None
    if value == "keybert":
        return KeyBERTInspired(top_n_words=12)
    if value == "mmr":
        return MaximalMarginalRelevance(diversity=0.3)
    if value == "keybert_mmr":
        return [KeyBERTInspired(top_n_words=12), MaximalMarginalRelevance(diversity=0.3)]
    raise ValueError("--representation-model must be one of: none, keybert, mmr, keybert_mmr")


def topic_words(topic_model: BERTopic, topic_id: int, n_words: int = 12) -> str:
    words = topic_model.get_topic(topic_id)
    if not words:
        return ""
    return "; ".join(f"{word}:{score:.3f}" for word, score in words[:n_words])


def select_industry(df: pd.DataFrame, industry: str) -> pd.DataFrame:
    aliases = INDUSTRY_ALIASES.get(industry, {industry})
    source = df.get("source_dataset", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
    keyword = df.get("industry_keyword", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
    return df[source.isin(aliases) | keyword.isin(aliases)].copy()


def vectorizer_stopwords(industry: str) -> list[str]:
    """Return generic plus occupational identity stopwords for topic labels."""
    return sorted(STOPWORDS | INDUSTRY_IDENTITY_STOPWORDS.get(industry, set()))


def representative_rows(
    topic_model: BERTopic,
    topic_id: int,
    topic_rows: dict[int, list[dict[str, Any]]],
    topic_doc_pairs: dict[int, list[tuple[str, dict[str, Any]]]],
    limit: int,
) -> list[dict[str, Any]]:
    pairs = topic_doc_pairs.get(topic_id, [])
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    for rep_doc in topic_model.get_representative_docs(topic_id) or []:
        for doc, row in pairs:
            comment_id = clean_text(row.get("comment_id"))
            if comment_id in used:
                continue
            if doc == rep_doc:
                chosen.append(row)
                used.add(comment_id)
                break
        if len(chosen) >= limit:
            return chosen
    for row in topic_rows.get(topic_id, []):
        comment_id = clean_text(row.get("comment_id"))
        if comment_id in used:
            continue
        chosen.append(row)
        used.add(comment_id)
        if len(chosen) >= limit:
            break
    return chosen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--industry", required=True, choices=sorted(INDUSTRY_ALIASES))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--min-topic-size", type=int, default=70)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=0.8)
    parser.add_argument("--ngram-max", type=int, default=3)
    parser.add_argument("--representation-model", default="keybert_mmr")
    parser.add_argument("--representative-comments-per-topic", type=int, default=10)
    parser.add_argument("--nr-topics", default="")
    parser.add_argument("--min-doc-chars", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = safe_filename(args.industry)

    df = pd.read_csv(input_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    industry_df = select_industry(df, args.industry)
    industry_df["industry"] = args.industry
    industry_df["bertopic_text_clean"] = industry_df["comment_body"].map(clean_text)
    usable = industry_df[
        industry_df["bertopic_text_clean"].str.len().ge(args.min_doc_chars)
        & industry_df["comment_body"].map(clean_text).astype(bool)
    ].copy()
    usable = usable.drop_duplicates(subset=["comment_id"], keep="first")

    docs = usable["bertopic_text_clean"].tolist()
    rows = usable.to_dict("records")
    print(f"Industry: {args.industry}")
    print(f"Total industry rows: {len(industry_df)}")
    print(f"Documents used: {len(docs)}")
    print(f"Excluded empty/unusable rows: {len(industry_df) - len(docs)}")
    print(f"Subreddits: {usable['subreddit'].nunique()}")

    input_cols = list(dict.fromkeys(list(industry_df.columns)))
    usable.to_csv(output_dir / f"{prefix}_bertopic_input.csv", index=False)

    embeddings_path = output_dir / f"{prefix}_embeddings_comment_body_all_minilm_l6_v2.npy"
    print(f"Loading embedding model: {args.embedding_model}")
    embedding_model = SentenceTransformer(args.embedding_model)
    if embeddings_path.exists():
        print(f"Loading cached embeddings: {embeddings_path}")
        embeddings = np.load(embeddings_path)
    else:
        print("Encoding documents")
        embeddings = embedding_model.encode(docs, show_progress_bar=True, batch_size=64, normalize_embeddings=True)
        np.save(embeddings_path, embeddings)

    if not args.nr_topics:
        nr_topics: str | int | None = None
    elif str(args.nr_topics).isdigit():
        nr_topics = int(args.nr_topics)
    else:
        nr_topics = args.nr_topics

    active_stopwords = vectorizer_stopwords(args.industry)
    vectorizer_model = CountVectorizer(
        stop_words=active_stopwords,
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
    )
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.min_topic_size,
        min_samples=args.min_samples,
        metric="euclidean",
        prediction_data=True,
        core_dist_n_jobs=1,
    )
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=make_representation_model(args.representation_model),
        min_topic_size=args.min_topic_size,
        nr_topics=nr_topics,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
    info = topic_model.get_topic_info()
    info.to_csv(output_dir / f"{prefix}_topic_info.csv", index=False)
    info.to_csv(output_dir / f"{prefix}_bertopic_topic_info.csv", index=False)
    topic_model.save(output_dir / f"{prefix}_bertopic_model", serialization="pickle")

    assigned = usable.copy()
    assigned["bertopic_topic"] = topics
    assigned["topic_probability"] = ""
    assigned.to_csv(output_dir / f"{prefix}_document_topic_assignments.csv", index=False)
    assigned.to_csv(output_dir / f"{prefix}_comments_with_bertopic_topics.csv", index=False)

    topic_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    topic_doc_pairs: dict[int, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for row, doc, topic_id in zip(rows, docs, topics):
        tid = int(topic_id)
        topic_rows[tid].append(row)
        topic_doc_pairs[tid].append((doc, row))

    summary_rows: list[dict[str, Any]] = []
    representative_output: list[dict[str, Any]] = []
    for _, info_row in info.iterrows():
        topic_id = int(info_row["Topic"])
        group = topic_rows.get(topic_id, [])
        reps = representative_rows(
            topic_model,
            topic_id,
            topic_rows,
            topic_doc_pairs,
            args.representative_comments_per_topic,
        )
        uk_counts = Counter(clean_text(row.get("uk_strong_relevant")) for row in group)
        unique_posts = {clean_text(row.get("post_id")) for row in group if clean_text(row.get("post_id"))}
        unique_subreddits = {clean_text(row.get("subreddit")) for row in group if clean_text(row.get("subreddit"))}
        summary_rows.append(
            {
                "topic": topic_id,
                "count": int(info_row["Count"]),
                "name": info_row.get("Name", ""),
                "representation": info_row.get("Representation", ""),
                "top_words": topic_words(topic_model, topic_id),
                "representative_comment_ids": "; ".join(clean_text(row.get("comment_id")) for row in reps),
                "representative_comments_short": " || ".join(short(row.get("comment_body"), 220) for row in reps[:5]),
                "unique_posts": len(unique_posts),
                "unique_subreddits": len(unique_subreddits),
                "subreddit_distribution": counter_text([row.get("subreddit", "") for row in group], 12),
                "top_post_months": counter_text([row.get("post_month", "") for row in group], 8),
                "top_comment_months": counter_text([row.get("comment_month", "") for row in group], 8),
                "uk_strong_1": uk_counts.get("1", 0),
                "uk_strong_0": uk_counts.get("0", 0),
                "share_uk_strong": round(uk_counts.get("1", 0) / len(group), 4) if group else 0,
            }
        )
        for rank, row in enumerate(reps, start=1):
            representative_output.append(
                {
                    "topic": topic_id,
                    "rank_in_topic": rank,
                    "comment_id": clean_text(row.get("comment_id")),
                    "post_id": clean_text(row.get("post_id")),
                    "industry": args.industry,
                    "subreddit": clean_text(row.get("subreddit")),
                    "uk_strong_relevant": clean_text(row.get("uk_strong_relevant")),
                    "post_month": clean_text(row.get("post_month")),
                    "comment_month": clean_text(row.get("comment_month")),
                    "post_title": row.get("post_title", ""),
                    "comment_body": short(row.get("comment_body"), 1200),
                }
            )
    summary_fields = [
        "topic",
        "count",
        "name",
        "representation",
        "top_words",
        "representative_comment_ids",
        "representative_comments_short",
        "unique_posts",
        "unique_subreddits",
        "subreddit_distribution",
        "top_post_months",
        "top_comment_months",
        "uk_strong_1",
        "uk_strong_0",
        "share_uk_strong",
    ]
    write_csv(output_dir / f"{prefix}_topic_summary.csv", summary_rows, summary_fields)
    write_csv(output_dir / f"{prefix}_bertopic_topic_summary.csv", summary_rows, summary_fields)
    write_csv(
        output_dir / f"{prefix}_representative_comments.csv",
        representative_output,
        [
            "topic",
            "rank_in_topic",
            "comment_id",
            "post_id",
            "industry",
            "subreddit",
            "uk_strong_relevant",
            "post_month",
            "comment_month",
            "post_title",
            "comment_body",
        ],
    )
    write_csv(
        output_dir / f"{prefix}_bertopic_representative_comments_top10.csv",
        representative_output,
        [
            "topic",
            "rank_in_topic",
            "comment_id",
            "post_id",
            "industry",
            "subreddit",
            "uk_strong_relevant",
            "post_month",
            "comment_month",
            "post_title",
            "comment_body",
        ],
    )

    post_out = month_out_of_window(assigned["post_month"])
    comment_out = month_out_of_window(assigned["comment_month"])
    out_cols = [
        "bertopic_topic",
        "subreddit",
        "post_id",
        "comment_id",
        "post_month",
        "comment_month",
        "post_title",
        "comment_body",
    ]
    assigned.loc[post_out | comment_out, out_cols].to_csv(
        output_dir / f"{prefix}_rows_outside_date_window.csv", index=False
    )
    date_audit = [
        {"metric": "rows_modelled", "value": len(assigned)},
        {"metric": "window_start", "value": START_MONTH},
        {"metric": "window_end", "value": END_MONTH},
        {"metric": "post_month_outside_window_rows", "value": int(post_out.sum())},
        {"metric": "comment_month_outside_window_rows", "value": int(comment_out.sum())},
        {
            "metric": "comment_month_outside_window_rate",
            "value": round(float(comment_out.sum() / len(assigned)), 6) if len(assigned) else 0,
        },
        {"metric": "post_month_min", "value": assigned["post_month"].replace("", pd.NA).dropna().min()},
        {"metric": "post_month_max", "value": assigned["post_month"].replace("", pd.NA).dropna().max()},
        {"metric": "comment_month_min", "value": assigned["comment_month"].replace("", pd.NA).dropna().min()},
        {"metric": "comment_month_max", "value": assigned["comment_month"].replace("", pd.NA).dropna().max()},
        {"metric": "out_of_window_comment_months", "value": counter_text(assigned.loc[comment_out, "comment_month"], 12)},
    ]
    write_csv(output_dir / f"{prefix}_date_window_audit_summary.csv", date_audit, ["metric", "value"])

    domination_rows: list[dict[str, Any]] = []
    for topic_id, group_df in assigned.groupby("bertopic_topic"):
        post_counts = group_df["post_id"].astype(str).str.strip().value_counts()
        subreddit_counts = group_df["subreddit"].astype(str).str.strip().value_counts()
        n = len(group_df)
        top3_post_n = int(post_counts.head(3).sum()) if len(post_counts) else 0
        domination_rows.append(
            {
                "topic": int(topic_id),
                "comments": n,
                "unique_posts": int(post_counts.size),
                "unique_subreddits": int(subreddit_counts.size),
                "largest_post_comment_n": int(post_counts.iloc[0]) if len(post_counts) else 0,
                "largest_post_comment_share": round(float(post_counts.iloc[0] / n), 4) if len(post_counts) else 0,
                "top_3_posts_comment_n": top3_post_n,
                "top_3_posts_comment_share": round(float(top3_post_n / n), 4) if n else 0,
                "largest_subreddit": subreddit_counts.index[0] if len(subreddit_counts) else "",
                "largest_subreddit_n": int(subreddit_counts.iloc[0]) if len(subreddit_counts) else 0,
                "largest_subreddit_share": round(float(subreddit_counts.iloc[0] / n), 4) if len(subreddit_counts) else 0,
                "top_post_ids": "; ".join(f"{idx}={value}" for idx, value in post_counts.head(3).items()),
                "top_subreddits": "; ".join(f"{idx}={value}" for idx, value in subreddit_counts.head(5).items()),
            }
        )
    domination = pd.DataFrame(domination_rows).sort_values(
        ["largest_post_comment_share", "top_3_posts_comment_share", "comments"],
        ascending=[False, False, False],
    )
    domination.to_csv(output_dir / f"{prefix}_topic_domination_audit.csv", index=False)
    flagged = domination[
        (domination["largest_post_comment_share"] >= 0.4)
        | (domination["top_3_posts_comment_share"] >= 0.6)
        | (domination["largest_subreddit_share"] >= 0.8)
    ].copy()
    flagged.to_csv(output_dir / f"{prefix}_topic_domination_flagged.csv", index=False)
    domination_summary = [
        {"metric": "topics_including_outlier", "value": len(set(topics))},
        {"metric": "topics_excluding_outlier", "value": len({topic for topic in topics if topic != -1})},
        {"metric": "flagged_topics", "value": len(flagged)},
        {
            "metric": "flagged_topics_share",
            "value": round(len(flagged) / len(domination), 4) if len(domination) else 0,
        },
        {"metric": "median_unique_posts_per_topic", "value": round(float(domination["unique_posts"].median()), 4)},
        {
            "metric": "median_largest_post_share",
            "value": round(float(domination["largest_post_comment_share"].median()), 4),
        },
        {
            "metric": "median_largest_subreddit_share",
            "value": round(float(domination["largest_subreddit_share"].median()), 4),
        },
    ]
    write_csv(output_dir / f"{prefix}_topic_domination_summary.csv", domination_summary, ["metric", "value"])

    outlier_count = int(sum(1 for topic in topics if topic == -1))
    model_overview = [
        {"metric": "industry", "value": args.industry},
        {"metric": "input_file", "value": str(input_path)},
        {"metric": "total_rows", "value": len(industry_df)},
        {"metric": "modelled_rows", "value": len(assigned)},
        {"metric": "excluded_empty_unusable_rows", "value": len(industry_df) - len(assigned)},
        {"metric": "text_definition", "value": "comment_body"},
        {"metric": "substantive_topics_excluding_outlier", "value": len({topic for topic in topics if topic != -1})},
        {"metric": "topics_found_including_outlier", "value": len(set(topics))},
        {"metric": "outlier_count", "value": outlier_count},
        {"metric": "outlier_rate", "value": round(outlier_count / len(assigned), 6) if len(assigned) else 0},
        {"metric": "embedding_model", "value": args.embedding_model},
        {"metric": "random_seed", "value": 42},
        {"metric": "min_topic_size", "value": args.min_topic_size},
        {"metric": "min_samples", "value": args.min_samples},
        {"metric": "min_df", "value": args.min_df},
        {"metric": "max_df", "value": args.max_df},
        {"metric": "ngram_range", "value": f"(1, {args.ngram_max})"},
        {"metric": "stopword_policy", "value": "generic_ai_reddit_terms_plus_occupational_identity_terms"},
        {"metric": "industry_identity_stopwords", "value": "; ".join(sorted(INDUSTRY_IDENTITY_STOPWORDS.get(args.industry, set())))},
        {"metric": "representation_model", "value": args.representation_model},
        {"metric": "nr_topics", "value": nr_topics if nr_topics is not None else "none"},
        {"metric": "calculate_probabilities", "value": 0},
        {"metric": "domination_flagged_topics", "value": len(flagged)},
        {"metric": "post_month_outside_window_rows", "value": int(post_out.sum())},
        {"metric": "comment_month_outside_window_rows", "value": int(comment_out.sum())},
        {"metric": "run_timestamp", "value": datetime.now().isoformat(timespec="seconds")},
    ]
    write_csv(output_dir / f"{prefix}_model_overview.csv", model_overview, ["metric", "value"])
    write_csv(output_dir / f"{prefix}_bertopic_run_summary.csv", model_overview, ["metric", "value"])

    manifest = {row["metric"]: row["value"] for row in model_overview}
    manifest["output_dir"] = str(output_dir)
    manifest["umap"] = {"n_neighbors": 15, "n_components": 5, "min_dist": 0.0, "metric": "cosine"}
    manifest["hdbscan"] = {
        "min_cluster_size": args.min_topic_size,
        "min_samples": args.min_samples,
        "metric": "euclidean",
        "prediction_data": True,
        "core_dist_n_jobs": 1,
    }
    manifest["audit_thresholds"] = {
        "largest_post_comment_share": 0.4,
        "top_3_posts_comment_share": 0.6,
        "largest_subreddit_share": 0.8,
    }
    manifest["stopword_policy"] = {
        "applied_stage": "CountVectorizer/c-TF-IDF topic labels only",
        "embedding_input": "unchanged full comment_body",
        "generic_stopwords": sorted(STOPWORDS),
        "industry_identity_stopwords": sorted(INDUSTRY_IDENTITY_STOPWORDS.get(args.industry, set())),
        "principle": "remove occupational identity/background terms, retain task/risk/workflow/trust terms",
    }
    (output_dir / f"{prefix}_config_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Topics found including outlier: {len(set(topics))}")
    print(f"Outliers (-1): {outlier_count}")
    print(f"Flagged domination topics: {len(flagged)}")
    print(f"Wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
