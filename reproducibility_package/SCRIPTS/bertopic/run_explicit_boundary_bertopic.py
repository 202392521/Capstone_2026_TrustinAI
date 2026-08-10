#!/usr/bin/env python3
"""Fit the final pooled BERTopic model for explicit-boundary comments.

The input is the locked 861-comment GPT-5 mini positive-boundary subset. The
model uses comment_body only and preserves all source metadata in the document
assignment export. Raw topic labels remain provisional pending human review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("REPRO_TEMP_DIR", "/tmp")) / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(os.environ.get("REPRO_TEMP_DIR", "/tmp")) / "numba_cache"))

import bertopic
import hdbscan
import numpy as np
import pandas as pd
import sentence_transformers
import sklearn
import umap
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
DEFAULT_INPUT = (
    OUTPUTS
    / "balanced_2000_final_stance_analysis_2026-08-01"
    / "balanced_2000_explicit_trust_boundary_subset.csv"
)
DEFAULT_OUTPUT = OUTPUTS / "explicit_boundary_bertopic_FINAL_2026-08-03"
INDUSTRIES = ["finance", "healthcare", "law", "software_engineering"]
RANDOM_SEED = 42
MIN_DOC_CHARS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--min-topic-size", type=int, default=20)
    parser.add_argument("--min-samples", type=int, default=5)
    # BERTopic vectorises one aggregated document per discovered topic. With
    # this 861-comment subset, min_df=5 can exceed max_df * n_topics, so the
    # small-corpus representation layer uses min_df=1.
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-df", type=float, default=0.8)
    parser.add_argument("--ngram-max", type=int, default=3)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pooled_stopwords() -> list[str]:
    words = set(STOPWORDS)
    for values in INDUSTRY_IDENTITY_STOPWORDS.values():
        words.update(values)
    return sorted(words)


def counter_text(values: pd.Series, limit: int = 12) -> str:
    counts = Counter(clean_text(value) for value in values if clean_text(value))
    return "; ".join(f"{key}={value}" for key, value in counts.most_common(limit))


def load_input(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required = {"comment_id", "post_id", "industry", "subreddit", "comment_body", "explicit_trust_boundary"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    if source["comment_id"].duplicated().any():
        raise ValueError("Input contains duplicate comment_id values")
    labels = source["explicit_trust_boundary"].map(clean_text).str.lower()
    if set(labels) != {"yes"}:
        raise ValueError(f"Input must contain only explicit-boundary positives; found {sorted(set(labels))}")
    if set(source["industry"].map(clean_text)) != set(INDUSTRIES):
        raise ValueError("Input does not contain exactly the four expected industries")

    source["bertopic_text_clean"] = source["comment_body"].map(clean_text)
    valid_mask = source["bertopic_text_clean"].str.len().ge(MIN_DOC_CHARS)
    valid = source.loc[valid_mask].copy()
    excluded = source.loc[~valid_mask].copy()
    valid["_model_position"] = np.arange(len(valid))
    return valid, excluded


def representative_rows(
    model: BERTopic,
    topic_id: int,
    topic_rows: dict[int, list[dict[str, Any]]],
    topic_doc_pairs: dict[int, list[tuple[str, dict[str, Any]]]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for representative in model.get_representative_docs(topic_id) or []:
        for doc, row in topic_doc_pairs.get(topic_id, []):
            comment_id = clean_text(row.get("comment_id"))
            if doc == representative and comment_id not in used:
                selected.append(row)
                used.add(comment_id)
                break
        if len(selected) >= limit:
            return selected
    for row in topic_rows.get(topic_id, []):
        comment_id = clean_text(row.get("comment_id"))
        if comment_id in used:
            continue
        selected.append(row)
        used.add(comment_id)
        if len(selected) >= limit:
            break
    return selected


def export_outputs(model: BERTopic, valid: pd.DataFrame, docs: list[str], topics: list[int], output: Path) -> dict[str, Any]:
    assigned = valid.copy()
    assigned["boundary_topic"] = topics
    assigned["boundary_topic_is_unassigned"] = assigned["boundary_topic"].eq(-1).astype(int)
    assigned.to_csv(output / "explicit_boundary_document_topic_assignments.csv", index=False)

    topic_info = model.get_topic_info()
    topic_info.to_csv(output / "explicit_boundary_topic_info.csv", index=False)

    records = assigned.to_dict("records")
    topic_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    topic_doc_pairs: dict[int, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for row, doc, topic in zip(records, docs, topics):
        topic_rows[int(topic)].append(row)
        topic_doc_pairs[int(topic)].append((doc, row))

    summary: list[dict[str, Any]] = []
    representative_export: list[dict[str, Any]] = []
    domination: list[dict[str, Any]] = []
    n_modelled = len(assigned)
    n_nonout = int(assigned["boundary_topic"].ne(-1).sum())

    for _, info in topic_info.iterrows():
        topic_id = int(info["Topic"])
        group = assigned[assigned["boundary_topic"].eq(topic_id)]
        reps = representative_rows(model, topic_id, topic_rows, topic_doc_pairs, 10)
        count = len(group)
        summary.append(
            {
                "topic": topic_id,
                "count": count,
                "share_all_modelled_comments": round(count / n_modelled, 6),
                "share_assigned_comments": round(count / n_nonout, 6) if topic_id != -1 and n_nonout else "",
                "algorithm_generated_name": info.get("Name", ""),
                "top_words": topic_words(model, topic_id),
                "representative_comment_ids": "; ".join(clean_text(row.get("comment_id")) for row in reps),
                "representative_comments_short": " || ".join(short(row.get("comment_body"), 240) for row in reps[:5]),
                "unique_posts": group["post_id"].map(clean_text).replace("", pd.NA).dropna().nunique(),
                "unique_subreddits": group["subreddit"].map(clean_text).replace("", pd.NA).dropna().nunique(),
                "industry_distribution": counter_text(group["industry"], 4),
                "subreddit_distribution": counter_text(group["subreddit"], 12),
                "post_month_distribution": counter_text(group["post_month"], 12),
                "researcher_assigned_label": "",
                "keep_merge_exclude": "",
                "confidence": "",
                "researcher_note": "",
            }
        )
        for rank, row in enumerate(reps, start=1):
            representative_export.append(
                {
                    "topic": topic_id,
                    "rank_in_topic": rank,
                    "comment_id": clean_text(row.get("comment_id")),
                    "post_id": clean_text(row.get("post_id")),
                    "industry": clean_text(row.get("industry")),
                    "subreddit": clean_text(row.get("subreddit")),
                    "post_title": clean_text(row.get("post_title")),
                    "comment_body": clean_text(row.get("comment_body")),
                }
            )

        post_counts = group["post_id"].map(clean_text).value_counts()
        subreddit_counts = group["subreddit"].map(clean_text).value_counts()
        largest_post = int(post_counts.iloc[0]) if len(post_counts) else 0
        top_three = int(post_counts.head(3).sum()) if len(post_counts) else 0
        largest_subreddit = int(subreddit_counts.iloc[0]) if len(subreddit_counts) else 0
        domination.append(
            {
                "topic": topic_id,
                "comments": count,
                "unique_posts": int(post_counts.size),
                "unique_subreddits": int(subreddit_counts.size),
                "largest_post_comment_n": largest_post,
                "largest_post_comment_share": round(largest_post / count, 6) if count else 0,
                "top_3_posts_comment_n": top_three,
                "top_3_posts_comment_share": round(top_three / count, 6) if count else 0,
                "largest_subreddit_n": largest_subreddit,
                "largest_subreddit_share": round(largest_subreddit / count, 6) if count else 0,
            }
        )

    pd.DataFrame(summary).to_csv(output / "explicit_boundary_topic_summary_for_human_review.csv", index=False)
    pd.DataFrame(representative_export).to_csv(output / "explicit_boundary_representative_comments_top10.csv", index=False)
    pd.DataFrame(domination).to_csv(output / "explicit_boundary_topic_domination_audit.csv", index=False)

    industry_counts = assigned.pivot_table(
        index="boundary_topic", columns="industry", values="comment_id", aggfunc="count", fill_value=0
    ).reset_index()
    industry_counts.to_csv(output / "explicit_boundary_topic_by_industry_counts.csv", index=False)
    industry_shares = industry_counts.copy()
    for industry in INDUSTRIES:
        if industry not in industry_shares:
            industry_shares[industry] = 0.0
        denominator = int((assigned["industry"] == industry).sum())
        industry_shares[industry] = industry_shares[industry] / denominator if denominator else 0.0
    industry_shares.to_csv(output / "explicit_boundary_topic_by_industry_within_industry_share.csv", index=False)

    overview = {
        "input_rows": 861,
        "modelled_documents": n_modelled,
        "excluded_short_or_blank_documents": 861 - n_modelled,
        "substantive_topics_excluding_unassigned": int(assigned.loc[assigned["boundary_topic"].ne(-1), "boundary_topic"].nunique()),
        "unassigned_count": int(assigned["boundary_topic"].eq(-1).sum()),
        "unassigned_rate": round(float(assigned["boundary_topic"].eq(-1).mean()), 6),
        "industry_counts": assigned["industry"].value_counts().sort_index().to_dict(),
    }
    pd.DataFrame([overview]).to_csv(output / "explicit_boundary_model_overview.csv", index=False)
    return overview


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()

    valid, excluded = load_input(input_path)
    excluded.to_csv(output / "explicit_boundary_excluded_short_or_blank_comments.csv", index=False)
    docs = valid["bertopic_text_clean"].tolist()

    print(f"Input boundary-positive rows: {len(valid) + len(excluded)}")
    print(f"Modelled comment_body documents: {len(valid)}")
    print(f"Excluded short/blank documents: {len(excluded)}")
    print(f"Industry distribution: {valid['industry'].value_counts().sort_index().to_dict()}")
    print(f"Loading embedding model: {args.embedding_model}")

    embedding_model = SentenceTransformer(args.embedding_model)
    embeddings = embedding_model.encode(
        docs, show_progress_bar=True, batch_size=64, normalize_embeddings=True
    )
    np.save(output / "explicit_boundary_comment_body_embeddings.npy", embeddings)

    vectorizer = CountVectorizer(
        stop_words=pooled_stopwords(),
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
    )
    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=RANDOM_SEED,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=args.min_topic_size,
            min_samples=args.min_samples,
            metric="euclidean",
            prediction_data=True,
            core_dist_n_jobs=1,
        ),
        vectorizer_model=vectorizer,
        representation_model=make_representation_model("keybert_mmr"),
        min_topic_size=args.min_topic_size,
        nr_topics=None,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = model.fit_transform(docs, embeddings=embeddings)
    topics = [int(topic) for topic in topics]
    overview = export_outputs(model, valid, docs, topics, output)
    model.save(output / "explicit_boundary_bertopic_model", serialization="pickle")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "raw_topics_generated_pending_human_interpretation",
        "input_file": str(input_path),
        "input_sha256": sha256(input_path),
        "script_file": str(script_path),
        "script_sha256": sha256(script_path),
        "text_definition": "comment_body only",
        "selection_rule": "explicit_trust_boundary == yes in the locked balanced 2000 sample",
        "embedding_model": args.embedding_model,
        "umap": {"n_neighbors": 15, "n_components": 5, "min_dist": 0.0, "metric": "cosine", "random_state": 42},
        "hdbscan": {"min_cluster_size": args.min_topic_size, "min_samples": args.min_samples, "metric": "euclidean"},
        "vectorizer": {"ngram_range": [1, args.ngram_max], "min_df": args.min_df, "max_df": args.max_df},
        "representation_model": "KeyBERTInspired(top_n_words=12) + MaximalMarginalRelevance(diversity=0.3)",
        "automatic_topic_reduction": False,
        "unassigned_topic_reassignment": False,
        "minimum_document_characters": MIN_DOC_CHARS,
        "overview": overview,
        "versions": {
            "python": platform.python_version(),
            "bertopic": getattr(bertopic, "__version__", "unknown"),
            "sentence_transformers": getattr(sentence_transformers, "__version__", "unknown"),
            "umap": getattr(umap, "__version__", "unknown"),
            "hdbscan": getattr(hdbscan, "__version__", "unknown"),
            "sklearn": getattr(sklearn, "__version__", "unknown"),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Explicit trust-boundary pooled BERTopic\n\n"
        "This model is fitted to the 861 GPT-5 mini comments classified as containing an explicit trust boundary in the locked balanced four-industry sample.\n\n"
        "- Input text: `comment_body` only.\n"
        "- This is a pooled model; industry is retained as metadata.\n"
        "- Topic -1 remains the model-designated unassigned class and is not reassigned.\n"
        "- Automatic topic reduction is disabled.\n"
        "- The small-corpus c-TF-IDF layer uses `min_df=1`; the full-corpus value of 5 is incompatible when only a few topic-documents are discovered.\n"
        "- Algorithm-generated topic names are provisional. Use `explicit_boundary_topic_summary_for_human_review.csv` to assign final researcher labels.\n"
        "- Counts describe the balanced 2,000-comment analytical sample, not the full Reddit corpus.\n",
        encoding="utf-8",
    )

    print(json.dumps(overview, indent=2))
    print(f"Output directory: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
