#!/usr/bin/env python3
"""Lexical analysis for model-identified explicit trust-boundary comments.

This script intentionally uses the frozen Prompt V2 predictions as input.
It does not call any model API, retrain a classifier, tune labels, or use
fine-grained trust classes to define the analytical subset.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
except Exception:  # pragma: no cover
    ENGLISH_STOP_WORDS = frozenset(
        """
        a about above after again against all am an and any are as at be because
        been before being below between both but by can did do does doing down
        during each few for from further had has have having he her here hers
        herself him himself his how i if in into is it its itself just me more
        most my myself of off on once only or other our ours ourselves out over
        own same she should so some such than that the their theirs them
        themselves then there these they this those through to too under until
        up very was we were what when where which while who whom why with you
        your yours yourself yourselves
        """.split()
    )


DEFAULT_INPUT = Path(
    "outputs/"
    "frozen_prompt_v2_total_posts_ranked_v3_annotation_2026-07-24/"
    "all_comments_v2_annotated.csv"
)
DEFAULT_VALIDATION_DIR = Path(
    "outputs/"
    "four_industry_full_schema_validation_100_performance_2026-07-30"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/"
    f"explicit_boundary_lexical_analysis_{date.today().isoformat()}"
)

BOUNDARY_COL = "gpt_has_explicit_trust_boundary"
COMMENT_ID_COL = "comment_id"
TARGET_TEXT_COL = "comment_body"
INDUSTRIES = ["finance", "healthcare", "law", "software_engineering"]

NEGATION_KEEP = {
    "no",
    "nor",
    "not",
    "never",
    "cannot",
    "cant",
    "can't",
    "dont",
    "don't",
    "doesnt",
    "doesn't",
    "didnt",
    "didn't",
    "wont",
    "won't",
    "wouldnt",
    "wouldn't",
    "shouldnt",
    "shouldn't",
    "without",
}
STOPWORDS = set(ENGLISH_STOP_WORDS) - NEGATION_KEEP
STOPWORDS |= {
    "im",
    "ive",
    "id",
    "youre",
    "theyre",
    "thats",
    "theres",
    "reddit",
    "comment",
    "post",
    "people",
    "thing",
    "things",
    "way",
    "lot",
    "really",
    "like",
    "just",
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
USER_SUB_RE = re.compile(r"(?:^|\s)[ru]/[A-Za-z0-9_]+", re.I)
DELETED_VALUES = {"[deleted]", "[removed]"}


@dataclass
class TermStats:
    freq: Counter
    doc_freq: Counter
    docs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--kwic-per-term", type=int, default=20)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def norm_label(value: object) -> str:
    return str(value or "").strip().lower()


def clean_text_for_tokens(text: object) -> str:
    s = html.unescape(str(text or ""))
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = URL_RE.sub(" ", s)
    s = USER_SUB_RE.sub(" ", s)
    s = re.sub(r"&\w+;", " ", s)
    s = s.lower()
    replacements = {
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def minimal_tokens(text: object) -> list[str]:
    return TOKEN_RE.findall(clean_text_for_tokens(text))


def content_tokens(text: object) -> list[str]:
    return [t for t in minimal_tokens(text) if t not in STOPWORDS and len(t) > 1]


def ngrams(tokens: list[str], n: int) -> list[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin


def preprocess_and_exclude(df: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    exclusions = []
    seen_ids: set[str] = set()
    for idx, row in df.iterrows():
        cid = str(row.get(COMMENT_ID_COL, "") or "").strip()
        industry = norm_label(row.get("industry"))
        text = str(row.get(TARGET_TEXT_COL, "") or "").strip()
        reason = ""
        if not cid:
            reason = "missing_comment_id"
        elif cid in seen_ids:
            reason = "duplicate_comment_id"
        elif not text:
            reason = "missing_or_empty_target_comment"
        elif text.lower().strip() in DELETED_VALUES:
            reason = "deleted_or_removed_comment"
        elif industry and industry not in INDUSTRIES:
            reason = "unexpected_industry"
        if reason:
            exclusions.append({"source_row": idx, "comment_id": cid, "industry": industry, "exclusion_reason": reason})
            continue
        seen_ids.add(cid)
        rows.append(row)
    kept = pd.DataFrame(rows).copy()
    log = pd.DataFrame(exclusions)
    log.to_csv(out_dir / "lexical_exclusions_log.csv", index=False)
    return kept, log


def add_token_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target_text_for_lexical_analysis"] = df[TARGET_TEXT_COL].astype(str)
    df["minimal_tokens"] = df[TARGET_TEXT_COL].map(minimal_tokens)
    df["content_tokens"] = df[TARGET_TEXT_COL].map(content_tokens)
    return df


def stats_for_docs(token_lists: Iterable[list[str]], n: int) -> TermStats:
    freq: Counter = Counter()
    doc_freq: Counter = Counter()
    docs = 0
    for toks in token_lists:
        terms = ngrams(toks, n)
        freq.update(terms)
        doc_freq.update(set(terms))
        docs += 1
    return TermStats(freq=freq, doc_freq=doc_freq, docs=docs)


def frequency_table(df: pd.DataFrame, n: int, min_df: int) -> pd.DataFrame:
    st = stats_for_docs(df["content_tokens"], n)
    rows = []
    for term, f in st.freq.items():
        d = st.doc_freq[term]
        if d >= min_df:
            rows.append(
                {
                    "ngram_n": n,
                    "term": term,
                    "frequency": f,
                    "document_frequency": d,
                    "document_prevalence": d / st.docs if st.docs else 0,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["document_frequency", "frequency", "term"], ascending=[False, False, True]).reset_index(drop=True)


def make_counts_by_label(df: pd.DataFrame, n: int) -> dict[str, TermStats]:
    out = {}
    for label, sub in df.groupby("boundary_pred"):
        out[label] = stats_for_docs(sub["content_tokens"], n)
    for label in ["yes", "no"]:
        out.setdefault(label, TermStats(Counter(), Counter(), 0))
    return out


def g2_stat(a: int, b: int, c: int, d: int) -> float:
    total = a + b + c + d
    if total == 0:
        return 0.0
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    cells = [(a, row1 * col1 / total), (b, row1 * col2 / total), (c, row2 * col1 / total), (d, row2 * col2 / total)]
    val = 0.0
    for obs, exp in cells:
        if obs > 0 and exp > 0:
            val += obs * math.log(obs / exp)
    return 2 * val


def weighted_log_odds_table(
    df: pd.DataFrame,
    n: int,
    min_total_df: int,
    min_group_df: int,
    group_col: str = "boundary_pred",
    positive: str = "yes",
    negative: str = "no",
) -> pd.DataFrame:
    stats = make_counts_by_label(df, n) if group_col == "boundary_pred" else {
        positive: stats_for_docs(df[df[group_col] == positive]["content_tokens"], n),
        negative: stats_for_docs(df[df[group_col] == negative]["content_tokens"], n),
    }
    pos = stats[positive]
    neg = stats[negative]
    vocab = set(pos.freq) | set(neg.freq)
    bg = Counter(pos.freq)
    bg.update(neg.freq)
    bg_total = sum(bg.values()) + len(vocab)
    pos_total = sum(pos.freq.values())
    neg_total = sum(neg.freq.values())
    rows = []
    for term in vocab:
        pos_f = pos.freq.get(term, 0)
        neg_f = neg.freq.get(term, 0)
        pos_df = pos.doc_freq.get(term, 0)
        neg_df = neg.doc_freq.get(term, 0)
        if (pos_df + neg_df) < min_total_df or max(pos_df, neg_df) < min_group_df:
            continue
        alpha = bg.get(term, 0) + 0.01
        alpha0 = bg_total
        pos_den = pos_total + alpha0 - pos_f - alpha
        neg_den = neg_total + alpha0 - neg_f - alpha
        if pos_den <= 0 or neg_den <= 0:
            continue
        delta = math.log((pos_f + alpha) / pos_den) - math.log((neg_f + alpha) / neg_den)
        var = 1 / (pos_f + alpha) + 1 / (neg_f + alpha)
        z = delta / math.sqrt(var)
        pos_non = max(pos.docs - pos_df, 0)
        neg_non = max(neg.docs - neg_df, 0)
        rows.append(
            {
                "ngram_n": n,
                "term": term,
                "positive_frequency": pos_f,
                "negative_frequency": neg_f,
                "positive_document_frequency": pos_df,
                "negative_document_frequency": neg_df,
                "positive_document_prevalence": pos_df / pos.docs if pos.docs else 0,
                "negative_document_prevalence": neg_df / neg.docs if neg.docs else 0,
                "log_odds_z": z,
                "log_likelihood_g2": g2_stat(pos_df, pos_non, neg_df, neg_non),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("log_odds_z", ascending=False).reset_index(drop=True)


def prevalence_tables(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for industry, sub in [("ALL", df)] + list(df.groupby("industry")):
        n = len(sub)
        yes = int((sub["boundary_pred"] == "yes").sum())
        no = int((sub["boundary_pred"] == "no").sum())
        lo, hi = wilson_ci(yes, n)
        rows.append(
            {
                "industry": industry,
                "predicted_rows": n,
                "explicit_boundary_positive": yes,
                "explicit_boundary_negative": no,
                "explicit_boundary_rate": yes / n if n else 0,
                "wilson_95_ci_low": lo,
                "wilson_95_ci_high": hi,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "explicit_boundary_prevalence_by_industry.csv", index=False)
    return out


def keyness_outputs(df: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for n in [1, 2, 3]:
        min_df = 10 if n == 1 else 5
        key = weighted_log_odds_table(df, n, min_df, min_group_df=5)
        key.to_csv(out_dir / f"keyness_boundary_positive_vs_negative_{n}gram.csv", index=False)
        outputs[f"{n}gram"] = key

    pub = pd.concat([v.assign(source=k) for k, v in outputs.items() if not v.empty], ignore_index=True)
    if not pub.empty:
        pub = pub.sort_values("log_odds_z", ascending=False).head(25)
        pub.to_csv(out_dir / "publication_table_terms_disproportionately_associated_with_explicit_boundaries.csv", index=False)
        pub.sort_values("log_odds_z").head(25).to_csv(out_dir / "diagnostic_terms_associated_with_boundary_negative_comments.csv", index=False)
    return outputs


def frequency_outputs(boundary_positive: pd.DataFrame, out_dir: Path) -> None:
    with pd.ExcelWriter(out_dir / "explicit_boundary_term_frequencies_by_industry.xlsx") as writer:
        for n in [1, 2, 3]:
            pooled = frequency_table(boundary_positive, n, 10 if n == 1 else 5)
            pooled.to_csv(out_dir / f"explicit_boundary_pooled_{n}gram_frequencies.csv", index=False)
            pooled.head(50).to_csv(out_dir / f"publication_top50_explicit_boundary_{n}grams.csv", index=False)
            for industry, sub in boundary_positive.groupby("industry"):
                tab = frequency_table(sub, n, 3)
                tab.to_excel(writer, sheet_name=f"{industry[:20]}_{n}g", index=False)


def industry_keyness_outputs(boundary_positive: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    outputs = {}
    with pd.ExcelWriter(out_dir / "industry_distinctive_terms_within_explicit_boundaries.xlsx") as writer:
        for n in [1, 2, 3]:
            all_rows = []
            for industry in INDUSTRIES:
                tmp = boundary_positive.copy()
                tmp["target_industry"] = np.where(tmp["industry"] == industry, industry, "other")
                key = weighted_log_odds_table(
                    tmp,
                    n=n,
                    min_total_df=5,
                    min_group_df=3,
                    group_col="target_industry",
                    positive=industry,
                    negative="other",
                )
                if not key.empty:
                    key.insert(0, "industry", industry)
                    all_rows.append(key)
            tab = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
            tab.to_excel(writer, sheet_name=f"{n}gram", index=False)
            outputs[f"{n}gram"] = tab
    return outputs


def pooled_keyness_industry_diagnostic(boundary_positive: pd.DataFrame, all_df: pd.DataFrame, keyness: dict[str, pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    combined = pd.concat([v.assign(kind=k) for k, v in keyness.items() if not v.empty], ignore_index=True)
    if combined.empty:
        out = pd.DataFrame()
        out.to_csv(out_dir / "pooled_keyness_industry_diagnostics.csv", index=False)
        return out
    top = combined.sort_values("log_odds_z", ascending=False).head(50)
    rows = []
    for _, termrow in top.iterrows():
        term = termrow["term"]
        n = int(termrow["ngram_n"])
        industry_rows = []
        pos_total_freq = 0
        industry_pos_freq = {}
        positive_direction_count = 0
        for industry in INDUSTRIES:
            ind = all_df[all_df["industry"] == industry]
            if ind.empty:
                continue
            tmp = ind.copy()
            pos = tmp[tmp["boundary_pred"] == "yes"]
            neg = tmp[tmp["boundary_pred"] == "no"]
            pos_terms = [ngrams(t, n) for t in pos["content_tokens"]]
            neg_terms = [ngrams(t, n) for t in neg["content_tokens"]]
            pos_freq = sum(ts.count(term) for ts in pos_terms)
            neg_freq = sum(ts.count(term) for ts in neg_terms)
            pos_df = sum(1 for ts in pos_terms if term in ts)
            neg_df = sum(1 for ts in neg_terms if term in ts)
            pos_total_freq += pos_freq
            industry_pos_freq[industry] = pos_freq
            direction = "positive" if (pos_df / len(pos) if len(pos) else 0) >= (neg_df / len(neg) if len(neg) else 0) else "negative"
            if direction == "positive" and pos_df > 0:
                positive_direction_count += 1
            industry_rows.append(
                f"{industry}: pos_df={pos_df}/{len(pos)}, neg_df={neg_df}/{len(neg)}, direction={direction}"
            )
        dominant_industry = max(industry_pos_freq, key=industry_pos_freq.get) if industry_pos_freq else ""
        dominant_share = industry_pos_freq.get(dominant_industry, 0) / pos_total_freq if pos_total_freq else 0
        if dominant_share >= 0.6 or positive_direction_count <= 1:
            label = "industry_concentrated"
        elif positive_direction_count == 2:
            label = "partial_cross_industry"
        else:
            label = "broad_cross_industry"
        rows.append(
            {
                "ngram_n": n,
                "term": term,
                "pooled_log_odds_z": termrow["log_odds_z"],
                "positive_direction_industries": positive_direction_count,
                "dominant_industry": dominant_industry,
                "dominant_industry_share_of_boundary_positive_occurrences": dominant_share,
                "diagnostic_label": label,
                "industry_details": " | ".join(industry_rows),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "pooled_keyness_industry_diagnostics.csv", index=False)
    return out


def collocation_outputs(boundary_positive: pd.DataFrame, keyness: dict[str, pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    unigram_key = keyness.get("1gram", pd.DataFrame())
    candidates = []
    if not unigram_key.empty:
        candidates.extend(unigram_key.sort_values("log_odds_z", ascending=False)["term"].head(20).tolist())
    freq = frequency_table(boundary_positive, 1, min_df=10)
    if not freq.empty:
        candidates.extend(freq["term"].head(20).tolist())
    keywords = list(dict.fromkeys([c for c in candidates if " " not in c]))[:30]
    all_tokens = list(boundary_positive["content_tokens"])
    total_freq = Counter(t for toks in all_tokens for t in toks)
    rows = []
    for keyword in keywords:
        kw_freq = total_freq[keyword]
        cooc = Counter()
        doc_cooc = Counter()
        for toks in all_tokens:
            seen = set()
            for i, tok in enumerate(toks):
                if tok != keyword:
                    continue
                window = toks[max(0, i - 5) : i] + toks[i + 1 : i + 6]
                cooc.update([w for w in window if w != keyword])
                seen.update([w for w in window if w != keyword])
            doc_cooc.update(seen)
        for collocate, count in cooc.most_common(50):
            if doc_cooc[collocate] < 5:
                continue
            denom = kw_freq + total_freq[collocate]
            logdice = 14 + math.log2((2 * count) / denom) if denom and count else float("nan")
            rows.append(
                {
                    "keyword": keyword,
                    "collocate": collocate,
                    "cooccurrence_frequency": count,
                    "cooccurrence_document_frequency": doc_cooc[collocate],
                    "keyword_frequency": kw_freq,
                    "collocate_frequency": total_freq[collocate],
                    "logdice": logdice,
                }
            )
    out = pd.DataFrame(rows).sort_values(["keyword", "logdice"], ascending=[True, False]) if rows else pd.DataFrame()
    out.to_csv(out_dir / "explicit_boundary_collocations_window5.csv", index=False)
    return out


def kwic_outputs(df: pd.DataFrame, keyness: dict[str, pd.DataFrame], out_dir: Path, seed: int, per_term: int) -> None:
    rng = random.Random(seed)
    combined = pd.concat([v.assign(kind=k) for k, v in keyness.items() if not v.empty], ignore_index=True)
    terms = combined.sort_values("log_odds_z", ascending=False)["term"].head(30).tolist() if not combined.empty else []
    rows = []
    for term in terms:
        pat = re.compile(r"\b" + re.escape(term).replace("\\ ", r"\s+") + r"\b", re.I)
        hits = []
        for _, row in df.iterrows():
            text = str(row.get(TARGET_TEXT_COL, "") or "")
            m = pat.search(clean_text_for_tokens(text))
            if not m:
                continue
            raw = text
            lower = clean_text_for_tokens(raw)
            start = max(0, m.start() - 120)
            end = min(len(lower), m.end() + 120)
            hits.append(
                {
                    "term": term,
                    "comment_id": row.get(COMMENT_ID_COL, ""),
                    "industry": row.get("industry", ""),
                    "subreddit": row.get("subreddit", ""),
                    "post_id": row.get("post_id", ""),
                    "post_title": row.get("post_title", ""),
                    "kwic_context": raw[start:end],
                    "parent_context_for_reference_only": row.get("parent_comment_body", ""),
                    "boundary_prediction": row.get(BOUNDARY_COL, ""),
                }
            )
        if len(hits) > per_term:
            hits = rng.sample(hits, per_term)
        rows.extend(hits)
    kwic = pd.DataFrame(rows)
    kwic.to_excel(out_dir / "kwic_all_candidate_terms.xlsx", index=False)
    manual = kwic.copy()
    manual["provisional_human_category"] = ""
    manual["human_interpretive_note"] = ""
    manual.to_excel(out_dir / "kwic_manual_review_sample.xlsx", index=False)


def candidate_evidence_workbook(keyness: dict[str, pd.DataFrame], colloc: pd.DataFrame, out_dir: Path) -> None:
    combined = pd.concat([v.assign(kind=k) for k, v in keyness.items() if not v.empty], ignore_index=True)
    if combined.empty:
        return
    rows = []
    for _, row in combined.sort_values("log_odds_z", ascending=False).head(100).iterrows():
        term = row["term"]
        related = ""
        if not colloc.empty and "keyword" in colloc and "collocate" in colloc:
            related = "; ".join(
                colloc[colloc["keyword"] == term].sort_values("logdice", ascending=False)["collocate"].head(8).tolist()
            )
        rows.append(
            {
                "candidate_term_or_phrase": term,
                "ngram_n": row["ngram_n"],
                "boundary_positive_document_frequency": row["positive_document_frequency"],
                "boundary_negative_document_frequency": row["negative_document_frequency"],
                "positive_document_prevalence": row["positive_document_prevalence"],
                "negative_document_prevalence": row["negative_document_prevalence"],
                "pooled_log_odds_z": row["log_odds_z"],
                "collocation_summary": related,
                "provisional_human_category": "",
                "include_in_writeup": "",
                "human_note": "",
            }
        )
    pd.DataFrame(rows).to_excel(out_dir / "candidate_boundary_lexical_evidence_workbook.xlsx", index=False)


def plot_prevalence(prev: pd.DataFrame, out_dir: Path) -> None:
    tab = prev[prev["industry"] != "ALL"].copy()
    tab = tab.sort_values("explicit_boundary_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = np.arange(len(tab))
    rate = tab["explicit_boundary_rate"].to_numpy()
    ax.barh(y, rate, color="#7cb7b8")
    ax.set_yticks(y)
    ax.set_yticklabels(tab["industry"].str.replace("_", " ").str.title())
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Explicit trust-boundary positive among predicted comments (%)")
    ax.set_title("Model-identified explicit trust boundaries by industry")
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    for i, (_, r) in enumerate(tab.iterrows()):
        ax.text(r["explicit_boundary_rate"] + 0.01, i, f"{r['explicit_boundary_rate']:.1%} ({int(r['explicit_boundary_positive'])}/{int(r['predicted_rows'])})", va="center", fontsize=9)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"figure_explicit_boundary_prevalence_by_industry.{ext}", dpi=300)
    plt.close(fig)


def plot_keyness(
    keyness: dict[str, pd.DataFrame],
    out_dir: Path,
    no_title: bool = False,
    filename_suffix: str = "",
) -> None:
    keyness_cmap = LinearSegmentedColormap.from_list(
        "boundary_keyness",
        ["#DDEFE8", "#A8D2C3", "#65A693", "#1F675F"],
    )
    ngram_labels = {"1gram": "unigrams", "2gram": "bigrams", "3gram": "trigrams"}

    for name, tab in keyness.items():
        if tab.empty:
            continue
        top = tab.sort_values("log_odds_z", ascending=False).head(20).sort_values("log_odds_z")
        values = top["log_odds_z"].to_numpy(dtype=float)
        value_range = float(values.max() - values.min())
        norm = Normalize(
            vmin=float(values.min()),
            vmax=float(values.max()) if value_range > 0 else float(values.min() + 1),
        )
        colors = keyness_cmap(norm(values))

        fig, ax = plt.subplots(figsize=(9.2, 7.4), facecolor="white")
        bars = ax.barh(
            top["term"],
            values,
            color=colors,
            edgecolor="none",
            height=0.72,
        )
        ax.set_xlabel("Weighted log-odds z-score")
        if not no_title:
            ax.set_title(
                "Terms most associated with explicit trust-boundary comments\n"
                f"({ngram_labels.get(name, name)})",
                pad=14,
            )
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#D9E1DF", linewidth=0.8, alpha=0.8)
        ax.grid(axis="y", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#8A9693")
        ax.spines["bottom"].set_color("#8A9693")
        ax.tick_params(axis="y", length=0, pad=8)
        ax.tick_params(axis="x", colors="#4C5755")
        ax.set_xlim(0, max(values) * 1.10)

        for bar, value in zip(bars, values):
            ax.text(
                value + max(values) * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=8.5,
                color="#33413E",
            )

        fig.tight_layout(pad=1.2)
        for ext in ["png", "pdf"]:
            fig.savefig(
                out_dir / f"figure_top_boundary_positive_keyness_{name}{filename_suffix}.{ext}",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
        plt.close(fig)


def plot_heatmap(boundary_positive: pd.DataFrame, keyness_diag: pd.DataFrame, out_dir: Path) -> None:
    if keyness_diag.empty:
        return
    terms = keyness_diag.head(20)["term"].tolist()
    rows = []
    for industry in INDUSTRIES:
        sub = boundary_positive[boundary_positive["industry"] == industry]
        docs = len(sub)
        for term in terms:
            n = len(term.split())
            df = sum(1 for toks in sub["content_tokens"] if term in ngrams(toks, n))
            rows.append({"industry": industry, "term": term, "prevalence": df / docs if docs else 0})
    pivot = pd.DataFrame(rows).pivot(index="industry", columns="term", values="prevalence").loc[INDUSTRIES]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([x.replace("_", " ").title() for x in pivot.index])
    ax.set_title("Boundary-associated term prevalence within boundary-positive comments")
    fig.colorbar(im, ax=ax, label="Document prevalence")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"figure_boundary_keyness_terms_by_industry_heatmap.{ext}", dpi=300)
    plt.close(fig)


def robustness_checks(df: pd.DataFrame, out_dir: Path) -> None:
    repeated = (
        df.assign(normalized_text=df[TARGET_TEXT_COL].map(lambda x: re.sub(r"\s+", " ", clean_text_for_tokens(x)).strip()))
        .groupby("normalized_text")
        .agg(n=("comment_id", "count"), industries=("industry", lambda x: ";".join(sorted(set(map(str, x))))))
        .reset_index()
        .query("n > 1")
        .sort_values("n", ascending=False)
    )
    repeated.to_csv(out_dir / "repeated_text_diagnostics_not_excluded.csv", index=False)
    rows = []
    for n in [1, 2, 3]:
        base = weighted_log_odds_table(df, n, min_total_df=10 if n == 1 else 5, min_group_df=5)
        base_terms = set(base.sort_values("log_odds_z", ascending=False).head(50)["term"]) if not base.empty else set()
        for threshold in [5, 10, 20]:
            comp = weighted_log_odds_table(df, n, min_total_df=threshold, min_group_df=max(3, threshold // 2))
            terms = set(comp.sort_values("log_odds_z", ascending=False).head(50)["term"]) if not comp.empty else set()
            jaccard = len(base_terms & terms) / len(base_terms | terms) if (base_terms | terms) else 0
            rows.append({"ngram_n": n, "comparison_min_total_df": threshold, "top50_jaccard_against_main": jaccard})
    pd.DataFrame(rows).to_excel(out_dir / "keyness_threshold_sensitivity.xlsx", index=False)
    (out_dir / "lexical_robustness_checks.md").write_text(
        "# Lexical Robustness Checks\n\n"
        "- Exact repeated target-comment text was diagnosed but not removed, consistent with the exclusion rules.\n"
        "- Keyness threshold sensitivity reports top-50 overlap across alternative document-frequency thresholds.\n"
        "- Frequencies are reported both as raw token frequencies and document frequencies.\n",
        encoding="utf-8",
    )


def load_validation(validation_dir: Path) -> dict[str, object]:
    out: dict[str, object] = {"validation_dir": str(validation_dir)}
    summary = validation_dir / "four_industry_full_schema_validation_100_gate_metrics_summary.csv"
    boot = validation_dir / "four_industry_full_schema_validation_100_explicit_boundary_bootstrap_ci.csv"
    if summary.exists():
        out["validation_gate_metrics_summary"] = pd.read_csv(summary).to_dict(orient="records")
    if boot.exists():
        out["explicit_boundary_bootstrap_ci"] = pd.read_csv(boot).to_dict(orient="records")
    return out


def write_report(metrics: dict[str, object], out_dir: Path) -> None:
    validation = metrics.get("validation", {})
    report = [
        "# Explicit Trust-Boundary Lexical Analysis",
        "",
        "Analytical subset: comments classified as containing an explicit trust boundary by the validated binary boundary classifier.",
        "",
        "This analysis does not use the substantive-trust gate or fine-grained trust construct/boundary labels to define or subdivide the sample. It uses only the target comment text for tokenisation. Parent context is retained only in KWIC outputs.",
        "",
        "Interpretive caution: this is a model-identified boundary-positive subset. The classifier is precision-oriented but not exhaustive, so results should not be read as all trust comments in the corpus.",
        "",
        "## Input",
        f"- Input file: `{metrics['input_file']}`",
        f"- Input SHA-256: `{metrics['input_sha256']}`",
        f"- Total input rows: {metrics['total_input_rows']}",
        f"- Rows with frozen explicit-boundary predictions: {metrics['predicted_rows']}",
        f"- Boundary-positive rows after exclusions: {metrics['boundary_positive_rows_after_exclusions']}",
        f"- Excluded rows: {metrics['excluded_rows']}",
        "",
        "## Validation context",
        f"- Validation directory: `{validation.get('validation_dir', '')}`",
        "- Exact validation metrics are copied into `explicit_boundary_lexical_analysis_metrics.json`.",
        "",
        "## Main outputs",
        "- `explicit_boundary_prevalence_by_industry.csv`",
        "- `explicit_boundary_pooled_1gram_frequencies.csv`, `explicit_boundary_pooled_2gram_frequencies.csv`, `explicit_boundary_pooled_3gram_frequencies.csv`",
        "- `keyness_boundary_positive_vs_negative_1gram.csv`, `2gram.csv`, `3gram.csv`",
        "- `pooled_keyness_industry_diagnostics.csv`",
        "- `industry_distinctive_terms_within_explicit_boundaries.xlsx`",
        "- `explicit_boundary_collocations_window5.csv`",
        "- `kwic_manual_review_sample.xlsx`",
        "- `candidate_boundary_lexical_evidence_workbook.xlsx`",
        "- `figures` are saved as PNG and PDF in this folder.",
    ]
    (out_dir / "explicit_boundary_lexical_analysis_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.input, low_memory=False)
    if BOUNDARY_COL not in df.columns:
        raise KeyError(f"Required column not found: {BOUNDARY_COL}")
    if TARGET_TEXT_COL not in df.columns:
        raise KeyError(f"Required target-comment column not found: {TARGET_TEXT_COL}")

    df["boundary_pred"] = df[BOUNDARY_COL].map(norm_label)
    predicted = df[df["boundary_pred"].isin(["yes", "no"])].copy()

    clean, exclusions = preprocess_and_exclude(predicted, args.output_dir)
    clean = add_token_columns(clean)
    boundary_positive = clean[clean["boundary_pred"] == "yes"].copy()
    boundary_positive.to_csv(args.output_dir / "explicit_boundary_positive_comments_for_lexical_analysis.csv", index=False)

    prevalence = prevalence_tables(clean, args.output_dir)
    frequency_outputs(boundary_positive, args.output_dir)
    keyness = keyness_outputs(clean, args.output_dir)
    industry_keyness_outputs(boundary_positive, args.output_dir)
    keyness_diag = pooled_keyness_industry_diagnostic(boundary_positive, clean, keyness, args.output_dir)
    colloc = collocation_outputs(boundary_positive, keyness, args.output_dir)
    kwic_outputs(boundary_positive, keyness, args.output_dir, seed=args.seed, per_term=args.kwic_per_term)
    candidate_evidence_workbook(keyness, colloc, args.output_dir)
    robustness_checks(clean, args.output_dir)

    plot_prevalence(prevalence, args.output_dir)
    plot_keyness(keyness, args.output_dir)
    plot_heatmap(boundary_positive, keyness_diag, args.output_dir)

    metrics = {
        "created_at": date.today().isoformat(),
        "input_file": str(args.input),
        "input_sha256": sha256_file(args.input),
        "total_input_rows": int(len(df)),
        "predicted_rows": int(len(predicted)),
        "boundary_positive_rows_before_exclusions": int((predicted["boundary_pred"] == "yes").sum()),
        "boundary_negative_rows_before_exclusions": int((predicted["boundary_pred"] == "no").sum()),
        "boundary_positive_rows_after_exclusions": int(len(boundary_positive)),
        "excluded_rows": int(len(exclusions)),
        "exclusion_counts": exclusions["exclusion_reason"].value_counts().to_dict() if not exclusions.empty else {},
        "industry_counts_predicted": clean["industry"].value_counts().to_dict(),
        "industry_counts_boundary_positive": boundary_positive["industry"].value_counts().to_dict(),
        "text_policy": "Only target comment text is tokenised; parent context is retained only for KWIC reference.",
        "sample_definition": "Comments classified as containing an explicit trust boundary by the validated binary boundary classifier.",
        "validation": load_validation(args.validation_dir),
    }
    (args.output_dir / "explicit_boundary_lexical_analysis_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(metrics, args.output_dir)

    print(f"Input rows: {len(df)}")
    print(f"Rows with explicit-boundary prediction: {len(predicted)}")
    print(f"Boundary-positive rows after exclusions: {len(boundary_positive)}")
    print(f"Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
