#!/usr/bin/env python3
"""Create a reproducible stratified annotation sample for Frozen Prompt V2.

The sample is balanced by industry and spread across subreddits/posts to avoid
letting a few large threads dominate. Existing annotations are copied into the
sample checkpoint so only unannotated sampled comments need new API calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


OUT = Path(os.environ.get("REPRO_OUTPUTS_DIR", "outputs"))
MASTER = OUT / "four_industry_ai_comments_master_harmonised_total_posts_ranked_v3.csv"
FULL_RAW = OUT / "frozen_prompt_v2_total_posts_ranked_v3_annotation_2026-07-24/raw_api_responses.jsonl"
SAMPLE_DIR = OUT / "frozen_prompt_v2_stratified_sample_2000_2026-07-24"
SAMPLE_INPUT = SAMPLE_DIR / "stratified_sample_input.csv"
SAMPLE_RAW = SAMPLE_DIR / "raw_api_responses.jsonl"
SAMPLE_SUMMARY = SAMPLE_DIR / "stratified_sample_summary.csv"
SUBREDDIT_SUMMARY = SAMPLE_DIR / "stratified_sample_by_subreddit.csv"
POST_SUMMARY = SAMPLE_DIR / "stratified_sample_by_post_top50.csv"


def read_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df = df.copy()
    df["industry"] = df["source_dataset"] if "source_dataset" in df.columns else df.get("industry_keyword", "")
    if "comment_id" not in df.columns:
        df["comment_id"] = ""
    df["comment_id"] = df["comment_id"].astype(str)
    missing = df["comment_id"].str.strip() == ""
    df.loc[missing, "comment_id"] = [f"row_{i + 1}" for i in df.index[missing]]
    df["annotation_id"] = df["comment_id"]
    duplicated = df["annotation_id"].duplicated(keep=False)
    df.loc[duplicated, "annotation_id"] = [
        f"{cid}__row_{idx + 1}" for idx, cid in zip(df.index[duplicated], df.loc[duplicated, "comment_id"])
    ]
    df["uk_status"] = df.get("uk_strong_relevant", "").map(lambda x: "UK_strong" if str(x) == "1" else "non_UK_or_unclear")
    return df


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def load_raw_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        annotation_id = str(row.get("annotation_id") or row.get("comment_id") or "").strip()
        if not annotation_id:
            continue
        if annotation_id not in rows or str(rows[annotation_id].get("parse_status")) != "ok":
            rows[annotation_id] = row
    return rows


def sqrt_allocations(counts: Dict[str, int], target: int) -> Dict[str, int]:
    positive = {key: value for key, value in counts.items() if value > 0}
    if not positive:
        return {}
    weights = {key: math.sqrt(value) for key, value in positive.items()}
    total_weight = sum(weights.values())
    alloc = {key: min(positive[key], max(1, int(round(target * weights[key] / total_weight)))) for key in positive}
    while sum(alloc.values()) > target:
        candidates = [key for key in alloc if alloc[key] > 1]
        if not candidates:
            break
        key = max(candidates, key=lambda item: alloc[item])
        alloc[key] -= 1
    while sum(alloc.values()) < target:
        candidates = [key for key in alloc if alloc[key] < positive[key]]
        if not candidates:
            break
        key = max(candidates, key=lambda item: positive[item] - alloc[item])
        alloc[key] += 1
    return alloc


def sample_group(group: pd.DataFrame, n: int, post_cap: int, rng: random.Random) -> pd.DataFrame:
    if len(group) <= n:
        return group.copy()
    shuffled = group.copy()
    shuffled["_rand"] = [rng.random() for _ in range(len(shuffled))]
    shuffled = shuffled.sort_values(["_rand"])
    selected = []
    post_counts: Counter[str] = Counter()
    for _, row in shuffled.iterrows():
        post_id = str(row.get("post_id") or "")
        if post_counts[post_id] >= post_cap:
            continue
        selected.append(row)
        post_counts[post_id] += 1
        if len(selected) >= n:
            break
    if len(selected) < n:
        selected_ids = {str(row.get("annotation_id")) for row in selected}
        for _, row in shuffled.iterrows():
            if str(row.get("annotation_id")) in selected_ids:
                continue
            selected.append(row)
            if len(selected) >= n:
                break
    return pd.DataFrame(selected).drop(columns=["_rand"], errors="ignore")


def build_sample(df: pd.DataFrame, target_per_industry: int, post_cap: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    pieces = []
    for industry, industry_df in df.groupby("industry", sort=True):
        target = min(target_per_industry, len(industry_df))
        counts = industry_df.groupby("subreddit").size().to_dict()
        alloc = sqrt_allocations(counts, target)
        industry_pieces = []
        for subreddit, n in alloc.items():
            sub_df = industry_df[industry_df["subreddit"] == subreddit]
            industry_pieces.append(sample_group(sub_df, n, post_cap, rng))
        sampled = pd.concat(industry_pieces, ignore_index=True) if industry_pieces else industry_df.head(0)
        if len(sampled) < target:
            remaining = industry_df[~industry_df["annotation_id"].isin(set(sampled["annotation_id"]))]
            sampled = pd.concat([sampled, sample_group(remaining, target - len(sampled), post_cap, rng)], ignore_index=True)
        pieces.append(sampled.head(target))
    sample = pd.concat(pieces, ignore_index=True)
    sample["_sample_rand"] = [rng.random() for _ in range(len(sample))]
    return sample.sort_values(["industry", "_sample_rand"]).drop(columns=["_sample_rand"], errors="ignore")


def write_seeded_raw(sample: pd.DataFrame, raw_by_id: Dict[str, Dict[str, Any]], path: Path) -> int:
    sample_ids = set(sample["annotation_id"].astype(str))
    rows = [row for key, row in raw_by_id.items() if key in sample_ids and str(row.get("parse_status")) == "ok"]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def write_summary(sample: pd.DataFrame, seeded_n: int, target_per_industry: int, post_cap: int, seed: int) -> None:
    summary_rows = []
    for industry, group in sample.groupby("industry"):
        existing = int(group["has_existing_annotation"].sum())
        summary_rows.append(
            {
                "industry": industry,
                "sample_comments": len(group),
                "existing_annotations_reused": existing,
                "new_annotations_needed": len(group) - existing,
                "unique_subreddits": group["subreddit"].nunique(),
                "unique_posts": group["post_id"].nunique(),
                "uk_strong_comments": int((group["uk_status"] == "UK_strong").sum()),
            }
        )
    summary_rows.append(
        {
            "industry": "TOTAL",
            "sample_comments": len(sample),
            "existing_annotations_reused": seeded_n,
            "new_annotations_needed": len(sample) - seeded_n,
            "unique_subreddits": sample["subreddit"].nunique(),
            "unique_posts": sample["post_id"].nunique(),
            "uk_strong_comments": int((sample["uk_status"] == "UK_strong").sum()),
        }
    )
    pd.DataFrame(summary_rows).to_csv(SAMPLE_SUMMARY, index=False)

    by_sub = (
        sample.groupby(["industry", "subreddit", "uk_status"], dropna=False)
        .agg(
            sample_comments=("annotation_id", "size"),
            existing_annotations_reused=("has_existing_annotation", "sum"),
            unique_posts=("post_id", "nunique"),
        )
        .reset_index()
        .sort_values(["industry", "sample_comments"], ascending=[True, False])
    )
    by_sub.to_csv(SUBREDDIT_SUMMARY, index=False)

    by_post = (
        sample.groupby(["industry", "subreddit", "post_id", "post_title"], dropna=False)
        .agg(sample_comments=("annotation_id", "size"))
        .reset_index()
        .sort_values("sample_comments", ascending=False)
        .head(50)
    )
    by_post.to_csv(POST_SUMMARY, index=False)

    run_config = {
        "seed": seed,
        "target_per_industry": target_per_industry,
        "post_cap": post_cap,
        "master": str(MASTER),
        "source_raw": str(FULL_RAW),
        "sample_input": str(SAMPLE_INPUT),
        "sample_raw_checkpoint": str(SAMPLE_RAW),
    }
    (SAMPLE_DIR / "sample_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a stratified Prompt V2 annotation sample.")
    parser.add_argument("--target-per-industry", type=int, default=500)
    parser.add_argument("--post-cap", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    df = read_master(MASTER)
    raw_by_id = load_raw_by_id(FULL_RAW)
    sample = build_sample(df, args.target_per_industry, args.post_cap, args.seed)
    sample["has_existing_annotation"] = sample["annotation_id"].isin(raw_by_id).astype(int)
    sample.to_csv(SAMPLE_INPUT, index=False, encoding="utf-8-sig")
    seeded_n = write_seeded_raw(sample, raw_by_id, SAMPLE_RAW)
    write_summary(sample, seeded_n, args.target_per_industry, args.post_cap, args.seed)
    print(f"Sample input: {SAMPLE_INPUT}")
    print(f"Sample rows: {len(sample)}")
    print(f"Seeded existing annotations: {seeded_n}")
    print(f"New annotations needed: {len(sample) - seeded_n}")
    print(f"Summary: {SAMPLE_SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
