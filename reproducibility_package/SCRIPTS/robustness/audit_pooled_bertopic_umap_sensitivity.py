#!/usr/bin/env python3
"""Audit pooled BERTopic clustering sensitivity to frozen UMAP choices.

This is a robustness audit, not a model-selection routine. It reuses the
locked pooled corpus and cached sentence-transformer embeddings, varies only
UMAP n_neighbors/min_dist, keeps HDBSCAN fixed, and compares every result with
the locked pooled topic assignments.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import sklearn
import umap
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from umap import UMAP


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FINAL_DIR = PROJECT_ROOT / "cross_industry_pooled_bertopic_FINAL_2026-07-27"
DEFAULT_OUTPUT_DIR = DEFAULT_FINAL_DIR / "audits" / "umap_parameter_sensitivity_2026-08-09"


@dataclass(frozen=True)
class UmapConfig:
    name: str
    n_neighbors: int
    min_dist: float


# Pre-specified before examining sensitivity outputs. The official setting is
# included as a reproducibility run. No configuration is selected as a new
# final model based on these results.
CONFIGS = (
    UmapConfig("nn5_md0", 5, 0.0),
    UmapConfig("nn10_md0", 10, 0.0),
    UmapConfig("nn15_md0_reproduction", 15, 0.0),
    UmapConfig("nn30_md0", 30, 0.0),
    UmapConfig("nn50_md0", 50, 0.0),
    UmapConfig("nn15_md01", 15, 0.1),
    UmapConfig("nn15_md05", 15, 0.5),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--config",
        action="append",
        choices=[config.name for config in CONFIGS],
        help="Run only the named pre-specified configuration; may be repeated.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def verify_alignment(index: pd.DataFrame, assignments: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    if len(index) != len(assignments) or len(index) != len(embeddings):
        raise ValueError(
            f"Row mismatch: index={len(index):,}, assignments={len(assignments):,}, "
            f"embeddings={len(embeddings):,}"
        )

    required = {"stable_comment_id", "_pooled_pos", "pooled_topic"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Assignment file lacks required columns: {sorted(missing)}")

    ordered = assignments.sort_values("_pooled_pos").reset_index(drop=True)
    positions = pd.to_numeric(ordered["_pooled_pos"], errors="raise").astype(int).to_numpy()
    expected = np.arange(len(ordered))
    if not np.array_equal(positions, expected):
        raise ValueError("_pooled_pos is not a complete zero-based sequence.")

    left = index["stable_comment_id"].astype(str).to_numpy()
    right = ordered["stable_comment_id"].astype(str).to_numpy()
    if not np.array_equal(left, right):
        mismatch = int(np.flatnonzero(left != right)[0])
        raise ValueError(f"Comment alignment failed at pooled position {mismatch}.")
    return ordered


def best_overlap_table(reference: np.ndarray, candidate: np.ndarray, config: UmapConfig) -> pd.DataFrame:
    ref_topics = sorted(int(x) for x in np.unique(reference) if x >= 0)
    cand_topics = sorted(int(x) for x in np.unique(candidate) if x >= 0)
    ref_sizes = {topic: int(np.sum(reference == topic)) for topic in ref_topics}
    cand_sizes = {topic: int(np.sum(candidate == topic)) for topic in cand_topics}
    rows: list[dict[str, object]] = []

    for ref_topic in ref_topics:
        ref_mask = reference == ref_topic
        best: dict[str, object] | None = None
        for cand_topic in cand_topics:
            intersection = int(np.sum(ref_mask & (candidate == cand_topic)))
            if intersection == 0:
                continue
            union = ref_sizes[ref_topic] + cand_sizes[cand_topic] - intersection
            row = {
                "config": config.name,
                "reference_topic": ref_topic,
                "candidate_topic": cand_topic,
                "reference_size": ref_sizes[ref_topic],
                "candidate_size": cand_sizes[cand_topic],
                "intersection": intersection,
                "jaccard": intersection / union,
                "reference_recall": intersection / ref_sizes[ref_topic],
                "candidate_precision": intersection / cand_sizes[cand_topic],
            }
            if best is None or (row["jaccard"], row["intersection"]) > (
                best["jaccard"],
                best["intersection"],
            ):
                best = row
        if best is None:
            best = {
                "config": config.name,
                "reference_topic": ref_topic,
                "candidate_topic": -1,
                "reference_size": ref_sizes[ref_topic],
                "candidate_size": int(np.sum(candidate == -1)),
                "intersection": int(np.sum(ref_mask & (candidate == -1))),
                "jaccard": 0.0,
                "reference_recall": 0.0,
                "candidate_precision": 0.0,
            }
        rows.append(best)
    return pd.DataFrame(rows)


def score_configuration(
    reference: np.ndarray,
    candidate: np.ndarray,
    config: UmapConfig,
) -> tuple[dict[str, object], pd.DataFrame]:
    if len(reference) != len(candidate):
        raise ValueError("Reference and candidate label vectors differ in length.")

    both_nonout = (reference >= 0) & (candidate >= 0)
    reference_nonout = reference >= 0
    candidate_nonout = candidate >= 0
    matches = best_overlap_table(reference, candidate, config)
    weights = matches["reference_size"].to_numpy(dtype=float)

    candidate_topic_sizes = pd.Series(candidate[candidate_nonout]).value_counts()
    metrics: dict[str, object] = {
        "config": config.name,
        "n_neighbors": config.n_neighbors,
        "min_dist": config.min_dist,
        "documents": int(len(reference)),
        "candidate_substantive_topics": int(len(candidate_topic_sizes)),
        "candidate_outlier_count": int(np.sum(candidate == -1)),
        "candidate_outlier_rate": float(np.mean(candidate == -1)),
        "largest_candidate_topic_share_nonout": (
            float(candidate_topic_sizes.max() / candidate_topic_sizes.sum()) if len(candidate_topic_sizes) else np.nan
        ),
        "ari_all": float(adjusted_rand_score(reference, candidate)),
        "ami_all": float(adjusted_mutual_info_score(reference, candidate)),
        "nmi_all": float(normalized_mutual_info_score(reference, candidate)),
        "ari_common_nonout": (
            float(adjusted_rand_score(reference[both_nonout], candidate[both_nonout]))
            if both_nonout.sum() > 1
            else np.nan
        ),
        "reference_nonout_retained_rate": float(np.mean(candidate[reference_nonout] >= 0)),
        "candidate_nonout_also_reference_nonout_rate": float(np.mean(reference[candidate_nonout] >= 0)),
        "outlier_status_agreement": float(np.mean((reference == -1) == (candidate == -1))),
        "weighted_best_topic_jaccard": float(np.average(matches["jaccard"], weights=weights)),
        "median_best_topic_jaccard": float(matches["jaccard"].median()),
        "weighted_reference_topic_recall": float(np.average(matches["reference_recall"], weights=weights)),
        "reference_topics_jaccard_ge_050": int((matches["jaccard"] >= 0.50).sum()),
        "reference_topics_jaccard_ge_033": int((matches["jaccard"] >= 0.33).sum()),
    }
    return metrics, matches


def fit_labels(embeddings: np.ndarray, config: UmapConfig) -> np.ndarray:
    reducer = UMAP(
        n_neighbors=config.n_neighbors,
        n_components=5,
        min_dist=config.min_dist,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )
    reduced = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=150,
        min_samples=5,
        metric="euclidean",
        prediction_data=True,
        core_dist_n_jobs=1,
    )
    return clusterer.fit_predict(reduced).astype(int)


def write_readme(output_dir: Path, metrics: pd.DataFrame) -> None:
    display_columns = [
        "config",
        "n_neighbors",
        "min_dist",
        "candidate_substantive_topics",
        "candidate_outlier_rate",
        "ari_all",
        "ami_all",
        "weighted_best_topic_jaccard",
        "weighted_reference_topic_recall",
    ]
    display = metrics[display_columns].copy()
    headers = list(display.columns)
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        values = [f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        markdown_rows.append("| " + " | ".join(values) + " |")
    table = "\n".join(markdown_rows)
    text = f"""# Pooled BERTopic UMAP Parameter Sensitivity Audit

Run completed: {datetime.now(timezone.utc).isoformat()}

## Purpose

This is a post-lock robustness audit, not a model-selection exercise. The
official pooled BERTopic model remains unchanged. Every pre-specified UMAP
configuration is reported, regardless of whether it produces a convenient
topic count or visually attractive solution.

## Frozen components

- Corpus: the 36,171 comments eligible for the locked pooled model.
- Embeddings: cached `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- UMAP output dimensions: 5.
- UMAP metric: cosine.
- Random seed: 42.
- HDBSCAN: `min_cluster_size=150`, `min_samples=5`, Euclidean metric.
- No automatic topic reduction and no outlier reassignment.

## Varied components

Only UMAP `n_neighbors` and `min_dist` were varied. The grid was declared in
the script before inspecting its outputs: `n_neighbors` 5, 10, 15, 30 and 50
at `min_dist=0`, plus `min_dist` 0.1 and 0.5 at `n_neighbors=15`.

## Interpretation

Exact topic numbers are not expected to remain identical because density-based
topic models may split or merge semantically related regions. ARI/AMI/NMI
measure assignment agreement with the locked model. Best-topic Jaccard and
reference recall quantify whether each locked topic is recoverable as a
document-overlapping candidate cluster. These diagnostics should be discussed
alongside the previously completed balanced-corpus and manual Topic 0 audits.

## Results

{table}
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    final_dir = args.final_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = final_dir / "corpus" / "pooled_comment_body_embeddings_all_minilm_l6_v2.npy"
    index_path = final_dir / "corpus" / "pooled_valid_document_index.csv"
    assignments_path = final_dir / "final_model" / "pooled_document_topic_assignments.csv"

    embeddings = np.load(embeddings_path, mmap_mode="r")
    index = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    assignments = pd.read_csv(assignments_path, low_memory=False)
    ordered = verify_alignment(index, assignments, embeddings)
    reference = pd.to_numeric(ordered["pooled_topic"], errors="raise").astype(int).to_numpy()

    selected = [config for config in CONFIGS if not args.config or config.name in set(args.config)]
    manifest = {
        "audit_type": "post-lock UMAP parameter robustness; not model selection",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "documents": len(reference),
        "reference_substantive_topics": int(len(set(reference)) - (1 if -1 in reference else 0)),
        "reference_outlier_rate": float(np.mean(reference == -1)),
        "embedding_path": str(embeddings_path),
        "assignment_path": str(assignments_path),
        "fixed_hdbscan": {"min_cluster_size": 150, "min_samples": 5, "metric": "euclidean"},
        "fixed_umap": {"n_components": 5, "metric": "cosine", "random_state": 42},
        "pre_specified_configs": [asdict(config) for config in CONFIGS],
        "executed_configs": [asdict(config) for config in selected],
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "umap_learn": umap.__version__,
        },
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    all_metrics: list[dict[str, object]] = []
    all_matches: list[pd.DataFrame] = []
    for position, config in enumerate(selected, start=1):
        labels_path = output_dir / f"labels_{config.name}.npy"
        print(f"[{position}/{len(selected)}] {config.name}: n_neighbors={config.n_neighbors}, min_dist={config.min_dist}")
        if labels_path.exists() and not args.overwrite:
            print(f"  loading checkpoint: {labels_path}")
            candidate = np.load(labels_path)
        else:
            candidate = fit_labels(embeddings, config)
            np.save(labels_path, candidate)

        metrics, matches = score_configuration(reference, candidate, config)
        all_metrics.append(metrics)
        all_matches.append(matches)
        pd.DataFrame(all_metrics).to_csv(output_dir / "umap_sensitivity_metrics_partial.csv", index=False)
        pd.concat(all_matches, ignore_index=True).to_csv(
            output_dir / "reference_topic_recovery_partial.csv", index=False
        )
        print(
            f"  topics={metrics['candidate_substantive_topics']}, "
            f"outliers={metrics['candidate_outlier_rate']:.1%}, "
            f"ARI={metrics['ari_all']:.3f}, "
            f"weighted Jaccard={metrics['weighted_best_topic_jaccard']:.3f}"
        )

    metrics_df = pd.DataFrame(all_metrics)
    matches_df = pd.concat(all_matches, ignore_index=True)
    metrics_df.to_csv(output_dir / "umap_sensitivity_metrics.csv", index=False)
    matches_df.to_csv(output_dir / "reference_topic_recovery.csv", index=False)
    write_readme(output_dir, metrics_df)
    print(f"Completed audit: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
