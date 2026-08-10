#!/usr/bin/env python3
"""Audit UMAP sensitivity for the four locked industry BERTopic models.

This script is a post-lock robustness audit, not a model-selection routine.
It reuses each final model's cached embeddings and HDBSCAN specification,
varies only UMAP n_neighbors/min_dist, and compares each partition with the
locked document-topic assignments. Every pre-specified result is retained.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
from umap import UMAP


PROJECT_ROOT = Path(__file__).resolve().parent
COMBINED_OUTPUT = PROJECT_ROOT / "bertopic_umap_sensitivity_ALL_FINAL_MODELS_2026-08-09"

# Import the shared, already verified scoring implementation used for pooled
# sensitivity. This keeps overlap and clustering metrics identical.
sys.path.insert(0, str(PROJECT_ROOT))
from audit_pooled_bertopic_umap_sensitivity import CONFIGS, UmapConfig, score_configuration  # noqa: E402


MODEL_SPECS = {
    "finance": {
        "folder": PROJECT_ROOT
        / "finance_top_commented_ai_threads"
        / "bertopic_finance_final_v3_comment_body_only_identity_stopwords",
        "min_cluster_size": 70,
        "min_samples": 10,
    },
    "healthcare": {
        "folder": PROJECT_ROOT
        / "healthcare_top_commented_ai_threads"
        / "bertopic_healthcare_final_v3_comment_body_only_identity_stopwords",
        "min_cluster_size": 70,
        "min_samples": 10,
    },
    "law": {
        "folder": PROJECT_ROOT
        / "law_top_commented_ai_threads"
        / "bertopic_law_final_v4_comment_body_only_identity_stopwords_min35_llm_degree_cleaned",
        "min_cluster_size": 35,
        "min_samples": 10,
    },
    "software_engineering": {
        "folder": PROJECT_ROOT
        / "software_engineering_top_commented_ai_threads"
        / "bertopic_software_engineering_final_v3_comment_body_only_identity_stopwords_sensitivity_min35_ms5",
        "min_cluster_size": 35,
        "min_samples": 5,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--industry", action="append", choices=sorted(MODEL_SPECS))
    parser.add_argument("--config", action="append", choices=[item.name for item in CONFIGS])
    parser.add_argument("--output-dir", type=Path, default=COMBINED_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def verify_inputs(industry: str, folder: Path) -> tuple[np.ndarray, np.ndarray]:
    assignments_path = folder / f"{industry}_document_topic_assignments.csv"
    input_path = folder / f"{industry}_bertopic_input.csv"
    embeddings_path = folder / f"{industry}_embeddings_comment_body_all_minilm_l6_v2.npy"

    assignments = pd.read_csv(assignments_path, low_memory=False)
    inputs = pd.read_csv(input_path, low_memory=False)
    embeddings = np.load(embeddings_path)

    if not (len(assignments) == len(inputs) == len(embeddings)):
        raise ValueError(
            f"{industry}: row mismatch assignments={len(assignments):,}, "
            f"inputs={len(inputs):,}, embeddings={len(embeddings):,}"
        )
    for frame_name, frame in (("assignments", assignments), ("input", inputs)):
        if "comment_id" not in frame.columns:
            raise ValueError(f"{industry}: {frame_name} lacks comment_id")
    if "bertopic_topic" not in assignments.columns:
        raise ValueError(f"{industry}: assignments lack bertopic_topic")

    left = assignments["comment_id"].fillna("").astype(str).to_numpy()
    right = inputs["comment_id"].fillna("").astype(str).to_numpy()
    if not np.array_equal(left, right):
        mismatch = int(np.flatnonzero(left != right)[0])
        raise ValueError(f"{industry}: input/assignment alignment failed at row {mismatch}")

    reference = pd.to_numeric(assignments["bertopic_topic"], errors="raise").astype(int).to_numpy()
    return reference, embeddings


def fit_labels(
    embeddings: np.ndarray,
    config: UmapConfig,
    min_cluster_size: int,
    min_samples: int,
) -> np.ndarray:
    reduced = UMAP(
        n_neighbors=config.n_neighbors,
        n_components=5,
        min_dist=config.min_dist,
        metric="cosine",
        random_state=42,
        low_memory=True,
    ).fit_transform(embeddings)
    return hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        prediction_data=True,
        core_dist_n_jobs=1,
    ).fit_predict(reduced).astype(int)


def markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_readme(output_dir: Path, metrics: pd.DataFrame) -> None:
    display_columns = [
        "industry",
        "config",
        "candidate_substantive_topics",
        "candidate_outlier_rate",
        "ari_all",
        "ari_common_nonout",
        "weighted_best_topic_jaccard",
        "weighted_reference_topic_recall",
        "reference_topics_jaccard_ge_050",
    ]
    text = f"""# Industry BERTopic UMAP Parameter Sensitivity Audit

Completed: {datetime.now(timezone.utc).isoformat()}

## Status

This is a post-lock robustness audit. It does not select replacement models,
alter the locked topic assignments, or change researcher-assigned labels.
All pre-specified UMAP configurations are reported.

## Frozen components

- Each industry's exact final model corpus and cached MiniLM embeddings.
- UMAP dimensions = 5, cosine metric, random seed = 42.
- Finance: HDBSCAN min cluster size 70, min samples 10.
- Healthcare: HDBSCAN min cluster size 70, min samples 10.
- Law: HDBSCAN min cluster size 35, min samples 10.
- Software engineering: HDBSCAN min cluster size 35, min samples 5.
- No automatic topic reduction or outlier reassignment.

## Varied components

Only UMAP `n_neighbors` and `min_dist` were varied: neighbors 5, 10, 15,
30 and 50 at min_dist 0, plus min_dist 0.1 and 0.5 at 15 neighbors. The
15/0 configuration is an exact reproducibility check.

## Metrics

ARI/AMI/NMI compare complete assignment vectors. Common-nonout ARI compares
documents assigned to a substantive topic in both partitions. Topic recovery
matches each locked topic to its highest-overlap candidate topic using
document-level Jaccard overlap. Exact topic counts are not treated as the sole
robustness criterion because related regions may split or merge.

## Results

{markdown_table(metrics[display_columns])}

## Interpretation boundary

The audit evaluates dependence on UMAP specification. It does not prove that
any topic partition is unique latent ground truth. Exact topic counts and
prevalence estimates remain conditional on the frozen BERTopic specification.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    industries = args.industry or list(MODEL_SPECS)
    selected_configs = [item for item in CONFIGS if not args.config or item.name in args.config]
    all_metrics: list[dict[str, object]] = []
    all_matches: list[pd.DataFrame] = []

    for industry in industries:
        spec = MODEL_SPECS[industry]
        folder = Path(spec["folder"])
        audit_dir = folder / "audits" / "umap_parameter_sensitivity_2026-08-09"
        audit_dir.mkdir(parents=True, exist_ok=True)
        reference, embeddings = verify_inputs(industry, folder)
        print(
            f"\n{industry}: documents={len(reference):,}; "
            f"locked topics={len(set(reference) - {-1})}; locked outliers={(reference == -1).sum():,}"
        )

        for config in selected_configs:
            labels_path = audit_dir / f"labels_{config.name}.npy"
            if labels_path.exists() and not args.overwrite:
                candidate = np.load(labels_path)
                print(f"  {config.name}: loaded checkpoint")
            else:
                print(f"  {config.name}: fitting")
                candidate = fit_labels(
                    embeddings,
                    config,
                    int(spec["min_cluster_size"]),
                    int(spec["min_samples"]),
                )
                np.save(labels_path, candidate)

            metrics, matches = score_configuration(reference, candidate, config)
            metrics.update(
                {
                    "industry": industry,
                    "locked_substantive_topics": int(len(set(reference) - {-1})),
                    "locked_outlier_count": int(np.sum(reference == -1)),
                    "locked_outlier_rate": float(np.mean(reference == -1)),
                    "hdbscan_min_cluster_size": int(spec["min_cluster_size"]),
                    "hdbscan_min_samples": int(spec["min_samples"]),
                }
            )
            matches.insert(0, "industry", industry)
            all_metrics.append(metrics)
            all_matches.append(matches)
            print(
                f"    topics={metrics['candidate_substantive_topics']}; "
                f"outliers={metrics['candidate_outlier_rate']:.1%}; "
                f"ARI={metrics['ari_all']:.3f}; "
                f"weighted Jaccard={metrics['weighted_best_topic_jaccard']:.3f}"
            )

    metrics_frame = pd.DataFrame(all_metrics)
    matches_frame = pd.concat(all_matches, ignore_index=True)
    metrics_frame.to_csv(args.output_dir / "umap_sensitivity_all_industry_models.csv", index=False)
    matches_frame.to_csv(args.output_dir / "reference_topic_recovery_all_industry_models.csv", index=False)
    write_readme(args.output_dir, metrics_frame)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "post-lock robustness audit; not model selection",
        "industries": industries,
        "configs": [item.__dict__ for item in selected_configs],
        "model_specs": {
            key: {name: str(value) if isinstance(value, Path) else value for name, value in MODEL_SPECS[key].items()}
            for key in industries
        },
    }
    (args.output_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nCombined metrics: {args.output_dir / 'umap_sensitivity_all_industry_models.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
