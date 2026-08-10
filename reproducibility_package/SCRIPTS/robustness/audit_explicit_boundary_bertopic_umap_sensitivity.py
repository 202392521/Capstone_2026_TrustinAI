#!/usr/bin/env python3
"""Post-lock UMAP sensitivity audit for the explicit-boundary BERTopic model."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
from umap import UMAP


ROOT = Path(__file__).resolve().parent
FINAL_DIR = ROOT / "explicit_boundary_bertopic_FINAL_2026-08-03"
OUTPUT_DIR = FINAL_DIR / "audits" / "umap_parameter_sensitivity_2026-08-09"
sys.path.insert(0, str(ROOT))
from audit_pooled_bertopic_umap_sensitivity import CONFIGS, score_configuration  # noqa: E402


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    assignments = pd.read_csv(FINAL_DIR / "explicit_boundary_document_topic_assignments.csv", low_memory=False)
    embeddings = np.load(FINAL_DIR / "explicit_boundary_comment_body_embeddings.npy")
    if len(assignments) != len(embeddings):
        raise ValueError("Assignment and embedding row counts differ.")
    positions = pd.to_numeric(assignments["_model_position"], errors="raise").astype(int).to_numpy()
    if not np.array_equal(positions, np.arange(len(assignments))):
        raise ValueError("_model_position is not an aligned zero-based sequence.")
    reference = pd.to_numeric(assignments["boundary_topic"], errors="raise").astype(int).to_numpy()

    metrics_rows: list[dict[str, object]] = []
    match_rows: list[pd.DataFrame] = []
    for config in CONFIGS:
        labels_path = OUTPUT_DIR / f"labels_{config.name}.npy"
        if labels_path.exists():
            candidate = np.load(labels_path)
            print(f"{config.name}: loaded checkpoint")
        else:
            print(f"{config.name}: fitting")
            reduced = UMAP(
                n_neighbors=config.n_neighbors,
                n_components=5,
                min_dist=config.min_dist,
                metric="cosine",
                random_state=42,
                low_memory=True,
            ).fit_transform(embeddings)
            candidate = hdbscan.HDBSCAN(
                min_cluster_size=12,
                min_samples=3,
                metric="euclidean",
                prediction_data=True,
                core_dist_n_jobs=1,
            ).fit_predict(reduced).astype(int)
            np.save(labels_path, candidate)
        metrics, matches = score_configuration(reference, candidate, config)
        metrics.update(
            {
                "model": "explicit_boundary",
                "locked_substantive_topics": int(len(set(reference) - {-1})),
                "locked_outlier_count": int(np.sum(reference == -1)),
                "locked_outlier_rate": float(np.mean(reference == -1)),
                "hdbscan_min_cluster_size": 12,
                "hdbscan_min_samples": 3,
            }
        )
        metrics_rows.append(metrics)
        matches.insert(0, "model", "explicit_boundary")
        match_rows.append(matches)
        print(
            f"  topics={metrics['candidate_substantive_topics']}; "
            f"outliers={metrics['candidate_outlier_rate']:.1%}; "
            f"ARI={metrics['ari_all']:.3f}; "
            f"weighted Jaccard={metrics['weighted_best_topic_jaccard']:.3f}"
        )

    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(OUTPUT_DIR / "umap_sensitivity_metrics.csv", index=False)
    pd.concat(match_rows, ignore_index=True).to_csv(OUTPUT_DIR / "reference_topic_recovery.csv", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "post-lock UMAP robustness audit; not model selection",
        "documents": int(len(reference)),
        "locked_topics": int(len(set(reference) - {-1})),
        "fixed_hdbscan": {"min_cluster_size": 12, "min_samples": 3, "metric": "euclidean"},
        "fixed_umap": {"n_components": 5, "metric": "cosine", "random_state": 42},
        "varied_configs": [item.__dict__ for item in CONFIGS],
    }
    (OUTPUT_DIR / "audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Metrics: {OUTPUT_DIR / 'umap_sensitivity_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
