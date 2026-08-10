# Industry BERTopic UMAP Parameter Sensitivity Audit

Completed: 2026-08-09T18:15:00.467951+00:00

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

| industry | config | candidate_substantive_topics | candidate_outlier_rate | ari_all | ari_common_nonout | weighted_best_topic_jaccard | weighted_reference_topic_recall | reference_topics_jaccard_ge_050 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| finance | nn5_md0 | 2 | 0.0456 | 0.1754 | 0.3084 | 0.3995 | 0.9735 | 1 |
| finance | nn10_md0 | 2 | 0.0428 | 0.1815 | 0.3191 | 0.4056 | 0.9826 | 1 |
| finance | nn15_md0_reproduction | 8 | 0.3163 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 8 |
| finance | nn30_md0 | 2 | 0.0281 | 0.1944 | 0.3500 | 0.4072 | 0.9968 | 1 |
| finance | nn50_md0 | 4 | 0.1788 | 0.5575 | 0.8853 | 0.7072 | 0.9369 | 3 |
| finance | nn15_md01 | 3 | 0.1010 | 0.2244 | 0.4080 | 0.4572 | 0.9913 | 2 |
| finance | nn15_md05 | 5 | 0.4942 | 0.3311 | 0.7553 | 0.4827 | 0.6191 | 2 |
| healthcare | nn5_md0 | 3 | 0.0109 | 0.0010 | 0.0143 | 0.0934 | 0.9879 | 1 |
| healthcare | nn10_md0 | 12 | 0.2185 | 0.4533 | 0.7976 | 0.6272 | 0.9301 | 8 |
| healthcare | nn15_md0_reproduction | 16 | 0.3607 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 16 |
| healthcare | nn30_md0 | 14 | 0.3347 | 0.6680 | 0.9615 | 0.7867 | 0.9120 | 14 |
| healthcare | nn50_md0 | 13 | 0.3414 | 0.4751 | 0.6389 | 0.5814 | 0.8642 | 11 |
| healthcare | nn15_md01 | 12 | 0.3721 | 0.4687 | 0.7546 | 0.5501 | 0.7611 | 7 |
| healthcare | nn15_md05 | 3 | 0.1887 | 0.0224 | 0.0781 | 0.1944 | 0.9171 | 2 |
| law | nn5_md0 | 8 | 0.0186 | 0.2537 | 0.4348 | 0.4643 | 0.9484 | 7 |
| law | nn10_md0 | 20 | 0.2742 | 0.7564 | 0.9833 | 0.8378 | 0.9060 | 16 |
| law | nn15_md0_reproduction | 19 | 0.2907 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 19 |
| law | nn30_md0 | 3 | 0.0043 | 0.1993 | 0.3317 | 0.4171 | 0.9933 | 2 |
| law | nn50_md0 | 3 | 0.0117 | 0.1971 | 0.3276 | 0.4130 | 0.9870 | 2 |
| law | nn15_md01 | 18 | 0.3884 | 0.7121 | 0.9997 | 0.8388 | 0.8493 | 17 |
| law | nn15_md05 | 3 | 0.2636 | 0.1699 | 0.3501 | 0.3778 | 0.8078 | 2 |
| software_engineering | nn5_md0 | 95 | 0.3443 | 0.3735 | 0.8985 | 0.5640 | 0.6827 | 28 |
| software_engineering | nn10_md0 | 72 | 0.3946 | 0.5269 | 0.9262 | 0.6546 | 0.7733 | 39 |
| software_engineering | nn15_md0_reproduction | 66 | 0.4085 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 66 |
| software_engineering | nn30_md0 | 54 | 0.4235 | 0.5637 | 0.9422 | 0.6861 | 0.8110 | 42 |
| software_engineering | nn50_md0 | 52 | 0.4626 | 0.5039 | 0.9365 | 0.6334 | 0.7446 | 32 |
| software_engineering | nn15_md01 | 3 | 0.0002 | 0.0154 | 0.0078 | 0.0656 | 0.9931 | 2 |
| software_engineering | nn15_md05 | 3 | 0.0419 | 0.0020 | 0.0075 | 0.0644 | 0.9694 | 2 |

## Interpretation boundary

The audit evaluates dependence on UMAP specification. It does not prove that
any topic partition is unique latent ground truth. Exact topic counts and
prevalence estimates remain conditional on the frozen BERTopic specification.
