# Topic models and meta-mapping master lock

This package records the final analytical state. It does not initiate model fitting.

## Industry-specific final models

- Finance: final v3, comment body only, occupational-identity stopwords in representation.
- Healthcare: final v3, comment body only, occupational-identity stopwords in representation.
- Law: final v4, comment body only; clearly degree-related `LLM` (Master of Laws) false positives removed before fitting.
- Software engineering and IT: final v3 sensitivity configuration, comment body only, `min_cluster_size=35`, `min_samples=5`.

## Pooled model

The cross-industry pooled BERTopic model was frozen on 2026-07-27. Topic 0 was retained as a broad, partly noisy shared theme after a 100-comment stratified manual audit. The final frozen mapping is `MAPPINGS/cross_industry_pooled_topic_meta_mapping_FINAL.csv`.

## Explicit-boundary model

The final explicit-boundary model maps 22 raw topics to 14 reporting themes and six meta-themes using the frozen CSV files in `MAPPINGS/`.

## Lock rule

Reporting-level labels, merge/exclude decisions, and topic mappings were frozen. No later result, robustness configuration, or single comment should be used to refit a model or change a mapping unless an explicit data or code error is discovered and separately documented.
