# Explicit Trust-Boundary Lexical Analysis

Analytical subset: comments classified as containing an explicit trust boundary by the validated binary boundary classifier.

This analysis does not use the substantive-trust gate or fine-grained trust construct/boundary labels to define or subdivide the sample. It uses only the target comment text for tokenisation. Parent context is retained only in KWIC outputs.

Interpretive caution: this is a model-identified boundary-positive subset. The classifier is precision-oriented but not exhaustive, so results should not be read as all trust comments in the corpus.

## Input
- Input file: `outputs/frozen_prompt_v2_stratified_sample_2000_2026-07-24/all_comments_v2_annotated.csv`
- Input SHA-256: `c8d59426e59b56b9765d615862cc56435be4fbac7bd1e223c61802a584f8afe2`
- Total input rows: 2000
- Rows with frozen explicit-boundary predictions: 2000
- Boundary-positive rows after exclusions: 861
- Excluded rows: 0

## Validation context
- Validation directory: `outputs/four_industry_full_schema_validation_100_performance_2026-07-30`
- Exact validation metrics are copied into `explicit_boundary_lexical_analysis_metrics.json`.

## Main outputs
- `explicit_boundary_prevalence_by_industry.csv`
- `explicit_boundary_pooled_1gram_frequencies.csv`, `explicit_boundary_pooled_2gram_frequencies.csv`, `explicit_boundary_pooled_3gram_frequencies.csv`
- `keyness_boundary_positive_vs_negative_1gram.csv`, `2gram.csv`, `3gram.csv`
- `pooled_keyness_industry_diagnostics.csv`
- `industry_distinctive_terms_within_explicit_boundaries.xlsx`
- `explicit_boundary_collocations_window5.csv`
- `kwic_manual_review_sample.xlsx`
- `candidate_boundary_lexical_evidence_workbook.xlsx`
- `figures` are saved as PNG and PDF in this folder.