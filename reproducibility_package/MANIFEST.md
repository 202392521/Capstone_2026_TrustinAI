# Package manifest

## Root files

- `README.md`: study overview, frozen workflow, and directory guide.
- `REPRODUCE.md`: shortest executable reconstruction route and expected checks.
- `DATA_AVAILABILITY.md`: source, exclusions, and data-governance statement.
- `requirements.txt` / `environment.yml`: recoverable Python environment.
- `.gitignore`: prevents sensitive corpora, models, caches, and credentials entering Git.
- `LICENSE` / `CITATION.cff`: code licensing and citation metadata.
- `TOPIC_MODELS_AND_META_MAPPING_MASTER_LOCK_2026-07-29.md`: frozen model/mapping status.
- `PACKAGE_UPDATE_2026-08-12.md`: scope, portability and privacy record for the final package additions.

## CONFIG

- `frozen_analysis_settings.yml`: portable final parameter ledger.
- `README.md`: provenance and use of the configuration records.
- `*_config_manifest.json`, `*_bertopic_run_summary.csv`, `pooled_bertopic_run_summary.csv`, and `explicit_boundary_run_manifest_source.json`: final-run configuration and aggregate model audit records.
- `final_annotation_provenance_2026-08-12.json`: sanitised model, hash, privacy, completion, and validation provenance for the final annotation work.

## PROMPTS

- `FROZEN_PROMPT_V2.txt`: final frozen annotation prompt/codebook.
- `README.md`: prompt-development, holdout, blinding, and model-comparison provenance.

## SCRIPTS

- `discovery/discover_sic_subreddits_reddit_search_selenium.py`: community-search discovery.
- `corpus/*.py`: active-month scoring, archive retrieval, tier harmonisation, cleaning, merge, and sample audit.
- `bertopic/*.py`: industry, pooled/JSD, explicit-boundary, and lexical analysis logic.
- `annotation/*.py`: balanced-sample and frozen-prompt annotation provenance.
- `validation/*.py`: validation and three-model comparison reconstruction.
- `robustness/*.py`: UMAP sensitivity analysis and summary logic.
- `reporting/*.py`: final table/figure generation.
- `SCRIPTS/README.md`: portability changes and execution caveats.

## MAPPINGS

- `cross_industry_pooled_topic_meta_mapping_FINAL.csv`: frozen pooled mapping and decisions.
- `explicit_boundary_original_topic_to_reporting_theme_mapping.csv`: 22 raw to 14 reporting-theme mapping.
- `explicit_boundary_meta_theme_to_reporting_theme_hierarchy.csv`: 14 reporting to six meta-theme hierarchy.
- `explicit_boundary_*_human_verified.csv`: aggregate final labels/counts.
- `MAPPINGS/README.md`: researcher-judgement and exclusion explanation.

## VALIDATION

- `holdout50/*`: non-sensitive healthcare-holdout aggregate metrics and matrices.
- `balanced100/*`: historical model-enriched 100-comment audit, retained for provenance and marked superseded.
- `model_comparison/*`: historical balanced-100 Mini/5.6/5.1 aggregate comparisons and figures.
- `random_audit200/*`: final two-batch prediction-blind random audit (N=200), including three-model summaries, per-class metrics, confusion matrices, and Table 3 sources. Row-level gold labels and comment text are excluded.

## ROBUSTNESS

- `BERTopic_UMAP_SENSITIVITY_AUDIT_2026-08-09.md`, `README.md`, and `README_PACKAGE.md`: audit protocol and interpretation bounds.
- `umap_sensitivity_*.csv`, `reference_topic_recovery_*.csv`, and `audit_manifest.json`: complete pre-specified grid results.
- `figure_umap_parameter_sensitivity_topic_recovery.*`: robustness visualisation.

## OUTPUTS

- `summary_tables/*`: safe aggregate corpus, topic, attitude/boundary, and industry composition source tables.
- `figures/*`: final non-sensitive BERTopic and trust-boundary publication figures.
- `lexical/*`: weighted-log-odds keyness tables, lexical diagnostics, and figures without KWIC or comment text.
- `stance_robustness/*`: aggregate GPT-5.1 attitude and three-model capability sensitivity tables and figures.

All other project artefacts, particularly comment-level files, IDs, annotations, prediction rows, fitted models, and embeddings, are deliberately excluded. [`MANIFEST_FILES.md`](MANIFEST_FILES.md) is the complete, per-file index for this release.
