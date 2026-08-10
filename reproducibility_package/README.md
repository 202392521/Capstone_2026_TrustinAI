# Professional AI Trust Boundaries on Reddit

This repository is the final reproducibility package for a dissertation on professional discussions of AI across finance and accounting, law, software engineering and IT, and healthcare. It contains frozen scripts, settings, researcher-led topic mappings, non-sensitive audit outputs, and publication figures. It intentionally excludes Reddit comment text, comment identifiers, usernames, raw human-coded rows, model-prediction rows, API credentials, cached embeddings, and fitted model binaries.

## Study scope

The collection window was March 2023 through January 2026. A post was eligible when its publication date fell inside that window; later comments attached to an eligible post could remain in the retrieved thread. Candidate communities were identified with SIC-informed occupation searches in Reddit Community Search, manually screened, and retained when they had at least 10 calendar months containing an AI-relevant post. The final harmonised corpus comprised 36,681 comments. Usernames were not collected.

## Pipeline

1. **Community discovery and screening**: `SCRIPTS/discovery/` and `SCRIPTS/corpus/rank_ai_discussion_subreddits.py` document community search and active-AI-month scoring.
2. **Corpus construction**: `SCRIPTS/corpus/` contains Arctic Shift collection, tier harmonisation, Law's Master-of-Laws `LLM` false-positive removal, merge, and sampling audit logic.
3. **Industry BERTopic**: `SCRIPTS/bertopic/bertopic_industry_comment_body_only.py` fits the four final comment-body-only models. Final settings are in `CONFIG/frozen_analysis_settings.yml`.
4. **Pooled BERTopic and JSD**: `cross_industry_pooled_bertopic.py` produces pooled assignments and industry-distribution diagnostics. Jensen-Shannon divergence compares `P(industry | topic)` against the pooled-modelled-corpus industry baseline.
5. **GPT-assisted annotation**: `PROMPTS/FROZEN_PROMPT_V2.txt` contains the frozen V2 coding instructions; the accompanying structured output schema is implemented in the annotation/validation scripts.
6. **Validation**: `VALIDATION/` contains aggregate healthcare holdout, final 100-comment, and blinded three-model comparison metrics. The non-healthcare segment of the final validation was enriched for model-identified substantive-trust content and is not an unconditional random sample.
7. **Weighted log-odds**: `run_explicit_boundary_lexical_analysis.py` compares explicit-boundary-positive and -negative comments using an empirical pooled-count Dirichlet prior.
8. **Explicit trust-boundary BERTopic**: `run_explicit_boundary_bertopic.py` fits the 861-comment positive-boundary subset (858 modelled after blank/short exclusion). `MAPPINGS/` records 22 raw topics, 14 reporting themes, and six meta-themes.
9. **Robustness**: `ROBUSTNESS/` preserves the pre-specified UMAP grid and all reported sensitivity outputs. It was post-hoc and was not used to select parameters or relabel topics.

## Dissertation-facing outputs

`OUTPUTS/figures/` holds final non-sensitive BERTopic and explicit-boundary figures. `OUTPUTS/summary_tables/` provides their source aggregates. `OUTPUTS/lexical/` holds non-sensitive keyness tables and figures. `VALIDATION/` provides metrics and confusion matrices. These are audit outputs, not a substitute for the non-distributable corpus.

## Frozen choices

All final BERTopic models use `comment_body` only, `sentence-transformers/all-MiniLM-L6-v2`, UMAP seed 42 with 15 neighbours, five dimensions, `min_dist=0`, and cosine distance. Occupational identity words were suppressed in topic representation only, not removed from embedding inputs. No automatic topic reduction or Topic -1 reassignment was used. See `CONFIG/frozen_analysis_settings.yml` and the copied run summaries for exact model-specific settings.

## Re-running

See `REPRODUCE.md`. Reproduction requires a lawfully obtained local corpus laid out under a user-selected data directory. The package does not distribute Reddit text, IDs, or human annotations. The final models and raw data cannot be regenerated from this repository alone.

## Directory guide

- `CONFIG/`: frozen parameters and archived run summaries.
- `PROMPTS/`: final Prompt V2 and its provenance.
- `SCRIPTS/`: portable copies of final logic; only path defaults were made configurable.
- `MAPPINGS/`: researcher-led frozen topic/reporting/meta-theme mappings.
- `VALIDATION/`: aggregate validation results without label rows or comment text.
- `ROBUSTNESS/`: full UMAP sensitivity evidence.
- `OUTPUTS/`: safe figures and aggregate source tables.

The executable scripts accept relative paths or `REPRO_OUTPUTS_DIR`, `REPRO_PROJECT_ROOT`, `REPRO_TEMP_DIR`, and `REPRO_EMBEDDING_MODEL` environment variables where applicable. Local cache paths from the original workstation were removed from the package.
