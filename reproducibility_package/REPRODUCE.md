# Reproduction workflow

These commands require non-distributable inputs prepared in a local `data/` directory. They are **not** executed by this package and must not be treated as a request to retrieve Reddit data without checking current platform terms and approval requirements.

```bash
conda env create -f environment.yml
conda activate occupational-ai-reddit
export REPRO_PROJECT_ROOT="$PWD"
export REPRO_OUTPUTS_DIR="$PWD/outputs"
export REPRO_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
mkdir -p "$REPRO_OUTPUTS_DIR"
```

1. Discover candidates and create screening input:
```bash
python SCRIPTS/discovery/discover_sic_subreddits_reddit_search_selenium.py --help
```
2. Retrieve/rank AI-active communities and construct the corpus using the scripts in `SCRIPTS/corpus/`. Provide local paths with each script's CLI arguments.
3. Fit industry models, once per industry:
```bash
python SCRIPTS/bertopic/bertopic_industry_comment_body_only.py --help
```
4. Fit the pooled model and compute its topic-by-industry outputs:
```bash
PYTHONPATH=SCRIPTS/bertopic python SCRIPTS/bertopic/cross_industry_pooled_bertopic.py --help
```
5. Run the explicit-boundary model only with a locally available copy of the frozen boundary-positive annotation subset; this subset is not distributed in the repository::
```bash
PYTHONPATH=SCRIPTS/bertopic python SCRIPTS/bertopic/run_explicit_boundary_bertopic.py --help
```
6. Run weighted log-odds on the balanced annotated sample:
```bash
PYTHONPATH=SCRIPTS/bertopic python SCRIPTS/bertopic/run_explicit_boundary_lexical_analysis.py --help
```
7. Reproduce parameter sensitivity summaries from stored assignment data with `SCRIPTS/robustness/`.

8. Reconstruct the final prediction-blind random-audit metrics from locally held human labels and the frozen model predictions:
```bash
python SCRIPTS/validation/evaluate_three_models_random_audit_batches.py --help
python SCRIPTS/validation/build_complete_model_performance_table_random_audit200.py --help
python SCRIPTS/validation/export_attitude_per_class_tables_random_audit200.py --help
python SCRIPTS/validation/export_capability_per_class_tables_random_audit200.py --help
```
The two human-coded row-level files are not distributed. Aggregate comparison targets are supplied in `VALIDATION/random_audit200/`.

9. Reproduce the final aggregate robustness tables and figures, after providing local copies of the balanced 2,000-comment predictions:
```bash
python SCRIPTS/reporting/create_gpt51_attitude_composition_by_industry.py --help
python SCRIPTS/reporting/create_three_model_capability_composition_figures.py --help
```

The annotation scripts in `SCRIPTS/annotation/` are provenance records. They can issue paid API requests and must not be executed merely to reproduce the stored aggregate results.

## Expected checks

- Final harmonised corpus audit: 36,681 comments.
- Pooled model: 36,171 modelled comments; 510 excluded before modelling (494 short, 16 URL-only); 18 substantive topics and Topic -1 with 5,478 comments.
- Explicit-boundary model: 861 input comments, 858 modelled comments, 22 raw topics, 14 reporting themes, six meta-themes, and 646 assigned comments in retained reporting themes.
- Balanced stance sample: 2,000 comments, exactly 500 in each industry.
- Final prediction-blind random audit: 200 comments in two non-overlapping batches; 50 per industry overall.
- Historical 50- and 100-comment evaluations remain in `VALIDATION/` for provenance but are not the final headline performance estimates.

The stored tables and figures in `OUTPUTS/`, `VALIDATION/`, and `ROBUSTNESS/` are the comparison targets. Their reproduction depends on access to the same data snapshot, manual-screening decisions, and researcher mappings.
