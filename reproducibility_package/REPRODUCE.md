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

## Expected checks

- Final harmonised corpus audit: 36,681 comments.
- Pooled model: 36,171 modelled comments; 510 excluded before modelling (494 short, 16 URL-only); 18 substantive topics and Topic -1 with 5,478 comments.
- Explicit-boundary model: 861 input comments, 858 modelled comments, 22 raw topics, 14 reporting themes, six meta-themes, and 646 assigned comments in retained reporting themes.
- Balanced stance sample: 2,000 comments, exactly 500 in each industry.
- Final validation: 100 comments, 25 in each industry.

The stored tables and figures in `OUTPUTS/`, `VALIDATION/`, and `ROBUSTNESS/` are the comparison targets. Their reproduction depends on access to the same data snapshot, manual-screening decisions, and researcher mappings.
