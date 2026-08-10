# BERTopic UMAP Parameter Sensitivity Audit

Date completed: 2026-08-09

## Decision

The criticism that the original reporting lacked a UMAP sensitivity analysis is valid. The stronger claim that this constitutes data leakage or demonstrates post-hoc cherry-picking is not supported by the project record.

- UMAP was used as an unsupervised dimensionality-reduction step. Human gold labels, stance predictions and research outcomes were not supplied to UMAP or HDBSCAN.
- The same `n_neighbors=15`, `n_components=5`, `min_dist=0`, cosine-distance and `random_state=42` specification appears throughout the early and final BERTopic scripts. The project record does not show a search across UMAP settings followed by selection of the most interpretable output.
- Researcher interpretation and topic mapping were frozen before this sensitivity audit. This audit reports every configuration in the documented grid and does not select a replacement model.

The audit nevertheless shows genuine parameter dependence. The locked models are reproducible, and several nearby specifications recover substantial parts of their structure, but exact topic counts, boundaries and prevalence estimates are conditional on the frozen UMAP/HDBSCAN specification. Finance is the least stable industry model and requires the greatest interpretive caution.

## Models Audited

Six locked BERTopic models were audited:

1. Cross-industry pooled corpus.
2. Finance and accounting.
3. Healthcare.
4. Law.
5. Software engineering and IT.
6. Explicit-trust-boundary subset.

Each audit reused the exact frozen analytical rows and cached `sentence-transformers/all-MiniLM-L6-v2` embeddings. Within each model, HDBSCAN parameters, five UMAP components, cosine distance and random seed 42 were held fixed.

## Parameter Grid

Only UMAP `n_neighbors` and `min_dist` were varied:

- `n_neighbors` = 5, 10, 15, 30 and 50 with `min_dist=0`;
- `min_dist` = 0.1 and 0.5 with `n_neighbors=15`.

The locked `15 / 0` configuration was rerun as an exact reproducibility check. No automatic topic reduction or Topic -1 reassignment was introduced.

## Evaluation Metrics

- **ARI/AMI/NMI:** agreement between complete document-assignment vectors.
- **Common-nonoutlier ARI:** agreement among documents assigned to a substantive topic in both partitions.
- **Weighted best-topic Jaccard:** for each locked topic, document overlap with its best matching candidate topic, weighted by locked-topic size. This is the principal topic-recovery measure in the summary figure.
- **Weighted locked-topic recall:** proportion of locked-topic documents contained in the best candidate match. This can remain high when several locked topics merge into one candidate topic and is therefore not interpreted alone.
- **Recovered topics at Jaccard >= .50:** count of locked topics with at least moderate document-level recovery.
- **Topic count and Topic -1 rate:** used to identify fragmentation, merger and assignment collapse.

## Main Findings

The locked `15 / 0` specification reproduced every model exactly: assignment agreement and weighted topic overlap were 1.00 for all six models. This confirms computational reproducibility under the recorded environment and seed.

### Pooled corpus

The `10 / 0` alternative recovered all 18 locked topics at Jaccard >= .50, with weighted Jaccard .801 and common-nonoutlier ARI .936, although it produced 23 rather than 18 topics. The `5 / 0` and `50 / 0` alternatives showed moderate recovery. `30 / 0` and larger `min_dist` settings caused extensive merging or high outlier assignment. The pooled model therefore has strong local recovery around the locked setting but is not invariant to larger perturbations.

### Finance and accounting

Finance was the most parameter-sensitive model. Most alternatives merged the eight locked topics into two to five clusters. Its strongest non-reference alternative was `50 / 0` (weighted Jaccard .707; four candidate topics), but only three of eight locked topics reached Jaccard >= .50. Finance findings should therefore be presented primarily as researcher-interpreted broad themes supported by representative comments, with exact topic prevalence treated as specification-dependent.

### Healthcare

Healthcare showed good local recovery at `30 / 0` (weighted Jaccard .787; 14 versus 16 topics; 14 of 16 topics at Jaccard >= .50) and moderate recovery at `10 / 0` and `50 / 0`. Extreme settings caused substantial merger. The main structure is locally recoverable, while exact granularity is conditional on the locked model.

### Law

Law showed strong local recovery at `10 / 0` (weighted Jaccard .838; 20 versus 19 topics; 16 of 19 topics at Jaccard >= .50) and `15 / 0.1` (weighted Jaccard .839; 18 topics; 17 of 19 recovered). Larger neighbourhoods merged the model into three topics. The locked interpretation is well supported locally but not globally invariant.

### Software engineering and IT

With `min_dist=0`, neighbours 5, 10, 30 and 50 yielded weighted Jaccard values from .564 to .686 and common-nonoutlier ARI from .898 to .942. Topic counts varied from 52 to 95 versus 66 in the locked model, indicating split/merge sensitivity despite substantial common structure. Changing `min_dist` to .1 or .5 collapsed the solution to three topics. Broad software themes are more defensible than a claim that all 66 raw clusters are uniquely determined.

### Explicit-trust-boundary subset

The explicit-boundary model showed moderate recovery. The strongest nearby alternatives were `15 / 0.1` (weighted Jaccard .686; 16 versus 22 topics) and `10 / 0` (.664; 20 topics). Other settings produced greater merger. The 14 researcher-led reporting themes should be treated as an interpretive organisation of the locked partition, not a unique latent taxonomy.

## Overall Interpretation

The results do not support describing the BERTopic solutions as parameter-invariant. They support a narrower conclusion:

1. The recorded locked models are exactly reproducible.
2. Several major structures are recoverable under nearby UMAP settings, especially for the pooled, healthcare, law and software models.
3. Topic granularity and some document assignments change materially under broader perturbations.
4. Finance is particularly sensitive.
5. The dissertation should emphasise broad, manually reviewed themes, representative comments and convergent evidence across pooled and industry-specific analyses rather than treating exact topic counts or percentages as ground truth.

The audit does not reopen model selection. Choosing a new configuration after inspecting this grid would create the post-hoc flexibility that the audit is intended to prevent.

## Thesis-Ready Methods Text

> UMAP parameters were fixed before substantive topic interpretation at `n_neighbors=15`, `n_components=5`, `min_dist=0`, cosine distance and `random_state=42`; they were not optimised against human labels or desired topic interpretations. After the models and researcher mappings had been frozen, a post-lock sensitivity audit reused the exact analytical rows, cached MiniLM embeddings and model-specific HDBSCAN settings while varying `n_neighbors` across 5, 10, 15, 30 and 50 at `min_dist=0`, and varying `min_dist` to 0.1 and 0.5 at 15 neighbours. All alternatives were retained in the audit. Stability was assessed using ARI, AMI, NMI, Topic -1 rates, topic counts and document-level overlap between each locked topic and its best-matching alternative topic.

## Thesis-Ready Results Text

> The locked `15/0` specification reproduced all six recorded BERTopic solutions exactly. Robustness to alternative UMAP settings was heterogeneous. The pooled model showed strong local recovery at `10/0` (weighted topic-overlap Jaccard=.801; common-nonoutlier ARI=.936), while law at `10/0` (.838), healthcare at `30/0` (.787) and software engineering at `30/0` (.686) retained substantial structure despite topic splitting or merging. Finance was less stable: its strongest non-reference alternative produced four rather than eight topics (weighted Jaccard=.707). Larger perturbations, particularly increases in `min_dist`, sometimes collapsed solutions or sharply increased outlier assignment. The analysis therefore treats broad manually reviewed themes as the principal findings and regards exact topic counts and prevalence estimates as conditional on the frozen model specification.

## Thesis-Ready Limitation

> BERTopic results remained partly dependent on UMAP specification. Although important topic structures were recoverable under several nearby settings, neither the exact number of topics nor every document assignment represented a unique parameter-invariant partition. This limitation was most pronounced for finance and for changes to `min_dist`; consequently, interpretation prioritised broad themes, representative comments and cross-model convergence over small differences in topic prevalence.

## Output Files

- Combined metrics: `umap_sensitivity_ALL_locked_bertopic_models.csv`
- Concise metrics: `umap_sensitivity_concise_report.csv`
- Summary figure: `figure_umap_parameter_sensitivity_topic_recovery.png` (also PDF and SVG)
- Industry topic-recovery detail: `reference_topic_recovery_all_industry_models.csv`
- Pooled and explicit-boundary detailed outputs remain in their model-specific `audits/umap_parameter_sensitivity_2026-08-09` directories.
