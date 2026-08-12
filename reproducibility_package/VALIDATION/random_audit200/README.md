# Final prediction-blind random audit (N=200)

This directory contains the final headline validation aggregates used in the dissertation.

- Two independently sampled, non-overlapping batches of 100 comments were drawn from the frozen balanced 2,000-comment analytical sample.
- Each batch contained 25 comments from each of the four industries; the combined audit therefore contains 50 comments per industry.
- Sampling did not use model predictions, predicted labels, keywords, or substantive-trust prefilters.
- Human coding was completed without displaying model predictions, human notes from earlier exercises, or model-specific selection information.
- Prompt/codebook definitions and the production-model decision were frozen before the second batch was coded.

The final production decision remained GPT-5 mini for attitude, capability assessment, and explicit trust boundary. GPT-5.1 attitude and GPT-5.6 capability outputs are sensitivity/model-comparison evidence; they do not replace the production labels.

## Included

Only non-sensitive aggregate artefacts are included: summary metrics, per-class metrics, confusion matrices, industry-stratified results, audit metadata, and dissertation-ready table sources.

## Deliberately excluded

The package does not include comment text, comment identifiers, human-coded rows, human notes, row-level model predictions, or disagreement tables. These materials cannot be reconstructed from the aggregate outputs in this directory.
