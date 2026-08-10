# Superseded binary-gate metrics in this directory

The files beginning `validation_100_binary_gate_` were copied from an earlier
calculation that read precomputed binary labels.  One human fine-grained
`unclear` trust-boundary label was incorrectly recorded as binary positive in
that earlier calculation.

For the authoritative final balanced-100 binary-gate metrics, use:

`../model_comparison/three_model_binary_gate_metrics.csv`

The final comparison derives gates directly from the frozen fine-grained
labels and treats `unclear` as not-positive.  For GPT-5 mini's explicit
trust-boundary gate, the final result is `TP=60, TN=19, FP=9, FN=12`, with
precision 0.870, recall 0.833, specificity 0.679, F1 0.851, and MCC 0.497.
