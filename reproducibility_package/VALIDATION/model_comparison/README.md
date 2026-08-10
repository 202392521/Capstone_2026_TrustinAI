# Final balanced-100 three-model comparison

Created 2026-08-09 without any API calls.

## Validation design

- 100 unique comments: 25 each from finance, healthcare, law and software engineering.
- GPT-5 mini predictions come from the frozen balanced analytical sample.
- GPT-5.6 and GPT-5.1 reuse 77 existing predictions and add the 23 previously missing healthcare predictions.
- The 23-row model input contained only `comment_id`, `parent_context` and `target_comment`.
- Human labels, gold labels and notes were absent from the model input.
- All evaluations use the same six-label codebook and the same Frozen Prompt V2.
- GPT-5 mini used an earlier richer response schema; GPT-5.6 and GPT-5.1 used the same six-field schema. Only the six shared labels are compared.

Audit verdict: **PASS**.

## Best macro-F1 by field

- Attitude: GPT-5.1 (0.595)
- Attitude target: GPT-5.1 (0.485)
- Capability assessment: GPT-5.6 (0.557)
- Fine-grained trust boundary: GPT-5 mini (0.471)
- Trust construct: GPT-5.6 (0.298)
- AI use: GPT-5 mini (0.465)

## Best balanced accuracy by binary gate

- Explicit trust boundary: GPT-5.6 (balanced accuracy 0.758; F1 0.787; MCC 0.466)
- Substantive trust content: GPT-5.6 (balanced accuracy 0.745; F1 0.794; MCC 0.441)

## Interpretation boundary

These are validation estimates from a stratified 100-comment sample. Per-industry estimates use only 25 comments and should be treated as transfer diagnostics rather than precise performance estimates.
