# Final two-batch random-audit performance (n=200)

This is an offline evaluation of frozen predictions against prediction-blind human coding. No API requests were made and no production labels were changed.

## Three models across three tasks

| Task | Model | Accuracy | Macro-F1 | Weighted-F1 | Balanced accuracy | Precision | Recall | F1 | MCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Attitude | GPT-5 mini | 0.555 | 0.498 | 0.532 | - | - | - | - | - |
| Attitude | GPT-5.6 Sol | 0.540 | 0.452 | 0.506 | - | - | - | - | - |
| Attitude | GPT-5.1 | 0.525 | 0.500 | 0.491 | - | - | - | - | - |
| Capability assessment | GPT-5 mini | 0.595 | 0.467 | 0.596 | - | - | - | - | - |
| Capability assessment | GPT-5.6 Sol| 0.620 | 0.481 | 0.622 | - | - | - | - | - |
| Capability assessment | GPT-5.1 | 0.655 | 0.503 | 0.649 | - | - | - | - | - |
| Explicit trust boundary | GPT-5 mini | 0.750 | - | - | 0.745 | 0.775 | 0.660 | 0.713 | 0.499 |
| Explicit trust boundary | GPT-5.6 Sol| 0.715 | - | - | 0.708 | 0.753 | 0.585 | 0.659 | 0.431 |
| Explicit trust boundary | GPT-5.1 | 0.680 | - | - | 0.664 | 0.826 | 0.404 | 0.543 | 0.390 |

## Interpretation lock

- GPT-5 mini remains the production model for all three production variables.
- GPT-5.6 Sol capability performance is reported only as a robustness/model-comparison result.
- Differences between batches or models do not trigger relabelling or a production-model switch.
- Evaluated comments: 200 (100 in each of two non-overlapping audit batches).
- Industry distribution: 50 comments per industry.
- Full per-class metrics, confusion matrices and industry-stratified metrics are saved alongside this report.
