# Script inventory and portability

Scripts are copies of the final workflow logic. Their substantive algorithms, parameter choices, prompts, and label definitions were not altered. Only workstation-specific default paths and local Hugging Face cache references were made relative/configurable in the packaged copies.

- `discovery/`: Selenium-based Reddit Community Search.
- `corpus/`: active-month scoring, Arctic Shift collection, tier harmonisation, Law LLM-degree cleaning, merge, and audit.
- `bertopic/`: industry, pooled/JSD, explicit-boundary, and lexical analyses.
- `annotation/`: balanced sample assembly and resumable frozen-prompt annotation. These files can make API calls when run; they are supplied for provenance and must not be run without explicit approval, a valid key, and data-governance review.
- `validation/`: aggregate validation metrics reconstruction from locally held non-distributable label/prediction rows.
- `robustness/`: UMAP sensitivity audit logic.
- `reporting/`: figure/table generators.

The package deliberately omits manually coded comment workbooks and raw corpora; scripts that require them will need locally reconstructed equivalents. See `DATA_AVAILABILITY.md`.
