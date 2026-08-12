# Configuration records

`frozen_analysis_settings.yml` is the portable, machine-readable final settings ledger. The copied run summaries/manifests are archival records from final runs; local embedding-cache locations were replaced with the public model identifier and project paths were made relative. Use the YAML file for a clean re-run.

Industry-specific occupational identity stopwords affected CountVectorizer/c-TF-IDF topic representation only. They were not removed from sentence-transformer embeddings.

`final_annotation_provenance_2026-08-12.json` is the sanitised final ledger for the balanced 2,000-comment model passes and the N=200 prediction-blind random audit. It records model identifiers, prompt/schema hashes, request privacy settings, completion counts, and the production/robustness decision without exposing local data paths, comment identifiers, or API credentials.
