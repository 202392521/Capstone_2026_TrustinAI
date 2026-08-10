# Configuration records

`frozen_analysis_settings.yml` is the portable, machine-readable final settings ledger. The copied run summaries/manifests are archival records from final runs; local embedding-cache locations were replaced with the public model identifier and project paths were made relative. Use the YAML file for a clean re-run.

Industry-specific occupational identity stopwords affected CountVectorizer/c-TF-IDF topic representation only. They were not removed from sentence-transformer embeddings.
