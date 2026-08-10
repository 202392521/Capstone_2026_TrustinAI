# Frozen researcher-led mappings

The mapping files preserve decisions made **after** model fitting. They do not change BERTopic assignments.

- `cross_industry_pooled_topic_meta_mapping_FINAL.csv`: final pooled-topic labels, inclusion decisions, and cross-industry status.
- `explicit_boundary_original_topic_to_reporting_theme_mapping.csv`: 22 raw explicit-boundary topics mapped to 14 reporting themes.
- `explicit_boundary_meta_theme_to_reporting_theme_hierarchy.csv`: reporting themes mapped to six broader meta-themes.
- `*_human_verified.csv`: researcher-assigned labels and counts used for reporting.

Topic -1 was retained for audit but excluded from substantive-theme prevalence. Any manually excluded incoherent or out-of-scope clusters remain visible in the mapping decision fields. These mappings are frozen interpretive records, not automatically learned labels.
