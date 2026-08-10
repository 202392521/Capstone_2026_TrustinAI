# UMAP sensitivity audit

This directory contains the complete pre-specified sensitivity grid for the four industry models, pooled model, and explicit-boundary model where audited. The frozen reporting configurations were set before this post-hoc audit. Audit results were not used to reselect parameters, refit reporting models, or change researcher-led labels.

The grid varied UMAP `n_neighbors` (5, 10, 15, 30, 50 at `min_dist=0`) and `min_dist` (0.1 and 0.5 at 15 neighbours), holding five dimensions, cosine metric, seed 42, corpus, embeddings, HDBSCAN, vectorizer, and representation settings fixed. The `nn15_md0_reproduction` row is an exact reproducibility check. ARI/NMI/AMI/Jaccard/topic-count/outlier metrics are retained in full. Exact boundaries and topic counts can vary; neighbouring settings nonetheless recover substantial structure for several models. Finance is comparatively more parameter-sensitive.
