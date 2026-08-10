# Frozen Prompt V2 provenance

`FROZEN_PROMPT_V2.txt` is the final annotation prompt/codebook used in the validated workflow. Prompt development drew on 150 manually coded finance, law, and software-engineering comments. After the prompt and schema were frozen, performance was assessed on an independent 50-comment healthcare holdout. The final four-industry validation used 100 comments (25 per industry): the non-healthcare 75-comment component and a fixed 25-comment healthcare component, including 23 subsequently blind-coded balanced-sample comments.

During all validation and production runs, model inputs contained text/context fields only; human labels and notes were not supplied to the model. Candidate-model comparison retained the frozen prompt/context approach. GPT-5 mini was the production model for the stored balanced-sample annotation workflow; GPT-5.6 and GPT-5.1 are retained as blinded candidate-model comparisons, not substituted production outputs.

No preliminary prompts, raw annotated examples, API keys, or request logs are included here.
