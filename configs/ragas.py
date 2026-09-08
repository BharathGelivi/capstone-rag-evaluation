# RAGAS-style metric configuration.

# Number of synthetic reverse-questions generated per answer for Answer Relevancy.
ANSWER_RELEVANCY_NUM_QUESTIONS = 3

# Weights (factual_f1, answer_similarity) combined into Answer Correctness. Matches ragas' defaults.
ANSWER_CORRECTNESS_WEIGHTS = (0.75, 0.25)

# Entailment threshold used by the NLI model when judging claim/sentence support
# for Context Recall and Answer Correctness (reference-based metrics).
RAGAS_ENTAILMENT_THRESHOLD = 0.5
