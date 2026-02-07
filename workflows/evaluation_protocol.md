# Evaluation Protocol Workflow

1.  **Dataset**: Use the same test queries for all three models.
2.  **Metrics**: Calculate:
    *   Answer Relevance (LLM-as-a-judge)
    *   Faithfulness (Factual consistency)
    *   Retrieval Precision@K
3.  **Performance**: Track latency (ms) and cost ($).
4.  **Reporting**: Aggregate results in `evaluation/results/` and run `compare.py`.
