# RAG Query Flow Workflow

1.  **Preprocessing**: Clean and normalize user query.
2.  **Retrieval**: Fetch Top-K (e.g., K=5) relevant chunks.
3.  **Augmentation**: Construct prompt with system instructions, retrieved context, and user query.
4.  **Generation**: Call LLM (OpenAI / Ollama).
5.  **Post-processing**: Format response and log metadata (latency, cost).
