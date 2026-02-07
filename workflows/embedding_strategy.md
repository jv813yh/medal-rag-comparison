# Embedding Strategy Workflow

1.  **OpenAI**: Use `text-embedding-3-small` (or latest recommended).
2.  **Local**: Use Ollama's `mxbai-embed-large` or `nomic-embed-text`.
3.  **Chunking**: Use consistent chunk size (e.g., 512 tokens) with 10% overlap across all implementations.
4.  **Storage**: Use FAISS for vector storage in OpenAI and Local implementations.
