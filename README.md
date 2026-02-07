# Medical Transcription RAG Comparison 🩺🤖

A comprehensive benchmarking project designed to evaluate and compare three different **Retrieval-Augmented Generation (RAG)** architectures using clinical medical transcription data.

## 📊 Dataset
The project uses the **Medical Transcriptions (MTSamples)** dataset from Kaggle.
*   **Source**: [Medical Transcriptions on Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
*   **Content**: Over 5,000 real-world medical transcriptions covering various medical specialties.
*   **Setup**: Download `mtsamples.csv` and place it in the `data/` directory.

---

## 🏗️ Project Structure & Component Details

### 🛠️ Global Tools (`/tools`)
*   `data_processor.py`: The heart of data handling. Contains functions for cleaning the CSV, normalizing medical text, and performing token-based chunking.
*   `export_to_markdown.py`: Converts CSV rows into individual `.md` files. This is crucial for the PageIndex RAG, which processes documents rather than raw table rows.

### ☁️ OpenAI RAG (`/openai-rag`)
A high-performance implementation using the official OpenAI API.
*   `config.py`: Centralized settings for API keys, model names (`gpt-4o-mini`), and FAISS parameters.
*   `ingest.py`: Loads data, generates embeddings via OpenAI API, and saves them into a FAISS index.
*   `query.py`: Handles the RAG loop: Query -> Embedding -> FAISS Search -> Prompt Augmentation -> LLM Answer.

### 🏠 Local Model RAG (`/local-model-rag`)
A fully private RAG running locally on your machine via Ollama.
*   `config.py`: Local settings for `Llama 3.2` and `mxbai-embed-large`.
*   `ingest.py`: Multi-threaded embedding generation (local) and FAISS indexing. It includes a "self-healing" feature to auto-pull missing models.
*   `query.py`: Uses local LLM for generation.

### 🌳 PageIndex RAG (`/pageindex-rag`)
Advanced reasoning-based RAG using [VectifyAI PageIndex](https://github.com/VectifyAI/PageIndex).
*   `config.py`: Configuration for PageIndex environment.
*   `ingest.py`: Builds a hierarchical tree-index from Markdown documents.
*   `query.py`: Performs reasoning-based retrieval across the document tree.

### � Evaluation Suite (`/evaluation`)
The benchmarking department.
*   `queries.json`: A standard set of 5 medical-domain questions to ensure a fair test.
*   `metrics.py`: Implements "LLM-as-a-judge" logic to score Relevance, Faithfulness, and Retrieval Precision.
*   `compare.py`: The orchestrator script that runs all three systems and generates a final comparison report.


---

## 🚀 Getting Started

### 1. Prerequisites
*   Python 3.10+
*   [Ollama](https://ollama.com/) installed and running (for Local RAG).
*   OpenAI API Key in a `.env` file.

### 2. Installation
```bash
pip install -r requirements.txt
ollama pull llama3.2
ollama pull mxbai-embed-large
```

### 3. Data Preparation
```bash
# Clean original CSV and export to Markdown for PageIndex
python tools/export_to_markdown.py
```

### 4. Ingestion
Run ingestion for each model to build the indices:
```bash
python openai-rag/ingest.py
python local-model-rag/ingest.py
python pageindex-rag/ingest.py
```

---

## 📊 Running Evaluation
To benchmark all models against the standard medical query set:
```bash
python evaluation/compare.py
```

The results will be summarized in the terminal and detailed JSON reports will be saved in `evaluation/results/`.

---

## 🔍 Metrics Tracked
*   **Answer Relevance**: Does the RAG answer the specific medical question?
*   **Faithfulness**: Is the answer derived strictly from the retrieved context? (Anti-hallucination)
*   **Precision@K**: How many of the top retrieved chunks were actually relevant?
*   **Latency**: How many seconds per query?
*   **Cost**: Total API spend for the run.

---

## ⚖️ Next Steps
Run the `compare.py` script to generate the final comparison table. Once the data is ready, this README will be updated with the performance breakdown.
