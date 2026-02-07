# Medical Transcription RAG Comparison 🩺🤖

A comprehensive benchmarking project designed to evaluate and compare three different **Retrieval-Augmented Generation (RAG)** architectures using clinical medical transcription data.

## 🎯 Purpose
The goal of this project is to provide a side-by-side comparison of modern RAG strategies applied to highly complex, specialized data (medical records). We measure **Accuracy**, **Faithfulness**, **Latency**, and **Cost** to identify which architecture best suits specific needs like privacy, speed, or reasoning capability.

---

## 🏗️ The Three Architectures

### 1. OpenAI RAG (Cloud-Native)
*   **Engine**: OpenAI `text-embedding-3-small` + `gpt-4o-mini`.
*   **Vector DB**: FAISS (FlatL2).
*   **Characteristics**: High speed, high accuracy, requires internet and API costs.
*   **Best for**: Prototyping and high-performance applications where data sharing is permitted.

### 2. Local Model RAG (Privacy-First)
*   **Engine**: Ollama - `mxbai-embed-large` + `Llama 3.2`.
*   **Vector DB**: FAISS (FlatL2).
*   **Characteristics**: Zero cost, 100% private, performance depends on local hardware.
*   **Best for**: Sensitive medical data where privacy is non-negotiable.

### 3. PageIndex RAG (Reasoning-Based)
*   **Engine**: VectifyAI PageIndex.
*   **Search Type**: Structure-aware tree navigation (Vectorless).
*   **Characteristics**: Decouples search from embeddings, focuses on navigating document hierarchy.
*   **Best for**: Complex documents where traditional vector search loses context.

---

## 🛠️ Project Structure
```text
├── data/               # Source MTSamples dataset
├── openai-rag/        # OpenAI pipeline implementation
├── local-model-rag/   # Ollama/Local pipeline implementation
├── pageindex-rag/     # PageIndex tree-based implementation
├── evaluation/        # Comparison suite and scoring metrics
├── tools/             # Data cleaning and Markdown export utilities
└── workflows/         # Core logic and strategy definitions
```

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
