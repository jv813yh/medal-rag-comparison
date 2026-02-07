"""
Local Model RAG Ingestion Script
--------------------------------
This script runs a fully private ingestion pipeline using Ollama. It manages 
local embedding generation and handles errors like missing models or 
context length overflows.
"""
import pandas as pd
import numpy as np

import faiss
import pickle
import os
import sys
import ollama
from config import (DATA_PATH, INDEX_PATH, METADATA_PATH, 
                   EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP)

# Add parent dir to path to import tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.data_processor import load_and_clean_data, normalize_data, get_token_chunks

def ensure_model_available(model_name):
    """
    Checks if the model is available in Ollama, pulls it if not.
    """
    print(f"Checking if model '{model_name}' is available...")
    try:
        ollama.show(model_name)
    except ollama.ResponseError:
        print(f"Model '{model_name}' not found. Pulling it now...")
        ollama.pull(model_name)
        print(f"Model '{model_name}' pulled successfully.")


def ingest():
    print(f"Loading data from {DATA_PATH}...")
    df = load_and_clean_data(DATA_PATH)
    df = normalize_data(df)

    # Ensure models are available before starting
    ensure_model_available(EMBED_MODEL)
    
    # Using a smaller subset for local processing stability - recommended for testing
    if len(df) > 100:
        print("Note: Dataset is large. Limiting to 100 samples for local testing. Edit ingest.py to change.")
        df = df.head(100)

    
    all_chunks = []
    print("Chunking transcriptions (tokens)...")
    for idx, row in df.iterrows():
        chunks = get_token_chunks(row['transcription'], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for chunk in chunks:

            all_chunks.append({
                'text': chunk,
                'medical_specialty': row['medical_specialty'],
                'sample_name': row['sample_name']
            })
    
    print(f"Total chunks: {len(all_chunks)}")
    
    print(f"Generating embeddings using {EMBED_MODEL}...")
    embeddings = []
    successful_chunks = []
    for i, chunk in enumerate(all_chunks):
        try:
            # We explicitly set a larger context locally just in case, 
            # though model architectural limits apply.
            response = ollama.embeddings(
                model=EMBED_MODEL, 
                prompt=chunk['text'],
                options={"num_ctx": 1024}
            )
            embeddings.append(response['embedding'])
            successful_chunks.append(chunk)
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(all_chunks)} chunks...")
        except Exception as e:
            print(f"Warning: Skipping chunk {i} due to error: {e}")
            continue
    
    if not embeddings:
        print("Error: No embeddings were generated. Check your Ollama logs.")
        return

    embeddings = np.array(embeddings).astype('float32')
    
    print(f"Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print(f"Saving index to {INDEX_PATH}...")
    faiss.write_index(index, INDEX_PATH)
    
    print(f"Saving metadata to {METADATA_PATH}...")
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(successful_chunks, f)

    
    print("Local Ingestion complete!")

if __name__ == "__main__":
    ingest()

