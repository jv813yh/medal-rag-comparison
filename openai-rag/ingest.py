"""
OpenAI RAG Ingestion Script
---------------------------
This script handles the process of loading medical transcriptions, chunking them 
into token-based segments, generating embeddings using OpenAI's API, 
and storing them in a FAISS vector index for fast retrieval.
"""
import pandas as pd
import numpy as np
import faiss
import pickle
import os
import sys
from openai import OpenAI
from config import (DATA_PATH, INDEX_PATH, METADATA_PATH, 
                   EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, OPENAI_API_KEY)

# Add parent dir to path to import tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.data_processor import load_and_clean_data, normalize_data, get_token_chunks

client = OpenAI(api_key=OPENAI_API_KEY)


def ingest():
    """
    Orchestrates the ingestion pipeline:
    1. Loads and cleans raw CSV data.
    2. Chunks transcriptions into 400-token segments.
    3. Converts text chunks into vector embeddings via OpenAI.
    4. Saves vectors into a FAISS index and stores text metadata in a PKL file.
    """
    print(f"Loading data from {DATA_PATH}...")
    df = load_and_clean_data(DATA_PATH)
    df = normalize_data(df)
    
    # For speed in testing, we use a subset
    # df = df.head(100) 
    
    all_chunks = []
    print("Chunking transcriptions (tokens)...")
    for idx, row in df.iterrows():
        chunks = get_token_chunks(row['transcription'], model=CHAT_MODEL, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for chunk in chunks:

            all_chunks.append({
                'text': chunk,
                'medical_specialty': row['medical_specialty'],
                'sample_name': row['sample_name']
            })
    
    print(f"Total chunks: {len(all_chunks)}")
    
    print("Generating embeddings...")
    texts = [c['text'] for c in all_chunks]
    
    # Process in batches to avoid rate limits/large payloads
    batch_size = 100
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        batch_embeddings = [record.embedding for record in response.data]
        embeddings.extend(batch_embeddings)
        print(f"Processed {min(i+batch_size, len(texts))}/{len(texts)} chunks...")

    embeddings = np.array(embeddings).astype('float32')
    
    print(f"Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print(f"Saving index to {INDEX_PATH}...")
    faiss.write_index(index, INDEX_PATH)
    
    print(f"Saving metadata to {METADATA_PATH}...")
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(all_chunks, f)
    
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest()

