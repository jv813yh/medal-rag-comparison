import faiss
import pickle
import numpy as np
import os
import sys
from openai import OpenAI
from config import (INDEX_PATH, METADATA_PATH, EMBEDDING_MODEL, 
                   CHAT_MODEL, TOP_K, OPENAI_API_KEY)

# Add parent dir to path to import tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.data_processor import normalize_query

client = OpenAI(api_key=OPENAI_API_KEY)

def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Index or Metadata not found. Run ingest.py first.")
    
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata

def query(question):
    # Step 1 of RAG Query Flow: Preprocessing
    question = normalize_query(question)
    print(f"Normalized Query: {question}")
    
    index, metadata = load_index()

    
    # 1. Embed the query
    response = client.embeddings.create(input=[question], model=EMBEDDING_MODEL)
    query_embedding = np.array([response.data[0].embedding]).astype('float32')
    
    # 2. Search FAISS
    distances, indices = index.search(query_embedding, TOP_K)
    
    retrieved_chunks = [metadata[i] for i in indices[0]]
    context = "\n\n---\n\n".join([c['text'] for c in retrieved_chunks])
    
    # 3. Generate Answer
    system_prompt = """You are a medical assistant. Use the following pieces of retrieved context 
    from medical transcriptions to answer the user's question. If you don't know the answer 
    based on the context, say that you don't know. Keep the answer professional and concise."""
    
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    answer = completion.choices[0].message.content
    print("\nAnswer:")
    print(answer)
    return answer, [c['text'] for c in retrieved_chunks]



if __name__ == "__main__":
    query("What are the symptoms and diagnosis for the patient in the records?")


