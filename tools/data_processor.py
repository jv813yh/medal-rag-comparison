import pandas as pd
import os
import tiktoken
import re

def normalize_query(text):
    """
    Step 1 of RAG Query Flow: Clean and normalize user query.
    """
    text = text.lower().strip()
    text = re.sub(r'[^\w\s?]', '', text)
    return text

def get_token_chunks(text, model="gpt-4o-mini", chunk_size=512, overlap=50):
    """
    Step 3 of Embedding Strategy: Consistent token-based chunking.
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i : i + chunk_size]
        chunks.append(encoding.decode(chunk_tokens))
    return chunks

def load_and_clean_data(file_path):
    """
    Loads the medical transcription dataset and performs basic cleaning.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    print(f"Reading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Basic cleaning as per workflow
    # Remove rows where transcription is missing
    initial_count = len(df)
    df = df.dropna(subset=['transcription'])
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    print(f"Cleaned data: {initial_count} -> {len(df)} rows.")
    return df

def normalize_data(df):
    """
    Normalizes text data.
    """
    # Standardize column names to lowercase
    df.columns = [col.lower() for col in df.columns]
    
    # Basic normalization (strip whitespace)
    df['transcription'] = df['transcription'].str.strip()
    
    return df

if __name__ == "__main__":
    # Test loading
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "mtsamples.csv")
    try:
        data = load_and_clean_data(data_path)
        data = normalize_data(data)
        print("Data normalization complete.")
        print(data.head())
    except Exception as e:
        print(f"Error: {e}")
