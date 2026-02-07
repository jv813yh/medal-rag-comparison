import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = "llama3.2:latest"
EMBED_MODEL = "mxbai-embed-large"
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data", "mtsamples.csv")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "vector_index.faiss")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "metadata.pkl")


CHUNK_SIZE = 400

CHUNK_OVERLAP = 50
TOP_K = 5


