import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data", "mtsamples.csv")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "vector_index.faiss")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "metadata.pkl")


EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 400

CHUNK_OVERLAP = 50
TOP_K = 5


