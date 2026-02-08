import os
import sys
from config import INDEX_PATH, MODEL

# Add the current directory to sys.path to find the local 'pageindex' shim
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add parent dir to path to import tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.data_processor import normalize_query

try:
    from pageindex import PageIndex
except ImportError as e:
    print(f"Error importing PageIndex: {e}")
    sys.exit(1)

def query(question):
    # Step 1 of RAG Query Flow: Preprocessing
    question = normalize_query(question)
    print(f"Normalized Query: {question}")

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Index not found at {INDEX_PATH}. Run ingest.py first.")

    # Load the index
    pi = PageIndex().load(INDEX_PATH)
    
    # Perform reasoning-based query
    # PageIndex navigates the tree structure to find the answer
    answer, chunks = pi.query(question, model=MODEL)
    
    print("\nPageIndex Answer:")
    print(answer)
    return answer, chunks


if __name__ == "__main__":
    query("What are the most common symptoms mentioned in these medical records?")

