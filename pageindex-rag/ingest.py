import os
import sys

# Add the current directory to sys.path to find the local 'pageindex' shim
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DOCS_DIR, INDEX_PATH

try:
    from pageindex import PageIndex
except ImportError as e:
    print(f"Error importing PageIndex: {e}")
    sys.exit(1)

def ingest():
    print(f"Indexing documents from {DOCS_DIR}...")
    
    # Verify docs exist
    if not os.path.exists(DOCS_DIR) or len(os.listdir(DOCS_DIR)) <= 1: # <=1 because of .gitkeep
        print(f"Error: {DOCS_DIR} is empty. Run tools/export_to_markdown.py first.")
        return

    # Initialize PageIndex
    # PageIndex uses CHATGPT_API_KEY from .env automatically
    pi = PageIndex()
    
    # Build the tree structure index
    pi.index(DOCS_DIR)
    
    # Save the index to disk
    pi.save(INDEX_PATH)
    
    print(f"PageIndex indexing complete! Index saved to {INDEX_PATH}")

if __name__ == "__main__":
    ingest()

