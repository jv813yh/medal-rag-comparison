import os
import pandas as pd
import sys

# Add parent dir to path to import tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.data_processor import load_and_clean_data, normalize_data

def export_to_markdown(limit=50):
    print("STARTING EXPORT...")
    data_path = os.path.join("data", "mtsamples.csv")
    output_dir = os.path.join("pageindex-rag", "docs")

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Loading data for export (limit={limit})...")
    df = load_and_clean_data(data_path)
    df = normalize_data(df)
    
    # Take a subset
    subset = df.head(limit)
    
    for idx, row in subset.iterrows():
        # Create a safe filename
        safe_name = "".join([c if c.isalnum() else "_" for c in str(row['sample_name'])])
        filename = f"{idx:03d}_{safe_name}.md"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Medical Transcription: {row['sample_name']}\n\n")
            f.write(f"**Specialty**: {row['medical_specialty']}\n")
            f.write(f"**Keywords**: {row['keywords']}\n\n")
            f.write(f"## Transcription\n\n")
            f.write(row['transcription'])
            
    print(f"Exported {len(subset)} files to {output_dir}")

if __name__ == "__main__":
    export_to_markdown()
