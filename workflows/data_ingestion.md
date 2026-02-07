# Data Ingestion Workflow

1.  **Download**: Obtain the `mtsamples.csv` from Kaggle.
2.  **Clean**: Remove empty rows or rows missing the `transcription` field.
3.  **Normalize**: Standardize headers and text formatting.
4.  **Partition**: Split data into ingestion and evaluation sets if necessary.
5.  **Verify**: Ensure row counts match across different RAG pipeline starts.
