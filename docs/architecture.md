# Architecture

The completed application will follow this flow:

1. The user uploads one or more supported documents.
2. The ingestion layer extracts and normalizes their text.
3. The chunking layer divides the text into retrieval units.
4. An embedding model converts chunks into vectors.
5. Chroma stores the vectors and document metadata.
6. A retrieval agent finds relevant chunks for the user's question.
7. A reasoning agent creates a grounded response.
8. A validation agent checks relevance, citations, and safety before display.

## Step 2 interface

The Streamlit interface now provides separate upload, question, and workflow tabs. It validates that documents and a meaningful question are present before allowing the request to continue. File summaries display document names, formats, sizes, and total upload size without processing document contents yet.
