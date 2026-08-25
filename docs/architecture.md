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

## Step 3 ingestion

The ingestion layer accepts PDF, TXT, CSV, and XLSX files up to 20 MB each. It extracts selectable PDF text by page, decodes text files, converts CSV rows into normalized text, and converts every Excel worksheet into labeled text. Each successful result includes source and format metadata. Errors are isolated per file so one invalid upload does not stop the remaining batch.

## Step 4 chunking

The chunking layer converts extracted text into overlapping retrieval units. It prefers paragraph, line, sentence, and word boundaries near the configured chunk size. Every chunk receives a stable ID, source filename, sequence index, character offsets, and inherited document metadata. Default settings use 1,000 characters with 150 characters of overlap and can be adjusted in the interface.

## Step 5 vector knowledge store

The vector layer generates one embedding per text chunk and upserts the vectors, chunk text, and scalar metadata into an in-memory Chroma collection configured for cosine distance. OpenAI `text-embedding-3-small` is available for production-quality semantic vectors when a local API key is configured. A deterministic 384-dimensional local provider supports offline development and repeatable tests without external calls.
