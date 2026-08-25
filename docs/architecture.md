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

## Step 6 semantic retrieval

The retrieval layer normalizes each question, embeds it with the same provider used for document chunks, and queries Chroma for the nearest passages. The interface allows one to eight results and displays each match in ranked order with a bounded relevance score, source filename, page or sheet label when available, stable chunk reference, and expandable source text. The retrieved passages become the evidence set for grounded generation in the next stage.

## Step 7 grounded answer generation

The generation layer sends the question and retrieved evidence to the OpenAI Responses API using `gpt-5-mini` by default. System-level grounding rules require evidence-only answers, inline `[S#]` labels, an explicit insufficient-evidence response, and treatment of all uploaded text as untrusted quoted data. The application validates that every cited label belongs to the retrieved evidence set before displaying the answer, and API response storage is disabled for each request.

## Step 8 specialized agent orchestration

The orchestration layer coordinates four focused components in a fixed, inspectable sequence. The planning agent normalizes the request and bounds retrieval. The retrieval agent searches the indexed documents. The reasoning agent delegates one grounded generation request to the Responses API. The validation agent checks the final text and source labels before release. Each completed stage produces an audit-friendly status record displayed in the interface. This design keeps control flow deterministic while reserving model inference for the reasoning task.
