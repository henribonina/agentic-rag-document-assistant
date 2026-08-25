# Agentic RAG Document Assistant

A Generative AI document question-answering application using Retrieval-Augmented Generation (RAG), semantic search, and autonomous AI agents.

## Capstone objective

Users will be able to upload PDF, TXT, CSV, and Excel documents, ask natural-language questions, and receive answers grounded in retrieved document content.

## Planned architecture

1. Streamlit user interface
2. Multi-format document ingestion
3. Text cleaning and chunking
4. Embedding generation
5. Chroma vector database
6. Semantic retrieval
7. LLM-based grounded generation
8. Planning, retrieval, reasoning, and validation agents
9. Reliability and safety controls

## Local setup

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create a local environment file:

```bash
copy .env.example .env
```

Add your API key to `.env`, then run:

```bash
streamlit run app.py
```

## Current status

- Step 1 complete: repository structure, dependencies, and configuration template.
- Step 2 complete: Streamlit interaction layer with multi-file upload, file summaries, question input, validation, and workflow guidance.
- Step 3 complete: document ingestion for PDF, TXT, CSV, and Excel, including normalized text, metadata, per-file error isolation, size limits, and interface previews.
- Step 4 complete: configurable, overlapping text chunks with natural boundary detection, stable chunk IDs, source metadata, and interface previews.
- Step 5 complete: embedding generation and Chroma vector indexing with production OpenAI embeddings or deterministic offline development embeddings.
- Step 6 complete: semantic question retrieval with configurable result counts, ranked relevance scores, source filenames, locations, and expandable evidence passages.
- Step 7 complete: grounded answer generation through the OpenAI Responses API with evidence-only instructions, inline source labels, citation validation, prompt-injection resistance, and an explicit insufficient-evidence response.
- Step 8 complete: coordinated planning, retrieval, reasoning, and validation agents with a visible execution trace and a single model call per question.
- Step 9 complete: deterministic input and output guardrails, bounded retries for transient API errors, human-review guidance, visible safety checks, and offline regression evaluations.

The capstone workflow is now complete from document upload through safe, evaluated answer delivery.

### Embedding modes

- **OpenAI:** Uses `text-embedding-3-small` when `OPENAI_API_KEY` is configured in the local `.env` file.
- **Local:** Uses deterministic 384-dimensional token-hash vectors for offline development and testing. Local mode is not a substitute for a production semantic embedding model.

### Grounded answers

The answer generator uses `gpt-5-mini` by default. It receives only the user question and retrieved passages, treats document text as untrusted data, and requires inline citations such as `[S1]`. Set `OPENAI_MODEL` in the local `.env` file to use another compatible model.

### Specialized agents

Each question moves through four bounded roles: the planning agent defines the search objective, the retrieval agent gathers ranked evidence, the reasoning agent creates the grounded response, and the validation agent checks the final output before display. The roles are coordinated locally and use one OpenAI model request, limiting unnecessary cost and latency.

### Safety and evaluation

The application rejects malformed or oversized questions, common instruction-bypass attempts, empty or oversized answers, and invalid evidence labels. Transient API failures receive one bounded retry. An offline evaluation tab tests expected behavior without an API call, while the interface keeps retrieved passages visible for human review.

## Security

Never commit `.env`, API keys, private documents, or generated vector databases.

## License

MIT
