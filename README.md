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

The ingestion and RAG controls displayed by the interface will be connected in later steps.

## Security

Never commit `.env`, API keys, private documents, or generated vector databases.

## License

MIT
