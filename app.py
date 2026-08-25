"""Streamlit interface for the Agentic RAG Document Assistant."""

import streamlit as st

from src.agents import run_agent_workflow
from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.document_loader import IngestionResult, load_documents
from src.retriever import retrieve_passages
from src.text_splitter import split_documents
from src.ui_helpers import build_file_records, total_upload_size
from src.vector_store import ChromaVectorStore, create_embedding_provider


st.set_page_config(
    page_title="Agentic RAG Document Assistant",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1100px; padding-top: 2rem;}
    [data-testid="stMetric"] {
        background: #f7f9fc;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Project status")
    st.success("Step 8: Specialized agent orchestration")
    st.write("Supported formats")
    st.code("PDF  TXT  CSV  XLSX", language=None)
    st.divider()
    st.subheader("Chunk settings")
    chunk_size = st.slider(
        "Chunk size (characters)",
        min_value=500,
        max_value=2_000,
        value=1_000,
        step=100,
    )
    chunk_overlap = st.slider(
        "Chunk overlap (characters)",
        min_value=50,
        max_value=300,
        value=150,
        step=25,
    )
    st.divider()
    default_mode = 0 if OPENAI_API_KEY else 1
    embedding_label = st.selectbox(
        "Embedding provider",
        options=["OpenAI", "Local"],
        index=default_mode,
        help="OpenAI provides production-quality semantic embeddings. Local mode "
        "is deterministic and intended for offline development.",
    )
    top_k = st.slider(
        "Passages to retrieve",
        min_value=1,
        max_value=8,
        value=4,
        help="The number of relevant document passages shown for each question.",
    )
    st.caption(f"Answer model: {OPENAI_MODEL}")
    st.divider()
    st.caption(
        "Files are used only for the current session. Do not upload confidential "
        "documents until production security controls are implemented."
    )

st.title("Agentic RAG Document Assistant")
st.write(
    "Upload enterprise documents and ask questions in natural language. "
    "The application now extracts, chunks, embeds, and indexes supported files "
    "in Chroma, retrieves relevant passages, and generates answers grounded in "
    "those sources through coordinated planning, retrieval, reasoning, and "
    "validation agents."
)

ingestion = IngestionResult()
chunks = ()
vector_index = None
vector_store = None

upload_tab, question_tab, about_tab = st.tabs(
    ["1. Upload documents", "2. Ask a question", "How it works"]
)

with upload_tab:
    st.subheader("Document workspace")
    uploaded_files = st.file_uploader(
        "Choose one or more files",
        type=["pdf", "txt", "csv", "xlsx"],
        accept_multiple_files=True,
        help="Supported file types: PDF, TXT, CSV, and Excel (.xlsx).",
    )

    if uploaded_files:
        file_records = build_file_records(uploaded_files)
        col1, col2, col3 = st.columns(3)
        col1.metric("Documents", len(uploaded_files))
        col2.metric("Total size", total_upload_size(uploaded_files))
        col3.metric("Status", "Selected")

        st.dataframe(
            file_records,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Name": st.column_config.TextColumn("File name"),
                "Type": st.column_config.TextColumn("Format"),
                "Size": st.column_config.TextColumn("Size"),
            },
        )
        with st.spinner("Extracting document content..."):
            ingestion = load_documents(uploaded_files)

        if ingestion.documents:
            total_characters = sum(
                document.character_count for document in ingestion.documents
            )
            chunks = split_documents(
                ingestion.documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            st.success(
                f"Processed {len(ingestion.documents)} document(s) and extracted "
                f"{total_characters:,} characters."
            )
            chunk_col1, chunk_col2 = st.columns(2)
            chunk_col1.metric("Searchable chunks", len(chunks))
            chunk_col2.metric("Overlap", f"{chunk_overlap} characters")

            try:
                with st.spinner("Generating embeddings and building vector index..."):
                    embedding_provider = create_embedding_provider(
                        embedding_label, api_key=OPENAI_API_KEY
                    )
                    vector_store = ChromaVectorStore(embedding_provider)
                    vector_index = vector_store.index_chunks(chunks)
                vector_col1, vector_col2, vector_col3 = st.columns(3)
                vector_col1.metric("Indexed vectors", vector_index.chunk_count)
                vector_col2.metric(
                    "Vector dimensions", vector_index.embedding_dimension
                )
                vector_col3.metric("Embedding mode", embedding_label)
                st.success("The vector knowledge store is ready for semantic search.")
            except Exception as exc:
                st.error(f"Vector indexing failed: {exc}")
            with st.expander("Review extracted-content previews"):
                for document in ingestion.documents:
                    st.markdown(f"**{document.source}**")
                    preview = document.text[:700]
                    if len(document.text) > 700:
                        preview += "..."
                    st.text(preview)

            with st.expander("Review chunk previews"):
                for chunk in chunks[:10]:
                    st.markdown(
                        f"**{chunk.chunk_id}** · {chunk.character_count} characters"
                    )
                    st.text(chunk.text[:500] + ("..." if len(chunk.text) > 500 else ""))
                if len(chunks) > 10:
                    st.caption(f"Showing 10 of {len(chunks)} chunks.")

        if ingestion.errors:
            st.warning(f"{len(ingestion.errors)} file(s) could not be processed.")
            for error in ingestion.errors:
                st.error(f"{error.source}: {error.message}")
    else:
        st.info("Upload at least one document to begin.")

with question_tab:
    st.subheader("Ask about your documents")
    st.caption(
        "For the best results, ask a focused question that can be answered from "
        "the uploaded content."
    )

    with st.form("question_form"):
        question = st.text_area(
            "Question",
            placeholder="Example: What are the main risks described in these documents?",
            height=120,
            max_chars=1_000,
        )
        submitted = st.form_submit_button(
            "Analyze documents", type="primary", use_container_width=True
        )

    if submitted:
        if not chunks or vector_index is None:
            st.error("Upload, process, and index at least one readable document first.")
        elif not question.strip():
            st.error("Enter a question before continuing.")
        elif len(question.strip()) < 5:
            st.error("Please enter a more specific question.")
        else:
            try:
                if OPENAI_API_KEY:
                    with st.spinner("Agents are analyzing your documents..."):
                        agentic_result = run_agent_workflow(
                            question=question,
                            vector_store=vector_store,
                            api_key=OPENAI_API_KEY,
                            model=OPENAI_MODEL,
                            top_k=top_k,
                        )
                    passages = agentic_result.passages
                else:
                    with st.spinner("Searching for relevant passages..."):
                        passages = retrieve_passages(
                            vector_store,
                            question,
                            top_k=top_k,
                        )
                if not passages:
                    st.warning("No relevant passages were found in the index.")
                else:
                    st.success(
                        f"Retrieved {len(passages)} passage(s) from "
                        f"{len(ingestion.documents)} document(s)."
                    )
                    if OPENAI_API_KEY:
                        grounded_answer = agentic_result.answer
                        st.subheader("Grounded answer")
                        st.markdown(grounded_answer.text)
                        if grounded_answer.citation_ids:
                            cited_sources = []
                            for citation_id in grounded_answer.citation_ids:
                                passage = passages[int(citation_id[1:]) - 1]
                                cited_sources.append(
                                    f"[{citation_id}] {passage.source} — "
                                    f"{passage.location}"
                                )
                            st.caption("Cited evidence: " + " | ".join(cited_sources))
                        st.caption(f"Generated with {grounded_answer.model}")
                        with st.expander("Agent workflow", expanded=False):
                            st.caption(
                                f"Objective: {agentic_result.plan.objective}"
                            )
                            for step in agentic_result.steps:
                                st.markdown(
                                    f"✅ **{step.agent}** — {step.detail}"
                                )
                    else:
                        st.warning(
                            "Add OPENAI_API_KEY to your local .env file to "
                            "generate the grounded answer. The retrieved "
                            "evidence is shown below."
                        )

                    st.subheader("Supporting evidence")
                    for passage in passages:
                        score = round(passage.relevance * 100)
                        label = (
                            f"{passage.rank}. {passage.source} · "
                            f"{passage.location} · {score}% relevance"
                        )
                        with st.expander(label, expanded=passage.rank == 1):
                            st.write(passage.text)
                            st.caption(f"Reference: {passage.chunk_id}")
                    st.info(
                        "Answers are restricted to the displayed evidence. "
                        "Always review the cited passages before relying on a response."
                    )
            except Exception as exc:
                st.error(f"Question answering failed: {exc}")

with about_tab:
    st.subheader("Planned processing workflow")
    st.markdown(
        """
        1. **Upload** PDF, TXT, CSV, or Excel documents.
        2. **Extract and clean** document content.
        3. **Chunk and embed** the content for semantic search.
        4. **Retrieve** the most relevant passages for the question.
        5. **Reason and validate** with specialized AI agents.
        6. **Answer with sources** grounded in the uploaded documents.
        """
    )
