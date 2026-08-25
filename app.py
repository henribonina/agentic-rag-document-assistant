"""Streamlit interface for the Agentic RAG Document Assistant."""

import streamlit as st

from src.document_loader import IngestionResult, load_documents
from src.text_splitter import split_documents
from src.ui_helpers import build_file_records, total_upload_size


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
    st.success("Step 4: Text chunking")
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
    st.caption(
        "Files are used only for the current session. Do not upload confidential "
        "documents until production security controls are implemented."
    )

st.title("Agentic RAG Document Assistant")
st.write(
    "Upload enterprise documents and ask questions in natural language. "
    "The application now extracts and chunks supported files; later steps will "
    "add embeddings, retrieval, agent reasoning, and grounded answer generation."
)

ingestion = IngestionResult()
chunks = ()

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
        if not chunks:
            st.error("Upload and process at least one readable document first.")
        elif not question.strip():
            st.error("Enter a question before continuing.")
        elif len(question.strip()) < 5:
            st.error("Please enter a more specific question.")
        else:
            st.success("Question accepted.")
            st.info(
                f"Your question is ready to search across "
                f"{len(chunks)} chunks from {len(ingestion.documents)} processed "
                "document(s). Embeddings and answer generation will be connected "
                "in upcoming steps."
            )

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
