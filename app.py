"""Streamlit interface for the Agentic RAG Document Assistant."""

import streamlit as st

from src.document_loader import IngestionResult, load_documents
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
    st.success("Step 3: Document ingestion")
    st.write("Supported formats")
    st.code("PDF  TXT  CSV  XLSX", language=None)
    st.divider()
    st.caption(
        "Files are used only for the current session. Do not upload confidential "
        "documents until production security controls are implemented."
    )

st.title("Agentic RAG Document Assistant")
st.write(
    "Upload enterprise documents and ask questions in natural language. "
    "The application now extracts content from supported files; later steps will "
    "add chunking, retrieval, agent reasoning, and grounded answer generation."
)

ingestion = IngestionResult()

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
            st.success(
                f"Processed {len(ingestion.documents)} document(s) and extracted "
                f"{total_characters:,} characters."
            )
            with st.expander("Review extracted-content previews"):
                for document in ingestion.documents:
                    st.markdown(f"**{document.source}**")
                    preview = document.text[:700]
                    if len(document.text) > 700:
                        preview += "..."
                    st.text(preview)

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
        if not ingestion.documents:
            st.error("Upload and process at least one readable document first.")
        elif not question.strip():
            st.error("Enter a question before continuing.")
        elif len(question.strip()) < 5:
            st.error("Please enter a more specific question.")
        else:
            st.success("Question accepted.")
            st.info(
                f"Your question is ready to search across "
                f"{len(ingestion.documents)} processed document(s). Text chunking "
                "and answer generation will be connected in upcoming steps."
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
