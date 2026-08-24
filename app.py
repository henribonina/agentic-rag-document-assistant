"""Streamlit interface for the Agentic RAG Document Assistant."""

import streamlit as st

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
    st.success("Step 2: User interface")
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
    "Later project steps will connect this interface to ingestion, retrieval, "
    "agent reasoning, and grounded answer generation."
)

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
        col3.metric("Status", "Ready")

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
        st.success("Documents selected successfully. Continue to the question tab.")
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
        if not uploaded_files:
            st.error("Upload at least one document before submitting a question.")
        elif not question.strip():
            st.error("Enter a question before continuing.")
        elif len(question.strip()) < 5:
            st.error("Please enter a more specific question.")
        else:
            st.success("Question accepted.")
            st.info(
                "Step 2 validates the user input. Document ingestion and answer "
                "generation will be connected in the upcoming steps."
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
