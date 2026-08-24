"""Entry point for the Agentic RAG Document Assistant."""

import streamlit as st


st.set_page_config(page_title="Agentic RAG Document Assistant", page_icon="📄")
st.title("Agentic RAG Document Assistant")
st.write(
    "Upload enterprise documents and ask questions grounded in their content. "
    "Document processing and AI retrieval will be added in the next steps."
)

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "csv", "xlsx"],
    accept_multiple_files=True,
)
question = st.text_input("Ask a question about your documents")

if uploaded_files:
    st.success(f"Selected {len(uploaded_files)} document(s).")

if st.button("Ask"):
    if not uploaded_files:
        st.warning("Please upload at least one document.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        st.info("The RAG response pipeline will be implemented in a later step.")
