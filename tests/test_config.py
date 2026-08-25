"""Basic tests for the project foundation."""

from io import BytesIO

from src.config import APP_NAME, SUPPORTED_EXTENSIONS
from src.document_loader import load_document, load_documents
from src.text_splitter import split_document, split_documents
from src.ui_helpers import build_file_records, format_bytes, total_upload_size


def test_app_name_is_defined() -> None:
    assert APP_NAME == "Agentic RAG Document Assistant"


def test_supported_extensions() -> None:
    assert SUPPORTED_EXTENSIONS == {"pdf", "txt", "csv", "xlsx"}


def test_format_bytes() -> None:
    assert format_bytes(500) == "500 B"
    assert format_bytes(1536) == "1.5 KB"


def test_file_display_helpers() -> None:
    class ExampleFile:
        name = "report.pdf"
        size = 2048

    files = [ExampleFile()]
    assert build_file_records(files) == [
        {"Name": "report.pdf", "Type": "PDF", "Size": "2.0 KB"}
    ]
    assert total_upload_size(files) == "2.0 KB"


class Upload(BytesIO):
    def __init__(self, name: str, content: bytes) -> None:
        super().__init__(content)
        self.name = name
        self.size = len(content)


def test_txt_ingestion() -> None:
    document = load_document(Upload("notes.txt", b"Grounded answer content"))
    assert document.source == "notes.txt"
    assert document.file_type == "txt"
    assert document.text == "Grounded answer content"
    assert document.metadata["encoding"] == "utf-8"


def test_csv_ingestion() -> None:
    document = load_document(Upload("records.csv", b"name,value\nalpha,10\n"))
    assert document.metadata["rows"] == 1
    assert "alpha,10" in document.text


def test_batch_isolates_errors() -> None:
    result = load_documents(
        [Upload("valid.txt", b"Valid text"), Upload("unsupported.docx", b"data")]
    )
    assert len(result.documents) == 1
    assert len(result.errors) == 1
    assert result.errors[0].source == "unsupported.docx"


def test_document_chunking_preserves_metadata() -> None:
    document = load_document(
        Upload("long.txt", ("First sentence. " * 100).encode("utf-8"))
    )
    chunks = split_document(document, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert chunks[0].chunk_id == "long-txt-0000"
    assert chunks[0].metadata["source"] == "long.txt"
    assert chunks[1].metadata["chunk_index"] == 1
    assert all(chunk.character_count <= 300 for chunk in chunks)


def test_multiple_document_chunk_order() -> None:
    first = load_document(Upload("first.txt", b"First document content."))
    second = load_document(Upload("second.txt", b"Second document content."))
    chunks = split_documents([first, second], chunk_size=200, chunk_overlap=20)
    assert [chunk.source for chunk in chunks] == ["first.txt", "second.txt"]
