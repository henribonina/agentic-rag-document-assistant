"""Basic tests for the project foundation."""

from io import BytesIO

from src.config import APP_NAME, SUPPORTED_EXTENSIONS
from src.document_loader import load_document, load_documents
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
