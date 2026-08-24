"""Basic tests for the project foundation."""

from src.config import APP_NAME, SUPPORTED_EXTENSIONS
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
