"""Basic tests for the project foundation."""

from src.config import APP_NAME, SUPPORTED_EXTENSIONS


def test_app_name_is_defined() -> None:
    assert APP_NAME == "Agentic RAG Document Assistant"


def test_supported_extensions() -> None:
    assert SUPPORTED_EXTENSIONS == {"pdf", "txt", "csv", "xlsx"}
