"""Multi-format document ingestion for PDF, TXT, CSV, and Excel files."""

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from pypdf import PdfReader


MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv", ".xlsx"}


@dataclass(frozen=True)
class LoadedDocument:
    """Normalized text and metadata extracted from one uploaded file."""

    source: str
    file_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def character_count(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class LoadError:
    """A safe, user-facing ingestion error."""

    source: str
    message: str


@dataclass(frozen=True)
class IngestionResult:
    """Successful documents and isolated file errors from one ingestion run."""

    documents: tuple[LoadedDocument, ...] = ()
    errors: tuple[LoadError, ...] = ()


def _read_bytes(file: Any) -> bytes:
    """Read an UploadedFile-like object and restore its cursor when possible."""
    if hasattr(file, "seek"):
        file.seek(0)
    data = file.read()
    if hasattr(file, "seek"):
        file.seek(0)
    if not isinstance(data, bytes):
        raise ValueError("The uploaded file could not be read as binary data.")
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise ValueError("The file exceeds the 20 MB ingestion limit.")
    return data


def _load_pdf(data: bytes) -> tuple[str, dict[str, Any]]:
    reader = PdfReader(BytesIO(data))
    sections = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            sections.append(f"## Page {page_number}\n{page_text}")
    if not sections:
        raise ValueError(
            "No selectable text was found. The PDF may require OCR before upload."
        )
    return "\n\n".join(sections), {"pages": len(reader.pages)}


def _load_txt(data: bytes) -> tuple[str, dict[str, Any]]:
    try:
        text = data.decode("utf-8-sig")
        encoding = "utf-8"
    except UnicodeDecodeError:
        text = data.decode("latin-1")
        encoding = "latin-1"
    text = text.strip()
    if not text:
        raise ValueError("No readable text was found in the TXT file.")
    return text, {"encoding": encoding}


def _load_csv(data: bytes) -> tuple[str, dict[str, Any]]:
    frame = pd.read_csv(BytesIO(data))
    if frame.empty and len(frame.columns) == 0:
        raise ValueError("No tabular content was found in the CSV file.")
    text = frame.fillna("").to_csv(index=False).strip()
    return text, {"rows": len(frame), "columns": list(map(str, frame.columns))}


def _load_xlsx(data: bytes) -> tuple[str, dict[str, Any]]:
    sheets = pd.read_excel(BytesIO(data), sheet_name=None, engine="openpyxl")
    if not sheets:
        raise ValueError("No worksheets were found in the Excel file.")
    sections = []
    row_count = 0
    for sheet_name, frame in sheets.items():
        row_count += len(frame)
        sheet_text = frame.fillna("").to_csv(index=False).strip()
        sections.append(f"## Sheet: {sheet_name}\n{sheet_text}")
    return "\n\n".join(sections), {
        "sheets": list(sheets.keys()),
        "rows": row_count,
    }


def load_document(file: Any) -> LoadedDocument:
    """Extract normalized text and metadata from one uploaded document."""
    source = Path(getattr(file, "name", "uploaded-file")).name
    extension = Path(source).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension or 'unknown'}.")

    data = _read_bytes(file)
    loaders = {
        ".pdf": _load_pdf,
        ".txt": _load_txt,
        ".csv": _load_csv,
        ".xlsx": _load_xlsx,
    }
    text, type_metadata = loaders[extension](data)
    metadata = {
        "source": source,
        "file_type": extension.lstrip("."),
        "size_bytes": len(data),
        "character_count": len(text),
        **type_metadata,
    }
    return LoadedDocument(
        source=source,
        file_type=extension.lstrip("."),
        text=text,
        metadata=metadata,
    )


def load_documents(files: list[Any]) -> IngestionResult:
    """Ingest files independently so one invalid file does not stop the batch."""
    documents = []
    errors = []
    for file in files:
        source = Path(getattr(file, "name", "uploaded-file")).name
        try:
            documents.append(load_document(file))
        except Exception as exc:  # File libraries can raise format-specific errors.
            errors.append(LoadError(source=source, message=str(exc)))
    return IngestionResult(documents=tuple(documents), errors=tuple(errors))
