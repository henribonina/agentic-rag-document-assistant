"""Presentation helpers for the Streamlit user interface."""

from collections.abc import Iterable
from typing import Any


def format_bytes(size: int) -> str:
    """Return a human-readable representation of a byte count."""
    value = float(max(size, 0))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def build_file_records(files: Iterable[Any]) -> list[dict[str, str]]:
    """Build display records from Streamlit UploadedFile-like objects."""
    records = []
    for file in files:
        name = getattr(file, "name", "Unnamed file")
        extension = name.rsplit(".", 1)[-1].upper() if "." in name else "UNKNOWN"
        records.append(
            {
                "Name": name,
                "Type": extension,
                "Size": format_bytes(getattr(file, "size", 0)),
            }
        )
    return records


def total_upload_size(files: Iterable[Any]) -> str:
    """Return the combined size of UploadedFile-like objects."""
    return format_bytes(sum(max(getattr(file, "size", 0), 0) for file in files))
