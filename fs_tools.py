"""File system tools for reading, searching, listing, and writing resume files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

try:
    from docx import Document
except ImportError:  # pragma: no cover
    Document = None

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        PdfReader = None


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


@dataclass
class FileMetadata:
    path: str
    name: str
    extension: str
    size_bytes: int
    modified_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
        }


def _metadata_for(path: Path) -> FileMetadata:
    stats = path.stat()
    modified = datetime.fromtimestamp(stats.st_mtime, tz=timezone.utc).isoformat()
    return FileMetadata(
        path=str(path.resolve()),
        name=path.name,
        extension=path.suffix.lower(),
        size_bytes=stats.st_size,
        modified_at=modified,
    )


def _error(message: str, filepath: str | None = None, **extra: Any) -> dict[str, Any]:
    payload = {"success": False, "error": message}
    if filepath is not None:
        payload["filepath"] = str(Path(filepath))
    payload.update(extra)
    return payload


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_docx(path: Path) -> str:
    if Document is None:
        raise RuntimeError("python-docx is not installed. Install dependencies from requirements.txt.")
    document = Document(path)
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs]
    return "\n".join(text for text in paragraphs if text)


def _read_pdf(path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed. Install dependencies from requirements.txt.")
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(page.strip() for page in pages if page.strip())


def read_file(filepath: str) -> dict[str, Any]:
    """Read TXT, PDF, or DOCX files and return structured content and metadata."""
    path = Path(filepath).expanduser()

    if not path.exists():
        return _error("File not found.", filepath)
    if not path.is_file():
        return _error("Path is not a file.", filepath)

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        return _error(
            "Unsupported file type.",
            filepath,
            supported_extensions=sorted(SUPPORTED_EXTENSIONS),
        )

    try:
        if extension == ".txt":
            content = _read_txt(path)
        elif extension == ".docx":
            content = _read_docx(path)
        else:
            content = _read_pdf(path)
    except UnicodeDecodeError:
        return _error("Could not decode file as UTF-8 text.", filepath)
    except Exception as exc:
        return _error(f"Failed to read file: {exc}", filepath)

    metadata = _metadata_for(path).as_dict()
    return {
        "success": True,
        "filepath": str(path.resolve()),
        "content": content,
        "metadata": metadata,
        "content_length": len(content),
    }


def list_files(directory: str, extension: str | None = None) -> list[dict[str, Any]]:
    """List files in a directory and optionally filter by extension."""
    path = Path(directory).expanduser()
    if not path.exists() or not path.is_dir():
        return []

    normalized_extension = None
    if extension:
        normalized_extension = extension if extension.startswith(".") else f".{extension}"
        normalized_extension = normalized_extension.lower()

    files: list[dict[str, Any]] = []
    for child in sorted(path.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_file():
            continue
        if normalized_extension and child.suffix.lower() != normalized_extension:
            continue
        files.append(_metadata_for(child).as_dict())
    return files


def write_file(filepath: str, content: str) -> dict[str, Any]:
    """Write content to disk, creating parent directories when required."""
    path = Path(filepath).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception as exc:
        return _error(f"Failed to write file: {exc}", filepath)

    return {
        "success": True,
        "filepath": str(path.resolve()),
        "bytes_written": len(content.encode("utf-8")),
        "metadata": _metadata_for(path).as_dict(),
    }


def search_in_file(filepath: str, keyword: str, context_chars: int = 60) -> dict[str, Any]:
    """Search for a keyword in a file and return case-insensitive matches with context."""
    if not keyword.strip():
        return _error("Keyword must not be empty.", filepath)

    read_result = read_file(filepath)
    if not read_result.get("success"):
        return read_result

    content = read_result["content"]
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    matches: list[dict[str, Any]] = []

    for match in pattern.finditer(content):
        start = max(0, match.start() - context_chars)
        end = min(len(content), match.end() + context_chars)
        snippet = content[start:end].replace("\n", " ").strip()
        matches.append(
            {
                "match": match.group(0),
                "start_index": match.start(),
                "end_index": match.end(),
                "context": snippet,
            }
        )

    return {
        "success": True,
        "filepath": read_result["filepath"],
        "keyword": keyword,
        "match_count": len(matches),
        "matches": matches,
        "metadata": read_result["metadata"],
    }
