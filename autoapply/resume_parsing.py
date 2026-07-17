"""pulls plain text out of an uploaded resume file - txt/md are trivial, pdf/docx need real parsing"""

from __future__ import annotations

import io

from docx import Document
from pypdf import PdfReader


class UnsupportedResumeFormatError(Exception):
    """file extension isn't one we know how to read"""


def extract_resume_text(filename: str, content: bytes) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext in ("txt", "md"):
        return content.decode("utf-8")
    if ext == "pdf":
        return _extract_pdf_text(content)
    if ext == "docx":
        return _extract_docx_text(content)
    raise UnsupportedResumeFormatError(f"can't read .{ext} files - use .txt, .md, .pdf, or .docx")


def _extract_pdf_text(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


def _extract_docx_text(content: bytes) -> str:
    doc = Document(io.BytesIO(content))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)
