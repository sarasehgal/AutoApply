"""
chunking for rag. resumes get split bullet-by-bullet (not fixed-size windows) so each chunk
is one citable claim - makes it way easier for the matcher to point at exactly which bullet
backs up a score instead of some window that cuts a bullet in half
"""

from __future__ import annotations

import re

_BLANK_LINE = re.compile(r"\n\s*\n")
_BULLET_START = re.compile(r"\n(?=[-*•]\s)")


def chunk_resume(text: str) -> list[str]:
    """one chunk per bullet/paragraph"""
    chunks: list[str] = []
    for block in _BLANK_LINE.split(text.strip()):
        block = block.strip()
        if not block:
            continue
        for piece in _BULLET_START.split(block):
            piece = piece.strip()
            if piece:
                chunks.append(piece)
    return chunks


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    """generic sliding-window chunker for longer text, e.g. postings"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    step = max(max_chars - overlap, 1)
    while start < len(text):
        end = start + max_chars
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start += step
    return chunks
