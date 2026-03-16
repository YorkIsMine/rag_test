"""Две стратегии чанкинга: fixed-size и structural."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loader import Document


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


# ── Стратегия 1: Fixed-size ──────────────────────────────────

def chunk_fixed(
    documents: list[Document],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[Chunk]:
    """Разбивает документы на чанки фиксированного размера с перекрытием."""
    chunks: list[Chunk] = []

    for doc in documents:
        text = doc.text
        start = 0
        idx = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            if not chunk_text.strip():
                start = end - overlap if overlap < chunk_size else end
                continue

            chunk = Chunk(
                text=chunk_text,
                metadata={
                    **doc.metadata,
                    "chunk_id": f"{doc.metadata.get('filename', 'unknown')}_fixed_{idx}",
                    "strategy": "fixed",
                    "section": f"chars {start}-{end}",
                    "char_count": len(chunk_text),
                },
            )
            chunks.append(chunk)
            idx += 1
            start = end - overlap if overlap < chunk_size else end

    return chunks


# ── Стратегия 2: Structural ──────────────────────────────────

def _split_markdown(text: str) -> list[tuple[str, str]]:
    """Разбивает markdown/txt по заголовкам. Возвращает [(section_name, text)]."""
    # Разбиваем по строкам-заголовкам (# ...)
    pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        # Нет заголовков — разбиваем по двойным переносам
        paragraphs = re.split(r"\n\s*\n", text)
        return [(f"paragraph_{i}", p) for i, p in enumerate(paragraphs) if p.strip()]

    sections: list[tuple[str, str]] = []

    # Текст до первого заголовка
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("preamble", preamble))

    for i, match in enumerate(matches):
        section_name = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((section_name, section_text))

    return sections


def _split_python(text: str) -> list[tuple[str, str]]:
    """Разбивает Python-код по функциям и классам."""
    pattern = re.compile(
        r"^((?:class|def|async\s+def)\s+\w+[^\n]*)",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))

    if not matches:
        return [("module", text)]

    sections: list[tuple[str, str]] = []

    # Импорты / код до первого определения
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("imports", preamble))

    for i, match in enumerate(matches):
        name = match.group(1).split("(")[0].strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        if block:
            sections.append((name, block))

    return sections


def chunk_structural(documents: list[Document]) -> list[Chunk]:
    """Разбивает документы по структуре: заголовки, функции, страницы."""
    chunks: list[Chunk] = []

    for doc in documents:
        file_type = doc.metadata.get("file_type", "txt")

        if file_type == "py":
            sections = _split_python(doc.text)
        elif file_type in ("md", "txt"):
            sections = _split_markdown(doc.text)
        elif file_type == "pdf":
            # PDF уже разбит по страницам через \n\n в loader
            pages = doc.text.split("\n\n")
            sections = [(f"page_{i+1}", p) for i, p in enumerate(pages) if p.strip()]
        else:
            sections = _split_markdown(doc.text)

        for idx, (section_name, section_text) in enumerate(sections):
            if not section_text.strip():
                continue

            chunk = Chunk(
                text=section_text,
                metadata={
                    **doc.metadata,
                    "chunk_id": f"{doc.metadata.get('filename', 'unknown')}_struct_{idx}",
                    "strategy": "structural",
                    "section": section_name,
                    "char_count": len(section_text),
                },
            )
            chunks.append(chunk)

    return chunks
