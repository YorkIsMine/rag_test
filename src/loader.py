"""Загрузка документов из папки: .txt, .md, .pdf, .py"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".py"}


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)


def _sanitize_text(text: str) -> str:
    """Убирает сурогатные символы, которые ломают UTF-8 сериализацию."""
    # Сначала убираем все сурогатные code-points (\uD800-\uDFFF)
    import re
    text = re.sub(r'[\ud800-\udfff]', '\ufffd', text)
    # Затем прогоняем через encode/decode для гарантии чистого UTF-8
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = _sanitize_text(text)
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_documents(docs_path: str) -> list[Document]:
    """Рекурсивно загружает документы из папки."""
    root = Path(docs_path)
    if not root.exists():
        raise FileNotFoundError(f"Папка не найдена: {docs_path}")

    documents: list[Document] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        try:
            if ext == ".pdf":
                text = load_pdf(path)
            else:
                text = load_text(path)
        except Exception as e:
            print(f"  [!] Ошибка при чтении {path}: {e}")
            continue

        if not text.strip():
            continue

        doc = Document(
            text=text,
            metadata={
                "source": str(path),
                "filename": path.name,
                "file_type": ext.lstrip("."),
                "rel_path": str(path.relative_to(root)),
            },
        )
        documents.append(doc)

    return documents
