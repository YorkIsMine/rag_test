"""FAISS-индекс + JSON-метаданные."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from chunking import Chunk


def save_index(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    index_dir: str,
) -> None:
    """Сохраняет FAISS-индекс и метаданные на диск."""
    path = Path(index_dir)
    path.mkdir(parents=True, exist_ok=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(path / "index.faiss"))

    meta = []
    for chunk in chunks:
        meta.append({
            "text": chunk.text,
            **chunk.metadata,
        })

    with open(path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  Сохранено: {len(chunks)} чанков -> {index_dir}")


def load_index(index_dir: str) -> tuple[faiss.Index, list[dict]]:
    """Загружает FAISS-индекс и метаданные."""
    path = Path(index_dir)

    index = faiss.read_index(str(path / "index.faiss"))

    with open(path / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


def search(
    index: faiss.Index,
    metadata: list[dict],
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> list[dict]:
    """Поиск top_k ближайших чанков. Возвращает метаданные с score."""
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        result = {**metadata[idx], "score": float(score)}
        results.append(result)

    return results
