"""Генерация эмбеддингов через OpenAI API."""

from __future__ import annotations

import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 2048


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY не задан. Укажите его в .env")
    return OpenAI(api_key=api_key)


def _sanitize_for_api(text: str) -> str:
    """Убирает сурогатные символы перед отправкой в API."""
    import re
    text = re.sub(r'[\ud800-\udfff]', '', text)
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def get_embeddings(texts: list[str]) -> np.ndarray:
    """Получить эмбеддинги для списка текстов. Возвращает нормализованный numpy array."""
    client = get_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = [_sanitize_for_api(t) for t in texts[i : i + BATCH_SIZE]]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend(item.embedding for item in response.data)

    arr = np.array(all_embeddings, dtype=np.float32)
    # L2-нормализация для cosine similarity через inner product
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    arr /= norms
    return arr
