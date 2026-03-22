"""
Реранкинг, фильтрация и query rewriting для RAG-пайплайна.

Три механизма улучшения поиска:
1. Threshold filter — отсечение по порогу cosine similarity
2. LLM reranker — переоценка релевантности через GPT
3. Query rewrite — переформулировка запроса для лучшего поиска
"""

from __future__ import annotations

import json

from openai import OpenAI

from embeddings import get_client

RERANK_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# 1. Threshold filter — простой порог по cosine similarity
# ---------------------------------------------------------------------------

def filter_by_threshold(
    results: list[dict],
    threshold: float = 0.3,
) -> list[dict]:
    """Отсекает результаты с score ниже порога."""
    return [r for r in results if r["score"] >= threshold]


# ---------------------------------------------------------------------------
# 2. LLM-based reranker — переоценка релевантности через модель
# ---------------------------------------------------------------------------

def rerank_with_llm(
    query: str,
    results: list[dict],
    top_k: int = 5,
    client: OpenAI | None = None,
) -> list[dict]:
    """
    Реранкинг результатов через LLM.

    Модель оценивает каждый чанк по шкале 0-10 на релевантность запросу.
    Результаты пересортировываются по новой оценке.
    """
    if not results:
        return []

    if client is None:
        client = get_client()

    # Формируем список чанков для оценки
    chunks_text = ""
    for i, r in enumerate(results):
        preview = r["text"][:500]
        chunks_text += f"\n[Chunk {i}]\n{preview}\n"

    prompt = (
        f"Запрос пользователя: {query}\n\n"
        f"Ниже приведены фрагменты документов. Оцени релевантность каждого фрагмента "
        f"запросу по шкале 0-10, где 10 — идеально релевантен, 0 — совсем не релевантен.\n"
        f"{chunks_text}\n\n"
        f"Верни JSON-массив объектов в формате: "
        f'[{{"chunk_index": 0, "relevance": 8, "reason": "краткое пояснение"}}, ...]\n'
        f"Только JSON, без пояснений."
    )

    response = client.chat.completions.create(
        model=RERANK_MODEL,
        messages=[
            {"role": "system", "content": "Ты — эксперт по оценке релевантности текстовых фрагментов."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()
    # Извлекаем JSON из ответа (может быть обёрнут в ```json ... ```)
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        # Если модель вернула невалидный JSON — возвращаем как есть
        return results[:top_k]

    # Присваиваем llm_score каждому результату
    scored = []
    for item in scores:
        idx = item.get("chunk_index", -1)
        relevance = item.get("relevance", 0)
        reason = item.get("reason", "")
        if 0 <= idx < len(results):
            r = dict(results[idx])
            r["llm_relevance"] = relevance
            r["rerank_reason"] = reason
            scored.append(r)

    # Сортируем по llm_relevance, при равенстве — по оригинальному score
    scored.sort(key=lambda r: (r["llm_relevance"], r["score"]), reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# 3. Query rewrite — переформулировка запроса
# ---------------------------------------------------------------------------

def rewrite_query(
    query: str,
    client: OpenAI | None = None,
) -> str:
    """
    Переформулирует запрос для улучшения поиска.

    Модель генерирует более точную и развёрнутую версию запроса,
    добавляя ключевые слова и уточнения.
    """
    if client is None:
        client = get_client()

    prompt = (
        f"Перефразируй следующий поисковый запрос, чтобы улучшить поиск по базе знаний. "
        f"Добавь синонимы и ключевые слова, сохрани смысл. "
        f"Верни ТОЛЬКО переформулированный запрос, без пояснений.\n\n"
        f"Оригинальный запрос: {query}"
    )

    response = client.chat.completions.create(
        model=RERANK_MODEL,
        messages=[
            {"role": "system", "content": "Ты — эксперт по поисковым запросам."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    rewritten = response.choices[0].message.content.strip()
    # Убираем кавычки, если модель обернула
    if rewritten.startswith('"') and rewritten.endswith('"'):
        rewritten = rewritten[1:-1]
    return rewritten


# ---------------------------------------------------------------------------
# 4. Полный пайплайн реранкинга
# ---------------------------------------------------------------------------

def rerank_pipeline(
    query: str,
    results: list[dict],
    *,
    threshold: float = 0.3,
    top_k: int = 5,
    use_reranker: bool = True,
    client: OpenAI | None = None,
) -> tuple[list[dict], dict]:
    """
    Полный пайплайн: threshold filter → LLM rerank.

    Возвращает (отфильтрованные результаты, статистику).
    """
    stats = {
        "before_filter": len(results),
        "threshold": threshold,
    }

    # Шаг 1: отсечение по порогу
    filtered = filter_by_threshold(results, threshold=threshold)
    stats["after_threshold"] = len(filtered)
    stats["removed_by_threshold"] = stats["before_filter"] - stats["after_threshold"]

    if not filtered:
        stats["after_rerank"] = 0
        return [], stats

    # Шаг 2: LLM rerank (если включён)
    if use_reranker:
        reranked = rerank_with_llm(query, filtered, top_k=top_k, client=client)
        stats["after_rerank"] = len(reranked)
        return reranked, stats

    stats["after_rerank"] = min(len(filtered), top_k)
    return filtered[:top_k], stats
