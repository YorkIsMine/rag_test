"""
RAG Pipeline — продвинутый механизм индексации документов.

Использование:
    python src/main.py index <docs_path> [--strategy fixed|structural|both]
    python src/main.py search <query> [--top_k 5] [--rerank] [--rewrite] [--threshold 0.3]
    python src/main.py chat [--top_k 5]
    python src/main.py smart_chat [--top_k 5]    # RAG + память задачи
    python src/main.py agent
    python src/main.py dual <query> [--top_k 5]
    python src/main.py compare <docs_path>
    python src/main.py compare_modes <query> [--top_k 5] [--threshold 0.3]
"""

from __future__ import annotations

import argparse
import sys
import os

# Добавляем src в path для импортов
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re

from loader import load_documents
from chunking import chunk_fixed, chunk_structural
from embeddings import get_embeddings, get_client
from index_store import save_index, load_index, search
from compare import compare_strategies, print_comparison
from reranker import rerank_pipeline, rewrite_query, filter_by_threshold
from smart_chat import run_smart_chat


def _clean_input(text: str) -> str:
    """Убирает сурогатные символы из пользовательского ввода (проблема локали терминала)."""
    # Убираем сурогатные code-points U+D800..U+DFFF
    text = re.sub(r'[\ud800-\udfff]', '', text)
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")

CHAT_MODEL = "gpt-4o"
RAG_INDEX_DIRS = ("rag_index_fixed", "rag_index_structural")

# Настройки по умолчанию
DEFAULT_THRESHOLD = 0.3
DEFAULT_FETCH_K = 20  # сколько достаём из индекса ДО фильтрации
RELEVANCE_FLOOR = 0.35  # ниже этого порога — модель обязана сказать "не знаю"


RAG_SYSTEM_PROMPT = """\
Ты — помощник, отвечающий на вопросы СТРОГО на основе предоставленного контекста из базы знаний.

ОБЯЗАТЕЛЬНЫЙ ФОРМАТ ОТВЕТА:

**Ответ:**
<твой ответ на вопрос, основанный только на контексте>

**Источники:**
- <source> | <section> | <chunk_id>
- ...

**Цитаты:**
- «<точная цитата из контекста, подтверждающая ответ>» — <source>, <section>
- ...

ПРАВИЛА:
1. Каждый ответ ОБЯЗАН содержать все три секции: Ответ, Источники и Цитаты.
2. Цитаты — это ДОСЛОВНЫЕ фрагменты из предоставленного контекста (в кавычках «»).
3. Каждая цитата должна подтверждать часть ответа.
4. Если контекст предоставлен, ты ОБЯЗАН попытаться ответить на его основе. Даже если информация неполная — дай ответ на основе того, что есть, и укажи что информация может быть неполной.
5. Говори «не знаю» ТОЛЬКО если предоставленный контекст АБСОЛЮТНО не связан с вопросом. В этом случае ответь:
   **Ответ:** К сожалению, в доступной базе знаний недостаточно информации для ответа на этот вопрос. Пожалуйста, уточните запрос или переформулируйте вопрос.
   **Источники:** нет релевантных
   **Цитаты:** нет релевантных
6. Отвечай на том же языке, на котором задан вопрос.
"""



def cmd_index(args: argparse.Namespace) -> None:
    """Индексация документов."""
    print(f"Загрузка документов из: {args.docs_path}")
    documents = load_documents(args.docs_path)
    print(f"  Загружено документов: {len(documents)}")

    if not documents:
        print("Нет документов для индексации.")
        return

    strategies = (
        ["fixed", "structural"] if args.strategy == "both"
        else [args.strategy]
    )

    for strategy in strategies:
        print(f"\nСтратегия: {strategy}")

        if strategy == "fixed":
            chunks = chunk_fixed(documents)
        else:
            chunks = chunk_structural(documents)

        print(f"  Чанков: {len(chunks)}")

        texts = [c.text for c in chunks]
        print(f"  Генерация эмбеддингов...")
        embeddings = get_embeddings(texts)
        print(f"  Эмбеддинги: {embeddings.shape}")

        index_dir = f"rag_index_{strategy}"
        save_index(chunks, embeddings, index_dir)

    print("\nИндексация завершена.")
    print(f"Для поиска: python src/main.py search \"запрос\" --index_dir rag_index_{strategies[0]}")


def cmd_search(args: argparse.Namespace) -> None:
    """Поиск по обоим индексам с опциональным реранкингом и фильтрацией."""
    use_enhanced = args.rerank or args.rewrite or args.threshold > 0

    if use_enhanced:
        print(f"Запрос: {args.query}")
        print(f"Режим: enhanced (rewrite={args.rewrite}, rerank={args.rerank}, "
              f"threshold={args.threshold}, fetch_k={args.fetch_k})\n")
        top_results, stats = _retrieve_enhanced(
            args.query,
            top_k=args.top_k,
            use_rewrite=args.rewrite,
            use_reranker=args.rerank,
            threshold=args.threshold,
            fetch_k=args.fetch_k,
            verbose=True,
        )
        print()
    else:
        top_results = _retrieve(args.query, top_k=args.top_k)
        stats = None
        print(f"Запрос: {args.query}\n")

    if not top_results:
        print("Результатов не найдено.")
        return

    for i, r in enumerate(top_results, 1):
        score = r.get("score", 0)
        text = r.get("text", "")
        llm_rel = r.get("llm_relevance")
        reason = r.get("rerank_reason", "")

        score_str = f"score: {score:.4f}"
        if llm_rel is not None:
            score_str += f", relevance: {llm_rel}/10"

        print(f"--- Результат {i} ({score_str}) ---")
        print(f"  Источник: {r.get('filename', '?')}")
        print(f"  Секция:   {r.get('section', '?')}")
        print(f"  Стратегия: {r.get('strategy', '?')}")
        if reason:
            print(f"  Причина:  {reason}")
        preview = text[:300].replace("\n", " ")
        print(f"  Текст:    {preview}{'...' if len(text) > 300 else ''}")
        print()


def _retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Базовый поиск по обоим индексам, возвращает top_k результатов (без фильтрации)."""
    query_emb = get_embeddings([query])
    all_results: list[dict] = []

    for idx_dir in RAG_INDEX_DIRS:
        try:
            index, metadata = load_index(idx_dir)
        except (FileNotFoundError, RuntimeError):
            continue
        results = search(index, metadata, query_emb[0], top_k=top_k)
        all_results.extend(results)

    all_results.sort(key=lambda r: r["score"], reverse=True)
    return all_results[:top_k]


def _retrieve_enhanced(
    query: str,
    top_k: int = 5,
    *,
    use_rewrite: bool = False,
    use_reranker: bool = True,
    threshold: float = DEFAULT_THRESHOLD,
    fetch_k: int = DEFAULT_FETCH_K,
    verbose: bool = False,
) -> tuple[list[dict], dict]:
    """
    Улучшенный поиск: query rewrite → расширенная выборка → threshold → LLM rerank.

    Возвращает (результаты, статистика).
    """
    stats: dict = {"original_query": query}

    # Шаг 0: Query rewrite
    if use_rewrite:
        rewritten = rewrite_query(query)
        stats["rewritten_query"] = rewritten
        if verbose:
            print(f"  Rewrite: {query!r} → {rewritten!r}")
        search_query = rewritten
    else:
        search_query = query

    # Шаг 1: Широкая выборка (fetch_k > top_k)
    raw_results = _retrieve(search_query, top_k=fetch_k)
    stats["fetched"] = len(raw_results)

    if verbose and raw_results:
        scores = [r["score"] for r in raw_results]
        print(f"  Fetch: {len(raw_results)} результатов, "
              f"score: {max(scores):.4f} .. {min(scores):.4f}")

    # Шаг 2: Threshold + Rerank
    final, pipeline_stats = rerank_pipeline(
        query,  # для реранкинга используем оригинальный запрос
        raw_results,
        threshold=threshold,
        top_k=top_k,
        use_reranker=use_reranker,
    )
    stats.update(pipeline_stats)

    if verbose:
        print(f"  Threshold ({threshold}): {pipeline_stats['before_filter']} → "
              f"{pipeline_stats['after_threshold']} "
              f"(отсечено: {pipeline_stats['removed_by_threshold']})")
        if use_reranker:
            print(f"  Rerank: → {pipeline_stats['after_rerank']} результатов")

    return final, stats


def _has_rag_index() -> bool:
    for idx_dir in RAG_INDEX_DIRS:
        try:
            load_index(idx_dir)
            return True
        except (FileNotFoundError, RuntimeError):
            continue
    return False


def _build_rag_context(
    query: str,
    top_k: int,
    *,
    use_rewrite: bool = False,
    use_reranker: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[str, list[dict], float]:
    """
    Возвращает (context_text, chunk_details, max_score).

    chunk_details — список словарей с полями:
        source, section, chunk_id, score, snippet (первые 200 символов текста).
    max_score — максимальный score среди найденных чанков (0.0 если пусто).
    """
    use_enhanced = use_rewrite or use_reranker or threshold > 0

    if use_enhanced:
        results, _ = _retrieve_enhanced(
            query, top_k=top_k,
            use_rewrite=use_rewrite,
            use_reranker=use_reranker,
            threshold=threshold,
        )
    else:
        results = _retrieve(query, top_k=top_k)

    if not results:
        return "(контекст не найден)", [], 0.0

    max_score = max(r.get("score", 0) for r in results)

    context_parts: list[str] = []
    chunk_details: list[dict] = []
    for i, r in enumerate(results, 1):
        source = r.get("filename", "?")
        section = r.get("section", "")
        chunk_id = r.get("chunk_id", f"chunk_{i}")
        score = r.get("score", 0)
        text = r["text"]

        # Контекст для модели — с метаданными, чтобы модель могла ссылаться
        context_parts.append(
            f"[Источник: {source} | Секция: {section} | ID: {chunk_id}]\n{text}"
        )
        chunk_details.append({
            "source": source,
            "section": section,
            "chunk_id": chunk_id,
            "score": score,
            "snippet": text[:200].strip(),
        })

    return "\n\n---\n\n".join(context_parts), chunk_details, max_score


def _format_low_relevance_response() -> str:
    """Ответ при низкой релевантности контекста."""
    return (
        "**Ответ:** К сожалению, в доступной базе знаний недостаточно информации "
        "для ответа на этот вопрос. Пожалуйста, уточните запрос или "
        "переформулируйте вопрос.\n\n"
        "**Источники:** нет релевантных\n\n"
        "**Цитаты:** нет релевантных"
    )


def cmd_chat(args: argparse.Namespace) -> None:
    """Интерактивный диалог с RAG-контекстом."""
    if not _has_rag_index():
        print("Индексы не найдены. Сначала: python src/main.py index <docs_path>")
        return

    client = get_client()
    history: list[dict] = []

    print("RAG-чат запущен. Введите вопрос (или 'выход' / 'exit' для завершения).")
    print(f"Контекст: top_{args.top_k} чанков из индекса\n")

    while True:
        try:
            question = _clean_input(input("Вы: ").strip())
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break

        if not question:
            continue
        if question.lower() in ("выход", "exit", "quit", "q"):
            print("До свидания!")
            break

        context, chunk_details, max_score = _build_rag_context(
            question, top_k=args.top_k, threshold=DEFAULT_THRESHOLD,
        )

        # Режим "не знаю": если лучший результат ниже порога релевантности
        if max_score < RELEVANCE_FLOOR or not chunk_details:
            answer = _format_low_relevance_response()
            print(f"\nБот (релевантность {max_score:.4f} < {RELEVANCE_FLOOR}):\n{answer}\n")
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            continue

        user_msg = f"Контекст из базы знаний:\n\n{context}\n\n---\n\nВопрос: {question}"

        messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_msg})

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.3,
        )

        answer = response.choices[0].message.content

        # Сохраняем в историю (без контекста, чтобы не раздувать)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        # Обрезаем историю до последних 10 пар
        if len(history) > 20:
            history = history[-20:]

        print(f"\nБот:\n{answer}")
        # Показываем детали чанков для прозрачности
        print(f"\n  [Найдено чанков: {len(chunk_details)}, "
              f"макс. score: {max_score:.4f}]")
        print()


def cmd_dual(args: argparse.Namespace) -> None:
    """Один запрос -> два ответа: с RAG и без RAG."""
    client = get_client()
    question = _clean_input(args.query)

    print(f"Запрос: {question}\n")

    if _has_rag_index():
        context, chunk_details, max_score = _build_rag_context(
            question, top_k=args.top_k, threshold=DEFAULT_THRESHOLD,
        )

        if max_score < RELEVANCE_FLOOR or not chunk_details:
            rag_answer = _format_low_relevance_response()
        else:
            rag_user_msg = f"Контекст из базы знаний:\n\n{context}\n\n---\n\nВопрос: {question}"
            rag_response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": rag_user_msg},
                ],
                temperature=0.3,
            )
            rag_answer = rag_response.choices[0].message.content
    else:
        rag_answer = (
            "RAG-ответ недоступен: индексы не найдены. "
            "Сначала запустите: python src/main.py index <docs_path>"
        )
        chunk_details = []
        max_score = 0.0

    plain_system_msg = (
        "Ты — помощник. Отвечай точно и по существу. "
        "Если данных недостаточно — честно сообщи об этом. "
        "Отвечай на том же языке, на котором задан вопрос."
    )
    plain_response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": plain_system_msg},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )
    plain_answer = plain_response.choices[0].message.content

    print("=== Ответ с RAG ===")
    print(rag_answer)
    if chunk_details:
        print(f"\n  [Чанков: {len(chunk_details)}, макс. score: {max_score:.4f}]")
    print("\n=== Ответ без RAG ===")
    print(plain_answer)


def cmd_agent(args: argparse.Namespace) -> None:
    """Интерактивный диалог с агентом без RAG-контекста."""
    client = get_client()
    history: list[dict] = []

    print("Агент без RAG запущен. Введите вопрос (или 'выход' / 'exit' для завершения).\n")

    while True:
        try:
            question = _clean_input(input("Вы: ").strip())
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break

        if not question:
            continue
        if question.lower() in ("выход", "exit", "quit", "q"):
            print("До свидания!")
            break

        system_msg = (
            "Ты — помощник. Отвечай точно и по существу. "
            "Если данных недостаточно — честно сообщи об этом. "
            "Отвечай на том же языке, на котором задан вопрос."
        )

        messages = [{"role": "system", "content": system_msg}]
        messages.extend(history)
        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.3,
        )

        answer = response.choices[0].message.content

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        if len(history) > 20:
            history = history[-20:]

        print(f"\nБот: {answer}\n")


def cmd_compare_modes(args: argparse.Namespace) -> None:
    """Сравнение режимов поиска: baseline vs enhanced (rewrite + rerank + threshold)."""
    if not _has_rag_index():
        print("Индексы не найдены. Сначала: python src/main.py index <docs_path>")
        return

    query = args.query
    top_k = args.top_k
    threshold = args.threshold

    print(f"{'=' * 70}")
    print(f"СРАВНЕНИЕ РЕЖИМОВ ПОИСКА")
    print(f"Запрос: {query}")
    print(f"top_k={top_k}, threshold={threshold}")
    print(f"{'=' * 70}\n")

    # Режим 1: Baseline (без фильтрации)
    print(">>> РЕЖИМ 1: Baseline (без фильтрации и реранкинга)")
    print("-" * 50)
    baseline = _retrieve(query, top_k=top_k)
    _print_results_compact(baseline, label="baseline")

    # Режим 2: Только threshold
    print("\n>>> РЕЖИМ 2: Threshold filter (порог отсечения)")
    print("-" * 50)
    raw = _retrieve(query, top_k=DEFAULT_FETCH_K)
    filtered = filter_by_threshold(raw, threshold=threshold)
    filtered = filtered[:top_k]
    print(f"  Из {len(raw)} результатов после threshold={threshold}: {len(filtered)}")
    _print_results_compact(filtered, label="threshold")

    # Режим 3: Threshold + Rerank
    print("\n>>> РЕЖИМ 3: Threshold + LLM Rerank")
    print("-" * 50)
    reranked, stats = _retrieve_enhanced(
        query, top_k=top_k,
        use_rewrite=False, use_reranker=True,
        threshold=threshold, verbose=True,
    )
    _print_results_compact(reranked, label="rerank", show_relevance=True)

    # Режим 4: Query Rewrite + Threshold + Rerank
    print("\n>>> РЕЖИМ 4: Query Rewrite + Threshold + LLM Rerank")
    print("-" * 50)
    full, full_stats = _retrieve_enhanced(
        query, top_k=top_k,
        use_rewrite=True, use_reranker=True,
        threshold=threshold, verbose=True,
    )
    _print_results_compact(full, label="full", show_relevance=True)

    # Сводная таблица
    print(f"\n{'=' * 70}")
    print("СВОДКА")
    print(f"{'=' * 70}")
    modes = [
        ("Baseline", baseline),
        ("Threshold", filtered),
        ("Threshold+Rerank", reranked),
        ("Rewrite+Threshold+Rerank", full),
    ]
    print(f"{'Режим':<30} {'Кол-во':>8} {'Avg score':>10} {'Min score':>10} {'Max score':>10}")
    print("-" * 70)
    for name, results in modes:
        if results:
            scores = [r["score"] for r in results]
            avg_s = sum(scores) / len(scores)
            print(f"{name:<30} {len(results):>8} {avg_s:>10.4f} {min(scores):>10.4f} {max(scores):>10.4f}")
        else:
            print(f"{name:<30} {'0':>8} {'—':>10} {'—':>10} {'—':>10}")


def _print_results_compact(
    results: list[dict],
    label: str = "",
    show_relevance: bool = False,
) -> None:
    """Компактный вывод результатов для сравнения."""
    if not results:
        print("  (нет результатов)")
        return

    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        llm_rel = r.get("llm_relevance")
        section = r.get("section", "?")
        strategy = r.get("strategy", "?")
        text_preview = r.get("text", "")[:100].replace("\n", " ")

        line = f"  {i}. [{score:.4f}]"
        if show_relevance and llm_rel is not None:
            line += f" [rel:{llm_rel}/10]"
        line += f" [{strategy}] {section}: {text_preview}..."
        print(line)


def cmd_compare(args: argparse.Namespace) -> None:
    """Сравнение стратегий чанкинга."""
    print(f"Загрузка документов из: {args.docs_path}")
    documents = load_documents(args.docs_path)
    print(f"  Загружено документов: {len(documents)}")

    if not documents:
        print("Нет документов для сравнения.")
        return

    result = compare_strategies(documents)
    print_comparison(result)

    # Опционально: индексируем оба
    if args.with_index:
        for strategy in ("fixed", "structural"):
            key = f"{strategy}_chunks"
            chunks = result[key]
            texts = [c.text for c in chunks]
            print(f"Генерация эмбеддингов ({strategy})...")
            embeddings = get_embeddings(texts)
            save_index(chunks, embeddings, f"rag_index_{strategy}")
        print("\nОба индекса сохранены.")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = sub.add_parser("index", help="Индексация документов")
    p_index.add_argument("docs_path", help="Путь к папке с документами")
    p_index.add_argument(
        "--strategy",
        choices=["fixed", "structural", "both"],
        default="both",
        help="Стратегия чанкинга (default: both)",
    )

    # search
    p_search = sub.add_parser("search", help="Поиск по обоим индексам")
    p_search.add_argument("query", help="Поисковый запрос")
    p_search.add_argument("--top_k", type=int, default=5, help="Кол-во результатов (после фильтрации)")
    p_search.add_argument("--fetch_k", type=int, default=DEFAULT_FETCH_K,
                          help="Кол-во результатов ДО фильтрации (default: 20)")
    p_search.add_argument("--threshold", type=float, default=0.0,
                          help="Порог отсечения по score (default: 0 — выключен)")
    p_search.add_argument("--rerank", action="store_true",
                          help="Включить LLM-реранкинг")
    p_search.add_argument("--rewrite", action="store_true",
                          help="Включить query rewriting")

    # chat
    p_chat = sub.add_parser("chat", help="Диалог с ботом + RAG-контекст")
    p_chat.add_argument("--top_k", type=int, default=5, help="Кол-во чанков контекста")

    # smart_chat (RAG + task state)
    p_smart = sub.add_parser("smart_chat", help="RAG-чат с памятью задачи и источниками")
    p_smart.add_argument("--top_k", type=int, default=5, help="Кол-во чанков контекста")
    p_smart.add_argument("--fast", action="store_true",
                         help="Быстрый режим (gpt-4o-mini, без LLM-реранкинга)")

    # agent (без RAG)
    sub.add_parser("agent", help="Диалог с агентом без RAG-контекста")

    # dual
    p_dual = sub.add_parser("dual", help="Один запрос -> два ответа: с RAG и без RAG")
    p_dual.add_argument("query", help="Запрос к модели")
    p_dual.add_argument("--top_k", type=int, default=5, help="Кол-во чанков контекста для RAG")

    # compare_modes
    p_cm = sub.add_parser("compare_modes", help="Сравнение режимов: baseline vs rerank vs rewrite")
    p_cm.add_argument("query", help="Поисковый запрос для сравнения")
    p_cm.add_argument("--top_k", type=int, default=5, help="Кол-во результатов")
    p_cm.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                      help=f"Порог отсечения (default: {DEFAULT_THRESHOLD})")

    # compare
    p_compare = sub.add_parser("compare", help="Сравнение стратегий чанкинга")
    p_compare.add_argument("docs_path", help="Путь к папке с документами")
    p_compare.add_argument(
        "--with-index",
        action="store_true",
        help="Также создать индексы для обеих стратегий",
    )

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "smart_chat":
        run_smart_chat(top_k=args.top_k, fast=args.fast)
    elif args.command == "agent":
        cmd_agent(args)
    elif args.command == "dual":
        cmd_dual(args)
    elif args.command == "compare_modes":
        cmd_compare_modes(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
