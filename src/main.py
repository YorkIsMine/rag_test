"""
RAG Pipeline — продвинутый механизм индексации документов.

Использование:
    python src/main.py index <docs_path> [--strategy fixed|structural|both]
    python src/main.py search <query> [--top_k 5]
    python src/main.py chat [--top_k 5]
    python src/main.py compare <docs_path>
"""

from __future__ import annotations

import argparse
import sys
import os

# Добавляем src в path для импортов
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loader import load_documents
from chunking import chunk_fixed, chunk_structural
from embeddings import get_embeddings, get_client
from index_store import save_index, load_index, search
from compare import compare_strategies, print_comparison

CHAT_MODEL = "gpt-4o"



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
    """Поиск по обоим индексам, результаты объединяются и ранжируются по score."""
    top_results = _retrieve(args.query, top_k=args.top_k)

    if not top_results:
        print("Индексы не найдены. Сначала запустите: python src/main.py index <docs_path>")
        return

    print(f"Запрос: {args.query}\n")
    for i, r in enumerate(top_results, 1):
        score = r.pop("score")
        text = r.pop("text")
        print(f"--- Результат {i} (score: {score:.4f}) ---")
        print(f"  Источник: {r.get('filename', '?')}")
        print(f"  Секция:   {r.get('section', '?')}")
        print(f"  Стратегия: {r.get('strategy', '?')}")
        print(f"  Chunk ID: {r.get('chunk_id', '?')}")
        preview = text[:300].replace("\n", " ")
        print(f"  Текст:    {preview}{'...' if len(text) > 300 else ''}")
        print()


def _retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Поиск по обоим индексам, возвращает top_k результатов."""
    query_emb = get_embeddings([query])
    all_results: list[dict] = []

    for idx_dir in ["rag_index_fixed", "rag_index_structural"]:
        try:
            index, metadata = load_index(idx_dir)
        except (FileNotFoundError, RuntimeError):
            continue
        results = search(index, metadata, query_emb[0], top_k=top_k)
        all_results.extend(results)

    all_results.sort(key=lambda r: r["score"], reverse=True)
    return all_results[:top_k]


def cmd_chat(args: argparse.Namespace) -> None:
    """Интерактивный диалог с RAG-контекстом."""
    # Проверяем наличие индексов
    has_index = False
    for idx_dir in ["rag_index_fixed", "rag_index_structural"]:
        try:
            load_index(idx_dir)
            has_index = True
        except (FileNotFoundError, RuntimeError):
            pass
    if not has_index:
        print("Индексы не найдены. Сначала: python src/main.py index <docs_path>")
        return

    client = get_client()
    history: list[dict] = []

    print("RAG-чат запущен. Введите вопрос (или 'выход' / 'exit' для завершения).")
    print(f"Контекст: top_{args.top_k} чанков из индекса\n")

    while True:
        try:
            question = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break

        if not question:
            continue
        if question.lower() in ("выход", "exit", "quit", "q"):
            print("До свидания!")
            break

        # Поиск релевантных чанков
        results = _retrieve(question, top_k=args.top_k)

        if results:
            context_parts = []
            sources = []
            for r in results:
                text = r["text"]
                source = r.get("filename", "?")
                section = r.get("section", "")
                context_parts.append(text)
                sources.append(f"{source} ({section})")

            context = "\n\n---\n\n".join(context_parts)
        else:
            context = "(контекст не найден)"
            sources = []

        system_msg = (
            "Ты — помощник, отвечающий на вопросы на основе предоставленного контекста из базы знаний. "
            "Отвечай точно и по существу. Если в контексте нет информации для ответа — честно скажи об этом. "
            "Отвечай на том же языке, на котором задан вопрос."
        )

        user_msg = f"Контекст из базы знаний:\n\n{context}\n\n---\n\nВопрос: {question}"

        # Формируем сообщения: system + история + текущий вопрос с контекстом
        messages = [{"role": "system", "content": system_msg}]
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

        print(f"\nБот: {answer}")
        if sources:
            print(f"\n  Источники: {', '.join(dict.fromkeys(sources))}")
        print()


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
    p_search.add_argument("--top_k", type=int, default=5, help="Кол-во результатов")

    # chat
    p_chat = sub.add_parser("chat", help="Диалог с ботом + RAG-контекст")
    p_chat.add_argument("--top_k", type=int, default=5, help="Кол-во чанков контекста")

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
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
