"""
Тестирование RAG-пайплайна: проверка источников, цитат и режима "не знаю".

Запуск:
    python src/test_citations.py
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embeddings import get_client
from main import (
    CHAT_MODEL,
    DEFAULT_THRESHOLD,
    RAG_SYSTEM_PROMPT,
    RELEVANCE_FLOOR,
    _build_rag_context,
    _format_low_relevance_response,
    _has_rag_index,
)

# ─── 10 тестовых вопросов ────────────────────────────────────────────
# 8 вопросов по книге (ожидаем ответ с источниками),
# 2 нерелевантных (ожидаем "не знаю")
TEST_QUESTIONS = [
    # Релевантные вопросы по книге "Deadline" Тома ДеМарко
    {"q": "Кто является главным героем книги?", "expect_answer": True},
    {"q": "Какие основные принципы управления проектами описаны в книге?", "expect_answer": True},
    {"q": "Что такое дедлайн в контексте книги?", "expect_answer": True},
    {"q": "Какие команды формировались в проекте?", "expect_answer": True},
    {"q": "Какова роль менеджера проекта по мнению автора?", "expect_answer": True},
    {"q": "Какие ошибки в управлении проектами описаны в книге?", "expect_answer": True},
    {"q": "Что говорится о мотивации сотрудников?", "expect_answer": True},
    {"q": "Какие уроки можно извлечь из книги для управления IT-проектами?", "expect_answer": True},
    # Нерелевантные вопросы — ожидаем "не знаю"
    {"q": "Какой рецепт борща самый популярный в Украине?", "expect_answer": False},
    {"q": "Как работает квантовый компьютер IBM Q System One?", "expect_answer": False},
]


def check_has_section(text: str, header: str) -> bool:
    """Проверяет наличие секции **Header:** в тексте."""
    pattern = rf"\*\*{re.escape(header)}[:：]\*\*"
    return bool(re.search(pattern, text))


def check_has_citations(text: str) -> bool:
    """Проверяет наличие цитат в кавычках «»."""
    return "«" in text and "»" in text


def check_idk_response(text: str) -> bool:
    """Проверяет, что ответ содержит явный отказ (режим 'не знаю').

    Определяем IDK строго: секция Источники должна содержать 'нет релевантных'.
    Это отличает полный отказ от ответа с оговорками о неполноте.
    """
    text_lower = text.lower()
    # Строгий маркер: источники = нет релевантных
    return "**источники:** нет релевантных" in text_lower


def run_single_test(
    client,
    question: str,
    expect_answer: bool,
    top_k: int = 5,
) -> dict:
    """Запускает один тест и возвращает результаты проверки."""
    context, chunk_details, max_score = _build_rag_context(
        question, top_k=top_k, threshold=DEFAULT_THRESHOLD,
    )

    # Если контекст слабый — используем режим "не знаю"
    if max_score < RELEVANCE_FLOOR or not chunk_details:
        answer = _format_low_relevance_response()
        used_idk_mode = True
    else:
        user_msg = f"Контекст из базы знаний:\n\n{context}\n\n---\n\nВопрос: {question}"
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
        answer = response.choices[0].message.content
        used_idk_mode = False

    # ─── Проверки ───
    has_answer_section = check_has_section(answer, "Ответ")
    has_sources_section = check_has_section(answer, "Источники")
    has_citations_section = check_has_section(answer, "Цитаты")
    has_actual_citations = check_has_citations(answer)
    is_idk = check_idk_response(answer)

    # Проверка: если ожидаем ответ — должны быть все секции и цитаты
    # Если ожидаем "не знаю" — должен быть отказ
    if expect_answer:
        passed = (
            has_answer_section
            and has_sources_section
            and has_citations_section
            and has_actual_citations
            and not is_idk
        )
    else:
        passed = is_idk

    return {
        "question": question,
        "expect_answer": expect_answer,
        "max_score": max_score,
        "num_chunks": len(chunk_details),
        "used_idk_mode": used_idk_mode,
        "has_answer_section": has_answer_section,
        "has_sources_section": has_sources_section,
        "has_citations_section": has_citations_section,
        "has_actual_citations": has_actual_citations,
        "is_idk": is_idk,
        "passed": passed,
        "answer_preview": answer[:500],
    }


def main() -> None:
    if not _has_rag_index():
        print("Индексы не найдены. Сначала: python src/main.py index <docs_path>")
        sys.exit(1)

    client = get_client()
    results: list[dict] = []
    total = len(TEST_QUESTIONS)

    print(f"{'=' * 70}")
    print(f"ТЕСТИРОВАНИЕ RAG: источники, цитаты, режим 'не знаю'")
    print(f"Вопросов: {total}")
    print(f"Порог релевантности (RELEVANCE_FLOOR): {RELEVANCE_FLOOR}")
    print(f"Порог фильтрации (DEFAULT_THRESHOLD): {DEFAULT_THRESHOLD}")
    print(f"{'=' * 70}\n")

    for i, tq in enumerate(TEST_QUESTIONS, 1):
        q = tq["q"]
        expect = tq["expect_answer"]

        print(f"--- Тест {i}/{total} ---")
        print(f"  Вопрос: {q}")
        print(f"  Ожидаем: {'ответ с цитатами' if expect else 'режим «не знаю»'}")

        result = run_single_test(client, q, expect)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(f"  Score: {result['max_score']:.4f}, Чанков: {result['num_chunks']}")
        print(f"  Секции: Ответ={result['has_answer_section']}, "
              f"Источники={result['has_sources_section']}, "
              f"Цитаты={result['has_citations_section']}")
        print(f"  Цитаты в «»: {result['has_actual_citations']}")
        print(f"  Режим 'не знаю': {result['is_idk']}")
        print(f"  Результат: [{status}]")
        print()

    # ─── Итоги ───
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    print(f"{'=' * 70}")
    print(f"ИТОГИ: {passed}/{total} тестов пройдено")
    print(f"{'=' * 70}")

    # Детальная таблица
    print(f"\n{'#':<3} {'Статус':<8} {'Score':>7} {'Чанк':>5} "
          f"{'Отв':>4} {'Ист':>4} {'Цит':>4} {'«»':>4} {'IDK':>4} {'Вопрос'}")
    print("-" * 90)
    for i, r in enumerate(results, 1):
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{i:<3} {status:<8} {r['max_score']:>7.4f} {r['num_chunks']:>5} "
              f"{'да' if r['has_answer_section'] else 'нет':>4} "
              f"{'да' if r['has_sources_section'] else 'нет':>4} "
              f"{'да' if r['has_citations_section'] else 'нет':>4} "
              f"{'да' if r['has_actual_citations'] else 'нет':>4} "
              f"{'да' if r['is_idk'] else 'нет':>4} "
              f"{r['question'][:40]}")

    if failed > 0:
        print(f"\nПроваленные тесты:")
        for i, r in enumerate(results, 1):
            if not r["passed"]:
                print(f"\n  Тест {i}: {r['question']}")
                print(f"  Ответ (превью):\n    {r['answer_preview'][:300]}")

    # Сохраняем полные результаты
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nПолные результаты сохранены: {out_path}")


if __name__ == "__main__":
    main()
