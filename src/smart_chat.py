"""
Мини-чат с RAG + источниками + памятью задачи (task state).

Хранит:
  - историю диалога (conversation history)
  - память задачи: цель диалога, уточнения пользователя, зафиксированные термины/ограничения
  - при каждом вопросе ищет контекст через RAG
  - всегда выводит источники

Использование:
    python src/main.py smart_chat [--top_k 5]

Или напрямую:
    python src/smart_chat.py [--top_k 5]
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import re
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embeddings import get_client
from index_store import load_index, search as faiss_search
from embeddings import get_embeddings
from reranker import rerank_pipeline

# ── Константы ──────────────────────────────────────────────────────

CHAT_MODEL = "gpt-4o"
CHAT_MODEL_FAST = "gpt-4o-mini"
RAG_INDEX_DIRS = ("rag_index_fixed", "rag_index_structural")
DEFAULT_THRESHOLD = 0.3
DEFAULT_FETCH_K = 20
RELEVANCE_FLOOR = 0.35
MAX_HISTORY_PAIRS = 15  # храним до 15 пар user/assistant


# ── Task State (память задачи) ─────────────────────────────────────

class TaskState:
    """
    Память задачи — фиксирует:
      - goal: цель диалога (зачем пользователь пришёл)
      - clarifications: что пользователь уточнил по ходу
      - constraints: ограничения и термины, зафиксированные в ходе разговора
      - key_facts: ключевые факты, которые уже были найдены
    """

    def __init__(self):
        self.goal: str = ""
        self.clarifications: list[str] = []
        self.constraints: list[str] = []
        self.key_facts: list[str] = []

    def to_prompt_block(self) -> str:
        """Форматирует task state для включения в system prompt."""
        parts = []
        if self.goal:
            parts.append(f"ЦЕЛЬ ДИАЛОГА: {self.goal}")
        if self.clarifications:
            items = "\n".join(f"  - {c}" for c in self.clarifications)
            parts.append(f"УТОЧНЕНИЯ ПОЛЬЗОВАТЕЛЯ:\n{items}")
        if self.constraints:
            items = "\n".join(f"  - {c}" for c in self.constraints)
            parts.append(f"ЗАФИКСИРОВАННЫЕ ОГРАНИЧЕНИЯ/ТЕРМИНЫ:\n{items}")
        if self.key_facts:
            items = "\n".join(f"  - {f}" for f in self.key_facts[-10:])
            parts.append(f"КЛЮЧЕВЫЕ ФАКТЫ (найденные ранее):\n{items}")
        return "\n\n".join(parts) if parts else "(Задача ещё не определена)"

    def is_empty(self) -> bool:
        return not self.goal and not self.clarifications and not self.constraints

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "clarifications": self.clarifications,
            "constraints": self.constraints,
            "key_facts": self.key_facts,
        }


# ── System prompt ──────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
Ты — интеллектуальный ассистент с доступом к базе знаний (RAG).
Ты ведёшь диалог с пользователем, помогая ему разобраться в теме.

ПАМЯТЬ ЗАДАЧИ (текущее состояние):
{task_state}

ПРАВИЛА ОТВЕТА:
1. Отвечай на основе предоставленного контекста из базы знаний.
2. Каждый ответ ОБЯЗАН содержать секции: **Ответ:** и **Источники:**
3. Если есть подтверждающие цитаты — добавь секцию **Цитаты:** с дословными фрагментами в «».
4. Если контекст неполный — дай ответ на основе того, что есть, и укажи это.
5. Если контекст АБСОЛЮТНО не связан с вопросом, ответь:
   **Ответ:** В базе знаний нет информации по этому вопросу.
   **Источники:** нет релевантных
6. Отвечай на том же языке, на котором задан вопрос.
7. Учитывай ПАМЯТЬ ЗАДАЧИ — не забывай цель диалога и ранее обсуждённое.

ВАЖНО — ОБЯЗАТЕЛЬНО определи цель диалога:
- Если цель ещё не задана (пусто) — ОБЯЗАТЕЛЬНО заполни поле "goal" уже в первом ответе.
  Цель — это то, что пользователь хочет узнать/понять. Сформулируй кратко (1 предложение).
- Если цель уже задана — обнови её только если пользователь явно сменил тему.

ДОПОЛНИТЕЛЬНО — после ответа верни JSON-блок для обновления памяти задачи:
```task_state_update
{{"action": "update", "goal": "<ОБЯЗАТЕЛЬНО заполни при первом вопросе — краткая цель диалога>", "new_clarification": "<новое уточнение или пустая строка>", "new_constraint": "<новое ограничение/термин или пустая строка>", "new_fact": "<ключевой факт из ответа или пустая строка>"}}
```

Этот блок ДОЛЖЕН быть в конце ответа. Если ничего не изменилось, оставь поля пустыми строками.
Поле "goal" ОБЯЗАТЕЛЬНО заполни, если память задачи пуста!
"""


# ── Retrieval ──────────────────────────────────────────────────────

def _retrieve_from_indices(query: str, top_k: int = 5) -> list[dict]:
    """Поиск по обоим FAISS-индексам."""
    query_emb = get_embeddings([query])
    all_results: list[dict] = []
    for idx_dir in RAG_INDEX_DIRS:
        try:
            index, metadata = load_index(idx_dir)
        except (FileNotFoundError, RuntimeError):
            continue
        results = faiss_search(index, metadata, query_emb[0], top_k=top_k)
        all_results.extend(results)
    all_results.sort(key=lambda r: r["score"], reverse=True)
    return all_results[:top_k]


def _retrieve_enhanced(
    query: str, top_k: int = 5, threshold: float = DEFAULT_THRESHOLD,
    *, use_reranker: bool = True,
) -> list[dict]:
    """Расширенный поиск: широкая выборка → threshold → rerank."""
    raw = _retrieve_from_indices(query, top_k=DEFAULT_FETCH_K)
    if not raw:
        return []
    final, _ = rerank_pipeline(query, raw, threshold=threshold, top_k=top_k, use_reranker=use_reranker)
    return final


def build_rag_context(query: str, top_k: int = 5, *, fast: bool = False) -> tuple[str, list[dict], float]:
    """
    Возвращает (context_text, chunk_details, max_score).
    """
    results = _retrieve_enhanced(query, top_k=top_k, use_reranker=not fast)
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

        context_parts.append(
            f"[Источник: {source} | Секция: {section} | ID: {chunk_id}]\n{text}"
        )
        chunk_details.append({
            "source": source,
            "section": section,
            "chunk_id": chunk_id,
            "score": round(score, 4),
        })

    return "\n\n---\n\n".join(context_parts), chunk_details, max_score


# ── Task state extraction ──────────────────────────────────────────

def _extract_task_update(text: str) -> tuple[str, dict | None]:
    """
    Извлекает JSON-блок task_state_update из ответа модели.
    Возвращает (очищенный текст, словарь обновления или None).
    """
    pattern = r"```task_state_update\s*\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return text, None
    json_str = match.group(1).strip()
    clean_text = text[:match.start()].rstrip()
    try:
        update = json.loads(json_str)
        return clean_text, update
    except json.JSONDecodeError:
        return clean_text, None


def apply_task_update(state: TaskState, update: dict) -> None:
    """Применяет обновление к task state."""
    if not update:
        return
    goal = update.get("goal", "")
    if goal:
        state.goal = goal
    clarification = update.get("new_clarification", "")
    if clarification:
        state.clarifications.append(clarification)
    constraint = update.get("new_constraint", "")
    if constraint:
        state.constraints.append(constraint)
    fact = update.get("new_fact", "")
    if fact:
        state.key_facts.append(fact)


# ── Утилиты ────────────────────────────────────────────────────────

def _infer_goal(first_question: str) -> str:
    """Формирует цель диалога из первого вопроса пользователя (фоллбэк)."""
    q = first_question.strip().rstrip("?").rstrip(".")
    if len(q) > 80:
        q = q[:77] + "..."
    return f"Узнать: {q}"


def _clean_input(text: str) -> str:
    text = re.sub(r'[\ud800-\udfff]', '', text)
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def _has_rag_index() -> bool:
    for idx_dir in RAG_INDEX_DIRS:
        try:
            load_index(idx_dir)
            return True
        except (FileNotFoundError, RuntimeError):
            continue
    return False


# ── Основной цикл чата ────────────────────────────────────────────

def run_smart_chat(
    top_k: int = 5,
    *,
    fast: bool = False,
    input_stream: io.TextIOBase | None = None,
    output_stream: io.TextIOBase | None = None,
) -> list[dict]:
    """
    Запускает интерактивный чат с RAG + памятью задачи.

    Args:
        top_k: кол-во чанков контекста
        fast: быстрый режим (gpt-4o-mini, без LLM-реранкинга) — для тестов
        input_stream: если задан, читает вопросы из потока (для тестирования)
        output_stream: если задан, пишет ответы в поток (для тестирования)

    Returns:
        Лог диалога: список словарей {role, content, sources?, task_state?}
    """
    if not _has_rag_index():
        msg = "Индексы не найдены. Сначала: python src/main.py index <docs_path>"
        if output_stream:
            output_stream.write(msg + "\n")
        else:
            print(msg)
        return []

    model = CHAT_MODEL_FAST if fast else CHAT_MODEL
    client = get_client()
    history: list[dict] = []  # для OpenAI messages
    task_state = TaskState()
    dialog_log: list[dict] = []

    def out(text: str):
        if output_stream:
            output_stream.write(text + "\n")
        else:
            print(text)

    out("╔══════════════════════════════════════════════════════════════╗")
    out("║  Smart RAG Chat — с памятью задачи и источниками          ║")
    out("║  Команды: 'выход' | 'состояние' | 'сброс'                ║")
    out("╚══════════════════════════════════════════════════════════════╝")
    out(f"  Контекст: top-{top_k} чанков | Модель: {model}"
        f"{' (fast)' if fast else ''}\n")

    turn = 0
    while True:
        # Ввод
        try:
            if input_stream:
                line = input_stream.readline()
                if not line:
                    break
                question = _clean_input(line.strip())
                out(f"Вы: {question}")
            else:
                question = _clean_input(input("Вы: ").strip())
        except (EOFError, KeyboardInterrupt):
            out("\nДо свидания!")
            break

        if not question:
            continue

        # Спецкоманды
        if question.lower() in ("выход", "exit", "quit", "q"):
            out("До свидания!")
            break

        if question.lower() in ("состояние", "state", "status"):
            out("\n── Память задачи ──")
            out(task_state.to_prompt_block())
            out(f"── Ходов в диалоге: {turn} ──\n")
            continue

        if question.lower() in ("сброс", "reset"):
            task_state = TaskState()
            history.clear()
            dialog_log.clear()
            turn = 0
            out("Память и история сброшены.\n")
            continue

        turn += 1

        # RAG поиск
        context, chunk_details, max_score = build_rag_context(question, top_k=top_k, fast=fast)

        # Низкая релевантность
        if max_score < RELEVANCE_FLOOR or not chunk_details:
            answer = (
                "**Ответ:** В базе знаний недостаточно информации для ответа "
                "на этот вопрос. Попробуйте переформулировать.\n\n"
                "**Источники:** нет релевантных"
            )
            out(f"\nБот (релевантность {max_score:.4f} < {RELEVANCE_FLOOR}):\n{answer}\n")
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            dialog_log.append({"role": "user", "content": question, "turn": turn})
            dialog_log.append({
                "role": "assistant", "content": answer,
                "sources": [], "max_score": max_score, "turn": turn,
            })
            continue

        # Формируем system prompt с task state
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            task_state=task_state.to_prompt_block()
        )

        user_msg = f"Контекст из базы знаний:\n\n{context}\n\n---\n\nВопрос: {question}"

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_msg})

        # Вызов LLM
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
        )
        raw_answer = response.choices[0].message.content

        # Извлекаем task state update
        clean_answer, task_update = _extract_task_update(raw_answer)
        if task_update:
            apply_task_update(task_state, task_update)

        # Авто-определение цели, если LLM не задал к 3-му ходу
        if not task_state.goal and turn <= 3:
            task_state.goal = _infer_goal(question)

        # Сохраняем в историю (без контекста)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": clean_answer})

        # Обрезка истории
        if len(history) > MAX_HISTORY_PAIRS * 2:
            history = history[-(MAX_HISTORY_PAIRS * 2):]

        # Вывод
        out(f"\nБот:\n{clean_answer}")
        out(f"\n  ┌─ RAG: {len(chunk_details)} чанков, "
            f"макс. score: {max_score:.4f}")
        for cd in chunk_details:
            out(f"  │  {cd['source']} | {cd['section']} | "
                f"{cd['chunk_id']} (score: {cd['score']})")
        out(f"  └─ Ход: {turn}")
        if not task_state.is_empty():
            out(f"  ┌─ Цель: {task_state.goal or '—'}")
            if task_state.clarifications:
                out(f"  │  Уточнений: {len(task_state.clarifications)}")
            if task_state.constraints:
                out(f"  │  Ограничений: {len(task_state.constraints)}")
            out(f"  └─")
        out("")

        # Лог
        dialog_log.append({"role": "user", "content": question, "turn": turn})
        dialog_log.append({
            "role": "assistant",
            "content": clean_answer,
            "sources": chunk_details,
            "max_score": max_score,
            "task_state": task_state.to_dict(),
            "turn": turn,
        })

    return dialog_log


# ── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Smart RAG Chat с памятью задачи")
    parser.add_argument("--top_k", type=int, default=5, help="Кол-во чанков контекста")
    args = parser.parse_args()
    run_smart_chat(top_k=args.top_k)


if __name__ == "__main__":
    main()
