"""
Тестирование Smart RAG Chat — 2 длинных сценария по 10-15 сообщений.

Проверяет:
  - ассистент не теряет цель диалога
  - каждый ответ содержит источники
  - task state обновляется корректно
  - при возврате к теме контекст сохраняется

Запуск:
    python src/test_smart_chat.py
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smart_chat import run_smart_chat, TaskState, _extract_task_update, apply_task_update


# ── Сценарии ───────────────────────────────────────────────────────

SCENARIO_1_NAME = "Управление проектами — роль менеджера"
SCENARIO_1_MESSAGES = [
    "Какова главная идея книги Deadline?",
    "Кто главный герой и чем он занимается?",
    "Расскажи подробнее о проектах, которые вёл главный герой",
    "Какие ошибки совершал менеджер в начале?",
    "А какие принципы управления оказались правильными?",
    "Вернёмся к началу — какова была исходная ситуация героя?",
    "Какие метафоры использует ДеМарко для объяснения проектного менеджмента?",
    "Как книга описывает работу с людьми в команде?",
    "Есть ли в книге советы по планированию сроков?",
    "Какие уроки можно извлечь из книги для реальных проектов?",
    "Напомни, какова была главная цель нашего разговора?",
    "Подведи итог: 5 главных тезисов книги по управлению проектами",
]

SCENARIO_2_NAME = "Команда и мотивация — глубокое исследование"
SCENARIO_2_MESSAGES = [
    "Что книга Deadline говорит о формировании команд?",
    "Какие типы людей описаны в книге?",
    "Как автор относится к давлению на команду?",
    "Меня особенно интересует мотивация — что ДеМарко пишет об этом?",
    "А что насчёт конфликтов в команде?",
    "Давай зафиксируем: мы исследуем тему мотивации и командной работы. Какие ещё аспекты затронуты?",
    "Как связаны мотивация и продуктивность по мнению автора?",
    "Есть ли в книге примеры провальных проектов из-за плохой мотивации?",
    "Вернёмся к типам людей — как они влияют на динамику команды?",
    "Какие конкретные практики предлагает автор для улучшения командной работы?",
    "Как автор описывает роль доверия в команде?",
    "Давай вспомним всё что обсудили — подведи итог по теме мотивации и команд",
    "Не забыл ли ты цель нашего разговора? Напомни её.",
    "Последний вопрос: чем отличается подход ДеМарко от классического менеджмента?",
]


# ── Валидация ──────────────────────────────────────────────────────

def validate_response(text: str, turn: int, scenario: str) -> list[str]:
    """Проверяет ответ на обязательные элементы. Возвращает список ошибок."""
    errors = []

    # Проверка наличия секции Ответ
    if "**Ответ:**" not in text and "Ответ:" not in text:
        errors.append(f"[{scenario}] Ход {turn}: отсутствует секция 'Ответ'")

    # Проверка наличия секции Источники
    if "**Источники:**" not in text and "Источники:" not in text:
        errors.append(f"[{scenario}] Ход {turn}: отсутствует секция 'Источники'")

    # Проверка что ответ не пустой (минимум 50 символов содержательного текста)
    clean = re.sub(r'\*\*\w+:\*\*', '', text).strip()
    if len(clean) < 50:
        errors.append(f"[{scenario}] Ход {turn}: слишком короткий ответ ({len(clean)} символов)")

    return errors


def validate_task_state_progression(log: list[dict], scenario: str) -> list[str]:
    """Проверяет что task state развивается по ходу диалога."""
    errors = []
    assistant_entries = [e for e in log if e["role"] == "assistant" and "task_state" in e]

    if not assistant_entries:
        errors.append(f"[{scenario}] Task state не был записан ни разу")
        return errors

    # Проверяем что к середине диалога цель определена
    mid = len(assistant_entries) // 2
    if mid > 0:
        mid_state = assistant_entries[mid].get("task_state", {})
        if not mid_state.get("goal"):
            errors.append(f"[{scenario}] К середине диалога (ход {mid+1}) цель не определена")

    # Проверяем что в последнем ответе есть хоть какая-то память
    last_state = assistant_entries[-1].get("task_state", {})
    total_items = (
        len(last_state.get("clarifications", []))
        + len(last_state.get("constraints", []))
        + len(last_state.get("key_facts", []))
    )
    if total_items == 0 and not last_state.get("goal"):
        errors.append(f"[{scenario}] В финальном task state пусто — память не работает")

    return errors


def validate_sources_present(log: list[dict], scenario: str) -> list[str]:
    """Проверяет что каждый ответ содержит источники."""
    errors = []
    for entry in log:
        if entry["role"] != "assistant":
            continue
        sources = entry.get("sources", [])
        content = entry.get("content", "")
        turn = entry.get("turn", "?")

        # Если ответ "нет релевантных" — это допустимо
        if "нет релевантных" in content.lower():
            continue

        if not sources:
            errors.append(f"[{scenario}] Ход {turn}: ответ без источников (sources пуст)")

    return errors


# ── Запуск сценария ────────────────────────────────────────────────

def run_scenario(name: str, messages: list[str], *, fast: bool = False) -> tuple[list[dict], list[str]]:
    """
    Запускает сценарий и возвращает (лог, ошибки).
    """
    header = (
        f"\n{'='*70}\n"
        f"СЦЕНАРИЙ: {name}\n"
        f"Сообщений: {len(messages)}{' (fast mode)' if fast else ''}\n"
        f"{'='*70}\n"
    )

    # Подготовка потоков
    input_text = "\n".join(messages) + "\nвыход\n"
    input_stream = io.StringIO(input_text)
    output_stream = io.StringIO()

    # Запуск
    t0 = time.time()
    log = run_smart_chat(
        top_k=5,
        fast=fast,
        input_stream=input_stream,
        output_stream=output_stream,
    )
    elapsed = time.time() - t0

    # Вывод
    output_text = output_stream.getvalue()
    print(header)
    print(output_text)
    print(f"  Время: {elapsed:.1f}s ({elapsed/max(len(messages),1):.1f}s/сообщение)")

    # Валидация
    all_errors = []

    # 1. Проверяем каждый ответ
    for entry in log:
        if entry["role"] == "assistant":
            errs = validate_response(entry["content"], entry.get("turn", "?"), name)
            all_errors.extend(errs)

    # 2. Проверяем источники
    all_errors.extend(validate_sources_present(log, name))

    # 3. Проверяем прогресс task state
    all_errors.extend(validate_task_state_progression(log, name))

    return log, all_errors


# ── Main ───────────────────────────────────────────────────────────

def main():
    fast = "--fast" in sys.argv

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Тестирование Smart RAG Chat — 2 длинных сценария          ║")
    if fast:
        print("║  Режим: FAST (gpt-4o-mini, без LLM-реранкинга)            ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    t_start = time.time()
    all_errors = []
    all_logs = {}

    # Запуск обоих сценариев параллельно
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut1 = pool.submit(run_scenario, SCENARIO_1_NAME, SCENARIO_1_MESSAGES, fast=fast)
        fut2 = pool.submit(run_scenario, SCENARIO_2_NAME, SCENARIO_2_MESSAGES, fast=fast)

        log1, errors1 = fut1.result()
        log2, errors2 = fut2.result()

    all_errors.extend(errors1)
    all_logs["scenario_1"] = {
        "name": SCENARIO_1_NAME,
        "messages": len(SCENARIO_1_MESSAGES),
        "responses": len([e for e in log1 if e["role"] == "assistant"]),
        "errors": errors1,
    }

    all_errors.extend(errors2)
    all_logs["scenario_2"] = {
        "name": SCENARIO_2_NAME,
        "messages": len(SCENARIO_2_MESSAGES),
        "responses": len([e for e in log2 if e["role"] == "assistant"]),
        "errors": errors2,
    }

    # Итоговый отчёт
    print(f"\n{'='*70}")
    print("ИТОГОВЫЙ ОТЧЁТ")
    print(f"{'='*70}")

    for key, info in all_logs.items():
        status = "✅ PASS" if not info["errors"] else "❌ FAIL"
        print(f"\n{status} {info['name']}")
        print(f"  Сообщений: {info['messages']}, Ответов: {info['responses']}")
        if info["errors"]:
            for err in info["errors"]:
                print(f"  ⚠ {err}")

    # Проверяем task state на финальном шаге
    for scenario_name, log in [("Сценарий 1", log1), ("Сценарий 2", log2)]:
        assistant_entries = [e for e in log if e["role"] == "assistant" and "task_state" in e]
        if assistant_entries:
            final_state = assistant_entries[-1]["task_state"]
            print(f"\n  Финальный task state ({scenario_name}):")
            print(f"    Цель: {final_state.get('goal', '—')}")
            print(f"    Уточнений: {len(final_state.get('clarifications', []))}")
            print(f"    Ограничений: {len(final_state.get('constraints', []))}")
            print(f"    Фактов: {len(final_state.get('key_facts', []))}")

    total = len(SCENARIO_1_MESSAGES) + len(SCENARIO_2_MESSAGES)
    elapsed_total = time.time() - t_start
    print(f"\n{'='*70}")
    if not all_errors:
        print(f"✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ ({total} сообщений, 0 ошибок)")
    else:
        print(f"❌ НАЙДЕНЫ ОШИБКИ: {len(all_errors)} из {total} сообщений")
    print(f"  Общее время: {elapsed_total:.1f}s")
    print(f"{'='*70}")

    # Сохраняем результаты
    results = {
        "total_messages": total,
        "total_errors": len(all_errors),
        "scenarios": all_logs,
    }
    with open("smart_chat_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены в smart_chat_test_results.json")

    return 0 if not all_errors else 1


if __name__ == "__main__":
    sys.exit(main())
