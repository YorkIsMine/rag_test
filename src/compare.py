"""Сравнение двух стратегий чанкинга."""

from __future__ import annotations

from loader import Document
from chunking import Chunk, chunk_fixed, chunk_structural


def _stats(chunks: list[Chunk]) -> dict:
    sizes = [c.metadata.get("char_count", len(c.text)) for c in chunks]
    sources = {}
    for c in chunks:
        fn = c.metadata.get("filename", "?")
        sources[fn] = sources.get(fn, 0) + 1

    return {
        "total_chunks": len(chunks),
        "avg_size": round(sum(sizes) / len(sizes)) if sizes else 0,
        "min_size": min(sizes) if sizes else 0,
        "max_size": max(sizes) if sizes else 0,
        "by_source": sources,
    }


def compare_strategies(documents: list[Document]) -> dict:
    """Сравнивает fixed и structural чанкинг, возвращает статистику."""
    fixed_chunks = chunk_fixed(documents)
    struct_chunks = chunk_structural(documents)

    fixed_stats = _stats(fixed_chunks)
    struct_stats = _stats(struct_chunks)

    return {
        "fixed": fixed_stats,
        "structural": struct_stats,
        "fixed_chunks": fixed_chunks,
        "structural_chunks": struct_chunks,
    }


def print_comparison(result: dict) -> None:
    """Выводит таблицу сравнения стратегий."""
    header = f"{'Метрика':<25} {'Fixed-size':>15} {'Structural':>15}"
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for key in ("total_chunks", "avg_size", "min_size", "max_size"):
        label = {
            "total_chunks": "Кол-во чанков",
            "avg_size": "Средний размер (симв.)",
            "min_size": "Мин. размер (симв.)",
            "max_size": "Макс. размер (симв.)",
        }[key]
        print(f"{label:<25} {result['fixed'][key]:>15} {result['structural'][key]:>15}")

    print(sep)

    print("\nРаспределение по файлам (fixed):")
    for fn, cnt in result["fixed"]["by_source"].items():
        print(f"  {fn}: {cnt} чанков")

    print("\nРаспределение по файлам (structural):")
    for fn, cnt in result["structural"]["by_source"].items():
        print(f"  {fn}: {cnt} чанков")

    print()
