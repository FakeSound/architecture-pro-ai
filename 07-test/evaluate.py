"""
evaluate.py — автоматическая оценка качества RAG-бота
======================================================
Запускает золотой набор вопросов, сравнивает результаты с ожиданиями,
пишет лог и выводит отчёт о покрытии базы знаний.

Запуск:
    python evaluate.py

Перед запуском:
    - Удалите 2-3 файла из knowledge_base/ для симуляции пробелов
    - Пересоберите индекс: python build_index.py
"""

import json
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, "04-bot")
from rag import RAGBot

GOLDEN_FILE = Path("golden_questions.jsonl")
EVAL_LOG    = Path("eval_log.jsonl")

NO_ANSWER_MARKER = "нет информации"


def load_questions() -> list[dict]:
    return [
        json.loads(line)
        for line in GOLDEN_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def is_answered(answer: str) -> bool:
    """Бот дал реальный ответ — не сказал 'не знаю'."""
    return NO_ANSWER_MARKER not in answer.lower() and len(answer) > 80


def evaluate_result(expect_answer: bool, answered: bool) -> tuple[bool, str]:
    """
    Возвращает (корректно: bool, статус: str).
    expect_answer=True  → бот должен был ответить
    expect_answer=False → бот должен был сказать 'не знаю'
    """
    if expect_answer and answered:
        return True, "✓ ОТВЕТИЛ (ожидалось)"
    if not expect_answer and not answered:
        return True, "✓ НЕ ОТВЕТИЛ (ожидалось)"
    if expect_answer and not answered:
        return False, "✗ НЕ ОТВЕТИЛ (ПРОБЕЛ В БАЗЕ)"
    return False, "✗ ОТВЕТИЛ (возможная галлюцинация)"


def main():
    print("Загрузка бота...")
    bot = RAGBot()

    questions = load_questions()
    print(f"Вопросов в наборе: {len(questions)}\n")
    print("=" * 60)

    results = []
    correct = 0

    for q in questions:
        result = bot.ask(q["question"])
        answered = is_answered(result.answer)
        is_correct, status = evaluate_result(q["expect_answer"], answered)

        if is_correct:
            correct += 1

        print(f"[{q['id']:02d}] {status}")
        print(f"     Q: {q['question']}")
        print(f"     Источники: {result.sources}")
        print()

        entry = {
            "timestamp":     datetime.now().isoformat(),
            "id":            q["id"],
            "question":      q["question"],
            "topic":         q["topic"],
            "expect_answer": q["expect_answer"],
            "answered":      answered,
            "is_correct":    is_correct,
            "status":        status,
            "sources":       result.sources,
            "chunks_found":  result.chunks_used,
            "answer_length": len(result.answer),
            "answer":        result.answer,
        }
        results.append(entry)

    # Сохраняем лог
    with EVAL_LOG.open("w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ─── Отчёт ───────────────────────────────────────────────────────────────

    print("=" * 60)
    print("ОТЧЁТ О ПОКРЫТИИ БАЗЫ ЗНАНИЙ")
    print("=" * 60)

    total = len(results)
    accuracy = correct / total * 100
    print(f"Точность: {correct}/{total} ({accuracy:.0f}%)\n")

    # Пробелы — вопросы где ожидался ответ, но бот не ответил
    gaps = [r for r in results if r["expect_answer"] and not r["answered"]]
    print(f"Пробелы в базе знаний ({len(gaps)}):")
    if gaps:
        for g in gaps:
            print(f"  - [{g['topic']}] {g['question']}")
    else:
        print("  Пробелов не обнаружено.")

    # Возможные галлюцинации — бот ответил там, где не должен был
    hallucinations = [r for r in results if not r["expect_answer"] and r["answered"]]
    print(f"\nВозможные галлюцинации ({len(hallucinations)}):")
    if hallucinations:
        for h in hallucinations:
            print(f"  - [{h['topic']}] {h['question']}")
    else:
        print("  Галлюцинаций не обнаружено.")

    # Нерелевантные источники — вопросы где sources пустые при успешном ответе
    irrelevant = [r for r in results if r["answered"] and not r["sources"]]
    print(f"\nОтветы без источников ({len(irrelevant)}):")
    if irrelevant:
        for i in irrelevant:
            print(f"  - {i['question']}")
    else:
        print("  Все ответы имеют источники.")

    print(f"\nРекомендации:")
    if gaps:
        topics = list({g["topic"] for g in gaps})
        print(f"  Добавить документы по темам: {', '.join(topics)}")
    if hallucinations:
        print("  Проверить промпт — модель отвечает на вопросы вне базы")
    if accuracy == 100:
        print("  База знаний покрывает все тестовые сценарии.")

    print(f"\nЛог сохранён: {EVAL_LOG}")


if __name__ == "__main__":
    main()