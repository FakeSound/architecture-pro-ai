"""
Шаг 4: Замена терминов в базе знаний
=====================================
Зависимости: нет (только стандартная библиотека)
Запуск:      python replace_terms.py

Читает terms_map.json, проходит по всем файлам в knowledge_base/
и заменяет оригинальные термины на вымышленные аналоги.

Важно: замены делаются от длинных к коротким, чтобы
"ESL Pro League" заменилось раньше, чем "ESL".
"""

import json
from pathlib import Path


KNOWLEDGE_BASE = Path("knowledge_base")
TERMS_MAP_FILE = Path("02-data/terms_map.json")


def load_terms(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Убираем служебные ключи-комментарии (начинаются с #)
    return {k: v for k, v in raw.items() if not k.startswith("#") and k != "_meta"}


def replace_in_text(text: str, terms: list[tuple]) -> tuple[str, int]:
    """Применяет замены, возвращает новый текст и количество замен."""
    total = 0
    for original, replacement in terms:
        count = text.count(original)
        if count > 0:
            text = text.replace(original, replacement)
            total += count
    return text, total


def main():
    if not TERMS_MAP_FILE.exists():
        print(f"[!] {TERMS_MAP_FILE} не найден")
        return

    if not KNOWLEDGE_BASE.exists():
        print(f"[!] Папка {KNOWLEDGE_BASE} не найдена. Сначала запусти clean_pages.py")
        return

    terms = load_terms(TERMS_MAP_FILE)

    # Сортируем: длинные термины первыми
    sorted_terms = sorted(terms.items(), key=lambda x: len(x[0]), reverse=True)

    files = sorted(KNOWLEDGE_BASE.rglob("*.txt"))
    if not files:
        print("[!] Нет файлов в knowledge_base/")
        return

    print(f"Словарь: {len(sorted_terms)} замен")
    print(f"Файлов:  {len(files)}\n")

    total_replacements = 0

    for filepath in files:
        content = filepath.read_text(encoding="utf-8")
        new_content, count = replace_in_text(content, sorted_terms)
        filepath.write_text(new_content, encoding="utf-8")

        marker = "✓ " if count > 0 else "–  "
        print(f"{marker} {filepath.relative_to(KNOWLEDGE_BASE)}: {count} замен")
        total_replacements += count

    print(f"\n{'=' * 40}")
    print(f"Всего замен: {total_replacements}")
    print(f"Файлов обработано: {len(files)}")
    print("\nПроверь несколько файлов вручную — убедись, что текст читаемый.")


if __name__ == "__main__":
    main()
