"""
Шаг 4б: Переименование файлов knowledge_base по terms_map.json
==============================================================
Запуск: python rename_files.py

Заменяет оригинальные термины в именах файлов на вымышленные.
Например: s1mple.txt → Zr0ne.txt, Natus_Vincere.txt → Nexus_Victus.txt
"""

import json
from pathlib import Path

KNOWLEDGE_BASE = Path("knowledge_base")
TERMS_MAP_FILE = Path("02-data/terms_map.json")


def main():
    terms = json.loads(TERMS_MAP_FILE.read_text(encoding="utf-8"))

    # Убираем служебные ключи, сортируем длинные первыми
    sorted_terms = sorted(
        ((k, v) for k, v in terms.items() if not k.startswith("#") and k != "_meta"),
        key=lambda x: len(x[0]),
        reverse=True,
    )

    files = sorted(KNOWLEDGE_BASE.rglob("*.txt"))
    renamed = 0

    for filepath in files:
        new_name = filepath.stem  # имя без расширения

        for original, replacement in sorted_terms:
            # В именах файлов пробелы заменены на _, учитываем оба варианта
            orig_underscored = original.replace(" ", "_")
            repl_underscored = replacement.replace(" ", "_")
            new_name = new_name.replace(orig_underscored, repl_underscored)
            new_name = new_name.replace(original, replacement)

        new_path = filepath.parent / (new_name + ".txt")

        if new_path != filepath:
            filepath.rename(new_path)
            print(f"✓  {filepath.name}  →  {new_path.name}")
            renamed += 1
        else:
            print(f"–  {filepath.name}  (без изменений)")

    print(f"\nПереименовано: {renamed} файлов")


if __name__ == "__main__":
    main()
