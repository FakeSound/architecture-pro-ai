"""
Шаг 3: Очистка wikitext → чистый текст
========================================
Зависимости: pip install mwparserfromhell
Запуск:      python clean_pages.py

Особенности:
  ==Gear and Settings== — параметры шаблонов сохраняются как текст,
                          иначе strip_code() выкинул бы всё содержимое
  Турниры               — вырезается только нужная секция согласно
                          raw_pages/tournaments/sections_config.json
  Остальные секции      — стандартная очистка через strip_code()
"""

import re
import json
from pathlib import Path

import mwparserfromhell


RAW_DIR = Path("raw_pages")
OUTPUT_DIR = Path("knowledge_base")
MIN_LENGTH = 200


# ─── Вспомогательные функции ─────────────────────────────────────────────────

def clean_artifact_lines(text: str) -> str:
    """Убирает артефакты после strip_code: обрывки таблиц, одиночные числа, мусор."""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            lines.append("")
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if re.fullmatch(r"[|\-=\s]+", line):
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def extract_gear_text(section) -> str:
    """
    Для секции ==Gear and Settings== извлекает параметры шаблонов как текст.
    strip_code() удалил бы всё содержимое шаблонов (мышь, монитор, прицел),
    поэтому обрабатываем эту секцию отдельно.

    Результат выглядит так:
        == Gear and Settings ==

        Mouse settings table:
          brand: Logitech
          model: G PRO X SUPERLIGHT
          dpi: 800
          ...
    """
    SKIP_TEMPLATES = {"cite", "cite_web", "flag", "abbr", "reflist", "ref"}
    SKIP_KEYS = {"date", "ref", "share_code"}

    lines = ["== Gear and Settings =="]

    for template in section.filter_templates():
        name = str(template.name).strip()
        if any(s in name.lower() for s in SKIP_TEMPLATES):
            continue

        params = {}
        for param in template.params:
            key = str(param.name).strip()
            if not key or re.fullmatch(r"\d+", key) or key in SKIP_KEYS:
                continue
            value = mwparserfromhell.parse(str(param.value)).strip_code().strip()
            if value:
                params[key] = value

        if params:
            lines.append(f"\n{name}:")
            for k, v in params.items():
                lines.append(f"  {k}: {v}")

    return "\n".join(lines) if len(lines) > 1 else ""


def extract_section_by_title(raw: str, section_title: str) -> str:
    """
    Вырезает конкретную секцию по заголовку.
    Используется для турнирных страниц, где нужна только одна секция
    (например, 'Finals' у ESL Pro League).
    """
    wikicode = mwparserfromhell.parse(raw)
    for section in wikicode.get_sections(flat=True):
        headings = section.filter_headings()
        if headings and str(headings[0].title).strip() == section_title:
            return clean_artifact_lines(
                section.strip_code(normalize=True, collapse=True)
            )
    return ""


def clean_wikitext(raw: str) -> str:
    """
    Основная очистка. Проходит по секциям:
    - ==Gear and Settings== → extract_gear_text()
    - всё остальное        → strip_code() + clean_artifact_lines()
    """
    wikicode = mwparserfromhell.parse(raw)
    parts = []

    for section in wikicode.get_sections(include_lead=True, flat=True):
        headings = section.filter_headings()
        title = str(headings[0].title).strip() if headings else ""

        if title == "Gear and Settings":
            gear = extract_gear_text(section)
            if gear:
                parts.append(gear)
        else:
            text = clean_artifact_lines(
                section.strip_code(normalize=True, collapse=True)
            )
            if text:
                parts.append(text)

    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(parts)).strip()


# ─── Основной цикл ───────────────────────────────────────────────────────────

def main():
    if not RAW_DIR.exists():
        print(f"[!] Папка {RAW_DIR} не найдена. Сначала запусти fetch_pages.py")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Конфиг секций для турниров — создаётся fetch_pages.py
    sections_config_path = RAW_DIR / "tournaments" / "sections_config.json"
    sections_config = {}
    if sections_config_path.exists():
        sections_config = json.loads(
            sections_config_path.read_text(encoding="utf-8")
        )

    ok, short, errors = [], [], []
    raw_files = sorted(f for f in RAW_DIR.rglob("*.txt"))

    if not raw_files:
        print("[!] Нет файлов для обработки в raw_pages/")
        return

    print(f"Найдено файлов: {len(raw_files)}\n")

    for raw_file in raw_files:
        relative = raw_file.relative_to(RAW_DIR)
        out_file = OUTPUT_DIR / relative
        out_file.parent.mkdir(parents=True, exist_ok=True)

        raw_text = raw_file.read_text(encoding="utf-8")

        try:
            section_to_extract = sections_config.get(raw_file.name)
            if section_to_extract:
                clean = extract_section_by_title(raw_text, section_to_extract)
                if not clean:
                    # Секция не найдена — берём весь текст
                    print(f"  ⚠  Секция '{section_to_extract}' не найдена в {raw_file.name}, беру весь текст")
                    clean = clean_wikitext(raw_text)
            else:
                clean = clean_wikitext(raw_text)

        except Exception as e:
            print(f"[!] Ошибка: {raw_file.name} — {e}")
            errors.append(raw_file.name)
            continue

        out_file.write_text(clean, encoding="utf-8")

        chars_before = len(raw_text)
        chars_after = len(clean)
        reduction = (1 - chars_after / chars_before) * 100 if chars_before else 0

        if chars_after < MIN_LENGTH:
            marker = "⚠️ "
            short.append(raw_file.name)
        else:
            marker = "✓  "
            ok.append(raw_file.name)

        print(f"{marker}{relative}: {chars_before:,} → {chars_after:,} символов  (-{reduction:.0f}%)")

    print(f"\n{'=' * 45}")
    print(f"Обработано:  {len(ok)}")
    print(f"Коротких:    {len(short)}  (проверь вручную)")
    print(f"Ошибок:      {len(errors)}")
    print(f"\nЧистые тексты сохранены в: {OUTPUT_DIR}/")

    if short:
        print("\nКороткие файлы:")
        for name in short:
            print(f"  - {name}")


if __name__ == "__main__":
    main()