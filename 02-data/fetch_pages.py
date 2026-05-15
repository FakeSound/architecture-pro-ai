"""
Шаг 2: Скачивание страниц с Liquipedia (CS2)
=============================================
Зависимости: pip install requests mwparserfromhell
Запуск:      python fetch_pages.py

Что скачивается:
  Игроки  — основная страница + /Achievements + /Settings
  Команды — основная страница
  Турниры — полная страница; нужная секция указана в sections_config.json
             и будет вырезана на шаге очистки (clean_pages.py)

ВАЖНО: замените email в HEADERS на свой.
"""

import requests
import time
import json
from pathlib import Path
from typing import Optional


# ─── Конфигурация ────────────────────────────────────────────────────────────

PLAYERS = [
    "s1mple", "ZywOo", "dev1ce", "NiKo", "karrigan",
    "Magisk", "dupreeh", "xyp9x", "gla1ve", "ElectroNic",
    "b1t", "Perfecto", "YEKINDAR", "apEX", "m0NESY",
    "sh1ro", "Ax1Le", "HObbit", "Boombl4", "frozen",
]

# Сабпейджи, которые качаем для каждого игрока
PLAYER_SUBPAGES = []  # Settings и Results — динамические шаблоны, данные в основной странице

TEAMS = [
    "Natus Vincere", "Astralis", "Team Vitality", "FaZe Clan",
    "G2 Esports", "Team Liquid", "MOUZ", "FURIA",
    "Cloud9", "ENCE",
]

# section — имя секции, которую вырежет clean_pages.py (None = весь текст)
TOURNAMENTS = [
    {"title": "ESL Pro League",        "section": None},
    {"title": "BLAST",                 "section": None},
    {"title": "Intel Extreme Masters", "section": None},
    {"title": "PGL",                   "section": None},
]


API_URL = "https://liquipedia.net/counterstrike/api.php"
HEADERS = {
    "User-Agent": "StudentRAGProject/1.0 (homework; your@email.com)",
    "Accept-Encoding": "gzip",
}
OUTPUT_DIR = Path("raw_pages")
DELAY_SECONDS = 2


# ─── API ─────────────────────────────────────────────────────────────────────

def fetch_page(title: str) -> Optional[str]:
    """Скачивает wikitext страницы. Возвращает текст или None."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
        "formatversion": "2",
    }
    try:
        response = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()

        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return None

        page = pages[0]
        if page.get("missing"):
            return None  # тихо, сабпейджей может не быть

        return page["revisions"][0]["slots"]["main"]["content"]

    except requests.RequestException as e:
        print(f"  [!] Ошибка запроса '{title}': {e}")
    except (KeyError, IndexError) as e:
        print(f"  [!] Ошибка разбора '{title}': {e}")

    return None


def save(directory: Path, filename: str, content: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / filename
    filepath.write_text(content, encoding="utf-8")
    return filepath


# ─── Загрузка по категориям ──────────────────────────────────────────────────

def fetch_players(stats: dict):
    print("\n── PLAYERS ─────────────────────────────")
    out_dir = OUTPUT_DIR / "players"
    total = len(PLAYERS) * (1 + len(PLAYER_SUBPAGES))
    count = 0

    for player in PLAYERS:
        pages_to_fetch = [player] + [f"{player}/{sub}" for sub in PLAYER_SUBPAGES]

        for title in pages_to_fetch:
            count += 1
            label = title.replace("/", " / ")
            print(f"[{count:03d}/{total}] {label} ...", end=" ", flush=True)

            content = fetch_page(title)
            if content:
                filename = title.replace("/", "_") + ".txt"
                path = save(out_dir, filename, content)
                print(f"✓  → {path.name}")
                stats["downloaded"].append({"title": title, "category": "players"})
            else:
                print("–  нет страницы")
                stats["skipped"].append(title)

            time.sleep(DELAY_SECONDS)


def fetch_teams(stats: dict):
    print("\n── TEAMS ───────────────────────────────")
    out_dir = OUTPUT_DIR / "teams"

    for i, team in enumerate(TEAMS, 1):
        print(f"[{i:02d}/{len(TEAMS)}] {team} ...", end=" ", flush=True)
        content = fetch_page(team)
        if content:
            filename = team.replace(" ", "_") + ".txt"
            path = save(out_dir, filename, content)
            print(f"✓  → {path.name}")
            stats["downloaded"].append({"title": team, "category": "teams"})
        else:
            print("✗  не найдено")
            stats["failed"].append(team)
        time.sleep(DELAY_SECONDS)


def fetch_tournaments(stats: dict):
    print("\n── TOURNAMENTS ─────────────────────────")
    out_dir = OUTPUT_DIR / "tournaments"
    sections_config = {}  # filename → section name для clean_pages.py

    for i, t in enumerate(TOURNAMENTS, 1):
        title = t["title"]
        section = t["section"]
        print(f"[{i:02d}/{len(TOURNAMENTS)}] {title} ...", end=" ", flush=True)

        content = fetch_page(title)
        if content:
            filename = title.replace(" ", "_") + ".txt"
            path = save(out_dir, filename, content)
            print(f"✓  → {path.name}  (секция: {section or 'весь текст'})")
            stats["downloaded"].append({"title": title, "category": "tournaments"})

            if section:
                sections_config[filename] = section
        else:
            print("✗  не найдено")
            stats["failed"].append(title)

        time.sleep(DELAY_SECONDS)

    # Сохраняем конфиг секций рядом с файлами — clean_pages.py его прочитает
    config_path = out_dir / "sections_config.json"
    config_path.write_text(
        json.dumps(sections_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Конфиг секций сохранён: {config_path}")


# ─── Точка входа ─────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    stats = {"downloaded": [], "failed": [], "skipped": []}

    fetch_players(stats)
    fetch_teams(stats)
    fetch_tournaments(stats)

    # Общий лог
    log_path = OUTPUT_DIR / "download_log.json"
    log_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n{'=' * 45}")
    print(f"Скачано:       {len(stats['downloaded'])}")
    print(f"Не найдено:    {len(stats['failed'])}")
    print(f"Нет сабпейджа: {len(stats['skipped'])}")
    print(f"Лог: {log_path}")

    if stats["failed"]:
        print("\nНе найдены (проверь название):")
        for t in stats["failed"]:
            print(f"  - {t}")


if __name__ == "__main__":
    main()
