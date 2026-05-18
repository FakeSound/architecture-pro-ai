"""
update_index.py — автоматическое обновление FAISS-индекса
==========================================================
Сканирует папку docs/ на наличие новых или изменённых файлов,
сравнивает с манифестом, добавляет новые чанки в существующий индекс.

Запуск вручную:   python update_index.py
Запуск через cron: см. cron_setup.md
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime

import tiktoken
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ─── Настройки ───────────────────────────────────────────────────────────────

DOCS_DIR       = Path("docs")            # папка с новыми документами
INDEX_DIR      = Path("faiss_index")     # существующий FAISS-индекс
MANIFEST_FILE  = Path("index_manifest.json")  # что уже обработано
LOG_DIR        = Path("logs")

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_TOKENS    = 500
CHUNK_OVERLAP   = 50


# ─── Логирование ─────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"update_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return logging.getLogger("update"), log_file


# ─── Манифест ────────────────────────────────────────────────────────────────

def load_manifest() -> dict:
    """Загружает манифест: {путь_к_файлу: время_изменения}."""
    if MANIFEST_FILE.exists():
        return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict):
    MANIFEST_FILE.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ─── Поиск новых файлов ──────────────────────────────────────────────────────

def file_hash(filepath: Path) -> str:
    """SHA256 хэш содержимого файла."""
    return hashlib.sha256(filepath.read_bytes()).hexdigest()


def find_new_files(manifest: dict) -> list[Path]:
    """
    Сравнивает файлы в docs/ с манифестом по хэшу содержимого.
    Возвращает файлы которых нет в манифесте или чей контент изменился.
    """
    DOCS_DIR.mkdir(exist_ok=True)
    new_files = []

    for filepath in sorted(DOCS_DIR.rglob("*.txt")):
        key = str(filepath)
        current_hash = file_hash(filepath)
        if key not in manifest or manifest[key] != current_hash:
            new_files.append(filepath)

    return new_files


# ─── Обработка файлов ────────────────────────────────────────────────────────

def process_files(files: list[Path]) -> list[Document]:
    """Читает файлы, разбивает на чанки, возвращает список Document."""
    enc = tiktoken.encoding_for_model("text-embedding-3-small")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=lambda t: len(enc.encode(t)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs = []
    for filepath in files:
        text = filepath.read_text(encoding="utf-8").strip()
        if not text:
            continue

        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source":      str(filepath),
                    "filename":    filepath.stem,
                    "category":    filepath.parent.name,
                    "chunk_id":    i,
                    "chunk_total": len(chunks),
                    "updated_at":  datetime.now().isoformat(),
                },
            ))

    return docs


# ─── Обновление индекса ──────────────────────────────────────────────────────

def update_index(new_docs: list[Document], api_key: str) -> int:
    """
    Добавляет новые чанки в существующий индекс.
    Если индекса нет — создаёт новый.
    Возвращает итоговое количество векторов в индексе.
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)

    if INDEX_DIR.exists():
        vectorstore = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vectorstore.add_documents(new_docs)
    else:
        vectorstore = FAISS.from_documents(new_docs, embeddings)

    vectorstore.save_local(str(INDEX_DIR))
    return vectorstore.index.ntotal


# ─── Точка входа ─────────────────────────────────────────────────────────────

def main():
    log, log_file = setup_logging()
    start = datetime.now()

    log.info("=" * 50)
    log.info(f"Запуск обновления индекса: {start.strftime('%Y-%m-%d %H:%M:%S')}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY не задан")
        return

    manifest = load_manifest()
    new_files = find_new_files(manifest)

    if not new_files:
        log.info("Новых файлов нет. Индекс актуален.")
        log.info("=" * 50)
        return

    log.info(f"Новых файлов: {len(new_files)}")
    for f in new_files:
        log.info(f"  + {f}")

    new_docs = process_files(new_files)
    log.info(f"Создано чанков: {len(new_docs)}")

    try:
        total_vectors = update_index(new_docs, api_key)
        log.info(f"Индекс обновлён. Векторов в индексе: {total_vectors}")

        # Обновляем манифест только после успешного обновления
        for filepath in new_files:
            manifest[str(filepath)] = file_hash(filepath)
        save_manifest(manifest)

    except Exception as e:
        log.error(f"Ошибка при обновлении индекса: {e}")
        raise

    duration = (datetime.now() - start).total_seconds()
    log.info(
        f"Готово: {len(new_files)} файлов, {len(new_docs)} чанков за {duration:.1f}с"
    )
    log.info(f"Лог сохранён: {log_file}")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
