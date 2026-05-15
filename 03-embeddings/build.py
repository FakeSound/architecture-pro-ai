"""
Задание 3: Построение векторного индекса
=========================================
Зависимости:
    pip install langchain langchain-community langchain-openai faiss-cpu tiktoken

Результат:
    faiss_index/   — сохранённый индекс (index.faiss + index.pkl)
"""

import os
import time
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # читает .env из текущей папки

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


# ─── Настройки ───────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = Path("knowledge_base")
INDEX_DIR      = Path("faiss_index")

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_TOKENS    = 500   # максимум токенов на чанк
CHUNK_OVERLAP   = 50    # перекрытие между чанками (токены)


# ─── Токенизатор для точного подсчёта ────────────────────────────────────────

_enc = tiktoken.encoding_for_model("text-embedding-3-small")

def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


# ─── Загрузка документов ─────────────────────────────────────────────────────

def load_documents() -> list[Document]:
    """Читает все .txt из knowledge_base/, возвращает список Document с метаданными."""
    docs = []
    files = sorted(KNOWLEDGE_BASE.rglob("*.txt"))

    if not files:
        raise FileNotFoundError(f"Нет файлов в {KNOWLEDGE_BASE}/. Запусти clean_pages.py и replace_terms.py")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for filepath in files:
        text = filepath.read_text(encoding="utf-8").strip()
        if not text:
            continue

        chunks = splitter.split_text(text)
        category = filepath.parent.name  # players / teams / tournaments

        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source":    str(filepath),
                    "filename":  filepath.stem,
                    "category":  category,
                    "chunk_id":  i,
                    "chunk_total": len(chunks),
                },
            ))

    return docs


# ─── Построение индекса ──────────────────────────────────────────────────────

def build_index(docs: list[Document]) -> FAISS:
    """Генерирует эмбеддинги и создаёт FAISS-индекс."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY не задан.\n"
            "Запусти: export OPENAI_API_KEY='sk-...'"
        )

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=api_key,
    )

    print(f"Генерация эмбеддингов ({len(docs)} чанков) ...")

    t0 = time.time()
    vectorstore = FAISS.from_documents(docs, embeddings)
    elapsed = time.time() - t0

    return vectorstore, elapsed


# ─── Тестовый поиск ──────────────────────────────────────────────────────────

def test_search(vectorstore: FAISS):
    """Несколько пробных запросов, чтобы убедиться, что индекс работает."""
    queries = [
        "Who is Zr0ne and what team does he play for?",
        "What mouse does Vexis use?",
        "Tell me about EVL Pro Circuit",
    ]

    print("\n" + "=" * 50)
    print("ТЕСТОВЫЙ ПОИСК")
    print("=" * 50)

    for query in queries:
        print(f"\nЗапрос: {query}")
        results = vectorstore.similarity_search(query, k=2)
        for j, doc in enumerate(results, 1):
            print(f"  [{j}] {doc.metadata['filename']} (чанк {doc.metadata['chunk_id']})")
            print(f"       {doc.page_content[:150].strip()}...")


# ─── Сохранение статистики ───────────────────────────────────────────────────

def save_stats(docs: list[Document], elapsed: float):
    categories = {}
    for doc in docs:
        cat = doc.metadata["category"]
        categories[cat] = categories.get(cat, 0) + 1

    stats = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": 1536,
        "knowledge_base": str(KNOWLEDGE_BASE),
        "total_chunks": len(docs),
        "chunks_by_category": categories,
        "chunk_size_tokens": CHUNK_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP,
        "build_time_seconds": round(elapsed, 1),
    }

    stats_path = INDEX_DIR / "index_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


# ─── Точка входа ─────────────────────────────────────────────────────────────

def main():
    print("Загрузка и нарезка документов...")
    docs = load_documents()

    sources = len({d.metadata["source"] for d in docs})
    print(f"Файлов: {sources}, чанков: {len(docs)}")

    vectorstore, elapsed = build_index(docs)

    INDEX_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))

    stats = save_stats(docs, elapsed)

    print(f"\n{'=' * 50}")
    print(f"Индекс сохранён: {INDEX_DIR}/")
    print(f"Модель:          {stats['embedding_model']} ({stats['embedding_dimensions']}d)")
    print(f"Чанков:          {stats['total_chunks']}")
    print(f"По категориям:   {stats['chunks_by_category']}")
    print(f"Время:           {elapsed:.1f} сек")

    test_search(vectorstore)


if __name__ == "__main__":
    main()
