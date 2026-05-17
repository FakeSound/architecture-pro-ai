"""
rag.py — ядро RAG-пайплайна
============================
Загружает FAISS-индекс, принимает запрос, ищет релевантные чанки,
формирует промпт (Few-shot + CoT) и вызывает OpenAI для генерации ответа.

Тест без бота:
    python rag.py
"""

import os
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ─── Настройки ───────────────────────────────────────────────────────────────

INDEX_DIR       = Path("faiss_index")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-4o-mini")
TOP_K           = 4       # сколько чанков достаём из индекса
TEMPERATURE     = 0.1     # низкая температура = меньше фантазии


# ─── Промпты ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Ты ассистент базы знаний по вселенной TacticalStrike — \
киберспортивной игре. Отвечай ТОЛЬКО на основе предоставленных фрагментов контекста.

Правила:
- Если ответ есть в контексте — рассуждай пошагово, затем дай ответ.
- Если ответа НЕТ в контексте — ответь строго: \
"В базе знаний нет информации по этому вопросу."
- Никогда не используй знания за пределами предоставленного контекста.
- Всегда указывай, из какого источника взята информация.

Формат рассуждения (Chain-of-Thought):
1. Что ищу: ...
2. Что говорит контекст: ...
3. Ответ: ..."""


# Few-shot примеры — 2 образца запрос/ответ из той же предметной области.
# Показывают модели ожидаемый формат CoT-рассуждения.

FEW_SHOT = [
    {
        "role": "user",
        "content": (
            "Контекст:\n"
            "Zr0ne is a TacticalStrike 2 professional player from Vashen. "
            "He is widely regarded as one of the best players in the world. "
            "He currently plays for Nexus Victus.\n\n"
            "Вопрос: За какую команду играет Zr0ne?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "1. Что ищу: команду, за которую выступает Zr0ne.\n"
            "2. Что говорит контекст: «He currently plays for Nexus Victus.»\n"
            "3. Ответ: Zr0ne играет за команду Nexus Victus."
        ),
    },
    {
        "role": "user",
        "content": (
            "Контекст:\n"
            "EVL Pro Circuit is a venture between EVL and EXEA to provide "
            "a TacticalStrike league with significant prize money. "
            "It was launched in 2015.\n\n"
            "Вопрос: Кто организует EVL Pro Circuit?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "1. Что ищу: организатора EVL Pro Circuit.\n"
            "2. Что говорит контекст: «a venture between EVL and EXEA».\n"
            "3. Ответ: EVL Pro Circuit организован совместно компаниями EVL и EXEA."
        ),
    },
]


# ─── Результат запроса ───────────────────────────────────────────────────────

@dataclass
class RAGResult:
    answer: str
    sources: list[str]
    chunks_used: int
    query: str


# ─── Основной класс ──────────────────────────────────────────────────────────

class RAGBot:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY не задан в .env")

        if not INDEX_DIR.exists():
            raise FileNotFoundError(
                f"Индекс не найден: {INDEX_DIR}/\n"
                "Сначала запусти build_index.py"
            )

        print("Загрузка индекса...", end=" ", flush=True)
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)
        self.vectorstore = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("готово.")

        self.client = OpenAI(api_key=api_key)

    def search(self, query: str) -> list:
        """Ищет TOP_K ближайших чанков по запросу."""
        return self.vectorstore.similarity_search(query, k=TOP_K)

    def build_messages(self, query: str, docs: list) -> list[dict]:
        """Собирает список сообщений: system + few-shot + текущий запрос."""
        context = "\n\n---\n\n".join(
            f"[{doc.metadata['filename']}]\n{doc.page_content}"
            for doc in docs
        )

        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + FEW_SHOT
            + [
                {
                    "role": "user",
                    "content": f"Контекст:\n{context}\n\nВопрос: {query}",
                }
            ]
        )
        return messages

    def ask(self, query: str) -> RAGResult:
        """Полный пайплайн: поиск → промпт → генерация → результат."""
        docs = self.search(query)
        messages = self.build_messages(query, docs)

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
        )

        answer = response.choices[0].message.content.strip()
        sources = list({doc.metadata["filename"] for doc in docs})

        return RAGResult(
            answer=answer,
            sources=sources,
            chunks_used=len(docs),
            query=query,
        )


# ─── Быстрый тест без бота ───────────────────────────────────────────────────

if __name__ == "__main__":
    bot = RAGBot()

    test_queries = [
        "Who is Zr0ne and what team does he play for?",
        "What mouse does Vexis use?",
        "Tell me about the EVL Pro Circuit format",
        "What is the capital of Mars?",           # должен ответить «не знаю»
        "Who won the 2050 World Championship?",   # должен ответить «не знаю»
    ]

    for query in test_queries:
        print(f"\n{'─' * 50}")
        print(f"Запрос: {query}")
        result = bot.ask(query)
        print(f"Ответ:\n{result.answer}")
        print(f"Источники: {result.sources}")
