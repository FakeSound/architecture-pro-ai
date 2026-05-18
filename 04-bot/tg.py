"""
bot.py — Telegram-интерфейс для RAG-бота
=========================================
Зависимости:
    pip install python-telegram-bot

Запуск:
    python bot.py
"""

import os
import logging
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from rag import RAGBot

load_dotenv()
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)

# Инициализируем RAG один раз при старте
rag_bot = RAGBot()


# ─── Хэндлеры ────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я ассистент базы знаний вселенной TacticalStrike.\n\n"
        "Задай вопрос об игроках, командах или турнирах.\n"
        "Если информации нет — честно скажу об этом.\n\n"
        "Пример: *Кто такой Zr0ne?*",
        parse_mode="Markdown",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Примеры вопросов:\n"
        "• Кто такой Zr0ne?\n"
        "• За какую команду играет Vexis?\n"
        "• Какую мышь использует Sph1nx?\n"
        "• Расскажи об EVL Pro Circuit\n\n"
        "Бот отвечает только на основе базы знаний — не выдумывает."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.strip()
    if not query:
        return

    # Показываем индикатор печати пока думаем
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing",
    )

    try:
        result = rag_bot.ask(query)

        # Форматируем ответ
        text = result.answer

        if result.sources:
            sources_str = ", ".join(f"`{s}`" for s in sorted(result.sources))
            text += f"\n\n📚 Источники: {sources_str}"

        # Telegram ограничивает сообщения 4096 символами
        if len(text) > 4096:
            text = text[:4090] + "…"

        await update.message.reply_text(text, parse_mode="Markdown")

    except Exception as e:
        logging.error(f"Ошибка при обработке запроса: {e}")
        await update.message.reply_text(
            "Произошла ошибка при обработке запроса. Попробуй ещё раз."
        )


# ─── Запуск ──────────────────────────────────────────────────────────────────

def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise EnvironmentError("TELEGRAM_BOT_TOKEN не задан в .env")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Бот запущен. Ctrl+C для остановки.")
    app.run_polling()


if __name__ == "__main__":
    main()