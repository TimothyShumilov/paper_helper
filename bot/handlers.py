"""
Telegram bot handlers.

Conversation state machine:
  IDLE           — no active session; waiting for arXiv ID
  READY          — paper loaded; Q&A and summarization available
  SESSION_ENDED  — session closed; showing rating then session picker

Commands (registered via set_my_commands):
  /start       — начать / проверить активную сессию
  /summarize   — суммаризировать текущую статью
  /new         — завершить сессию и выбрать следующую
"""
import logging
import uuid
from pathlib import Path

from telegram import BotCommand, Update, Message
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ConversationHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import settings
from core import arxiv_loader, pdf_parser, chunker, embedder, vector_store
from core import db, rag_pipeline
from core.langfuse_client import log_session_to_langfuse
from bot.keyboards import (
    CB_ALLOW_TRACE,
    CB_DENY_TRACE,
    CB_RATE_DOWN,
    CB_RATE_UP,
    rating_keyboard,
    trace_permission_keyboard,
)

logger = logging.getLogger(__name__)

# ConversationHandler states
IDLE, READY, SESSION_ENDED = range(3)

COMMANDS_HINT = "\n\nПодсказка: /help — список доступных команд"

HELP_TEXT = (
    "<b>Доступные команды:</b>\n\n"
    "/start — начать работу или вернуться к активной сессии\n"
    "/summarize — краткое изложение текущей статьи\n"
    "/new [id/ссылка] — завершить сессию (без аргумента) или открыть статью по ID или ссылке arXiv\n"
    "/help — показать этот список"
)

WELCOME_TEXT = (
    "Привет! Я научный ассистент для работы со статьями arXiv.\n\n"
    "Что я умею:\n"
    "• Скачать и проиндексировать статью по её ID\n"
    "• Ответить на вопросы по содержанию статьи — просто напишите ваш вопрос в диалог\n"
    "• Создать краткое структурированное изложение (/summarize)\n\n"
    "Отправь мне ID статьи (например: <code>2301.07041</code>) или ссылку на неё.\n\n"
    "Полный список команд — /help"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session_id(context: ContextTypes.DEFAULT_TYPE) -> str | None:
    return context.user_data.get("session_id")


async def _send_typing(update: Update) -> None:
    await update.effective_chat.send_action(ChatAction.TYPING)


# ---------------------------------------------------------------------------
# /start
# ---------------------------------------------------------------------------

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    active = await db.get_active_session(user_id)
    if active:
        context.user_data["session_id"] = str(active["id"])
        title = active.get("paper_title") or active["arxiv_id"]
        await update.message.reply_text(
            f"Активна сессия по статье <b>{title}</b>.\n"
            "Задавайте вопросы или используйте <code>/new &lt;id&gt;</code> для переключения."
            + COMMANDS_HINT,
            parse_mode="HTML",
        )
        return READY

    await update.message.reply_text(WELCOME_TEXT, parse_mode="HTML")
    return IDLE


# ---------------------------------------------------------------------------
# /help
# ---------------------------------------------------------------------------

async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(HELP_TEXT, parse_mode="HTML")
    return READY if _session_id(context) else IDLE


# ---------------------------------------------------------------------------
# /new [arxiv_id] — open paper or end session
# ---------------------------------------------------------------------------

async def new_session_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    args = context.args or []
    raw = args[0] if args else None
    session_id = _session_id(context)
    user_id = update.effective_user.id

    if raw:
        arxiv_id = arxiv_loader.validate_arxiv_id(raw)
        if arxiv_id is None:
            await update.message.reply_text(
                "Не могу распознать ID статьи. Примеры:\n"
                "<code>/new 2301.07041</code>\n"
                "<code>/new https://arxiv.org/abs/2301.07041</code>",
                parse_mode="HTML",
            )
            return SESSION_ENDED if session_id else IDLE

        if session_id:
            # End current session and ask for rating; open new paper after rating
            try:
                await db.end_session(uuid.UUID(session_id))
            except Exception as exc:
                logger.warning("Could not end session %s: %s", session_id, exc)
            context.user_data["last_ended_session_id"] = session_id
            context.user_data["pending_arxiv_id"] = arxiv_id
            context.user_data.pop("session_id", None)
            await update.message.reply_text(
                "Сессия завершена. Как оцените работу бота в этой сессии?",
                reply_markup=rating_keyboard(),
            )
            return SESSION_ENDED

        # No active session — open paper directly
        return await _open_by_arxiv_id(context, user_id, arxiv_id, update.message.reply_text)

    # /new without args: end session and ask for rating
    if session_id:
        try:
            await db.end_session(uuid.UUID(session_id))
        except Exception as exc:
            logger.warning("Could not end session %s: %s", session_id, exc)
        context.user_data["last_ended_session_id"] = session_id
        context.user_data.pop("session_id", None)
        await update.message.reply_text(
            "Сессия завершена. Как оцените работу бота в этой сессии?",
            reply_markup=rating_keyboard(),
        )
        return SESSION_ENDED

    await update.message.reply_text(WELCOME_TEXT, parse_mode="HTML")
    return IDLE


# ---------------------------------------------------------------------------
# Shared helper: open paper by validated arxiv_id
# ---------------------------------------------------------------------------

async def _open_by_arxiv_id(
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    arxiv_id: str,
    reply_fn,
) -> int:
    """Resume existing session or start ingestion for the given arxiv_id.

    reply_fn — async callable(text, **kwargs) -> Message (e.g. update.message.reply_text
    or context.bot.send_message with chat_id pre-bound).
    """
    existing = await db.get_session_by_arxiv_id(user_id, arxiv_id)
    if existing:
        if existing["status"] == "ended":
            pool = await db.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE sessions SET status='active', ended_at=NULL WHERE id=$1",
                    existing["id"],
                )
        context.user_data["session_id"] = str(existing["id"])
        title = existing.get("paper_title") or arxiv_id
        await reply_fn(
            f"Найдена существующая сессия по статье <b>{title}</b>.\n"
            "Продолжаем работу с ней." + COMMANDS_HINT,
            parse_mode="HTML",
        )
        return READY

    processing_msg = await reply_fn(
        f"ID получен: <b>{arxiv_id}</b>. Начинаю загрузку и обработку статьи...\n"
        "(это может занять несколько минут)",
        parse_mode="HTML",
    )
    return await _process_paper(processing_msg, user_id, arxiv_id, context)


# ---------------------------------------------------------------------------
# arXiv ID input (state: IDLE / SESSION_ENDED)
# ---------------------------------------------------------------------------

async def arxiv_id_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    arxiv_id = arxiv_loader.validate_arxiv_id(text)

    if arxiv_id is None:
        await update.message.reply_text(
            "Не могу распознать ID статьи. Пожалуйста, отправьте ID в формате "
            "<code>2312.12456</code> или ссылку вида "
            "<code>https://arxiv.org/abs/2312.12456</code>.",
            parse_mode="HTML",
        )
        return IDLE

    return await _open_by_arxiv_id(
        context, update.effective_user.id, arxiv_id, update.message.reply_text
    )


async def _process_paper(
    processing_msg: Message,
    user_id: int,
    arxiv_id: str,
    context: ContextTypes.DEFAULT_TYPE,
) -> int:
    """Full ingestion pipeline. Returns READY on success, IDLE on failure."""
    session_record = None
    try:
        # 1. Fetch metadata
        await processing_msg.edit_text(
            f"<b>{arxiv_id}</b>: получаю метаданные...", parse_mode="HTML"
        )
        metadata = await arxiv_loader.fetch_paper_metadata(arxiv_id)
        title = metadata["title"]

        # 2. Create session in DB
        session_record = await db.create_session(user_id, arxiv_id, title)
        session_id = str(session_record["id"])

        # 3. Download PDF
        await processing_msg.edit_text(
            f"<b>{arxiv_id}</b>: скачиваю PDF...", parse_mode="HTML"
        )
        pdf_path = await arxiv_loader.download_pdf(arxiv_id)

        # 4. Extract text
        await processing_msg.edit_text(
            f"<b>{arxiv_id}</b>: извлекаю текст...", parse_mode="HTML"
        )
        pages = pdf_parser.extract_text_from_pdf(pdf_path)
        if not pages:
            raise ValueError("Не удалось извлечь текст из PDF. Возможно, статья содержит только изображения.")

        # 5. Chunk
        await processing_msg.edit_text(
            f"<b>{arxiv_id}</b>: разбиваю на фрагменты...", parse_mode="HTML"
        )
        chunks = chunker.chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)

        # 6. Embed
        await processing_msg.edit_text(
            f"<b>{arxiv_id}</b>: векторизую {len(chunks)} фрагментов...", parse_mode="HTML"
        )
        texts = [c.text for c in chunks]
        embeddings = await embedder.embed_texts(texts)
        page_nums = [c.page_num for c in chunks]

        # 7. Upsert to Qdrant
        await vector_store.upsert_chunks(
            session_id=session_id,
            user_id=user_id,
            arxiv_id=arxiv_id,
            texts=texts,
            embeddings=embeddings,
            page_nums=page_nums,
        )

        # 8. Save context to bot state
        context.user_data["session_id"] = session_id

        # 9. Notify user
        authors_str = ", ".join(metadata.get("authors", [])[:3])
        if len(metadata.get("authors", [])) > 3:
            authors_str += " и др."
        await processing_msg.edit_text(
            f"Готово! Статья загружена и проиндексирована.\n\n"
            f"<b>{title}</b>\n"
            f"Авторы: {authors_str}\n"
            f"Опубликована: {metadata.get('published', '')}\n\n"
            f"Фрагментов: {len(chunks)}\n\n"
            "Задайте вопрос по статье или используйте команды:"
            + COMMANDS_HINT,
            parse_mode="HTML",
        )
        return READY

    except Exception as exc:
        logger.exception("Error processing paper %s: %s", arxiv_id, exc)
        if session_record:
            try:
                await db.end_session(uuid.UUID(str(session_record["id"])))
            except Exception:
                pass
        await processing_msg.edit_text(
            f"Произошла ошибка при обработке статьи <b>{arxiv_id}</b>:\n"
            f"<code>{str(exc)[:300]}</code>\n\n"
            "Пожалуйста, проверьте ID статьи и попробуйте снова.",
            parse_mode="HTML",
        )
        return IDLE


# ---------------------------------------------------------------------------
# Q&A (state: READY)
# ---------------------------------------------------------------------------

async def question_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    session_id = _session_id(context)
    if not session_id:
        await update.message.reply_text(
            "Нет активной сессии. Отправьте ID статьи, чтобы начать."
        )
        return IDLE

    session = await db.get_session_by_id(uuid.UUID(session_id))
    if not session or session["status"] != "active":
        context.user_data.pop("session_id", None)
        await update.message.reply_text(
            "Сессия завершена или истекла. Отправьте новый ID статьи."
        )
        return IDLE

    await _send_typing(update)
    question = update.message.text.strip()

    try:
        response = await rag_pipeline.answer_question(
            session_id, update.effective_user.id, question
        )
        await update.message.reply_text(response)
    except Exception as exc:
        logger.exception("RAG error: %s", exc)
        await update.message.reply_text(
            "Произошла ошибка при генерации ответа. Попробуйте ещё раз."
        )

    return READY


# ---------------------------------------------------------------------------
# /summarize
# ---------------------------------------------------------------------------

async def summarize_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    session_id = _session_id(context)
    if not session_id:
        await update.message.reply_text("Нет активной сессии.")
        return IDLE

    await _send_typing(update)
    try:
        summary = await rag_pipeline.summarize_paper(
            session_id, update.effective_user.id
        )
        await update.message.reply_text(summary)
    except Exception as exc:
        logger.exception("Summarization error: %s", exc)
        await update.message.reply_text(
            "Не удалось сформировать краткое изложение. Попробуйте ещё раз."
        )
    return READY


# ---------------------------------------------------------------------------
# Callback query handler (rating + session navigation)
# ---------------------------------------------------------------------------

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data
    user_id = update.effective_user.id

    # --- Rating ---
    if data in (CB_RATE_UP, CB_RATE_DOWN):
        rating = 1 if data == CB_RATE_UP else -1

        # Find session to rate: prefer the one we just ended, else search recent
        sid_str = context.user_data.pop("last_ended_session_id", None)
        session_to_rate = None
        if sid_str:
            session_to_rate = await db.get_session_by_id(uuid.UUID(sid_str))
        if session_to_rate is None:
            recent = await db.get_recent_sessions(user_id)
            ended = [s for s in recent if s["status"] in ("ended", "expired") and s["rating"] is None]
            session_to_rate = ended[0] if ended else None

        if session_to_rate:
            try:
                await db.rate_session(session_to_rate["id"], rating)
            except Exception as exc:
                logger.warning("Could not save rating: %s", exc)

        if data == CB_RATE_UP:
            # Rating saved to DB. If session already expired, log to Langfuse immediately.
            if session_to_rate:
                refreshed = await db.get_session_by_id(session_to_rate["id"])
                if refreshed and refreshed["status"] == "expired":
                    await log_session_to_langfuse(refreshed, rating, None)

            pending_arxiv_id = context.user_data.pop("pending_arxiv_id", None)
            if pending_arxiv_id:
                await query.edit_message_text("Спасибо за оценку!", parse_mode="HTML")
                chat_id = query.message.chat_id
                reply_fn = lambda text, **kw: context.bot.send_message(
                    chat_id=chat_id, text=text, **kw
                )
                return await _open_by_arxiv_id(context, user_id, pending_arxiv_id, reply_fn)

            await query.edit_message_text(
                "Спасибо за оценку!\n\nОтправьте ID статьи или используйте "
                "<code>/new 2301.07041</code> для открытия статьи.",
                parse_mode="HTML",
            )
            return IDLE

        else:
            # Negative: ask permission to save conversation history.
            # pending_arxiv_id stays in user_data until after consent.
            if session_to_rate:
                context.user_data["pending_session_id"] = str(session_to_rate["id"])
            context.user_data["pending_rating"] = rating
            await query.edit_message_text(
                "Разрешаете сохранить историю этого диалога для улучшения качества бота?",
                reply_markup=trace_permission_keyboard(),
            )
            return SESSION_ENDED

    # --- Trace permission (after negative rating) ---
    if data in (CB_ALLOW_TRACE, CB_DENY_TRACE):
        pending_sid = context.user_data.pop("pending_session_id", None)
        pending_rating = context.user_data.pop("pending_rating", -1)
        pending_arxiv_id = context.user_data.pop("pending_arxiv_id", None)

        if pending_sid:
            consent = (data == CB_ALLOW_TRACE)
            try:
                await db.set_trace_consent(uuid.UUID(pending_sid), consent)
            except Exception as exc:
                logger.warning("Could not save trace consent: %s", exc)

            # If session already expired, log to Langfuse immediately.
            session = await db.get_session_by_id(uuid.UUID(pending_sid))
            if session and session["status"] == "expired":
                messages = await db.get_session_messages(session["id"]) if consent else None
                await log_session_to_langfuse(session, pending_rating, messages)

        if pending_arxiv_id:
            await query.edit_message_text("Спасибо!", parse_mode="HTML")
            chat_id = query.message.chat_id
            reply_fn = lambda text, **kw: context.bot.send_message(
                chat_id=chat_id, text=text, **kw
            )
            return await _open_by_arxiv_id(context, user_id, pending_arxiv_id, reply_fn)

        await query.edit_message_text(
            "Спасибо!\n\nОтправьте ID статьи или используйте "
            "<code>/new 2301.07041</code> для открытия статьи.",
            parse_mode="HTML",
        )
        return IDLE

    # Unknown callback
    logger.warning("Unknown callback data: %s", data)
    return IDLE


# ---------------------------------------------------------------------------
# Error handler
# ---------------------------------------------------------------------------

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled exception:", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "Произошла непредвиденная ошибка. Попробуйте снова или отправьте /start."
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Application builder
# ---------------------------------------------------------------------------

async def _set_commands(app: Application) -> None:
    await app.bot.set_my_commands([
        BotCommand("start",      "Начать / проверить активную сессию"),
        BotCommand("summarize",  "Суммаризировать текущую статью"),
        BotCommand("new",        "Открыть статью: /new <ID или ссылку>"),
        BotCommand("help",       "Показать список команд"),
    ])


def build_application() -> Application:
    app = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .post_init(_set_commands)
        .build()
    )

    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("start", start_handler),
            CommandHandler("new", new_session_handler),
            CommandHandler("help", help_handler),
            MessageHandler(filters.TEXT & ~filters.COMMAND, arxiv_id_handler),
        ],
        states={
            IDLE: [
                CommandHandler("new", new_session_handler),
                CommandHandler("help", help_handler),
                MessageHandler(filters.TEXT & ~filters.COMMAND, arxiv_id_handler),
                CallbackQueryHandler(callback_handler),
            ],
            READY: [
                CommandHandler("summarize", summarize_handler),
                CommandHandler("new", new_session_handler),
                CommandHandler("help", help_handler),
                CallbackQueryHandler(callback_handler),
                MessageHandler(filters.TEXT & ~filters.COMMAND, question_handler),
            ],
            SESSION_ENDED: [
                CallbackQueryHandler(callback_handler),
                CommandHandler("new", new_session_handler),
                CommandHandler("help", help_handler),
                MessageHandler(filters.TEXT & ~filters.COMMAND, arxiv_id_handler),
            ],
        },
        fallbacks=[
            CommandHandler("start", start_handler),
            CommandHandler("new", new_session_handler),
            CommandHandler("help", help_handler),
        ],
    )

    app.add_handler(conv_handler)
    app.add_error_handler(error_handler)

    return app
