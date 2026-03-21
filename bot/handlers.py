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
from bot.keyboards import (
    CB_RATE_DOWN,
    CB_RATE_UP,
    CB_RETURN_PREFIX,
    CB_SESSIONS_PAGE,
    new_session_keyboard,
    rating_keyboard,
)

logger = logging.getLogger(__name__)

# ConversationHandler states
IDLE, READY, SESSION_ENDED = range(3)

COMMANDS_HINT = "\n\nКоманды: /summarize · /new"

WELCOME_TEXT = (
    "Привет! Я научный ассистент для работы со статьями arXiv.\n\n"
    "Что я умею:\n"
    "• Скачать и проиндексировать статью по её ID\n"
    "• Ответить на вопросы по содержанию статьи\n"
    "• Создать краткое структурированное изложение (/summarize)\n\n"
    "Отправь мне ID статьи (например: <code>2301.07041</code>) или ссылку на неё."
)

SESSION_PICKER_TEXT = "Выберите сессию для продолжения или отправьте ID новой статьи:"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session_id(context: ContextTypes.DEFAULT_TYPE) -> str | None:
    return context.user_data.get("session_id")


async def _send_typing(update: Update) -> None:
    await update.effective_chat.send_action(ChatAction.TYPING)


async def _show_session_picker(target, user_id: int, page: int = 0) -> None:
    """Send or edit a message showing the paginated session list."""
    sessions = await db.get_recent_sessions(user_id)
    if sessions:
        keyboard = new_session_keyboard(sessions, page=page)
        if hasattr(target, "edit_message_text"):
            await target.edit_message_text(SESSION_PICKER_TEXT, reply_markup=keyboard)
        else:
            await target.reply_text(SESSION_PICKER_TEXT, reply_markup=keyboard)
    else:
        if hasattr(target, "edit_message_text"):
            await target.edit_message_text(WELCOME_TEXT, parse_mode="HTML")
        else:
            await target.reply_text(WELCOME_TEXT, parse_mode="HTML")


# ---------------------------------------------------------------------------
# /start
# ---------------------------------------------------------------------------

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    active = await db.get_active_session(user_id)
    if active:
        context.user_data["session_id"] = str(active["id"])
        await update.message.reply_text(
            f"У вас есть активная сессия по статье <b>{active['arxiv_id']}</b>.\n"
            "Можете продолжать задавать вопросы или выбрать команду."
            + COMMANDS_HINT,
            parse_mode="HTML",
        )
        return READY

    await update.message.reply_text(WELCOME_TEXT, parse_mode="HTML")
    return IDLE


# ---------------------------------------------------------------------------
# /new — end current session (if any), ask for rating, then show session picker
# ---------------------------------------------------------------------------

async def new_session_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    session_id = _session_id(context)
    user_id = update.effective_user.id

    if session_id:
        # End current session and ask for rating
        try:
            await db.end_session(uuid.UUID(session_id))
        except Exception as exc:
            logger.warning("Could not end session %s: %s", session_id, exc)
        context.user_data.pop("session_id", None)
        await update.message.reply_text(
            "Сессия завершена. Как оцените работу бота в этой сессии?",
            reply_markup=rating_keyboard(),
        )
        return SESSION_ENDED

    # No current session — show picker immediately
    await _show_session_picker(update.message, user_id)
    return IDLE


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

    processing_msg = await update.message.reply_text(
        f"ID получен: <b>{arxiv_id}</b>. Начинаю загрузку и обработку статьи...\n"
        "(это может занять несколько минут)",
        parse_mode="HTML",
    )

    return await _process_paper(
        processing_msg,
        update.effective_user.id,
        arxiv_id,
        context,
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
        recent = await db.get_recent_sessions(user_id)
        ended = [s for s in recent if s["status"] == "ended" and s["rating"] is None]
        if ended:
            try:
                await db.rate_session(ended[0]["id"], rating)
            except Exception as exc:
                logger.warning("Could not save rating: %s", exc)

        # Show session picker after rating
        sessions = await db.get_recent_sessions(user_id)
        if sessions:
            await query.edit_message_text(
                "Спасибо за оценку!\n\n" + SESSION_PICKER_TEXT,
                reply_markup=new_session_keyboard(sessions),
            )
        else:
            await query.edit_message_text(
                "Спасибо за оценку!\n\n" + WELCOME_TEXT,
                parse_mode="HTML",
            )
        return SESSION_ENDED

    # --- Session page navigation ---
    if data.startswith(CB_SESSIONS_PAGE):
        try:
            page = int(data[len(CB_SESSIONS_PAGE):])
        except ValueError:
            return IDLE
        sessions = await db.get_recent_sessions(user_id)
        await query.edit_message_reply_markup(
            reply_markup=new_session_keyboard(sessions, page=page)
        )
        return IDLE

    # --- Return to session ---
    if data.startswith(CB_RETURN_PREFIX):
        sid_str = data[len(CB_RETURN_PREFIX):]
        try:
            sid = uuid.UUID(sid_str)
        except ValueError:
            await query.edit_message_text("Неверный идентификатор сессии.")
            return SESSION_ENDED

        session = await db.get_session_by_id(sid)
        if not session:
            await query.edit_message_text("Сессия не найдена.")
            return SESSION_ENDED

        if session["status"] == "expired":
            await query.edit_message_text(
                "Эта сессия истекла (>24 часов). Отправьте ID статьи, чтобы начать новую."
            )
            return IDLE

        # Reactivate ended session
        if session["status"] == "ended":
            from core.db import get_pool
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE sessions SET status='active', ended_at=NULL WHERE id=$1",
                    sid,
                )

        context.user_data["session_id"] = str(sid)
        messages = await db.get_session_messages(sid)
        history_preview = ""
        if messages:
            last = messages[-2:] if len(messages) >= 2 else messages
            for m in last:
                role_label = "Вы" if m["role"] == "user" else "Бот"
                preview = m["content"][:200] + ("..." if len(m["content"]) > 200 else "")
                history_preview += f"\n<b>{role_label}:</b> {preview}"

        await query.edit_message_text(
            f"Возврат к сессии по статье <b>{session['arxiv_id']}</b>.\n"
            f"{history_preview}\n\n"
            "Продолжайте задавать вопросы или используйте команды:"
            + COMMANDS_HINT,
            parse_mode="HTML",
        )
        return READY

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
        BotCommand("new",        "Завершить сессию и выбрать следующую"),
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
            MessageHandler(filters.TEXT & ~filters.COMMAND, arxiv_id_handler),
        ],
        states={
            IDLE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, arxiv_id_handler),
                CallbackQueryHandler(callback_handler),
            ],
            READY: [
                CommandHandler("summarize", summarize_handler),
                CommandHandler("new", new_session_handler),
                CallbackQueryHandler(callback_handler),
                MessageHandler(filters.TEXT & ~filters.COMMAND, question_handler),
            ],
            SESSION_ENDED: [
                CallbackQueryHandler(callback_handler),
                CommandHandler("new", new_session_handler),
                MessageHandler(filters.TEXT & ~filters.COMMAND, arxiv_id_handler),
            ],
        },
        fallbacks=[
            CommandHandler("start", start_handler),
            CommandHandler("new", new_session_handler),
        ],
    )

    app.add_handler(conv_handler)
    app.add_error_handler(error_handler)

    return app
