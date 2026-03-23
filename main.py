"""Entry point for the arXiv RAG Telegram bot."""
import asyncio
import logging
import uuid

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram.ext import Application

from config import settings
from core import db
from core.db import close_pool, expire_old_sessions, init_db
from core.embedder import warmup
from core.langfuse_client import log_session_to_langfuse
from core.vector_store import delete_session_chunks, init_collection
from bot.handlers import build_application
from bot.keyboards import rating_keyboard

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def cleanup_expired_sessions(app: Application) -> None:
    """Hourly job: expire stale sessions, remove Qdrant vectors, and handle Langfuse logging."""
    try:
        expired_sessions = await expire_old_sessions()
        for session in expired_sessions:
            session_id_str = str(session["id"])
            await delete_session_chunks(session_id_str)

            if session["rating"] is not None:
                # Rating already saved — log to Langfuse now
                include_messages = session.get("trace_consent") is True
                messages = None
                if include_messages:
                    messages = await db.get_session_messages(uuid.UUID(session_id_str))
                await log_session_to_langfuse(session, session["rating"], messages)
            else:
                # No rating yet — notify user and show rating keyboard
                title = session.get("paper_title") or session["arxiv_id"]
                try:
                    app.user_data.setdefault(session["user_id"], {})[
                        "last_ended_session_id"
                    ] = session_id_str
                    await app.bot.send_message(
                        chat_id=session["user_id"],
                        text=(
                            f"Ваша сессия по статье <b>{title}</b> истекла.\n"
                            "Как оцените работу бота в этой сессии?"
                        ),
                        parse_mode="HTML",
                        reply_markup=rating_keyboard(),
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not notify user %s about expired session: %s",
                        session["user_id"], exc,
                    )
    except Exception as exc:
        logger.error("Session cleanup error: %s", exc)


async def startup(app: Application) -> None:
    """Called by python-telegram-bot after the bot is initialized."""
    logger.info("Initializing database...")
    await init_db()
    logger.info("Initializing Qdrant collection...")
    await init_collection()
    logger.info("Warming up embedding model (this may take a while on first run)...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, warmup)
    logger.info("Startup complete. Bot is ready.")


async def shutdown(app: Application) -> None:
    """Called by python-telegram-bot before shutdown."""
    await close_pool()
    logger.info("Database pool closed.")


def main() -> None:
    app = build_application()
    app.post_init = startup
    app.post_shutdown = shutdown

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        cleanup_expired_sessions,
        trigger="interval",
        hours=1,
        id="session_cleanup",
        kwargs={"app": app},
    )
    scheduler.start()
    logger.info("Scheduler started (session cleanup every 1 hour).")

    app.run_polling(
        allowed_updates=["message", "callback_query"],
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
