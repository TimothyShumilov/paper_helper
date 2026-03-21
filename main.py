"""Entry point for the arXiv RAG Telegram bot."""
import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram.ext import Application

from config import settings
from core.db import close_pool, expire_old_sessions, init_db
from core.embedder import warmup
from core.vector_store import delete_session_chunks, init_collection
from bot.handlers import build_application

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def cleanup_expired_sessions() -> None:
    """Hourly job: expire stale sessions and remove their Qdrant vectors."""
    try:
        expired_ids = await expire_old_sessions()
        for session_id in expired_ids:
            await delete_session_chunks(session_id)
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
    )
    scheduler.start()
    logger.info("Scheduler started (session cleanup every 1 hour).")

    app.run_polling(
        allowed_updates=["message", "callback_query"],
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
