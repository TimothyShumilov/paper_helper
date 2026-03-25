"""
Langfuse integration (optional).

If LANGFUSE_PUBLIC_KEY is not set, all functions are no-ops and the bot
operates normally without any observability.

Trace model:
  - One trace per session (id = session UUID)
  - Each user→assistant exchange logged as a generation
  - Rating logged as a score on the trace (1.0 = positive, 0.0 = negative)
"""
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_client = None  # lazy singleton


def get_langfuse():
    """Return the Langfuse singleton, or None if not configured."""
    global _client
    if _client is not None:
        return _client

    from config import settings
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None

    try:
        from langfuse import Langfuse
        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("Langfuse client initialised (host=%s)", settings.langfuse_host)
    except Exception as exc:
        logger.warning("Failed to initialise Langfuse: %s", exc)

    return _client


async def log_session_to_langfuse(
    session: dict,
    rating: int,
    messages: Optional[list[dict]],
) -> None:
    """
    Log a completed session to Langfuse.

    Args:
        session:  Session dict from db.get_session_by_id (has id, user_id, arxiv_id, paper_title).
        rating:   1 (positive) or -1 (negative).
        messages: Full conversation history from db.get_session_messages, or None to skip
                  logging message content (score only).
    """
    lf = get_langfuse()
    if lf is None:
        return

    try:
        trace_id = str(session["id"])
        score_value = 1.0 if rating == 1 else 0.0

        trace = lf.trace(
            id=trace_id,
            name="arxiv_qa_session",
            user_id=str(session["user_id"]),
            metadata={
                "arxiv_id": session["arxiv_id"],
                "paper_title": session.get("paper_title"),
            },
            input={"arxiv_id": session["arxiv_id"]},
        )

        if messages:
            # Log each user→assistant pair as a generation
            i = 0
            while i < len(messages):
                msg = messages[i]
                if msg["role"] == "user" and i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                    trace.generation(
                        name="qa",
                        input=msg["content"],
                        output=messages[i + 1]["content"],
                        metadata={"is_summary": messages[i + 1].get("is_summary", False)},
                    )
                    i += 2
                else:
                    i += 1

        lf.score(
            trace_id=trace_id,
            name="user_rating",
            value=score_value,
            comment="👍 Полезно" if rating == 1 else "👎 Не полезно",
        )

        await asyncio.to_thread(lf.flush)
        logger.debug("Langfuse: logged session %s (rating=%s, messages=%s)", trace_id, rating, bool(messages))

    except Exception as exc:
        logger.warning("Langfuse logging failed for session %s: %s", session.get("id"), exc)
