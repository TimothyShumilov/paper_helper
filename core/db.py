"""PostgreSQL database layer using asyncpg."""
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

import asyncpg

from config import settings

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS sessions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       BIGINT NOT NULL,
    arxiv_id      VARCHAR(32) NOT NULL,
    paper_title   TEXT,
    status        VARCHAR(16) NOT NULL DEFAULT 'active',
    rating        SMALLINT,
    trace_consent BOOLEAN,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at      TIMESTAMPTZ,
    expires_at    TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '24 hours'),
    CONSTRAINT chk_status CHECK (status IN ('active', 'ended', 'expired')),
    CONSTRAINT chk_rating CHECK (rating IN (-1, 1))
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id    ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_sessions_status     ON sessions(status);

CREATE TABLE IF NOT EXISTS messages (
    id          BIGSERIAL PRIMARY KEY,
    session_id  UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        VARCHAR(16) NOT NULL,
    content     TEXT NOT NULL,
    is_summary  BOOLEAN NOT NULL DEFAULT FALSE,
    token_count INT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_role CHECK (role IN ('user', 'assistant', 'system'))
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
"""


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=settings.postgres_dsn,
            min_size=2,
            max_size=10,
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def init_db() -> None:
    """Create tables if they don't exist. Called once at startup."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)
        # Migration: add trace_consent column if it doesn't exist (for existing DBs)
        await conn.execute(
            "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS trace_consent BOOLEAN;"
        )
    logger.info("Database schema initialized.")


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

async def create_session(
    user_id: int,
    arxiv_id: str,
    paper_title: Optional[str] = None,
) -> dict:
    """Create a new session and return the row as a dict."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO sessions (user_id, arxiv_id, paper_title)
            VALUES ($1, $2, $3)
            RETURNING id, user_id, arxiv_id, paper_title, status,
                      rating, created_at, ended_at, expires_at
            """,
            user_id,
            arxiv_id,
            paper_title,
        )
    return dict(row)


async def get_active_session(user_id: int) -> Optional[dict]:
    """Return the most recent active (non-expired) session for a user."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, user_id, arxiv_id, paper_title, status,
                   rating, created_at, ended_at, expires_at
            FROM sessions
            WHERE user_id = $1
              AND status = 'active'
              AND expires_at > NOW()
            ORDER BY created_at DESC
            LIMIT 1
            """,
            user_id,
        )
    return dict(row) if row else None


async def get_session_by_id(session_id: uuid.UUID) -> Optional[dict]:
    """Return a session row by its UUID."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, user_id, arxiv_id, paper_title, status,
                   rating, created_at, ended_at, expires_at
            FROM sessions WHERE id = $1
            """,
            session_id,
        )
    return dict(row) if row else None


async def get_session_by_arxiv_id(user_id: int, arxiv_id: str) -> Optional[dict]:
    """Return the most recent non-expired session for this user and paper, or None."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, user_id, arxiv_id, paper_title, status,
                   rating, created_at, ended_at, expires_at
            FROM sessions
            WHERE user_id = $1 AND arxiv_id = $2 AND status != 'expired'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            user_id,
            arxiv_id,
        )
    return dict(row) if row else None


async def get_recent_sessions(user_id: int, hours: int = 24) -> list[dict]:
    """Return all sessions within the TTL window (regardless of status)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, arxiv_id, paper_title, status,
                   rating, created_at, ended_at, expires_at
            FROM sessions
            WHERE user_id = $1
              AND created_at > NOW() - ($2 || ' hours')::INTERVAL
              AND status != 'expired'
            ORDER BY created_at DESC
            LIMIT 10
            """,
            user_id,
            str(hours),
        )
    return [dict(r) for r in rows]


async def end_session(session_id: uuid.UUID) -> None:
    """Mark session as ended."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE sessions SET status='ended', ended_at=NOW() WHERE id=$1",
            session_id,
        )


async def rate_session(session_id: uuid.UUID, rating: int) -> None:
    """Store user rating (1 or -1)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE sessions SET rating=$1 WHERE id=$2",
            rating,
            session_id,
        )


async def set_trace_consent(session_id: uuid.UUID, consent: bool) -> None:
    """Store user consent to save conversation history for Langfuse tracing."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE sessions SET trace_consent=$1 WHERE id=$2",
            consent,
            session_id,
        )


async def expire_old_sessions() -> list[dict]:
    """
    Mark expired sessions and return their full rows (for Qdrant cleanup + Langfuse logging).
    Called by the hourly scheduler.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            UPDATE sessions
            SET status = 'expired'
            WHERE expires_at < NOW()
              AND status = 'active'
            RETURNING id, user_id, rating, trace_consent, arxiv_id, paper_title
            """
        )
    expired = [dict(r) for r in rows]
    if expired:
        logger.info("Expired %d sessions.", len(expired))
    return expired


# ---------------------------------------------------------------------------
# Message CRUD
# ---------------------------------------------------------------------------

async def add_message(
    session_id: uuid.UUID,
    role: str,
    content: str,
    is_summary: bool = False,
    token_count: Optional[int] = None,
) -> int:
    """Insert a message and return its id."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO messages (session_id, role, content, is_summary, token_count)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
            """,
            session_id,
            role,
            content,
            is_summary,
            token_count,
        )
    return row["id"]


async def get_session_messages(session_id: uuid.UUID) -> list[dict]:
    """Return all messages for a session ordered by creation time."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, session_id, role, content, is_summary, token_count, created_at
            FROM messages
            WHERE session_id = $1
            ORDER BY created_at ASC
            """,
            session_id,
        )
    return [dict(r) for r in rows]


async def get_total_token_count(session_id: uuid.UUID) -> int:
    """Sum of estimated token_count for all messages in a session."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT COALESCE(SUM(token_count), 0) AS total FROM messages WHERE session_id=$1",
            session_id,
        )
    return row["total"]


async def replace_history_with_summary(
    session_id: uuid.UUID,
    summary_content: str,
    summary_token_count: int,
) -> None:
    """
    Delete all existing messages for a session and insert a single
    assistant summary message. Used for context window compression.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM messages WHERE session_id=$1", session_id
            )
            await conn.execute(
                """
                INSERT INTO messages (session_id, role, content, is_summary, token_count)
                VALUES ($1, 'assistant', $2, TRUE, $3)
                """,
                session_id,
                summary_content,
                summary_token_count,
            )
