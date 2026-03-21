"""
RAG pipeline orchestrator.

Handles question answering, paper summarization, and
conversation history management (including compression
when the context window approaches its limit).
"""
import logging
import uuid
from typing import Optional

from groq import AsyncGroq

from config import settings
from core.db import (
    add_message,
    get_session_messages,
    get_total_token_count,
    replace_history_with_summary,
)
from core.embedder import embed_query, embed_texts
from core.vector_store import search_chunks

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты научный ассистент. Отвечай на вопросы пользователя ТОЛЬКО на основе "
    "предоставленных выдержек из статьи. Если ответа нет в тексте, скажи: "
    "'В статье нет информации об этом.' Используй академический, вежливый стиль. "
    "Предупреди пользователя, что ответы могут содержать ошибки, особенно "
    "в сложной математике, и требуют проверки."
)

SUMMARIZE_QUERY = (
    "цель задачи методология датасет архитектура модели результаты выводы ограничения"
)

SUMMARIZE_PROMPT_PREFIX = (
    "Создай структурированное краткое изложение данной научной статьи. "
    "Включи: (1) Цель и задачи, (2) Методология и данные, "
    "(3) Ключевые результаты, (4) Выводы и ограничения. "
    "Опирайся строго на предоставленные выдержки.\n\n"
)

COMPRESS_PROMPT = (
    "Сделай краткую выжимку этого диалога, сохранив ключевые факты и "
    "текущий предмет обсуждения. Это резюме будет использовано для продолжения диалога."
)

HISTORY_COMPRESS_THRESHOLD = 0.95  # compress when history reaches 95% of limit
# ~6 000 tokens of content per batch — safe under Groq free tier 12 000 TPM limit
SUMMARIZE_BATCH_CHAR_LIMIT = 24_000


def _batch_chunks(chunks: list[dict], char_limit: int) -> list[list[dict]]:
    """Split chunks into batches where each batch's combined text fits within char_limit."""
    batches, current, current_len = [], [], 0
    for chunk in chunks:
        n = len(chunk["text"])
        if current and current_len + n > char_limit:
            batches.append(current)
            current, current_len = [chunk], n
        else:
            current.append(chunk)
            current_len += n
    if current:
        batches.append(current)
    return batches


def _estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters (conservative for RU/EN mix)."""
    return max(1, len(text) // 4)


def _build_context_string(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Выдержка {i} (стр. {chunk.get('page_num', '?')})]:\n{chunk['text']}")
    return "\n\n".join(parts)


def _build_messages(
    history: list[dict],
    context_string: str,
    user_content: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict]:
    """
    Assemble the messages list for the OpenRouter API:
    [system] + [history] + [user with context]
    """
    messages = [{"role": "system", "content": system_prompt}]

    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    user_message = (
        f"[Извлечённая информация из статьи:]\n{context_string}\n\n{user_content}"
    )
    messages.append({"role": "user", "content": user_message})
    return messages


async def _call_groq(messages: list[dict]) -> str:
    """Call Groq Chat Completions API and return response text."""
    client = AsyncGroq(api_key=settings.groq_api_key)
    completion = await client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
    )
    return completion.choices[0].message.content


async def _compress_history(session_id: uuid.UUID) -> None:
    """
    Compress conversation history when it approaches the context limit.
    Replaces all messages with a single assistant summary message.
    """
    logger.info("Compressing history for session %s.", session_id)
    messages = await get_session_messages(session_id)
    if not messages:
        return

    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )
    compress_messages = [
        {"role": "system", "content": COMPRESS_PROMPT},
        {"role": "user", "content": transcript},
    ]
    summary = await _call_groq(compress_messages)
    summary_text = (
        "[Краткое изложение предыдущего диалога — используй при ответах]\n"
        + summary
    )
    await replace_history_with_summary(
        session_id,
        summary_text,
        _estimate_tokens(summary_text),
    )
    logger.info("History compressed for session %s.", session_id)


async def answer_question(
    session_id: str,
    user_id: int,
    question: str,
) -> str:
    """
    Full RAG pipeline for a user question:
    1. Embed query
    2. Retrieve top-k chunks from Qdrant
    3. Load conversation history from DB
    4. Optionally compress history if context budget is exceeded
    5. Call LLM via OpenRouter
    6. Persist messages to DB
    7. Return response text
    """
    sid = uuid.UUID(session_id)

    # 1. Embed query and retrieve chunks
    query_vector = await embed_query(question)
    chunks = await search_chunks(session_id, query_vector, top_k=settings.top_k_chunks)

    if not chunks:
        return "Не удалось найти релевантные фрагменты статьи. Попробуйте переформулировать вопрос."

    # 2. Load history
    history_rows = await get_session_messages(sid)
    history = [{"role": r["role"], "content": r["content"]} for r in history_rows]

    # 3. Check context budget and compress if needed
    total_tokens = await get_total_token_count(sid)
    context_tokens = _estimate_tokens(_build_context_string(chunks)) + _estimate_tokens(question)
    if (total_tokens + context_tokens) > settings.max_history_tokens * HISTORY_COMPRESS_THRESHOLD:
        await _compress_history(sid)
        history_rows = await get_session_messages(sid)
        history = [{"role": r["role"], "content": r["content"]} for r in history_rows]

    # 4. Build messages and call LLM
    context_string = _build_context_string(chunks)
    messages = _build_messages(history, context_string, f"Вопрос: {question}")
    response = await _call_groq(messages)

    # 5. Save to DB
    await add_message(sid, "user", question, token_count=_estimate_tokens(question))
    await add_message(sid, "assistant", response, token_count=_estimate_tokens(response))

    return response


async def summarize_paper(session_id: str, user_id: int) -> str:
    """
    Retrieval-augmented summarization.
    Uses a broad query to fetch a wider coverage of chunks (top_k=15).
    """
    sid = uuid.UUID(session_id)

    query_vector = await embed_query(SUMMARIZE_QUERY)
    chunks = await search_chunks(session_id, query_vector, top_k=15)

    if not chunks:
        return "Не удалось найти фрагменты статьи для суммаризации."

    batches = _batch_chunks(chunks, SUMMARIZE_BATCH_CHAR_LIMIT)

    if len(batches) == 1:
        context_string = _build_context_string(batches[0])
        user_content = SUMMARIZE_PROMPT_PREFIX + "Статья:\n" + context_string
    else:
        logger.info("Summarizing in %d batches (map-reduce).", len(batches))
        partial_summaries = []
        for i, batch in enumerate(batches):
            ctx = _build_context_string(batch)
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Это часть {i + 1} из {len(batches)} статьи. "
                    f"{SUMMARIZE_PROMPT_PREFIX}Статья (часть {i + 1}):\n{ctx}"
                )},
            ]
            partial_summaries.append(await _call_groq(msgs))

        combined = "\n\n".join(
            f"[Часть {i + 1}]:\n{s}" for i, s in enumerate(partial_summaries)
        )
        user_content = (
            "На основе следующих частичных изложений создай единое структурированное краткое изложение. "
            "Включи: (1) Цель и задачи, (2) Методология, (3) Ключевые результаты, (4) Выводы и ограничения.\n\n"
            + combined
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    response = await _call_groq(messages)

    # Save summary to history
    await add_message(sid, "user", "/summarize", token_count=5)
    await add_message(sid, "assistant", response, token_count=_estimate_tokens(response))

    return response
