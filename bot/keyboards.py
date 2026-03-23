"""Inline keyboard builders and callback data constants."""
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

# Callback data constants (must stay ≤ 64 bytes each)
CB_RATE_UP     = "action:rate:1"
CB_RATE_DOWN   = "action:rate:-1"
CB_ALLOW_TRACE = "action:trace:allow"
CB_DENY_TRACE  = "action:trace:deny"


def rating_keyboard() -> InlineKeyboardMarkup:
    """Ask for session rating."""
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("👍 Полезно", callback_data=CB_RATE_UP),
            InlineKeyboardButton("👎 Не полезно", callback_data=CB_RATE_DOWN),
        ]]
    )


def trace_permission_keyboard() -> InlineKeyboardMarkup:
    """Ask user's consent to save conversation history for Langfuse tracing."""
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("✅ Разрешить", callback_data=CB_ALLOW_TRACE),
            InlineKeyboardButton("❌ Отказать",  callback_data=CB_DENY_TRACE),
        ]]
    )
