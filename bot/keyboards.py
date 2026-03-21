"""Inline keyboard builders and callback data constants."""
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

# Callback data constants (must stay ≤ 64 bytes each)
CB_RATE_UP = "action:rate:1"
CB_RATE_DOWN = "action:rate:-1"
CB_RETURN_PREFIX = "action:return:"   # + full UUID (36 chars) = 50 total, within limit
CB_SESSIONS_PAGE = "action:sessions:page:"  # + page number (1-2 digits) = 23-24 total


def rating_keyboard() -> InlineKeyboardMarkup:
    """Ask for session rating."""
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("👍 Полезно", callback_data=CB_RATE_UP),
            InlineKeyboardButton("👎 Не полезно", callback_data=CB_RATE_DOWN),
        ]]
    )


def new_session_keyboard(
    sessions: list[dict],
    page: int = 0,
    page_size: int = 5,
) -> InlineKeyboardMarkup:
    """
    Paginated session picker.
    Each row: one button per session showing paper title (truncated) or arxiv_id.
    Last row (if needed): [← Назад] [Вперёд →] navigation.
    """
    total = len(sessions)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = max(0, min(page, total_pages - 1))

    start = page * page_size
    page_sessions = sessions[start : start + page_size]

    rows = []
    for session in page_sessions:
        sid = str(session["id"])
        label_raw = session.get("paper_title") or session["arxiv_id"]
        label = label_raw[:40] + "…" if len(label_raw) > 40 else label_raw
        rows.append(
            [InlineKeyboardButton(label, callback_data=f"{CB_RETURN_PREFIX}{sid}")]
        )

    nav = []
    if page > 0:
        nav.append(
            InlineKeyboardButton("← Назад", callback_data=f"{CB_SESSIONS_PAGE}{page - 1}")
        )
    if page < total_pages - 1:
        nav.append(
            InlineKeyboardButton("Вперёд →", callback_data=f"{CB_SESSIONS_PAGE}{page + 1}")
        )
    if nav:
        rows.append(nav)

    return InlineKeyboardMarkup(rows)
