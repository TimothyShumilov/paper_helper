"""Word-based sliding window text chunker."""
from dataclasses import dataclass

from core.pdf_parser import PageText


@dataclass
class Chunk:
    text: str
    chunk_index: int
    page_num: int  # page of the first word in this chunk


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """
    Word-based sliding window chunker.

    Algorithm:
    1. Flatten all pages into a list of (word, page_num) pairs.
    2. Slide a window of `chunk_size` words with step = chunk_size - overlap.
    3. Each window becomes one Chunk; page_num is taken from the first word.

    Word-based (not token-based) for simplicity: 512 words ≈ 394 tokens
    on average, well within the 8192-token limit of Qwen3-Embedding.
    """
    if not pages:
        return []

    # Flatten to (word, page_num) pairs
    word_pages: list[tuple[str, int]] = []
    for page in pages:
        words = page.text.split()
        for word in words:
            word_pages.append((word, page.page_num))

    if not word_pages:
        return []

    step = max(1, chunk_size - overlap)
    chunks: list[Chunk] = []
    chunk_index = 0
    start = 0

    while start < len(word_pages):
        end = min(start + chunk_size, len(word_pages))
        window = word_pages[start:end]
        text = " ".join(w for w, _ in window)
        page_num = window[0][1]
        chunks.append(Chunk(text=text, chunk_index=chunk_index, page_num=page_num))
        chunk_index += 1
        start += step

    return chunks
