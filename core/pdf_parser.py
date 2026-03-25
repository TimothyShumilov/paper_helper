"""PDF text extraction using PyMuPDF (fitz)."""
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class PageText:
    page_num: int
    text: str


def extract_text_from_pdf(pdf_path: Path) -> list[PageText]:
    """
    Extract plain text from each page.
    Pages with fewer than 50 characters (likely blank or image-only) are skipped.
    Returns list of PageText in page order.
    """
    pages: list[PageText] = []
    doc = fitz.open(str(pdf_path))
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            raw = page.get_text("text")
            cleaned = clean_text(raw)
            if len(cleaned) >= 50:
                pages.append(PageText(page_num=page_index + 1, text=cleaned))
    finally:
        doc.close()
    return pages


def clean_text(raw: str) -> str:
    """
    - Remove soft hyphenation at line breaks (word-\\nbreak → wordbreak)
    - Normalize unicode to NFKC
    - Collapse runs of 3+ newlines to 2
    - Collapse horizontal whitespace runs to a single space
    """
    # Remove hyphenation at line breaks
    text = re.sub(r"-\n(\w)", r"\1", raw)
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse horizontal whitespace (spaces/tabs) but preserve newlines
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
