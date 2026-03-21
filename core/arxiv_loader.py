"""arXiv article downloader: metadata via Atom API, PDF via direct URL."""
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

from config import settings

logger = logging.getLogger(__name__)

ARXIV_ID_RE = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")
ARXIV_ABS_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)")

ARXIV_API_URL = "https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"

NS = {"atom": "http://www.w3.org/2005/Atom"}


def validate_arxiv_id(raw: str) -> str | None:
    """
    Accept:
    - "2312.12456", "2312.12456v2"
    - "https://arxiv.org/abs/2312.12456"
    - "arxiv.org/pdf/2312.12456v1"
    Returns canonical ID (without version suffix) or None if invalid.
    """
    raw = raw.strip()

    # Try direct match
    m = ARXIV_ID_RE.match(raw)
    if m:
        return m.group(1)

    # Try extracting from URL
    m = ARXIV_ABS_URL_RE.search(raw)
    if m:
        full_id = m.group(1)
        base = ARXIV_ID_RE.match(full_id)
        if base:
            return base.group(1)

    return None


async def fetch_paper_metadata(arxiv_id: str) -> dict:
    """
    Query arXiv Atom feed and return:
    {'title': str, 'authors': list[str], 'abstract': str, 'published': str}
    Raises ValueError if paper not found.
    """
    url = ARXIV_API_URL.format(arxiv_id=arxiv_id)
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    root = ET.fromstring(resp.text)
    entry = root.find("atom:entry", NS)
    if entry is None:
        raise ValueError(f"Paper '{arxiv_id}' not found on arXiv.")

    title_el = entry.find("atom:title", NS)
    title = title_el.text.strip().replace("\n", " ") if title_el is not None else arxiv_id

    authors = [
        a.find("atom:name", NS).text.strip()
        for a in entry.findall("atom:author", NS)
        if a.find("atom:name", NS) is not None
    ]

    abstract_el = entry.find("atom:summary", NS)
    abstract = abstract_el.text.strip() if abstract_el is not None else ""

    published_el = entry.find("atom:published", NS)
    published = published_el.text[:10] if published_el is not None else ""

    return {"title": title, "authors": authors, "abstract": abstract, "published": published}


async def download_pdf(arxiv_id: str) -> Path:
    """
    Download PDF to pdf_download_dir/{arxiv_id}.pdf.
    Streams response to avoid holding the full file in RAM.
    Returns the local Path.
    """
    dest_dir = Path(settings.pdf_download_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{arxiv_id}.pdf"

    if dest_path.exists():
        logger.info("PDF already cached at %s.", dest_path)
        return dest_path

    url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
    logger.info("Downloading PDF from %s ...", url)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10, read=180, write=10, pool=10),
        follow_redirects=True,
    ) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with dest_path.open("wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    f.write(chunk)

    logger.info("PDF saved to %s (%d bytes).", dest_path, dest_path.stat().st_size)
    return dest_path
