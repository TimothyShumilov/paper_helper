"""
Embedding generation using Qwen3-Embedding-0.6B-Q4_K_M-GGUF via llama-cpp-python.

The GGUF file is downloaded from HuggingFace on first use and cached locally.
Embeddings are L2-normalized before return so Qdrant cosine similarity is correct.
"""
import asyncio
import logging
import math
from functools import lru_cache
from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from config import settings

logger = logging.getLogger(__name__)


def _download_model() -> str:
    """Download GGUF file from HuggingFace Hub if not already cached."""
    cache_dir = Path(settings.models_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_path = hf_hub_download(
        repo_id=settings.embedding_repo_id,
        filename=settings.embedding_filename,
        cache_dir=str(cache_dir),
    )
    logger.info("Embedding model ready at: %s", local_path)
    return local_path


@lru_cache(maxsize=1)
def _get_llama() -> Llama:
    """
    Load the Llama model in embedding mode.
    lru_cache ensures the model is loaded only once per process.
    """
    model_path = _download_model()
    logger.info("Loading embedding model...")
    llm = Llama(
        model_path=model_path,
        embedding=True,
        n_ctx=8192,
        n_threads=4,
        verbose=False,
    )
    logger.info("Embedding model loaded.")
    return llm


def _normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def _encode_batch(texts: list[str]) -> list[list[float]]:
    """Synchronous encoding — runs in executor thread."""
    llm = _get_llama()
    result = llm.create_embedding(texts)
    vectors = [item["embedding"] for item in result["data"]]
    return [_normalize(v) for v in vectors]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Async wrapper: runs llama-cpp inference in a thread pool
    so it doesn't block the asyncio event loop.
    Processes in batches of 16 to avoid OOM on large documents.
    """
    loop = asyncio.get_event_loop()
    batch_size = 16
    all_vectors: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors = await loop.run_in_executor(None, _encode_batch, batch)
        all_vectors.extend(vectors)

    return all_vectors


async def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    results = await embed_texts([query])
    return results[0]


def warmup() -> None:
    """Pre-load the model synchronously at startup to avoid first-query delay."""
    _get_llama()
