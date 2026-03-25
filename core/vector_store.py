"""Qdrant vector store wrapper."""
import logging
import uuid
from typing import Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PayloadSchemaType,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from config import settings

logger = logging.getLogger(__name__)

_client: Optional[AsyncQdrantClient] = None


def get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
    return _client


async def init_collection() -> None:
    """
    Create the 'arxiv_chunks' collection if it doesn't exist,
    and ensure payload indexes are in place.
    """
    client = get_client()
    collection_name = settings.qdrant_collection

    existing = await client.get_collections()
    existing_names = [c.name for c in existing.collections]

    if collection_name not in existing_names:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection '%s'.", collection_name)
    else:
        logger.info("Qdrant collection '%s' already exists.", collection_name)

    # Create payload indexes for fast filtering
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="session_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="user_id",
        field_schema=PayloadSchemaType.INTEGER,
    )


async def upsert_chunks(
    session_id: str,
    user_id: int,
    arxiv_id: str,
    texts: list[str],
    embeddings: list[list[float]],
    page_nums: list[int],
) -> None:
    """Batch upsert all chunk vectors for a new session."""
    client = get_client()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload={
                "session_id": session_id,
                "user_id": user_id,
                "arxiv_id": arxiv_id,
                "chunk_index": i,
                "text": texts[i],
                "page_num": page_nums[i],
            },
        )
        for i in range(len(texts))
    ]
    await client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )
    logger.info(
        "Upserted %d chunks for session %s (arxiv: %s).",
        len(points),
        session_id,
        arxiv_id,
    )


async def search_chunks(
    session_id: str,
    query_vector: list[float],
    top_k: int = 5,
) -> list[dict]:
    """
    Cosine similarity search filtered strictly to session_id.
    Returns list of {'text': str, 'score': float, 'chunk_index': int}.
    """
    client = get_client()
    results = await client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="session_id",
                    match=MatchValue(value=session_id),
                )
            ]
        ),
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "text": r.payload["text"],
            "score": r.score,
            "chunk_index": r.payload.get("chunk_index", 0),
            "page_num": r.payload.get("page_num", 0),
        }
        for r in results
    ]


async def delete_session_chunks(session_id: str) -> None:
    """Delete all points for a given session_id."""
    client = get_client()
    await client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="session_id",
                    match=MatchValue(value=session_id),
                )
            ]
        ),
    )
    logger.info("Deleted Qdrant chunks for session %s.", session_id)
