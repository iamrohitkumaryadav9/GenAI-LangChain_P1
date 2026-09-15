"""
DocQA Retrieval Module.

Configures and provides vector store retrievers for the query pipeline.
Supports configurable k, search type, and similarity scoring.
"""

import logging
import time
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import get_settings
from app.ingestion import get_embedding_function

logger = logging.getLogger(__name__)


def get_vectorstore(collection_name: str | None = None) -> Chroma:
    """Get a ChromaDB vector store instance.

    Args:
        collection_name: Name of the ChromaDB collection. Uses config default if None.

    Returns:
        Chroma vector store instance connected to the persisted collection.
    """
    settings = get_settings()
    collection_name = collection_name or settings.chroma_collection_name

    embedding_fn = get_embedding_function()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_fn,
        persist_directory=settings.chroma_persist_dir,
    )

    logger.debug(
        "Connected to ChromaDB collection '%s' (%d documents)",
        collection_name,
        vectorstore._collection.count(),
    )
    return vectorstore


def get_retriever(
    collection_name: str | None = None,
    k: int | None = None,
    search_type: str | None = None,
) -> Any:
    """Get a configured LangChain retriever from the vector store.

    Args:
        collection_name: ChromaDB collection name. Uses config default if None.
        k: Number of documents to retrieve. Uses config default if None.
        search_type: Search type ('similarity' or 'mmr'). Uses config default if None.

    Returns:
        A LangChain VectorStoreRetriever.
    """
    settings = get_settings()
    k = k or settings.retriever_k
    search_type = search_type or settings.retriever_search_type

    vectorstore = get_vectorstore(collection_name)

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )

    logger.info(
        "Created retriever (collection='%s', k=%d, search_type='%s')",
        collection_name or settings.chroma_collection_name,
        k,
        search_type,
    )
    return retriever


def retrieve_with_scores(
    query: str,
    collection_name: str | None = None,
    k: int | None = None,
) -> list[tuple[Document, float]]:
    """Retrieve documents with similarity scores for grounding checks.

    Uses similarity_search_with_relevance_scores which returns scores
    normalized between 0 and 1 (higher = more similar).

    Args:
        query: The user's question.
        collection_name: ChromaDB collection name. Uses config default if None.
        k: Number of documents to retrieve. Uses config default if None.

    Returns:
        List of (Document, score) tuples, sorted by relevance (highest first).
    """
    settings = get_settings()
    k = k or settings.retriever_k

    vectorstore = get_vectorstore(collection_name)

    start = time.perf_counter()

    # Use similarity_search_with_score (returns distance, lower = better)
    results_with_distance = vectorstore.similarity_search_with_score(query, k=k)

    elapsed = time.perf_counter() - start

    # Convert distances to similarity scores (1 - normalized_distance)
    # ChromaDB uses L2 distance by default; we normalize for a 0-1 range
    results_with_scores: list[tuple[Document, float]] = []
    for doc, distance in results_with_distance:
        # L2 distance: 0 = identical, higher = more different
        # Convert to similarity: use 1/(1+distance) for a 0-1 range
        similarity = 1.0 / (1.0 + distance)
        results_with_scores.append((doc, similarity))

    logger.info(
        "Retrieved %d documents in %.3fs (query='%s...')",
        len(results_with_scores),
        elapsed,
        query[:50],
    )
    for i, (doc, score) in enumerate(results_with_scores):
        logger.debug(
            "  [%d] score=%.4f | source=%s, page=%s | %s...",
            i,
            score,
            doc.metadata.get("source", "?"),
            doc.metadata.get("page", "?"),
            doc.page_content[:60],
        )

    return results_with_scores


def get_collection_stats(collection_name: str | None = None) -> dict[str, Any]:
    """Get statistics about a ChromaDB collection.

    Args:
        collection_name: Collection name. Uses config default if None.

    Returns:
        Dict with collection stats: name, document count, metadata.
    """
    settings = get_settings()
    collection_name = collection_name or settings.chroma_collection_name

    try:
        vectorstore = get_vectorstore(collection_name)
        count = vectorstore._collection.count()

        return {
            "collection_name": collection_name,
            "document_count": count,
            "persist_dir": settings.chroma_persist_dir,
            "status": "ready" if count > 0 else "empty",
        }
    except Exception as e:
        logger.error("Failed to get collection stats: %s", e)
        return {
            "collection_name": collection_name,
            "document_count": 0,
            "persist_dir": settings.chroma_persist_dir,
            "status": f"error: {e}",
        }
