"""
DocQA Ingestion Pipeline.

Handles the full document ingestion workflow:
1. Load PDF documents using PyPDFLoader
2. Split into chunks with RecursiveCharacterTextSplitter
3. Embed chunks using configurable embedding model
4. Store in ChromaDB with source metadata preserved

Each chunk retains metadata: source filename, page number, chunk index.
"""

import logging
import time
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

logger = logging.getLogger(__name__)


def load_pdf(file_path: str | Path) -> list[Document]:
    """Load a PDF file and return a list of Documents, one per page.

    Each Document's metadata includes the source filename and page number.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of Document objects with page_content and metadata.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the PDF file is empty or unreadable.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    if not file_path.suffix.lower() == ".pdf":
        raise ValueError(f"Expected a PDF file, got: {file_path.suffix}")

    logger.info("Loading PDF: %s", file_path.name)
    start = time.perf_counter()

    try:
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
    except Exception as e:
        raise ValueError(f"Failed to load PDF '{file_path.name}': {e}") from e

    if not documents:
        raise ValueError(f"PDF file is empty or unreadable: {file_path.name}")

    # Ensure metadata has the source filename (not full path for privacy)
    for doc in documents:
        doc.metadata["source"] = file_path.name
        # PyPDFLoader uses 0-indexed pages; convert to 1-indexed for display
        if "page" in doc.metadata:
            doc.metadata["page"] = doc.metadata["page"] + 1

    elapsed = time.perf_counter() - start
    logger.info(
        "Loaded %d pages from '%s' in %.2fs",
        len(documents),
        file_path.name,
        elapsed,
    )
    return documents


def load_multiple_pdfs(file_paths: list[str | Path]) -> list[Document]:
    """Load multiple PDF files and return combined Documents.

    Args:
        file_paths: List of paths to PDF files.

    Returns:
        Combined list of Document objects from all PDFs.

    Raises:
        ValueError: If no valid documents were loaded from any file.
    """
    all_documents: list[Document] = []
    errors: list[str] = []

    for path in file_paths:
        try:
            docs = load_pdf(path)
            all_documents.extend(docs)
        except (FileNotFoundError, ValueError) as e:
            logger.error("Skipping file: %s", e)
            errors.append(str(e))

    if not all_documents:
        raise ValueError(
            f"No valid documents loaded. Errors: {'; '.join(errors)}"
        )

    logger.info(
        "Loaded %d total pages from %d files (%d errors)",
        len(all_documents),
        len(file_paths) - len(errors),
        len(errors),
    )
    return all_documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents into chunks using RecursiveCharacterTextSplitter.

    Preserves original metadata (source, page) and adds chunk_index.

    Args:
        documents: List of Document objects to split.
        chunk_size: Target chunk size in characters. Uses config default if None.
        chunk_overlap: Overlap between chunks. Uses config default if None.

    Returns:
        List of chunked Document objects with preserved metadata.

    Raises:
        ValueError: If documents list is empty.
    """
    if not documents:
        raise ValueError("Cannot chunk an empty document list.")

    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    logger.info(
        "Chunking %d documents (chunk_size=%d, overlap=%d)",
        len(documents),
        chunk_size,
        chunk_overlap,
    )
    start = time.perf_counter()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index metadata for traceability
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx

    elapsed = time.perf_counter() - start
    logger.info(
        "Created %d chunks in %.2fs (avg %.0f chars/chunk)",
        len(chunks),
        elapsed,
        sum(len(c.page_content) for c in chunks) / max(len(chunks), 1),
    )
    return chunks


def get_embedding_function() -> Any:
    """Get the configured embedding function.

    Returns OpenAI embeddings, sentence-transformers, or ChromaDB's built-in
    default embeddings based on config.

    Returns:
        An embedding function compatible with ChromaDB / LangChain.
    """
    settings = get_settings()

    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        logger.info("Using OpenAI embeddings: %s", settings.openai_embedding_model)
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )
    elif settings.embedding_provider == "chromadb":
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        from langchain_core.embeddings import Embeddings

        class ChromaDBEmbeddings(Embeddings):
            """LangChain-compatible wrapper for ChromaDB's built-in embeddings.

            Uses ChromaDB's default ONNX-based all-MiniLM-L6-v2 model.
            No scipy dependency required.
            """

            def __init__(self) -> None:
                self._ef = DefaultEmbeddingFunction()

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                """Embed a list of documents."""
                return self._ef(texts)

            def embed_query(self, text: str) -> list[float]:
                """Embed a single query."""
                return self._ef([text])[0]

        logger.info("Using ChromaDB built-in embeddings (ONNX all-MiniLM-L6-v2)")
        return ChromaDBEmbeddings()
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        logger.info(
            "Using sentence-transformers: %s", settings.st_embedding_model
        )
        return HuggingFaceEmbeddings(
            model_name=settings.st_embedding_model,
        )


def embed_and_store(
    chunks: list[Document],
    collection_name: str | None = None,
) -> dict[str, Any]:
    """Embed document chunks and store in ChromaDB.

    Creates or updates a persistent ChromaDB collection with the given chunks.

    Args:
        chunks: List of chunked Document objects to embed and store.
        collection_name: ChromaDB collection name. Uses config default if None.

    Returns:
        Dict with storage stats: collection_name, num_chunks, persist_dir.

    Raises:
        ValueError: If chunks list is empty.
        RuntimeError: If embedding or storage fails.
    """
    if not chunks:
        raise ValueError("Cannot store empty chunks list.")

    settings = get_settings()
    collection_name = collection_name or settings.chroma_collection_name

    logger.info(
        "Embedding and storing %d chunks in collection '%s'",
        len(chunks),
        collection_name,
    )
    start = time.perf_counter()

    try:
        embedding_fn = get_embedding_function()

        from langchain_chroma import Chroma

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_fn,
            collection_name=collection_name,
            persist_directory=settings.chroma_persist_dir,
        )

        elapsed = time.perf_counter() - start
        stats = {
            "collection_name": collection_name,
            "num_chunks": len(chunks),
            "persist_dir": settings.chroma_persist_dir,
            "embedding_time_s": round(elapsed, 2),
        }
        logger.info(
            "Stored %d chunks in %.2fs (collection='%s', dir='%s')",
            len(chunks),
            elapsed,
            collection_name,
            settings.chroma_persist_dir,
        )
        return stats

    except Exception as e:
        raise RuntimeError(f"Failed to embed and store chunks: {e}") from e


def ingest_pipeline(
    file_paths: list[str | Path],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    collection_name: str | None = None,
) -> dict[str, Any]:
    """Run the full ingestion pipeline: load → chunk → embed → store.

    Args:
        file_paths: List of PDF file paths to ingest.
        chunk_size: Optional override for chunk size.
        chunk_overlap: Optional override for chunk overlap.
        collection_name: Optional override for ChromaDB collection name.

    Returns:
        Dict with pipeline stats: files loaded, pages, chunks, timing.

    Raises:
        ValueError: If no files provided or all files fail to load.
        RuntimeError: If embedding/storage fails.
    """
    if not file_paths:
        raise ValueError("No file paths provided for ingestion.")

    logger.info("=" * 60)
    logger.info("INGESTION PIPELINE START — %d file(s)", len(file_paths))
    logger.info("=" * 60)
    pipeline_start = time.perf_counter()

    # Step 1: Load PDFs
    documents = load_multiple_pdfs(file_paths)

    # Step 2: Chunk documents
    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Step 3: Embed and store
    storage_stats = embed_and_store(chunks, collection_name=collection_name)

    pipeline_elapsed = time.perf_counter() - pipeline_start

    result = {
        "files_processed": len(file_paths),
        "total_pages": len(documents),
        "total_chunks": len(chunks),
        "storage": storage_stats,
        "total_time_s": round(pipeline_elapsed, 2),
    }

    logger.info("=" * 60)
    logger.info("INGESTION PIPELINE COMPLETE in %.2fs", pipeline_elapsed)
    logger.info("  Files: %d | Pages: %d | Chunks: %d", 
                result["files_processed"], result["total_pages"], result["total_chunks"])
    logger.info("=" * 60)

    return result
