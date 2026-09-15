"""
DocQA FastAPI Application.

Provides REST API endpoints for the DocQA RAG system:
- POST /upload — Upload PDF files for ingestion
- POST /query — Ask questions against ingested documents
- GET /health — System health check
- DELETE /session/{session_id} — Clear session history
"""

import logging
import os
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings, setup_langsmith
from app.generation import clear_session_history, query_rag
from app.ingestion import ingest_pipeline
from app.retrieval import get_collection_stats

logger = logging.getLogger(__name__)


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logging.basicConfig(
        level=getattr(logging, get_settings().log_level, logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    setup_langsmith()
    logger.info("DocQA API starting up...")
    yield
    # Shutdown
    logger.info("DocQA API shutting down...")


# =============================================================================
# App & Middleware
# =============================================================================

app = FastAPI(
    title="DocQA API",
    description="RAG Document Q&A System — Upload PDFs and ask questions with grounded, cited answers.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request/Response Models
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for the /query endpoint."""

    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask.")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Session ID for multi-turn.")
    collection_name: str | None = Field(default=None, description="ChromaDB collection to search.")
    k: int | None = Field(default=None, ge=1, le=20, description="Number of chunks to retrieve.")


class QueryResponse(BaseModel):
    """Response model for the /query endpoint."""

    answer: str
    citations: list[dict[str, str]]
    confidence: float
    refused: bool
    sources: list[dict[str, Any]]
    session_id: str
    retrieval_time_ms: float
    generation_time_ms: float


class UploadResponse(BaseModel):
    """Response model for the /upload endpoint."""

    message: str
    files_processed: int
    total_pages: int
    total_chunks: int
    collection_name: str
    total_time_s: float


class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""

    status: str
    collection: dict[str, Any]
    settings: dict[str, str]


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str


# =============================================================================
# Endpoints
# =============================================================================

@app.post("/upload", response_model=UploadResponse, responses={400: {"model": ErrorResponse}})
async def upload_documents(
    files: list[UploadFile] = File(..., description="PDF files to ingest"),
    collection_name: str | None = None,
):
    """Upload one or more PDF files for ingestion into the vector store.

    Extracts text, chunks, embeds, and stores in ChromaDB with metadata.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Validate file types
    for f in files:
        if not f.filename or not f.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: '{f.filename}'. Only PDF files are accepted.",
            )

    # Save uploaded files to temp directory
    temp_dir = tempfile.mkdtemp(prefix="docqa_upload_")
    temp_paths: list[str] = []

    try:
        for f in files:
            temp_path = os.path.join(temp_dir, f.filename)
            content = await f.read()

            if len(content) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{f.filename}' is empty.",
                )

            with open(temp_path, "wb") as fp:
                fp.write(content)
            temp_paths.append(temp_path)

        # Run ingestion pipeline
        result = ingest_pipeline(
            file_paths=temp_paths,
            collection_name=collection_name,
        )

        return UploadResponse(
            message=f"Successfully ingested {result['files_processed']} file(s)",
            files_processed=result["files_processed"],
            total_pages=result["total_pages"],
            total_chunks=result["total_chunks"],
            collection_name=result["storage"]["collection_name"],
            total_time_s=result["total_time_s"],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during upload")
        raise HTTPException(status_code=500, detail=f"Internal error: {type(e).__name__}: {e}")
    finally:
        # Clean up temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/query", response_model=QueryResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def query_documents(request: QueryRequest):
    """Ask a question against the ingested documents.

    Returns a grounded answer with inline citations and confidence score.
    Refuses to answer when context is insufficient.
    """
    try:
        result = query_rag(
            question=request.question,
            session_id=request.session_id,
            collection_name=request.collection_name,
            k=request.k,
        )

        return QueryResponse(
            answer=result["answer"],
            citations=result["citations"],
            confidence=result["confidence"],
            refused=result["refused"],
            sources=result["sources"],
            session_id=request.session_id,
            retrieval_time_ms=result["retrieval_time_ms"],
            generation_time_ms=result["generation_time_ms"],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        error_msg = str(e)
        if "429" in error_msg or "rate_limit" in error_msg.lower() or "quota" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="LLM API rate limit or quota exceeded. Please try again later or check your API credits.",
            )
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        logger.exception("Unexpected error during query")
        raise HTTPException(status_code=500, detail=f"Internal error: {type(e).__name__}: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health and loaded collection stats."""
    settings = get_settings()
    stats = get_collection_stats()

    return HealthResponse(
        status="healthy",
        collection=stats,
        settings={
            "llm_provider": settings.llm_provider,
            "embedding_provider": settings.embedding_provider,
            "chunk_size": str(settings.chunk_size),
            "chunk_overlap": str(settings.chunk_overlap),
            "retriever_k": str(settings.retriever_k),
        },
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Clear conversation history for a session."""
    clear_session_history(session_id)
    return {"message": f"Session '{session_id}' history cleared"}


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler to prevent stack traces in responses."""
    logger.exception("Unhandled exception: %s", exc)
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
