"""
DocQA Configuration Module.

Loads application settings from environment variables and .env file.
Provides a centralized configuration object used across all pipeline stages.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


# Project root directory (docqa/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default .env file path
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file.

    All settings can be overridden via environment variables or a .env file
    located at the project root.
    """

    # --- LLM Configuration ---
    llm_provider: Literal["openai", "anthropic", "ollama"] = Field(
        default="openai",
        description="LLM provider to use for generation.",
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for GPT models and embeddings.",
    )
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key for Claude models.",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model name for generation.",
    )
    anthropic_model: str = Field(
        default="claude-haiku-4-20250414",
        description="Anthropic model name for generation.",
    )
    ollama_model: str = Field(
        default="llama3.2",
        description="Ollama model name for local generation.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL.",
    )

    # --- Embedding Configuration ---
    embedding_provider: Literal["openai", "sentence-transformers", "chromadb"] = Field(
        default="openai",
        description="Embedding provider: 'openai', 'sentence-transformers', or 'chromadb' (built-in).",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name.",
    )
    st_embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model name for local embeddings.",
    )

    # --- Chunking Configuration ---
    chunk_size: int = Field(
        default=1000,
        description="Target chunk size in characters for text splitting.",
        ge=100,
        le=10000,
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between consecutive chunks in characters.",
        ge=0,
    )

    # --- Retrieval Configuration ---
    retriever_k: int = Field(
        default=4,
        description="Number of chunks to retrieve per query.",
        ge=1,
        le=20,
    )
    retriever_search_type: Literal["similarity", "mmr"] = Field(
        default="similarity",
        description="Search type for the retriever.",
    )

    # --- ChromaDB Configuration ---
    chroma_persist_dir: str = Field(
        default=str(PROJECT_ROOT / "chroma_db"),
        description="Directory for ChromaDB persistence.",
    )
    chroma_collection_name: str = Field(
        default="docqa_default",
        description="Default ChromaDB collection name.",
    )

    # --- Refusal Configuration ---
    refusal_similarity_threshold: float = Field(
        default=0.3,
        description="Minimum similarity score to consider context relevant.",
        ge=0.0,
        le=1.0,
    )
    refusal_confidence_threshold: float = Field(
        default=0.5,
        description="Minimum confidence score from grounding check.",
        ge=0.0,
        le=1.0,
    )

    # --- Observability ---
    langsmith_api_key: str = Field(
        default="",
        description="LangSmith API key for tracing (optional).",
    )
    langsmith_project: str = Field(
        default="docqa",
        description="LangSmith project name.",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )

    # --- Server Configuration ---
    api_host: str = Field(default="0.0.0.0", description="FastAPI host.")
    api_port: int = Field(default=8000, description="FastAPI port.")
    streamlit_port: int = Field(default=8501, description="Streamlit port.")

    model_config = {
        "env_file": str(ENV_FILE),
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


def get_settings() -> Settings:
    """Get application settings, loading from .env if available.

    Returns:
        Settings: Validated application settings.
    """
    return Settings()


def setup_langsmith() -> None:
    """Configure LangSmith tracing if API key is available.

    Sets the required environment variables for LangSmith integration.
    If no API key is configured, this is a no-op.
    """
    settings = get_settings()
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
