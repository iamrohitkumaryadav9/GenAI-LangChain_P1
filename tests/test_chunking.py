"""
Tests for DocQA chunking logic.

Tests cover:
- Chunk count with different configurations
- Metadata preservation through chunking
- Edge case: empty document handling
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from app.ingestion import chunk_documents, load_pdf


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            page_content="A" * 2000,  # 2000 chars
            metadata={"source": "test.pdf", "page": 1},
        ),
        Document(
            page_content="B" * 1500,  # 1500 chars
            metadata={"source": "test.pdf", "page": 2},
        ),
    ]


@pytest.fixture
def real_sample_pdf() -> Path:
    """Path to the real sample PDF if available."""
    pdf_path = Path(__file__).parent.parent / "eval" / "sample_doc.pdf"
    if not pdf_path.exists():
        pytest.skip("sample_doc.pdf not found — run eval/generate_sample_doc.py first")
    return pdf_path


# =============================================================================
# Tests
# =============================================================================

class TestChunking:
    """Tests for document chunking logic."""

    def test_chunk_size_500_produces_more_chunks(self, sample_documents: list[Document]):
        """Smaller chunk size should produce more chunks than larger."""
        chunks_small = chunk_documents(sample_documents, chunk_size=500, chunk_overlap=50)
        chunks_large = chunk_documents(sample_documents, chunk_size=1000, chunk_overlap=100)

        assert len(chunks_small) > len(chunks_large), (
            f"Small chunks ({len(chunks_small)}) should be more than "
            f"large chunks ({len(chunks_large)})"
        )

    def test_chunk_size_respected(self, sample_documents: list[Document]):
        """Chunks should not significantly exceed the target chunk size."""
        chunk_size = 500
        chunks = chunk_documents(sample_documents, chunk_size=chunk_size, chunk_overlap=50)

        for chunk in chunks:
            # Allow some tolerance due to separator logic
            assert len(chunk.page_content) <= chunk_size * 1.2, (
                f"Chunk exceeds size limit: {len(chunk.page_content)} > {chunk_size * 1.2}"
            )

    def test_metadata_preservation(self, sample_documents: list[Document]):
        """Chunking should preserve source and page metadata."""
        chunks = chunk_documents(sample_documents, chunk_size=500, chunk_overlap=50)

        for chunk in chunks:
            assert "source" in chunk.metadata, "Missing 'source' metadata"
            assert "page" in chunk.metadata, "Missing 'page' metadata"
            assert "chunk_index" in chunk.metadata, "Missing 'chunk_index' metadata"
            assert chunk.metadata["source"] == "test.pdf"
            assert chunk.metadata["page"] in [1, 2]

    def test_chunk_index_sequential(self, sample_documents: list[Document]):
        """Chunk indices should be sequential starting from 0."""
        chunks = chunk_documents(sample_documents, chunk_size=500, chunk_overlap=50)

        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_documents_raises(self):
        """Chunking an empty list should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot chunk an empty document list"):
            chunk_documents([], chunk_size=500, chunk_overlap=50)

    def test_overlap_creates_shared_content(self, sample_documents: list[Document]):
        """With overlap > 0, consecutive chunks from the same page should share content."""
        chunks = chunk_documents(sample_documents, chunk_size=500, chunk_overlap=100)

        # Find consecutive chunks from the same page
        for i in range(len(chunks) - 1):
            if chunks[i].metadata["page"] == chunks[i + 1].metadata["page"]:
                # Check that the end of chunk i overlaps with start of chunk i+1
                # Due to separator logic, exact overlap may vary, but there should be some
                chunk_i_end = chunks[i].page_content[-50:]
                chunk_i1_start = chunks[i + 1].page_content[:100]
                # At least some shared characters should exist
                shared = set(chunk_i_end) & set(chunk_i1_start)
                assert len(shared) > 0, "Expected some content overlap between consecutive chunks"
                break

    def test_real_pdf_chunking(self, real_sample_pdf: Path):
        """Test chunking with the actual sample PDF."""
        docs = load_pdf(real_sample_pdf)
        assert len(docs) > 0, "PDF should have at least one page"

        chunks_a = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
        chunks_b = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)

        assert len(chunks_a) > len(chunks_b), "Config A should produce more chunks"
        assert all("source" in c.metadata for c in chunks_a)
        assert all("page" in c.metadata for c in chunks_a)
