"""
Tests for DocQA citation formatting and extraction.

Tests cover:
- Citation extraction from LLM output
- Confidence score extraction
- Answer cleaning (removing confidence line)
- Context formatting
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.generation import (
    extract_citations,
    extract_confidence,
    clean_answer,
    format_context,
)
from langchain_core.documents import Document


class TestCitationExtraction:
    """Tests for citation extraction from LLM output."""

    def test_extract_single_citation(self):
        """Should extract a single citation."""
        text = "ML was coined by Arthur Samuel [Source: sample_doc.pdf, Page 2]."
        citations = extract_citations(text)
        assert len(citations) == 1
        assert citations[0]["source"] == "sample_doc.pdf"
        assert citations[0]["page"] == "2"

    def test_extract_multiple_citations(self):
        """Should extract multiple distinct citations."""
        text = (
            "Supervised learning [Source: doc.pdf, Page 3] includes "
            "classification [Source: doc.pdf, Page 4] and regression."
        )
        citations = extract_citations(text)
        assert len(citations) == 2
        assert citations[0]["page"] == "3"
        assert citations[1]["page"] == "4"

    def test_deduplicate_citations(self):
        """Should deduplicate repeated citations."""
        text = (
            "ML is great [Source: doc.pdf, Page 2]. "
            "Really great [Source: doc.pdf, Page 2]."
        )
        citations = extract_citations(text)
        assert len(citations) == 1

    def test_no_citations(self):
        """Should return empty list when no citations present."""
        text = "This is a plain answer with no citations."
        citations = extract_citations(text)
        assert len(citations) == 0


class TestConfidenceExtraction:
    """Tests for confidence score extraction."""

    def test_extract_confidence_normal(self):
        """Should extract a valid confidence score."""
        text = "ML is a subset of AI.\nCONFIDENCE: 0.85"
        confidence = extract_confidence(text)
        assert confidence == 0.85

    def test_extract_confidence_default(self):
        """Should return 0.5 default when no confidence found."""
        text = "ML is a subset of AI."
        confidence = extract_confidence(text)
        assert confidence == 0.5

    def test_extract_confidence_clamped(self):
        """Should clamp confidence to 0.0-1.0 range."""
        text = "Answer.\nCONFIDENCE: 1.5"
        confidence = extract_confidence(text)
        assert confidence == 1.0

        text2 = "Answer.\nCONFIDENCE: -0.3"
        confidence2 = extract_confidence(text2)
        assert confidence2 == 0.0


class TestAnswerCleaning:
    """Tests for answer cleaning."""

    def test_clean_removes_confidence(self):
        """Should remove the CONFIDENCE line."""
        text = "ML is a subset of AI.\nCONFIDENCE: 0.85"
        cleaned = clean_answer(text)
        assert "CONFIDENCE" not in cleaned
        assert cleaned == "ML is a subset of AI."

    def test_clean_preserves_content(self):
        """Should preserve answer content."""
        text = "ML is a subset of AI with many applications."
        cleaned = clean_answer(text)
        assert cleaned == text


class TestContextFormatting:
    """Tests for context formatting."""

    def test_format_context_includes_metadata(self):
        """Should include source and page in formatted context."""
        docs = [
            (Document(
                page_content="ML is great",
                metadata={"source": "test.pdf", "page": 1},
            ), 0.95),
        ]
        context = format_context(docs)
        assert "test.pdf" in context
        assert "Page 1" in context
        assert "ML is great" in context
        assert "0.950" in context

    def test_format_context_multiple_chunks(self):
        """Should format multiple chunks with separators."""
        docs = [
            (Document(page_content="Chunk 1", metadata={"source": "a.pdf", "page": 1}), 0.9),
            (Document(page_content="Chunk 2", metadata={"source": "b.pdf", "page": 2}), 0.8),
        ]
        context = format_context(docs)
        assert "Chunk 1" in context
        assert "Chunk 2" in context
        assert "---" in context  # separator
