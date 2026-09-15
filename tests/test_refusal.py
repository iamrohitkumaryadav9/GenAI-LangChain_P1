"""
Tests for DocQA refusal detection logic.

Tests cover:
- Retrieval quality check with high/low scores
- Answer grounding check with refusal phrases
- Combined grounding evaluation
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from app.refusal import (
    check_retrieval_quality,
    check_answer_grounding,
    evaluate_grounding,
    REFUSAL_MESSAGE,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def high_score_docs() -> list[tuple[Document, float]]:
    """Documents with high similarity scores."""
    return [
        (Document(page_content="ML content", metadata={"source": "test.pdf", "page": 1}), 0.85),
        (Document(page_content="More ML", metadata={"source": "test.pdf", "page": 2}), 0.72),
    ]


@pytest.fixture
def low_score_docs() -> list[tuple[Document, float]]:
    """Documents with low similarity scores."""
    return [
        (Document(page_content="Irrelevant", metadata={"source": "test.pdf", "page": 1}), 0.15),
        (Document(page_content="Also irrelevant", metadata={"source": "test.pdf", "page": 2}), 0.10),
    ]


# =============================================================================
# Tests
# =============================================================================

class TestRetrievalQuality:
    """Tests for retrieval quality check."""

    def test_high_scores_pass(self, high_score_docs):
        """High similarity scores should pass the quality check."""
        passed, best_score, reason = check_retrieval_quality(
            high_score_docs, threshold=0.3
        )
        assert passed is True
        assert best_score == 0.85

    def test_low_scores_fail(self, low_score_docs):
        """Low similarity scores should fail the quality check."""
        passed, best_score, reason = check_retrieval_quality(
            low_score_docs, threshold=0.3
        )
        assert passed is False
        assert best_score == 0.15

    def test_empty_docs_fail(self):
        """Empty document list should fail the quality check."""
        passed, best_score, reason = check_retrieval_quality([], threshold=0.3)
        assert passed is False
        assert best_score == 0.0


class TestAnswerGrounding:
    """Tests for answer grounding check."""

    def test_high_confidence_passes(self):
        """High confidence score should pass grounding check."""
        passed, reason = check_answer_grounding(
            confidence=0.9,
            answer_text="ML is a field of study.",
            threshold=0.5,
        )
        assert passed is True

    def test_low_confidence_fails(self):
        """Low confidence score should fail grounding check."""
        passed, reason = check_answer_grounding(
            confidence=0.2,
            answer_text="Maybe ML is something.",
            threshold=0.5,
        )
        assert passed is False

    def test_refusal_phrase_fails(self):
        """Answer containing refusal phrase should fail regardless of confidence."""
        passed, reason = check_answer_grounding(
            confidence=0.9,
            answer_text="I don't have enough information to answer this question.",
            threshold=0.5,
        )
        assert passed is False
        assert "insufficient context" in reason.lower() or "explicitly" in reason.lower()


class TestCombinedGrounding:
    """Tests for the combined grounding evaluation."""

    def test_both_pass_no_refusal(self, high_score_docs):
        """Both checks passing should result in no refusal."""
        result = evaluate_grounding(
            docs_with_scores=high_score_docs,
            confidence=0.9,
            answer_text="ML is great based on the docs",
            retrieval_threshold=0.3,
            confidence_threshold=0.5,
        )
        assert result.is_grounded is True
        assert result.should_refuse is False
        assert result.retrieval_quality_passed is True
        assert result.answer_grounding_passed is True

    def test_low_retrieval_triggers_refusal(self, low_score_docs):
        """Low retrieval scores should trigger refusal even with high confidence."""
        result = evaluate_grounding(
            docs_with_scores=low_score_docs,
            confidence=0.9,
            answer_text="Some answer",
            retrieval_threshold=0.3,
            confidence_threshold=0.5,
        )
        assert result.should_refuse is True
        assert result.retrieval_quality_passed is False

    def test_llm_refusal_triggers_system_refusal(self, high_score_docs):
        """LLM saying it can't answer should trigger system refusal."""
        result = evaluate_grounding(
            docs_with_scores=high_score_docs,
            confidence=0.2,
            answer_text="I don't have enough information to answer this question.",
            retrieval_threshold=0.3,
            confidence_threshold=0.5,
        )
        assert result.should_refuse is True
        assert result.answer_grounding_passed is False
