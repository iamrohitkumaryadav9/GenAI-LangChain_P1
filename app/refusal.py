"""
DocQA Refusal Module.

Implements dual-layer grounding checks to determine whether the system should
refuse to answer a question:

Layer 1 — Retrieval Quality Check:
    Examines similarity scores from retrieved chunks. If the best score is
    below a configurable threshold, the context is likely irrelevant.

Layer 2 — Answer Grounding Check:
    Uses the LLM's self-reported confidence score (extracted from structured
    output) to assess whether the answer is well-supported by the context.

Both layers must pass for the system to provide an answer. If either fails,
the system returns a polite refusal message.
"""

import logging
from dataclasses import dataclass, field

from langchain_core.documents import Document

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class GroundingResult:
    """Result of the grounding/refusal check.

    Attributes:
        is_grounded: Whether the answer is sufficiently grounded in context.
        should_refuse: Whether the system should refuse to answer.
        confidence: Combined confidence score (0.0 - 1.0).
        reason: Human-readable explanation for the decision.
        retrieval_quality_passed: Whether retrieval quality check passed.
        answer_grounding_passed: Whether answer grounding check passed.
    """

    is_grounded: bool
    should_refuse: bool
    confidence: float
    reason: str
    retrieval_quality_passed: bool
    answer_grounding_passed: bool


REFUSAL_MESSAGE = (
    "I don't have enough information in the provided documents to answer "
    "this question confidently. The retrieved context doesn't appear to "
    "contain relevant information for this query."
)


def check_retrieval_quality(
    docs_with_scores: list[tuple[Document, float]],
    threshold: float | None = None,
) -> tuple[bool, float, str]:
    """Check if retrieved documents are relevant enough to answer.

    Examines the similarity scores of retrieved documents. If the best
    score is below the threshold, the context is likely irrelevant.

    Args:
        docs_with_scores: List of (Document, similarity_score) tuples.
        threshold: Minimum similarity score to consider relevant.
                   Uses config default if None.

    Returns:
        Tuple of (passed, best_score, reason).
    """
    settings = get_settings()
    threshold = threshold if threshold is not None else settings.refusal_similarity_threshold

    if not docs_with_scores:
        return False, 0.0, "No documents retrieved"

    scores = [score for _, score in docs_with_scores]
    best_score = max(scores)
    avg_score = sum(scores) / len(scores)

    passed = best_score >= threshold

    if passed:
        reason = (
            f"Retrieval quality OK: best_score={best_score:.4f} >= "
            f"threshold={threshold:.4f} (avg={avg_score:.4f})"
        )
    else:
        reason = (
            f"Retrieval quality LOW: best_score={best_score:.4f} < "
            f"threshold={threshold:.4f} (avg={avg_score:.4f})"
        )

    logger.info("Retrieval quality check: passed=%s | %s", passed, reason)
    return passed, best_score, reason


def check_answer_grounding(
    confidence: float,
    answer_text: str,
    threshold: float | None = None,
) -> tuple[bool, str]:
    """Check if the LLM's answer is sufficiently grounded in context.

    Uses the confidence score reported by the LLM in its structured output,
    plus checks for refusal phrases in the answer text.

    Args:
        confidence: The LLM's self-reported confidence score (0.0-1.0).
        answer_text: The generated answer text.
        threshold: Minimum confidence to consider grounded.
                   Uses config default if None.

    Returns:
        Tuple of (passed, reason).
    """
    settings = get_settings()
    threshold = threshold if threshold is not None else settings.refusal_confidence_threshold

    # Check for explicit refusal phrases in the answer
    refusal_phrases = [
        "don't have enough information",
        "do not have enough information",
        "cannot answer",
        "not enough context",
        "no relevant information",
        "insufficient context",
        "not mentioned in",
        "not found in the provided",
    ]

    has_refusal_phrase = any(
        phrase in answer_text.lower() for phrase in refusal_phrases
    )

    if has_refusal_phrase:
        reason = "LLM explicitly indicated insufficient context in its answer"
        logger.info("Answer grounding check: passed=False | %s", reason)
        return False, reason

    passed = confidence >= threshold

    if passed:
        reason = (
            f"Answer grounding OK: confidence={confidence:.2f} >= "
            f"threshold={threshold:.2f}"
        )
    else:
        reason = (
            f"Answer grounding LOW: confidence={confidence:.2f} < "
            f"threshold={threshold:.2f}"
        )

    logger.info("Answer grounding check: passed=%s | %s", passed, reason)
    return passed, reason


def evaluate_grounding(
    docs_with_scores: list[tuple[Document, float]],
    confidence: float,
    answer_text: str,
    retrieval_threshold: float | None = None,
    confidence_threshold: float | None = None,
) -> GroundingResult:
    """Run the full dual-layer grounding check.

    Combines retrieval quality check and answer grounding check to determine
    whether the system should refuse to answer.

    Args:
        docs_with_scores: List of (Document, similarity_score) tuples.
        confidence: LLM's self-reported confidence score.
        answer_text: The generated answer text.
        retrieval_threshold: Override for retrieval quality threshold.
        confidence_threshold: Override for answer confidence threshold.

    Returns:
        GroundingResult with combined assessment.
    """
    # Layer 1: Retrieval quality
    retrieval_passed, best_score, retrieval_reason = check_retrieval_quality(
        docs_with_scores, threshold=retrieval_threshold
    )

    # Layer 2: Answer grounding
    answer_passed, answer_reason = check_answer_grounding(
        confidence, answer_text, threshold=confidence_threshold
    )

    # Combined decision: both must pass
    is_grounded = retrieval_passed and answer_passed
    should_refuse = not is_grounded

    # Combined confidence: geometric mean of retrieval score and LLM confidence
    combined_confidence = (best_score * confidence) ** 0.5 if best_score > 0 else 0.0

    reason_parts = []
    if not retrieval_passed:
        reason_parts.append(f"RETRIEVAL: {retrieval_reason}")
    if not answer_passed:
        reason_parts.append(f"GROUNDING: {answer_reason}")
    if is_grounded:
        reason_parts.append("Both retrieval quality and answer grounding checks passed")

    combined_reason = " | ".join(reason_parts)

    result = GroundingResult(
        is_grounded=is_grounded,
        should_refuse=should_refuse,
        confidence=round(combined_confidence, 4),
        reason=combined_reason,
        retrieval_quality_passed=retrieval_passed,
        answer_grounding_passed=answer_passed,
    )

    logger.info(
        "GROUNDING RESULT: grounded=%s, refuse=%s, confidence=%.4f | %s",
        result.is_grounded,
        result.should_refuse,
        result.confidence,
        result.reason,
    )

    return result
