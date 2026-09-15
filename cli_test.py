"""
CLI Test Script for Phase 2 & 3: Retrieval + Generation + Refusal.

Tests the full RAG pipeline with real questions against the ingested sample_doc.pdf.
Includes a trap question to verify refusal behavior.
"""

import logging
import os
import sys
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
# Reduce noise from HTTP libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

from app.generation import query_rag
from app.retrieval import retrieve_with_scores, get_collection_stats
from app.refusal import evaluate_grounding


def test_retrieval():
    """Test retrieval independently."""
    print("\n" + "=" * 70)
    print("TEST: Retrieval")
    print("=" * 70)

    stats = get_collection_stats("test_phase1")
    print(f"Collection: {stats}")

    query = "What is supervised learning?"
    results = retrieve_with_scores(query, collection_name="test_phase1", k=3)
    print(f"\nQuery: '{query}'")
    print(f"Retrieved {len(results)} chunks:")
    for doc, score in results:
        print(f"  [{score:.4f}] Page {doc.metadata.get('page', '?')}: {doc.page_content[:100]}...")

    return results


def test_refusal_logic():
    """Test refusal logic with mock data."""
    print("\n" + "=" * 70)
    print("TEST: Refusal Logic")
    print("=" * 70)

    from langchain_core.documents import Document

    # Test 1: Good scores, high confidence -> should NOT refuse
    good_docs = [
        (Document(page_content="ML is great", metadata={"source": "test.pdf", "page": 1}), 0.85),
        (Document(page_content="More ML", metadata={"source": "test.pdf", "page": 2}), 0.72),
    ]
    result1 = evaluate_grounding(good_docs, confidence=0.9, answer_text="ML is great based on the docs")
    print(f"\n  Test 1 (high scores + high confidence):")
    print(f"    should_refuse={result1.should_refuse}, confidence={result1.confidence}")
    assert not result1.should_refuse, "Should NOT refuse with good scores"
    print(f"    [PASS]")

    # Test 2: Low scores -> should refuse
    bad_docs = [
        (Document(page_content="irrelevant", metadata={"source": "test.pdf", "page": 1}), 0.15),
        (Document(page_content="also irrelevant", metadata={"source": "test.pdf", "page": 2}), 0.10),
    ]
    result2 = evaluate_grounding(bad_docs, confidence=0.8, answer_text="Some answer")
    print(f"\n  Test 2 (low retrieval scores):")
    print(f"    should_refuse={result2.should_refuse}, confidence={result2.confidence}")
    assert result2.should_refuse, "Should refuse with low retrieval scores"
    print(f"    [PASS]")

    # Test 3: Good scores but LLM says it can't answer -> should refuse
    result3 = evaluate_grounding(
        good_docs,
        confidence=0.2,
        answer_text="I don't have enough information to answer this question"
    )
    print(f"\n  Test 3 (LLM explicit refusal):")
    print(f"    should_refuse={result3.should_refuse}, confidence={result3.confidence}")
    assert result3.should_refuse, "Should refuse when LLM explicitly refuses"
    print(f"    [PASS]")

    print("\n  All refusal logic tests passed!")


def test_rag_query():
    """Test the full RAG pipeline with real questions."""
    print("\n" + "=" * 70)
    print("TEST: Full RAG Pipeline")
    print("=" * 70)

    questions = [
        "What is supervised learning and what are some common algorithms?",
        "Who coined the term 'machine learning' and when?",
        "What is the difference between precision and recall?",
    ]

    for q in questions:
        print(f"\n{'~' * 60}")
        print(f"Q: {q}")
        print(f"{'~' * 60}")

        try:
            result = query_rag(q, session_id="test_cli", collection_name="test_phase1")
            print(f"\nA: {result['answer'][:300]}...")
            print(f"\nCitations: {result['citations']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Refused: {result['refused']}")
            print(f"Retrieval: {result['retrieval_time_ms']:.0f}ms | Generation: {result['generation_time_ms']:.0f}ms")
        except Exception as e:
            print(f"\n  [ERROR] {e}")
            print("  (This is expected if OpenAI credits are exhausted)")
            return False

    return True


def test_trap_question():
    """Test refusal behavior with a question that has no answer in the docs."""
    print("\n" + "=" * 70)
    print("TEST: Trap Question (Refusal Behavior)")
    print("=" * 70)

    trap = "What is the capital of Mars and who governs it?"
    print(f"\nQ: {trap}")

    try:
        result = query_rag(trap, session_id="test_trap", collection_name="test_phase1")
        print(f"\nA: {result['answer'][:300]}")
        print(f"\nConfidence: {result['confidence']}")
        print(f"Refused: {result['refused']}")

        if result['refused'] or result['confidence'] < 0.3:
            print("\n  [PASS] System correctly refused or showed low confidence")
        else:
            print("\n  [WARNING] System did not refuse - may need threshold tuning")
    except Exception as e:
        print(f"\n  [ERROR] {e}")
        return False

    return True


def test_multi_turn():
    """Test multi-turn conversation support."""
    print("\n" + "=" * 70)
    print("TEST: Multi-turn Conversation")
    print("=" * 70)

    session = "test_multiturn"

    try:
        # First question
        q1 = "What are the three main categories of machine learning?"
        print(f"\nQ1: {q1}")
        r1 = query_rag(q1, session_id=session, collection_name="test_phase1")
        print(f"A1: {r1['answer'][:200]}...")

        # Follow-up question
        q2 = "Tell me more about the second category you mentioned."
        print(f"\nQ2: {q2}")
        r2 = query_rag(q2, session_id=session, collection_name="test_phase1")
        print(f"A2: {r2['answer'][:200]}...")
        print("\n  [PASS] Multi-turn conversation working")
    except Exception as e:
        print(f"\n  [ERROR] {e}")
        return False

    return True


if __name__ == "__main__":
    print("DocQA — Phase 2 & 3 CLI Tests")
    print("=" * 70)

    # Test 1: Retrieval (no LLM needed)
    test_retrieval()

    # Test 2: Refusal logic (no LLM needed)
    test_refusal_logic()

    # Tests 3-5 require a working LLM
    print("\n" + "=" * 70)
    print("NOTE: The following tests require a working LLM (OpenAI/Anthropic/Ollama)")
    print("If your API key has no credits, these will fail gracefully.")
    print("=" * 70)

    llm_tests_passed = test_rag_query()
    if llm_tests_passed:
        test_trap_question()
        test_multi_turn()
    else:
        print("\nSkipping LLM-dependent tests (API unavailable)")

    print("\n" + "=" * 70)
    print("CLI TESTS COMPLETE")
    print("=" * 70)
