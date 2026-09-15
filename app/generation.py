"""
DocQA Generation Module.

Builds the RAG generation chain using LangChain Expression Language (LCEL).
Handles:
- LLM factory (OpenAI / Anthropic / Ollama)
- Prompt construction with citation enforcement
- Citation formatting and extraction
- Multi-turn conversation support with session memory
"""

import logging
import re
import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.config import get_settings
from app.retrieval import retrieve_with_scores

logger = logging.getLogger(__name__)

# In-memory session storage for multi-turn conversations
_session_histories: dict[str, list[dict[str, str]]] = {}


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are DocQA, a precise document question-answering assistant.
Your job is to answer questions based ONLY on the provided context chunks.

RULES:
1. Answer ONLY using information from the provided context.
2. For EVERY claim in your answer, include an inline citation in the format: [Source: <filename>, Page <page_number>]
3. If the context does not contain enough information to answer the question confidently, you MUST respond with:
   "I don't have enough information in the provided documents to answer this question."
   Do NOT guess, speculate, or use external knowledge.
4. Be concise but thorough. Prefer direct answers over lengthy explanations.
5. If the question is a follow-up to a previous conversation, use the chat history for context but still only cite the provided documents.

CONFIDENCE ASSESSMENT:
After your answer, on a new line, provide a confidence assessment in exactly this format:
CONFIDENCE: <score>
Where <score> is a float between 0.0 and 1.0 indicating how confident you are that your answer is fully supported by the context.
- 1.0 = answer is directly and completely stated in the context
- 0.7-0.9 = answer is well-supported but requires minor inference
- 0.3-0.6 = answer is partially supported, some gaps
- 0.0-0.2 = context is insufficient, you should refuse to answer"""


RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", """Context chunks:
{context}

Question: {question}"""),
])


# =============================================================================
# LLM Factory
# =============================================================================

def get_llm() -> Any:
    """Get the configured LLM based on environment settings.

    Returns:
        A LangChain BaseChatModel instance (OpenAI, Anthropic, or Ollama).

    Raises:
        ValueError: If the configured provider is invalid or API key is missing.
    """
    settings = get_settings()

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        logger.info("Using OpenAI LLM: %s", settings.openai_model)
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
        )
    elif settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        logger.info("Using Anthropic LLM: %s", settings.anthropic_model)
        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=0.1,
        )
    elif settings.llm_provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        logger.info("Using Ollama LLM: %s", settings.ollama_model)
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


# =============================================================================
# Context Formatting
# =============================================================================

def format_context(docs_with_scores: list[tuple[Document, float]]) -> str:
    """Format retrieved documents into a context string for the prompt.

    Each chunk is labeled with its source, page number, and relevance score.

    Args:
        docs_with_scores: List of (Document, similarity_score) tuples.

    Returns:
        Formatted context string with source attributions.
    """
    context_parts = []
    for i, (doc, score) in enumerate(docs_with_scores):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Chunk {i + 1}] Source: {source}, Page {page} (relevance: {score:.3f})\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# Citation Extraction
# =============================================================================

def extract_citations(answer_text: str) -> list[dict[str, str]]:
    """Extract inline citations from the LLM's answer.

    Looks for patterns like [Source: filename.pdf, Page 3].

    Args:
        answer_text: The raw answer text from the LLM.

    Returns:
        List of dicts with 'source' and 'page' keys.
    """
    pattern = r'\[Source:\s*([^,\]]+),\s*Page\s*(\d+)\]'
    matches = re.findall(pattern, answer_text)

    citations = []
    seen = set()
    for source, page in matches:
        key = (source.strip(), page.strip())
        if key not in seen:
            seen.add(key)
            citations.append({
                "source": source.strip(),
                "page": page.strip(),
            })

    return citations


def extract_confidence(answer_text: str) -> float:
    """Extract the confidence score from the LLM's answer.

    Looks for 'CONFIDENCE: <score>' at the end of the answer.

    Args:
        answer_text: The raw answer text from the LLM.

    Returns:
        Confidence score between 0.0 and 1.0, or 0.5 as default.
    """
    pattern = r'CONFIDENCE:\s*(-?[\d.]+)'
    match = re.search(pattern, answer_text)
    if match:
        try:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        except ValueError:
            pass
    return 0.5  # Default if not found


def clean_answer(answer_text: str) -> str:
    """Remove the confidence line from the answer for display.

    Args:
        answer_text: The raw answer text from the LLM.

    Returns:
        Clean answer text without the confidence line.
    """
    # Remove CONFIDENCE: line
    cleaned = re.sub(r'\n*CONFIDENCE:\s*[\d.]+\s*$', '', answer_text).strip()
    return cleaned


# =============================================================================
# Session Management
# =============================================================================

def get_session_history(session_id: str) -> list[dict[str, str]]:
    """Get conversation history for a session.

    Args:
        session_id: Unique session identifier.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    if session_id not in _session_histories:
        _session_histories[session_id] = []
    return _session_histories[session_id]


def add_to_session_history(
    session_id: str,
    question: str,
    answer: str,
    max_history: int = 10,
) -> None:
    """Add a Q&A exchange to session history.

    Keeps only the most recent `max_history` exchanges.

    Args:
        session_id: Unique session identifier.
        question: The user's question.
        answer: The system's answer.
        max_history: Maximum number of exchanges to retain.
    """
    history = get_session_history(session_id)
    history.append({"role": "human", "content": question})
    history.append({"role": "assistant", "content": answer})

    # Trim to max_history exchanges (each exchange = 2 messages)
    max_messages = max_history * 2
    if len(history) > max_messages:
        _session_histories[session_id] = history[-max_messages:]


def clear_session_history(session_id: str) -> None:
    """Clear conversation history for a session.

    Args:
        session_id: Unique session identifier.
    """
    _session_histories.pop(session_id, None)


def _build_chat_history_messages(session_id: str) -> list[Any]:
    """Convert session history to LangChain message objects.

    Args:
        session_id: Unique session identifier.

    Returns:
        List of HumanMessage/AIMessage objects.
    """
    history = get_session_history(session_id)
    messages = []
    for msg in history:
        if msg["role"] == "human":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages


# =============================================================================
# Main Query Function
# =============================================================================

def query_rag(
    question: str,
    session_id: str = "default",
    collection_name: str | None = None,
    k: int | None = None,
) -> dict[str, Any]:
    """Run the full RAG query pipeline.

    Retrieves relevant chunks, constructs a grounded prompt with chat history,
    generates an answer with citations, and returns structured results.

    Args:
        question: The user's natural language question.
        session_id: Session ID for multi-turn conversation support.
        collection_name: ChromaDB collection to search. Uses config default if None.
        k: Number of chunks to retrieve. Uses config default if None.

    Returns:
        Dict with keys:
        - answer: Clean answer text
        - raw_answer: Full LLM response including confidence
        - citations: List of citation dicts
        - confidence: Float 0.0-1.0
        - sources: List of source documents with metadata
        - retrieval_time_ms: Retrieval latency
        - generation_time_ms: Generation latency
        - refused: Whether the system refused to answer

    Raises:
        RuntimeError: If LLM call fails.
    """
    logger.info("=" * 50)
    logger.info("RAG QUERY: '%s' (session=%s)", question[:80], session_id)
    logger.info("=" * 50)

    # Step 1: Retrieve with scores
    retrieval_start = time.perf_counter()
    docs_with_scores = retrieve_with_scores(
        query=question,
        collection_name=collection_name,
        k=k,
    )
    retrieval_time = (time.perf_counter() - retrieval_start) * 1000

    # Step 2: Format context
    context = format_context(docs_with_scores)

    # Step 3: Build chat history
    chat_history = _build_chat_history_messages(session_id)

    # Step 4: Generate answer via LCEL chain
    generation_start = time.perf_counter()
    try:
        llm = get_llm()
        chain = RAG_PROMPT_TEMPLATE | llm | StrOutputParser()

        raw_answer = chain.invoke({
            "context": context,
            "question": question,
            "chat_history": chat_history,
        })
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        raise RuntimeError(f"LLM generation failed: {e}") from e

    generation_time = (time.perf_counter() - generation_start) * 1000

    # Step 5: Parse answer
    confidence = extract_confidence(raw_answer)
    clean_ans = clean_answer(raw_answer)
    citations = extract_citations(raw_answer)

    # Step 6: Determine refusal
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
    refused = any(phrase in clean_ans.lower() for phrase in refusal_phrases)

    # Step 7: Update session history
    add_to_session_history(session_id, question, clean_ans)

    # Step 8: Build result
    sources = [
        {
            "content": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""),
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "score": round(score, 4),
        }
        for doc, score in docs_with_scores
    ]

    result = {
        "answer": clean_ans,
        "raw_answer": raw_answer,
        "citations": citations,
        "confidence": confidence,
        "sources": sources,
        "retrieval_time_ms": round(retrieval_time, 1),
        "generation_time_ms": round(generation_time, 1),
        "refused": refused,
    }

    logger.info(
        "RAG RESULT: confidence=%.2f, refused=%s, citations=%d, "
        "retrieval=%.0fms, generation=%.0fms",
        confidence,
        refused,
        len(citations),
        retrieval_time,
        generation_time,
    )

    return result
