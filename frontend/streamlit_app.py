"""
DocQA Streamlit Frontend.

Provides a chat-based interface for the DocQA RAG system with:
- File upload sidebar (multi-file PDF)
- Chat interface with conversation history
- Citation display with expandable source chunks
- Refusal display with distinct styling
- Session management
"""

import os
import sys
import uuid
import time
import tempfile
import logging
from pathlib import Path

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.ingestion import ingest_pipeline
from app.generation import query_rag, clear_session_history
from app.retrieval import get_collection_stats

logger = logging.getLogger(__name__)

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="DocQA - Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS
# =============================================================================

st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    /* Citation box */
    .citation-box {
        background-color: #F0F9FF;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
        font-size: 0.85rem;
    }
    /* Refusal box */
    .refusal-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    /* Stats */
    .stats-box {
        background-color: #F3F4F6;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-size: 0.8rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialization
# =============================================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False


# =============================================================================
# Sidebar — File Upload & Settings
# =============================================================================

with st.sidebar:
    st.markdown("## 📄 DocQA")
    st.markdown("Upload PDFs and ask questions with cited answers.")
    st.divider()

    # File Upload
    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to ask questions about.",
    )

    collection_name = st.text_input(
        "Collection Name",
        value="docqa_default",
        help="Name for the document collection in the vector store.",
    )

    if uploaded_files:
        if st.button("📥 Ingest Documents", type="primary", use_container_width=True):
            with st.spinner("Processing documents..."):
                try:
                    # Save uploaded files to temp directory
                    temp_dir = tempfile.mkdtemp(prefix="docqa_st_")
                    temp_paths = []
                    for f in uploaded_files:
                        temp_path = os.path.join(temp_dir, f.name)
                        with open(temp_path, "wb") as fp:
                            fp.write(f.getbuffer())
                        temp_paths.append(temp_path)

                    # Run ingestion
                    result = ingest_pipeline(
                        file_paths=temp_paths,
                        collection_name=collection_name,
                    )

                    st.session_state.documents_loaded = True
                    st.success(
                        f"Ingested {result['files_processed']} file(s): "
                        f"{result['total_pages']} pages, "
                        f"{result['total_chunks']} chunks "
                        f"({result['total_time_s']:.1f}s)"
                    )

                    # Cleanup
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)

                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.divider()

    # Collection Stats
    st.markdown("### Collection Info")
    try:
        stats = get_collection_stats(collection_name)
        if stats["status"] == "ready":
            st.metric("Documents", stats["document_count"])
            st.session_state.documents_loaded = True
        else:
            st.info("No documents loaded yet.")
    except Exception:
        st.info("No documents loaded yet.")

    st.divider()

    # Settings
    st.markdown("### Settings")
    settings = get_settings()

    k_value = st.slider(
        "Retrieval chunks (k)",
        min_value=1,
        max_value=10,
        value=settings.retriever_k,
        help="Number of document chunks to retrieve per query.",
    )

    st.divider()

    # Session Management
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        clear_session_history(st.session_state.session_id)
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")


# =============================================================================
# Main Chat Interface
# =============================================================================

st.markdown('<div class="main-header">DocQA - Document Q&A</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ask questions about your documents. '
    'Answers include citations and confidence scores.</div>',
    unsafe_allow_html=True,
)

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show citations for assistant messages
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]

            # Refusal notice
            if meta.get("refused"):
                st.markdown(
                    '<div class="refusal-box">⚠️ The system determined it does not have '
                    'enough context to answer this question confidently.</div>',
                    unsafe_allow_html=True,
                )

            # Stats
            st.markdown(
                f'<div class="stats-box">'
                f'Confidence: {meta.get("confidence", 0):.0%} | '
                f'Retrieval: {meta.get("retrieval_time_ms", 0):.0f}ms | '
                f'Generation: {meta.get("generation_time_ms", 0):.0f}ms'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Citations
            if meta.get("citations"):
                with st.expander(f"📎 {len(meta['citations'])} Citation(s)"):
                    for cite in meta["citations"]:
                        st.markdown(f"- **{cite['source']}**, Page {cite['page']}")

            # Source chunks
            if meta.get("sources"):
                with st.expander(f"📚 {len(meta['sources'])} Source Chunk(s)"):
                    for i, src in enumerate(meta["sources"]):
                        st.markdown(
                            f'<div class="citation-box">'
                            f'<strong>[{i+1}] {src["source"]}, Page {src["page"]}</strong> '
                            f'(score: {src["score"]:.3f})<br/>'
                            f'{src["content"]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )


# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = query_rag(
                    question=prompt,
                    session_id=st.session_state.session_id,
                    collection_name=collection_name,
                    k=k_value,
                )

                # Display answer
                st.markdown(result["answer"])

                # Refusal notice
                if result["refused"]:
                    st.markdown(
                        '<div class="refusal-box">⚠️ The system determined it does not have '
                        'enough context to answer this question confidently.</div>',
                        unsafe_allow_html=True,
                    )

                # Stats
                st.markdown(
                    f'<div class="stats-box">'
                    f'Confidence: {result["confidence"]:.0%} | '
                    f'Retrieval: {result["retrieval_time_ms"]:.0f}ms | '
                    f'Generation: {result["generation_time_ms"]:.0f}ms'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Citations
                if result["citations"]:
                    with st.expander(f"📎 {len(result['citations'])} Citation(s)"):
                        for cite in result["citations"]:
                            st.markdown(f"- **{cite['source']}**, Page {cite['page']}")

                # Source chunks
                if result["sources"]:
                    with st.expander(f"📚 {len(result['sources'])} Source Chunk(s)"):
                        for i, src in enumerate(result["sources"]):
                            st.markdown(
                                f'<div class="citation-box">'
                                f'<strong>[{i+1}] {src["source"]}, Page {src["page"]}</strong> '
                                f'(score: {src["score"]:.3f})<br/>'
                                f'{src["content"]}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                # Save to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "metadata": {
                        "citations": result["citations"],
                        "confidence": result["confidence"],
                        "refused": result["refused"],
                        "sources": result["sources"],
                        "retrieval_time_ms": result["retrieval_time_ms"],
                        "generation_time_ms": result["generation_time_ms"],
                    },
                })

            except RuntimeError as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    st.error(
                        "⚠️ LLM API quota exceeded. Please check your API credits "
                        "or switch to a local model (Ollama) in the .env file."
                    )
                else:
                    st.error(f"Error: {e}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {e}",
                })

            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {e}",
                })
