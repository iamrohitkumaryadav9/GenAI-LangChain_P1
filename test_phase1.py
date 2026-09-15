"""
Phase 1 Integration Test: Test the ingestion pipeline end-to-end.

Loads sample_doc.pdf, chunks it, embeds with OpenAI, and stores in ChromaDB.
Prints stats and verifies metadata preservation.
"""

import logging
import os
import sys
from pathlib import Path

# Force UTF-8 output on Windows
os.environ["PYTHONIOENCODING"] = "utf-8"

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

from app.ingestion import ingest_pipeline, load_pdf, chunk_documents


def test_ingestion():
    """Run the full ingestion pipeline and verify results."""
    sample_pdf = Path(__file__).parent / "eval" / "sample_doc.pdf"

    print("\n" + "=" * 70)
    print("PHASE 1 TEST: Ingestion Pipeline")
    print("=" * 70)

    # Test 1: Load PDF
    print("\n--- Test 1: Load PDF ---")
    docs = load_pdf(sample_pdf)
    print(f"  [OK] Loaded {len(docs)} pages")
    for doc in docs[:3]:
        print(f"    Page {doc.metadata['page']}: {len(doc.page_content)} chars | Source: {doc.metadata['source']}")
    print(f"    ... ({len(docs)} pages total)")

    # Test 2: Chunk with config A (500/50)
    print("\n--- Test 2: Chunk Config A (chunk_size=500, overlap=50) ---")
    chunks_a = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    print(f"  [OK] Created {len(chunks_a)} chunks")
    print(f"    Avg chunk size: {sum(len(c.page_content) for c in chunks_a) / len(chunks_a):.0f} chars")
    print(f"    Sample metadata: {chunks_a[0].metadata}")

    # Test 3: Chunk with config B (1000/200)
    print("\n--- Test 3: Chunk Config B (chunk_size=1000, overlap=200) ---")
    chunks_b = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)
    print(f"  [OK] Created {len(chunks_b)} chunks")
    print(f"    Avg chunk size: {sum(len(c.page_content) for c in chunks_b) / len(chunks_b):.0f} chars")

    # Test 4: Metadata preservation
    print("\n--- Test 4: Metadata Preservation ---")
    for chunk in chunks_a[:3]:
        meta = chunk.metadata
        assert "source" in meta, "Missing 'source' metadata"
        assert "page" in meta, "Missing 'page' metadata"
        assert "chunk_index" in meta, "Missing 'chunk_index' metadata"
        print(f"    Chunk {meta['chunk_index']}: source={meta['source']}, page={meta['page']}")
    print("  [OK] All metadata fields preserved")

    # Test 5: Full pipeline with embed + store
    print("\n--- Test 5: Full Ingestion Pipeline (embed + store) ---")
    result = ingest_pipeline(
        file_paths=[str(sample_pdf)],
        collection_name="test_phase1",
    )
    print(f"  [OK] Pipeline complete:")
    print(f"    Files: {result['files_processed']}")
    print(f"    Pages: {result['total_pages']}")
    print(f"    Chunks: {result['total_chunks']}")
    print(f"    Embedding time: {result['storage']['embedding_time_s']}s")
    print(f"    Total time: {result['total_time_s']}s")

    # Test 6: Verify Chroma collection
    print("\n--- Test 6: Verify ChromaDB Collection ---")
    from langchain_chroma import Chroma
    from app.ingestion import get_embedding_function
    from app.config import get_settings

    settings = get_settings()
    vectorstore = Chroma(
        collection_name="test_phase1",
        embedding_function=get_embedding_function(),
        persist_directory=settings.chroma_persist_dir,
    )
    collection_count = vectorstore._collection.count()
    print(f"  [OK] ChromaDB collection 'test_phase1' has {collection_count} documents")

    # Quick similarity search test
    results = vectorstore.similarity_search("What is supervised learning?", k=2)
    print(f"  [OK] Similarity search returned {len(results)} results")
    for r in results:
        print(f"    - Page {r.metadata['page']}: {r.page_content[:80]}...")

    print("\n" + "=" * 70)
    print("ALL PHASE 1 TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_ingestion()
