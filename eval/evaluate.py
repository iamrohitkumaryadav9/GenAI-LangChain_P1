"""
DocQA Evaluation Harness.

Runs the RAG pipeline against a test set of Q&A pairs and scores with RAGAS.
Produces results.md with per-question and aggregate scores.

Evaluates two chunking configurations:
- Config A: chunk_size=500, overlap=50
- Config B: chunk_size=1000, overlap=200

Usage:
    python eval/evaluate.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
# Reduce noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


CHUNKING_CONFIGS = {
    "config_a": {"chunk_size": 500, "chunk_overlap": 50, "label": "500/50 (small chunks)"},
    "config_b": {"chunk_size": 1000, "chunk_overlap": 200, "label": "1000/200 (large chunks)"},
}


def load_test_set(path: str | Path) -> list[dict]:
    """Load the Q&A test set from JSON.

    Args:
        path: Path to qa_test_set.json.

    Returns:
        List of test case dicts.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ingest_for_config(
    pdf_path: str,
    config_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    """Ingest the sample document with a specific chunking configuration.

    Args:
        pdf_path: Path to the sample PDF.
        config_name: Config identifier (used as collection name).
        chunk_size: Chunk size in characters.
        chunk_overlap: Overlap between chunks.

    Returns:
        Collection name.
    """
    from app.ingestion import ingest_pipeline

    collection_name = f"eval_{config_name}"

    logger.info(
        "Ingesting with %s (chunk_size=%d, overlap=%d)",
        config_name, chunk_size, chunk_overlap,
    )

    result = ingest_pipeline(
        file_paths=[pdf_path],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        collection_name=collection_name,
    )

    logger.info(
        "Ingestion complete: %d chunks in %.2fs",
        result["total_chunks"],
        result["total_time_s"],
    )
    return collection_name


def run_rag_on_test_set(
    test_set: list[dict],
    collection_name: str,
) -> list[dict]:
    """Run the RAG pipeline on all test questions.

    Args:
        test_set: List of test case dicts.
        collection_name: ChromaDB collection to query.

    Returns:
        List of result dicts with question, answer, contexts, ground_truth.
    """
    from app.generation import query_rag

    results = []
    for i, test_case in enumerate(test_set):
        question = test_case["question"]
        ground_truth = test_case["ground_truth"]
        is_trap = test_case.get("is_trap", False)

        logger.info("[%d/%d] Q: %s", i + 1, len(test_set), question[:60])

        try:
            rag_result = query_rag(
                question=question,
                session_id=f"eval_{collection_name}_{i}",
                collection_name=collection_name,
            )

            results.append({
                "question": question,
                "answer": rag_result["answer"],
                "contexts": [s["content"] for s in rag_result["sources"]],
                "ground_truth": ground_truth,
                "is_trap": is_trap,
                "refused": rag_result["refused"],
                "confidence": rag_result["confidence"],
                "retrieval_time_ms": rag_result["retrieval_time_ms"],
                "generation_time_ms": rag_result["generation_time_ms"],
                "citations": rag_result["citations"],
            })

            logger.info(
                "  -> confidence=%.2f, refused=%s, citations=%d",
                rag_result["confidence"],
                rag_result["refused"],
                len(rag_result["citations"]),
            )

        except Exception as e:
            logger.error("  -> ERROR: %s", e)
            results.append({
                "question": question,
                "answer": f"ERROR: {e}",
                "contexts": [],
                "ground_truth": ground_truth,
                "is_trap": is_trap,
                "refused": False,
                "confidence": 0.0,
                "retrieval_time_ms": 0,
                "generation_time_ms": 0,
                "citations": [],
                "error": str(e),
            })

    return results


def score_with_ragas(results: list[dict]) -> dict | None:
    """Score results using RAGAS metrics.

    Args:
        results: List of result dicts from run_rag_on_test_set.

    Returns:
        Dict with per-question and aggregate scores, or None if RAGAS fails.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        # Filter out error results and trap questions for RAGAS scoring
        valid_results = [
            r for r in results
            if "error" not in r and not r["is_trap"]
        ]

        if not valid_results:
            logger.error("No valid results to score")
            return None

        # Build RAGAS dataset
        data = {
            "question": [r["question"] for r in valid_results],
            "answer": [r["answer"] for r in valid_results],
            "contexts": [r["contexts"] for r in valid_results],
            "ground_truth": [r["ground_truth"] for r in valid_results],
        }
        dataset = Dataset.from_dict(data)

        logger.info("Running RAGAS evaluation on %d questions...", len(valid_results))

        ragas_result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )

        scores = {
            "aggregate": {
                "faithfulness": ragas_result.get("faithfulness", None),
                "answer_relevancy": ragas_result.get("answer_relevancy", None),
                "context_precision": ragas_result.get("context_precision", None),
                "context_recall": ragas_result.get("context_recall", None),
            },
            "per_question": ragas_result.to_pandas().to_dict("records") if hasattr(ragas_result, "to_pandas") else [],
        }

        logger.info("RAGAS scores: %s", scores["aggregate"])
        return scores

    except ImportError as e:
        logger.warning("RAGAS not available: %s. Skipping RAGAS scoring.", e)
        return None
    except Exception as e:
        logger.error("RAGAS scoring failed: %s", e)
        return None


def generate_results_md(
    config_results: dict[str, dict],
    output_path: str | Path,
) -> None:
    """Generate the results.md file with evaluation results.

    Args:
        config_results: Dict mapping config names to their results.
        output_path: Where to write results.md.
    """
    lines = [
        "# DocQA Evaluation Results",
        "",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
    ]

    for config_name, data in config_results.items():
        config_label = data.get("label", config_name)
        results = data["results"]
        ragas_scores = data.get("ragas_scores")
        ingestion_stats = data.get("ingestion_stats", {})

        lines.append(f"## Configuration: {config_label}")
        lines.append("")
        lines.append(f"- **Chunk size**: {data.get('chunk_size', 'N/A')}")
        lines.append(f"- **Chunk overlap**: {data.get('chunk_overlap', 'N/A')}")
        lines.append(f"- **Total chunks**: {ingestion_stats.get('total_chunks', 'N/A')}")
        lines.append("")

        # Per-question results table
        lines.append("### Per-Question Results")
        lines.append("")
        lines.append("| # | Question (truncated) | Confidence | Refused | Citations | Retrieval (ms) | Generation (ms) |")
        lines.append("|---|---------------------|-----------|---------|-----------|---------------|----------------|")

        for i, r in enumerate(results):
            q_short = r["question"][:40] + ("..." if len(r["question"]) > 40 else "")
            is_trap_marker = " [TRAP]" if r.get("is_trap") else ""
            error_marker = " [ERROR]" if "error" in r else ""
            lines.append(
                f"| {i+1} | {q_short}{is_trap_marker}{error_marker} | "
                f"{r['confidence']:.2f} | "
                f"{'Yes' if r['refused'] else 'No'} | "
                f"{len(r.get('citations', []))} | "
                f"{r.get('retrieval_time_ms', 0):.0f} | "
                f"{r.get('generation_time_ms', 0):.0f} |"
            )

        lines.append("")

        # Trap question analysis
        trap_results = [r for r in results if r.get("is_trap")]
        if trap_results:
            lines.append("### Trap Question Analysis")
            lines.append("")
            for r in trap_results:
                passed = r["refused"] or r["confidence"] < 0.3
                status = "PASSED (correctly refused)" if passed else "FAILED (did not refuse)"
                lines.append(f"- **Q**: {r['question']}")
                lines.append(f"- **Status**: {status}")
                lines.append(f"- **Confidence**: {r['confidence']:.2f}")
                lines.append(f"- **Answer**: {r['answer'][:200]}...")
                lines.append("")

        # RAGAS scores
        if ragas_scores and ragas_scores.get("aggregate"):
            agg = ragas_scores["aggregate"]
            lines.append("### RAGAS Scores (Aggregate)")
            lines.append("")
            lines.append("| Metric | Score |")
            lines.append("|--------|-------|")
            for metric, score in agg.items():
                score_str = f"{score:.4f}" if score is not None else "N/A"
                lines.append(f"| {metric} | {score_str} |")
            lines.append("")
        else:
            lines.append("### RAGAS Scores")
            lines.append("")
            lines.append("*RAGAS scoring was not available or failed. "
                         "Ensure `ragas` is installed and API keys are configured.*")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Comparison section
    if len(config_results) > 1:
        lines.append("## Chunking Configuration Comparison")
        lines.append("")
        lines.append("| Metric | " + " | ".join(
            data.get("label", name) for name, data in config_results.items()
        ) + " |")
        lines.append("|--------| " + " | ".join("------" for _ in config_results) + " |")

        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        for metric in metrics:
            row = f"| {metric} |"
            for name, data in config_results.items():
                ragas = data.get("ragas_scores", {})
                agg = ragas.get("aggregate", {}) if ragas else {}
                score = agg.get(metric)
                row += f" {score:.4f} |" if score is not None else " N/A |"
            lines.append(row)

        lines.append("")

        # Average performance metrics
        lines.append("### Performance Metrics")
        lines.append("")
        lines.append("| Metric | " + " | ".join(
            data.get("label", name) for name, data in config_results.items()
        ) + " |")
        lines.append("|--------| " + " | ".join("------" for _ in config_results) + " |")

        for metric_name, metric_key in [
            ("Avg Confidence", "confidence"),
            ("Avg Retrieval (ms)", "retrieval_time_ms"),
            ("Avg Generation (ms)", "generation_time_ms"),
        ]:
            row = f"| {metric_name} |"
            for name, data in config_results.items():
                valid = [r for r in data["results"] if "error" not in r and not r.get("is_trap")]
                if valid:
                    avg = sum(r[metric_key] for r in valid) / len(valid)
                    row += f" {avg:.2f} |"
                else:
                    row += " N/A |"
            lines.append(row)

        lines.append("")

    # Write file
    output_path = Path(output_path)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Results written to %s", output_path)


def main():
    """Run the full evaluation harness."""
    eval_dir = Path(__file__).parent
    project_root = eval_dir.parent

    pdf_path = str(eval_dir / "sample_doc.pdf")
    test_set_path = eval_dir / "qa_test_set.json"
    results_path = project_root / "results.md"

    # Verify files exist
    if not Path(pdf_path).exists():
        logger.error("sample_doc.pdf not found. Run eval/generate_sample_doc.py first.")
        sys.exit(1)
    if not test_set_path.exists():
        logger.error("qa_test_set.json not found.")
        sys.exit(1)

    # Load test set
    test_set = load_test_set(test_set_path)
    logger.info("Loaded %d test questions (%d trap)", len(test_set), sum(1 for t in test_set if t.get("is_trap")))

    config_results = {}

    for config_name, config in CHUNKING_CONFIGS.items():
        logger.info("=" * 60)
        logger.info("EVALUATING: %s — %s", config_name, config["label"])
        logger.info("=" * 60)

        # Ingest with this config
        try:
            collection_name = ingest_for_config(
                pdf_path=pdf_path,
                config_name=config_name,
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"],
            )
        except Exception as e:
            logger.error("Ingestion failed for %s: %s", config_name, e)
            continue

        # Run RAG on test set
        results = run_rag_on_test_set(test_set, collection_name)

        # Score with RAGAS
        ragas_scores = score_with_ragas(results)

        config_results[config_name] = {
            "label": config["label"],
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
            "results": results,
            "ragas_scores": ragas_scores,
            "ingestion_stats": {"total_chunks": len(results)},
        }

    # Generate results.md
    if config_results:
        generate_results_md(config_results, results_path)
        logger.info("Evaluation complete! Results at: %s", results_path)
    else:
        logger.error("No configs evaluated successfully.")
        sys.exit(1)


if __name__ == "__main__":
    main()
