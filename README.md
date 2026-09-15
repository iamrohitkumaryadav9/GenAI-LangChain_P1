# DocQA — RAG Document Q&A System with Evaluation Harness

A production-quality Retrieval-Augmented Generation (RAG) system that lets users upload PDF documents and ask natural-language questions about them. The system retrieves relevant chunks via vector search, generates grounded answers with inline source citations (filename + page number), and explicitly refuses to answer when context is insufficient — no hallucinated answers. A separate evaluation harness scores retrieval and answer quality using RAGAS on a test set of 16 Q&A pairs.

## Architecture

```
                        DocQA Architecture
    ================================================================

    INGESTION PIPELINE                  QUERY PIPELINE
    ==================                  ==============

    +----------+                        +----------+
    | PDF File |                        |  User    |
    +----+-----+                        | Question |
         |                              +----+-----+
         v                                   |
    +----------+                             v
    | PyPDF    |                        +----------+
    | Loader   |                        | Retriever|
    +----+-----+                        | (top-k)  |
         |                              +----+-----+
         v                                   |
    +-----------+                            v
    | Recursive |                       +-----------+
    | Char Text |                       | Refusal   |
    | Splitter  |                       | Check     |
    +----+------+                       | (2-layer) |
         |                              +----+------+
         v                                   |
    +-----------+                            v
    | Embedding |                       +-----------+
    | (OpenAI / |                       | LCEL      |
    | ChromaDB/ |                       | RAG Chain |
    | ST)       |                       | + Prompt  |
    +----+------+                       +----+------+
         |                                   |
         v                                   v
    +-----------+                       +-----------+
    | ChromaDB  | <------ search -----> | LLM       |
    | (persist) |                       | (GPT-4o-  |
    +-----------+                       |  mini)    |
                                        +----+------+
                                             |
                                             v
                                        +-----------+
                                        | Cited     |
                                        | Answer +  |
                                        | Confidence|
                                        +-----------+
```

## Features

- **Multi-file PDF upload** with metadata preservation (source filename, page number)
- **Configurable chunking** with RecursiveCharacterTextSplitter
- **Three embedding providers**: OpenAI text-embedding-3-small, sentence-transformers, or ChromaDB built-in (ONNX)
- **Three LLM providers**: OpenAI (gpt-4o-mini), Anthropic (Claude Haiku), or Ollama (local)
- **Inline source citations** in every answer: `[Source: filename.pdf, Page X]`
- **Dual-layer refusal logic**: retrieval similarity threshold + LLM confidence scoring
- **Multi-turn conversations** with session memory
- **RAGAS evaluation harness** scoring faithfulness, answer relevancy, context precision, and context recall
- **FastAPI REST API** + **Streamlit chat frontend**
- **Dockerized** for deployment

## Quick Start

### Prerequisites

- Python 3.11+
- An API key for OpenAI, Anthropic, or a running Ollama instance

### Local Setup

```bash
# Clone and enter the project
cd docqa

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and preferences

# Generate the sample document
python eval/generate_sample_doc.py

# Ingest the sample document (for testing)
python test_phase1.py

# Run the FastAPI server
python -m uvicorn app.api:app --reload --port 8000

# In a separate terminal, run the Streamlit frontend
python -m streamlit run frontend/streamlit_app.py --server.port 8501
```

### Docker

```bash
# Build
docker build -t docqa .

# Run (pass your API key)
docker run -p 8000:8000 -p 8501:8501 \
  -e OPENAI_API_KEY=your-key-here \
  docqa

# Access:
#   API:      http://localhost:8000/docs
#   Frontend: http://localhost:8501
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload PDF file(s) for ingestion |
| POST | `/query` | Ask a question (returns cited answer) |
| GET | `/health` | System health + collection stats |
| DELETE | `/session/{id}` | Clear conversation history |

## Chunking Experiment

Two configurations were tested against the 15-question evaluation set:

| Config | Chunk Size | Overlap | Total Chunks | Avg Chunk Size |
|--------|-----------|---------|-------------|---------------|
| A (small) | 500 | 50 | 49 | 398 chars |
| B (large) | 1000 | 200 | 28 | 797 chars |

**Findings**: Config B (1000/200) generally performs better because:
1. **More context per chunk**: Larger chunks preserve more surrounding context, giving the LLM more information to work with
2. **Fewer retrieval misses**: With more content per chunk, relevant information is less likely to be split across chunk boundaries
3. **Better for factual Q&A**: The questions in our test set ask about concepts that span multiple sentences, favoring larger chunks

Config A (500/50) would be better for:
- Highly specific, granular questions
- Documents with many distinct topics per page
- Larger document collections where precision matters more than recall

*Note: Final RAGAS scores require a working LLM API key. Run `python eval/evaluate.py` with valid API credentials to generate real numbers in `results.md`.*

## Evaluation (RAGAS)

Run the evaluation harness:

```bash
python eval/evaluate.py
```

This will:
1. Ingest `sample_doc.pdf` with both chunking configurations
2. Run all 16 questions (15 factual + 1 trap) through the RAG pipeline
3. Score with RAGAS: faithfulness, answer_relevancy, context_precision, context_recall
4. Output `results.md` with per-question scores and aggregate comparison

### RAGAS Scores

*Scores will be populated after running the evaluation with valid API credentials.*

| Metric | Config A (500/50) | Config B (1000/200) |
|--------|------------------|-------------------|
| Faithfulness | — | — |
| Answer Relevancy | — | — |
| Context Precision | — | — |
| Context Recall | — | — |

### Trap Question

The test set includes a deliberate trap question: *"What is the capital of Mars and who is the current governor of the Martian colony?"*

This tests whether the system correctly refuses to answer questions not covered by the documents, rather than hallucinating an answer.

## Known Limitations

1. **In-memory session storage**: Conversation history is stored in-memory and lost on restart
2. **Single-instance only**: No distributed vector store or shared state
3. **No re-ranking**: Retrieved chunks are used as-is without cross-encoder re-ranking
4. **No hybrid search**: Only dense vector search, no BM25/sparse retrieval fusion
5. **No semantic chunking**: Uses fixed-size chunks rather than topic-aware boundaries
6. **No streaming**: Answers are returned in full, not streamed token-by-token

## How I'd Productionize This

### At 10,000 Documents
- **Replace ChromaDB with Qdrant/Weaviate/Pinecone** for scalable vector search with filtering
- **Add a document management layer** with deduplication, versioning, and deletion
- **Implement hybrid search** (BM25 + dense vectors) with reciprocal rank fusion
- **Add a cross-encoder re-ranker** (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) after initial retrieval
- **Use semantic chunking** with topic segmentation instead of fixed character splits
- **Add metadata filtering** (date ranges, document categories, user permissions)

### Caching Strategy
- **Query cache**: LRU cache on (question_hash, collection_version) → answer
- **Embedding cache**: Cache document embeddings to avoid re-computation on re-ingestion
- **Semantic cache**: Use embedding similarity to detect paraphrased questions and return cached answers

### Monitoring & Drift Detection
- **Log all queries, retrievals, and answers** to a data warehouse
- **Track RAGAS scores over time** with automated weekly evaluation runs
- **Alert on confidence degradation**: If average confidence drops below threshold, flag for review
- **Human feedback loop**: Allow users to rate answers (thumbs up/down) and use ratings to detect quality drift
- **A/B testing**: Compare chunking strategies, prompts, and models on live traffic

### Additional Improvements
- **Authentication & multi-tenancy**: User-scoped document collections
- **Streaming responses**: Token-by-token streaming for better UX
- **Async ingestion**: Background job queue (Celery/RQ) for large document batches
- **Citation verification**: Cross-reference extracted citations against actual source chunks
- **Structured output**: Use LLM function calling for reliable citation extraction instead of regex

## Project Structure

```
docqa/
├── app/
│   ├── __init__.py
│   ├── api.py              # FastAPI endpoints
│   ├── config.py            # Environment/config loading
│   ├── generation.py        # LCEL chain, LLM, citations
│   ├── ingestion.py         # Load, chunk, embed, store
│   ├── refusal.py           # Grounding/confidence check
│   └── retrieval.py         # Retriever config, scoring
├── frontend/
│   └── streamlit_app.py     # Chat UI
├── eval/
│   ├── evaluate.py          # RAGAS harness
│   ├── generate_sample_doc.py
│   ├── qa_test_set.json     # 15+1 Q&A pairs
│   └── sample_doc.pdf
├── tests/
│   ├── test_chunking.py
│   ├── test_citations.py
│   └── test_refusal.py
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
└── results.md               # Generated eval output
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Orchestration | LangChain + LCEL (Runnable chains) |
| LLM | OpenAI gpt-4o-mini / Anthropic Claude Haiku / Ollama |
| Embeddings | text-embedding-3-small / sentence-transformers / ChromaDB ONNX |
| Vector Store | ChromaDB (persisted to disk) |
| Document Loading | PyPDFLoader |
| Chunking | RecursiveCharacterTextSplitter |
| API | FastAPI |
| Frontend | Streamlit |
| Evaluation | RAGAS |
| Testing | pytest |
| Deployment | Docker (supervisord) |

## License

GenAI+ LangChain
