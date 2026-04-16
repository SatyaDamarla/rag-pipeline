# RAG Pipeline — Production-Grade Document Q&A System

A configurable, end-to-end Retrieval-Augmented Generation pipeline for document Q&A, built with pluggable chunking strategies, semantic retrieval with reranking, and automated evaluation using RAGAS.

## Key features

- **Pluggable chunking strategies** — recursive, token-based, and sentence-window chunkers behind a common interface
- **Semantic retrieval + reranking** — FAISS vector search followed by cross-encoder reranking for higher precision
- **Source citations** — every answer includes the source document and page number
- **Automated evaluation** — RAGAS-based eval harness measuring faithfulness, answer relevance, and context recall
- **Config-driven** — swap chunking strategies, vector stores, or LLMs via YAML without code changes
- **Interactive CLI** — query the system in natural language from the terminal

## Architecture

```
Query → Embed → Vector search (top-k) → Rerank → LLM generation → Answer + sources

rag-pipeline/
├── src/
│   ├── ingestion/       # document loaders, chunkers, embedders, indexers
│   ├── retrieval/       # retriever, reranker, RAG orchestrator
│   └── generation/      # prompts, LLM generator
├── benchmarks/          # RAGAS eval harness + ground-truth Q&A
├── configs/             # pipeline.yaml — all parameters externalized
├── data/                # source documents + vector store
└── main.py              # interactive Q&A CLI

## Tech stack

- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: Google `gemini-embedding-001`
- **Vector store**: FAISS
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, no API)
- **Framework**: LangChain
- **Evaluation**: RAGAS
- **PDF parsing**: PyMuPDF

## Evaluation results (RAGAS)

Measured on a ground-truth Q&A dataset from the indexed document:

| Metric             | Score   |
|--------------------|---------|
| Answer relevancy   | 0.920   |
| Faithfulness       | _TBD_   |
| Context recall     | _TBD_   |

- **Faithfulness** — answers are grounded in retrieved context, not hallucinated
- **Answer relevancy** — answers directly address the question asked
- **Context recall** — retrieval finds relevant chunks from the corpus

_Results will be updated as the full benchmark is run across the complete Q&A set._

## Senior engineering decisions

1. **Abstraction over chunking** — each strategy (recursive, token, sentence-window) implements a common `BaseChunker` interface. Swapping strategies is a one-line config change, enabling apples-to-apples benchmarking.

2. **Reranking beyond similarity search** — added a cross-encoder reranker on top of FAISS semantic search. Cross-encoders are slower but significantly more accurate than bi-encoder similarity alone.

3. **Deterministic generation** — `temperature=0` for factual Q&A to eliminate hallucination drift across runs.

4. **Source attribution built in** — every answer includes filename and page number. No answer without a citation.

5. **Config-driven pipeline** — all tunable parameters (chunk size, top-k, model name, reranker type) live in `configs/pipeline.yaml`. Non-engineers can tune without touching Python.

6. **Retry logic on embedding API** — exponential backoff on transient API failures. Production-grade robustness.

## Quick start

### Prerequisites

- Python 3.11+
- Google AI Studio API key ([get one here](https://aistudio.google.com/app/apikey))

### Setup

```bash
git clone https://github.com/SatyaDamarla/rag-pipeline.git
cd rag-pipeline

python -m venv .venv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

### Configure

Create a `.env` file:

GOOGLE_API_KEY=your_key_here

Drop documents into `data/docs/`:

```bash
mkdir -p data/docs data/vector_store
cp your_document.pdf data/docs/
```

### Run

Build the index:

```python
# In main.py, uncomment the ingestion lines once
from src.ingestion.pipeline import IngestionPipeline
pipeline = IngestionPipeline("configs/pipeline.yaml")
pipeline.run(["data/docs/your_document.pdf"])
```

Start the interactive Q&A:

```bash
python main.py
```

Example:
You: What is the main thesis of the document?
A: The document argues that [...] [Source: document.pdf, page 3]
Sources: ['data/docs/document.pdf p.3', 'data/docs/document.pdf p.5']

### Run evaluation

```bash
python benchmarks/ragas_eval.py
```

Results are saved to `benchmarks/ragas_results.json`.

## Configuration

All pipeline behavior is controlled via `configs/pipeline.yaml`:

```yaml
chunking:
  strategy: recursive         # recursive | token | sentence_window
  chunk_size: 512
  chunk_overlap: 64

vector_store:
  type: faiss                 # faiss | chroma
  path: ./data/vector_store

retrieval:
  top_k: 10                   # candidates from vector search
  top_n: 5                    # kept after reranking
  reranker: local             # local | cohere

generation:
  model: models/gemini-2.5-flash
  temperature: 0.0
```

## Roadmap

- [x] Document ingestion with pluggable chunking
- [x] Semantic retrieval + reranking
- [x] Source-cited answer generation
- [x] Interactive CLI
- [x] RAGAS eval harness
- [ ] Full benchmark across all chunking strategies
- [ ] CI/CD with GitHub Actions running eval on every PR
- [ ] Streamlit / Gradio UI for public demo
- [ ] Deploy to Hugging Face Spaces

## License

MIT