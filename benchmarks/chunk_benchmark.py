from typing import List, Dict
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from src.ingestion.chunkers import get_chunker
import json, time

# Load your ground-truth Q&A pairs
# Format: [{"question": "...", "answer_source_chunk": "..."}]
with open("benchmarks/qa_pairs.json") as f:
    qa_pairs = json.load(f)

def reciprocal_rank(retrieved: List[str], relevant: str) -> float:
    for i, doc in enumerate(retrieved):
        if relevant.lower() in doc.lower():
            return 1.0 / (i + 1)
    return 0.0

def evaluate_strategy(strategy: str, docs: List[Document], k: int = 5) -> Dict:
    chunker = get_chunker(strategy)

    t0 = time.time()
    chunks = chunker.chunk(docs)
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    store = FAISS.from_documents(chunks, embedder)
    index_time = round(time.time() - t0, 2)

    rr_scores, recall_hits = [], []
    for qa in qa_pairs:
        results = store.similarity_search(qa["question"], k=k)
        retrieved_texts = [r.page_content for r in results]
        rr_scores.append(reciprocal_rank(retrieved_texts, qa["answer_source_chunk"]))
        recall_hits.append(any(qa["answer_source_chunk"].lower() in t.lower() for t in retrieved_texts))

    return {
        "strategy":    strategy,
        "num_chunks":  len(chunks),
        "mrr":         round(sum(rr_scores) / len(rr_scores), 3),
        "recall_at_k": round(sum(recall_hits) / len(recall_hits), 3),
        "index_time_s": index_time,
    }


if __name__ == "__main__":
    from src.ingestion.loaders import PDFLoader
    docs = PDFLoader().load("data/sample.pdf")

    for strategy in ["recursive", "token", "sentence_window"]:
        print(f"\nBenchmarking: {strategy}")
        results = evaluate_strategy(strategy, docs)
        for k, v in results.items():
            print(f"  {k}: {v}")