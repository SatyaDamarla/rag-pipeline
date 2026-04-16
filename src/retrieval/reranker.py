from typing import List
from langchain_core.documents import Document


class CohereReranker:
    """
    Reranks retrieved chunks using Cohere's cross-encoder.
    Cross-encoders are slower but significantly more accurate than
    bi-encoder similarity search alone.
    """
    def __init__(self, model: str = "rerank-english-v3.0", top_n: int = 5):
        import cohere
        self.client = cohere.Client()
        self.model = model
        self.top_n = top_n

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        texts = [d.page_content for d in documents]
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=texts,
            top_n=self.top_n,
        )

        # preserve original metadata, re-order by rerank score
        reranked = []
        for result in response.results:
            doc = documents[result.index]
            doc.metadata["rerank_score"] = round(result.relevance_score, 4)
            reranked.append(doc)

        return reranked


class LocalReranker:
    """
    Fallback reranker using a local cross-encoder — no API key needed.
    Useful for offline dev and cost-free benchmarking.
    """
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_n: int = 5):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model)
        self.top_n = top_n

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        pairs = [(query, d.page_content) for d in documents]
        scores = self.model.predict(pairs)

        scored = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True
        )[:self.top_n]

        for score, doc in scored:
            doc.metadata["rerank_score"] = round(float(score), 4)

        return [doc for _, doc in scored]