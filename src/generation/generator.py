from typing import List
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from .prompts import RAG_PROMPT


class RAGGenerator:
    def __init__(self, model: str = "models/gemini-2.5-flash", temperature: float = 0.0):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        self.prompt = RAG_PROMPT

    def generate(self, question: str, documents: List[Document]) -> dict:
        context = self._format_context(documents)
        chain = self.prompt | self.llm
        response = chain.invoke({"context": context, "question": question})

        return {
            "answer": response.content,
            "sources": self._extract_sources(documents),
            "num_chunks_used": len(documents),
        }

    def _format_context(self, documents: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            page_str = f", page {page}" if page else ""
            parts.append(f"[{i}] (source: {source}{page_str})\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, documents: List[Document]) -> List[dict]:
        seen = set()
        sources = []
        for doc in documents:
            key = (doc.metadata.get("source"), doc.metadata.get("page"))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page"),
                    "rerank_score": doc.metadata.get("rerank_score"),
                })
        return sources