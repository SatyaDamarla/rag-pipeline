from typing import List, Optional, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class Retriever:
    def __init__(
        self,
        store_path: str = "./data/vector_store",
        top_k: int = 10,
    ):
        self.top_k = top_k
        self._embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.store = FAISS.load_local(
            store_path,
            self._embedder,
            allow_dangerous_deserialization=True
        )

    def retrieve(
        self,
        query: str,
        filter: Optional[Dict] = None,
    ) -> List[Document]:
        """
        Semantic search with optional metadata filtering.
        e.g. filter={"file_type": ".pdf"} to restrict to PDFs only.
        """
        kwargs = {"k": self.top_k}
        if filter:
            kwargs["filter"] = filter

        return self.store.similarity_search(query, **kwargs)