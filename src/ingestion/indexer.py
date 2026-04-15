from typing import List, Literal
from langchain.schema import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class VectorIndexer:
    def __init__(
        self,
        store_type: Literal["faiss", "chroma"] = "faiss",
        persist_path: str = "./vector_store",
    ):
        self.store_type = store_type
        self.persist_path = persist_path
        self._embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    def build(self, chunks: List[Document]):
        print(f"Indexing {len(chunks)} chunks into {self.store_type}...")
        if self.store_type == "faiss":
            store = FAISS.from_documents(chunks, self._embedder)
            store.save_local(self.persist_path)
        elif self.store_type == "chroma":
            store = Chroma.from_documents(
                chunks, self._embedder, persist_directory=self.persist_path
            )
        print(f"Index saved to {self.persist_path}")
        return store

    def load(self):
        if self.store_type == "faiss":
            return FAISS.load_local(
                self.persist_path, self._embedder, allow_dangerous_deserialization=True
            )
        return Chroma(
            persist_directory=self.persist_path, embedding_function=self._embedder
        )