import time
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document


class BatchEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 64):
        self.embedder = OpenAIEmbeddings(model=model)
        self.batch_size = batch_size

    def embed_documents(self, chunks: List[Document]) -> List[List[float]]:
        texts = [c.page_content for c in chunks]
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self._embed_with_retry(batch)
            all_embeddings.extend(embeddings)
            print(f"  Embedded {min(i + self.batch_size, len(texts))}/{len(texts)} chunks")
        return all_embeddings

    def _embed_with_retry(self, texts: List[str], retries: int = 3) -> List[List[float]]:
        for attempt in range(retries):
            try:
                return self.embedder.embed_documents(texts)
            except Exception as e:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"  Embedding failed ({e}), retrying in {wait}s...")
                time.sleep(wait)