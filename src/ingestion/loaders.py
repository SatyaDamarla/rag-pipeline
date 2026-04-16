from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
import fitz  # PyMuPDF

class BaseLoader(ABC):
    @abstractmethod
    def load(self,source:str) -> List[Document]:
        ...

    def _make_metadata(self, source: str,extra: dict = {}) -> dict:
        return {"source": source, "file_type": Path(source).suffix, **extra}

class PDFLoader(BaseLoader):
    def load(self, source: str) -> List[Document]:
        doc = fitz.open(source)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if not text:
                continue
            pages.append(Document(
                page_content=text,
                metadata=self._make_metadata(source, {"page": i + 1, "total_pages": len(doc)})
            ))
        return pages

class WebLoader(BaseLoader):
    def load(self, source: str) -> List[Document]:
        loader = WebBaseLoader(source)
        docs = loader.load()
        for doc in docs:
            doc.metadata.update(self._make_metadata(source))
        return docs


class LoaderRegistry:
    @classmethod
    def get(cls, path: str) -> BaseLoader:
        if path.startswith("http://") or path.startswith("https://"):
            return WebLoader()
        
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            return PDFLoader()
        elif ext == ".txt":
            return PDFLoader()  
        else:
            raise ValueError(f"No loader registered for: {path} (extension: {ext})")