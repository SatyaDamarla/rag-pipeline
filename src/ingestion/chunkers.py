from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document
from langchain.text_splitter import (RecursiveCharacterTextSplitter, TokenTextSplitter,)

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        ...
    
    def _propagate_metadata(self, parent:Document, child:Document, idx:int) -> Document:
        child.metadata = {**parent.metadata, "chunk_index":idx}
        return child

class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators= ["\n\n", "\n", ". ", " ", ""],
        )
    
    def chunk(self, documents: List[Document]) -> List[Document]:
        chunks = []
        for doc in documents:
            splits = self.splitter.split_documents([doc])
            chunks.extend(
                self._propagate_metadata(doc,s,i) for i,s in enumerate(splits)
            )
        return chunks

class TokenChunker(BaseChunker):
    """Chunks by token count - better for LLM context window alignment"""
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32):
        self.splitter = TokenTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
        )
        
    def chunk(self, documents: List[Document]) -> List[Document]:
        chunks = []
        for doc in documents:
            splits = self.splitter.split_documents([doc])
            chunks.extend(
                self._propagate_metadata(doc,s,i) for i,s in enumerate(splits)
            )
        return chunks

class SentenceWindowChunker(BaseChunker):
    """Keeps surrounding sentences as context - good for dense technical docs"""
    def __init__(self, window_size: int =3):
        import nltk
        nltk.download("punkt",quiet=True)
        self.window_size = window_size
    
    def chunk(self, documents: List[Document]) -> List[Document]:
        import nltk
        chunks = []
        for doc in documents:
            sentences = nltk.sent_tokenize(doc.page_content)
            for i, sent in enumerate(sentences):
                start = max(0, i - self.window_size)
                end = min(len(sentences), i + self.window_size + 1)
                window_text = " ".join(sentences[start:end])
                chunk = Document(
                    page_content = window_text,
                    metadata = {**doc.metadata, "chunk_index": i,"core_sentence": sent},
                )
                chunks.append(chunk)
        return chunks

#Factory - swap strategies via config, not code changes
CHUNKER_REGISTRY = {
    "recursive": RecursiveChunker,
    "token": TokenChunker,
    "sentence_window": SentenceWindowChunker,
}

def get_chunker(strategy: str, **kwargs) -> BaseChunker:
    if strategy not in CHUNKER_REGISTRY:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Choose from {list(CHUNKER_REGISTRY)}")
    return CHUNKER_REGISTRY[strategy](**kwargs)
                         