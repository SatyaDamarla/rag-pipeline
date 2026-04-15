import yaml
from pathlib import Path
from typing import List
from .loaders import LoaderRegistry
from .chunkers import get_chunker
from .indexer import VectorIndexer


class IngestionPipeline:
    def __init__(self, config_path: str = "configs/pipeline.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.chunker = get_chunker(
            self.cfg["chunking"]["strategy"],
            chunk_size=self.cfg["chunking"].get("chunk_size", 512),
            chunk_overlap=self.cfg["chunking"].get("chunk_overlap", 64),
        )
        self.indexer = VectorIndexer(
            store_type=self.cfg["vector_store"]["type"],
            persist_path=self.cfg["vector_store"]["path"],
        )

    def run(self, sources: List[str]):
        all_chunks = []
        for source in sources:
            print(f"\nLoading: {source}")
            loader = LoaderRegistry.get(source)
            docs = loader.load(source)
            chunks = self.chunker.chunk(docs)
            print(f"  {len(docs)} pages → {len(chunks)} chunks")
            all_chunks.extend(chunks)

        print(f"\nTotal: {len(all_chunks)} chunks across {len(sources)} sources")
        store = self.indexer.build(all_chunks)
        return store