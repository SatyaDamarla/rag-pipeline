import yaml
from .retriever import Retriever
from .reranker import LocalReranker, CohereReranker
from src.generation.generator import RAGGenerator


class RAGPipeline:
    def __init__(self, config_path: str = "configs/pipeline.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.retriever = Retriever(
            store_path=cfg["vector_store"]["path"],
            top_k=cfg["retrieval"].get("top_k", 10),
        )

        reranker_type = cfg["retrieval"].get("reranker", "local")
        self.reranker = (
            CohereReranker(top_n=cfg["retrieval"].get("top_n", 5))
            if reranker_type == "cohere"
            else LocalReranker(top_n=cfg["retrieval"].get("top_n", 5))
        )

        self.generator = RAGGenerator(
            model=cfg["generation"].get("model", "gpt-4o-mini"),
            temperature=cfg["generation"].get("temperature", 0.0),
        )

    def query(self, question: str, filter: dict = None) -> dict:
        # step 1: retrieve
        candidates = self.retriever.retrieve(question, filter=filter)

        # step 2: rerank
        reranked = self.reranker.rerank(question, candidates)

        # step 3: generate
        result = self.generator.generate(question, reranked)
        result["question"] = question
        return result