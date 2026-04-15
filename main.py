# main.py
from dotenv import load_dotenv
load_dotenv()

import time
from src.retrieval.rag_pipeline import RAGPipeline

if __name__ == "__main__":
    print("Loading RAG pipeline...")
    rag = RAGPipeline("configs/pipeline.yaml")
    print("Ready! Type your question or 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        result = rag.query(question)
        print(f"\nA: {result['answer']}")
        print(f"Sources: {[s['source'] + ' p.' + str(s['page']) for s in result['sources']]}")
        print(f"Chunks used: {result['num_chunks_used']}\n")
        
        time.sleep(3)  # small delay to avoid rate limit