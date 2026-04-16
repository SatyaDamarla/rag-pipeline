# benchmarks/ragas_eval.py
import json
import time
import sys
import os
import math
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.retrieval.rag_pipeline import RAGPipeline


def avg(scores):
    scores = [s for s in scores if s is not None and not (isinstance(s, float) and math.isnan(s))]
    return sum(scores) / len(scores) if scores else float("nan")


def clean_score(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return None
    return round(s, 3)


def fmt(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "N/A (insufficient data)"
    return f"{s:.3f}"


def run_eval():
    with open("benchmarks/qa_dataset.json") as f:
        qa_pairs = json.load(f)

    qa_pairs = qa_pairs[:3]

    print(f"Running eval on {len(qa_pairs)} questions...\n")

    rag = RAGPipeline("configs/pipeline.yaml")

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, qa in enumerate(qa_pairs):
        print(f"[{i+1}/{len(qa_pairs)}] {qa['question']}")
        try:
            result = rag.query(qa["question"])
            questions.append(qa["question"])
            answers.append(result["answer"])
            contexts.append([doc.page_content for doc in rag.last_retrieved_docs])
            ground_truths.append(qa["ground_truth"])
            time.sleep(30)
        except Exception as e:
            print(f"  Error: {e}")
            continue

    if not questions:
        print("No questions collected — check rate limits and try again tomorrow.")
        return

    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            temperature=0.0
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
    )

    faithfulness.llm = llm
    answer_relevancy.llm = llm
    answer_relevancy.embeddings = embeddings
    context_recall.llm = llm

    print("\nRunning RAGAS evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
    )

    faith_score = avg(results["faithfulness"]) if isinstance(results["faithfulness"], list) else results["faithfulness"]
    ar_score = avg(results["answer_relevancy"]) if isinstance(results["answer_relevancy"], list) else results["answer_relevancy"]
    cr_score = avg(results["context_recall"]) if isinstance(results["context_recall"], list) else results["context_recall"]

    print("\n=== RAGAS Results ===")
    print(f"Faithfulness:     {fmt(faith_score)}")
    print(f"Answer relevancy: {fmt(ar_score)}")
    print(f"Context recall:   {fmt(cr_score)}")

    results_dict = {
        "faithfulness":     clean_score(faith_score),
        "answer_relevancy": clean_score(ar_score),
        "context_recall":   clean_score(cr_score),
        "num_questions":    len(questions),
        "note":             "Results may be partial if daily quota was hit",
    }

    with open("benchmarks/ragas_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    print("\nSaved to benchmarks/ragas_results.json")
    return results_dict


if __name__ == "__main__":
    run_eval()