"""
evaluation/evaluate.py

Evaluates the LexiAssist RAG pipeline using custom-implemented metrics.
Measures faithfulness, answer relevancy, context precision, and context recall
across a curated test dataset using the OpenAI API directly.

Run from the project root:
    python evaluation/evaluate.py
"""

import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EVAL_DATASET_PATH = PROJECT_ROOT / "evaluation" / "eval_dataset.json"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "eval_results.json"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_eval_dataset() -> list[dict]:
    """Loads the hand-curated evaluation samples from disk."""
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data['samples'])} evaluation samples")
    return data["samples"]


def run_rag_on_samples(samples: list[dict]) -> list[dict]:
    """
    Runs each question through the RAG pipeline and collects
    the question, answer, retrieved contexts, and ground truth.
    """
    from backend.rag_chain import ask
    from backend.embeddings import get_retriever

    retriever = get_retriever()
    results = []

    for i, sample in enumerate(samples):
        question = sample["question"]
        ground_truth = sample["ground_truth"]

        logger.info(f"[{i+1}/{len(samples)}] Evaluating: {question[:60]}...")

        rag_result = ask(question=question, chat_history=[])
        answer = rag_result["answer"]

        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]

        results.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )

    return results


def score_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Measures whether the answer is grounded in the retrieved contexts.
    Asks the LLM to judge if each claim in the answer is supported.
    Score: 0.0 to 1.0
    """
    context_text = "\n\n".join(contexts[:3])
    prompt = f"""You are an evaluator. Given the following context and answer, 
rate how faithful the answer is to the context on a scale from 0 to 10.
A score of 10 means every claim in the answer is directly supported by the context.
A score of 0 means the answer contains information not present in the context.

Context:
{context_text}

Answer:
{answer}

Respond with ONLY a single integer from 0 to 10."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,
    )
    try:
        score = int(response.choices[0].message.content.strip())
        return round(min(max(score, 0), 10) / 10.0, 4)
    except Exception:
        return 0.5


def score_answer_relevancy(question: str, answer: str) -> float:
    """
    Measures whether the answer actually addresses the question asked.
    Score: 0.0 to 1.0
    """
    prompt = f"""You are an evaluator. Rate how relevant the answer is to the question
on a scale from 0 to 10.
A score of 10 means the answer directly and completely addresses the question.
A score of 0 means the answer is completely off-topic.

Question: {question}

Answer: {answer}

Respond with ONLY a single integer from 0 to 10."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,
    )
    try:
        score = int(response.choices[0].message.content.strip())
        return round(min(max(score, 0), 10) / 10.0, 4)
    except Exception:
        return 0.5


def score_context_precision(question: str, contexts: list[str]) -> float:
    """
    Measures what proportion of retrieved chunks are actually relevant
    to the question. Score: 0.0 to 1.0
    """
    relevant = 0
    for ctx in contexts:
        prompt = f"""Is the following context relevant to answering the question?
Question: {question}
Context: {ctx[:500]}
Respond with ONLY 'yes' or 'no'."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        answer = response.choices[0].message.content.strip().lower()
        if "yes" in answer:
            relevant += 1

    return round(relevant / len(contexts), 4) if contexts else 0.0


def score_context_recall(answer: str, ground_truth: str, contexts: list[str]) -> float:
    """
    Measures whether the retrieved context contains the information
    needed to produce the ground truth answer. Score: 0.0 to 1.0
    """
    context_text = "\n\n".join(contexts[:3])
    prompt = f"""You are an evaluator. Given the context and the ground truth answer,
rate how well the context covers the information needed to produce the ground truth.
Score from 0 to 10. A score of 10 means the context fully supports the ground truth.

Context:
{context_text}

Ground Truth Answer:
{ground_truth}

Respond with ONLY a single integer from 0 to 10."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,
    )
    try:
        score = int(response.choices[0].message.content.strip())
        return round(min(max(score, 0), 10) / 10.0, 4)
    except Exception:
        return 0.5


def run_evaluation(results: list[dict]) -> tuple[dict, list[dict]]:
    """
    Runs all four metrics across all samples and returns
    aggregate scores and per-sample detailed results.
    """
    logger.info("Running evaluation metrics...")

    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []
    recall_scores = []
    detailed = []

    for i, r in enumerate(results):
        logger.info(f"  Scoring sample {i+1}/{len(results)}: {r['question'][:50]}...")

        f = score_faithfulness(r["answer"], r["contexts"])
        a = score_answer_relevancy(r["question"], r["answer"])
        p = score_context_precision(r["question"], r["contexts"])
        rc = score_context_recall(r["answer"], r["ground_truth"], r["contexts"])

        faithfulness_scores.append(f)
        relevancy_scores.append(a)
        precision_scores.append(p)
        recall_scores.append(rc)

        detailed.append(
            {
                "question": r["question"],
                "answer": r["answer"],
                "ground_truth": r["ground_truth"],
                "scores": {
                    "faithfulness": f,
                    "answer_relevancy": a,
                    "context_precision": p,
                    "context_recall": rc,
                },
            }
        )

    aggregate = {
        "faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 4),
        "answer_relevancy": round(sum(relevancy_scores) / len(relevancy_scores), 4),
        "context_precision": round(sum(precision_scores) / len(precision_scores), 4),
        "context_recall": round(sum(recall_scores) / len(recall_scores), 4),
    }

    return aggregate, detailed


def save_results(samples: list[dict], scores: dict, detailed: list[dict]) -> None:
    """Saves the full evaluation results and scores to disk."""
    output = {
        "metadata": {
            "evaluated_at": datetime.now().isoformat(),
            "total_samples": len(samples),
            "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            "evaluation_method": "custom LLM-as-judge metrics",
        },
        "scores": scores,
        "detailed_results": detailed,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {RESULTS_PATH}")


def print_summary(scores: dict) -> None:
    """Prints a clean summary table of evaluation scores."""
    print("\n" + "=" * 50)
    print("  LexiAssist — RAG Evaluation Results")
    print("=" * 50)
    print(f"  {'Metric':<25} {'Score':<10} {'Rating'}")
    print("-" * 50)

    for metric, score in scores.items():
        if score is None:
            rating = "Skipped"
            score_str = "N/A"
        else:
            score_str = f"{score:.4f}"
            if score >= 0.8:
                rating = "✅ Excellent"
            elif score >= 0.6:
                rating = "🟡 Good"
            else:
                rating = "🔴 Needs improvement"
        print(f"  {metric:<25} {score_str:<10} {rating}")

    print("=" * 50)


def main():
    logger.info("=" * 60)
    logger.info("  LexiAssist — RAG Evaluation Pipeline")
    logger.info("=" * 60)

    logger.info("\n📋 Step 1: Loading evaluation dataset...")
    samples = load_eval_dataset()

    logger.info("\n🤖 Step 2: Running RAG pipeline on all samples...")
    rag_results = run_rag_on_samples(samples)

    logger.info("\n📊 Step 3: Scoring with LLM-as-judge metrics...")
    scores, detailed = run_evaluation(rag_results)

    logger.info("\n💾 Step 4: Saving results...")
    save_results(samples, scores, detailed)

    print_summary(scores)


if __name__ == "__main__":
    main()
