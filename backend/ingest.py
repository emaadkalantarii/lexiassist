"""
backend/ingest.py

Downloads AI/ML research papers from ArXiv and saves them as structured
documents for the RAG ingestion pipeline.
"""

import arxiv
import json
import os
import time
import logging

from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "papers.json"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "documents.json"

PAPERS_PER_TOPIC = 10
API_DELAY_SECONDS = 1

SEARCH_TOPICS = [
    "retrieval augmented generation",
    "large language model fine-tuning",
    "transformer architecture attention mechanism",
    "chain of thought prompting",
    "LLM agents tool use",
    "vector database semantic search",
    "instruction following language models",
    "reinforcement learning from human feedback",
    "hallucination detection language models",
    "efficient transformers",
    "knowledge graph language model",
    "multimodal language models",
    "LLM evaluation benchmarks",
    "prompt engineering techniques",
    "parameter efficient fine-tuning LoRA",
]


def fetch_papers_for_topic(topic: str, max_results: int) -> list[dict]:
    """Fetch papers from ArXiv for a single search topic."""
    logger.info(f"Fetching papers for topic: '{topic}'")

    search = arxiv.Search(
        query=topic, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    client = arxiv.Client()

    for result in client.results(search):
        authors = [author.name for author in result.authors]

        paper = {
            "arxiv_id": result.entry_id,
            "title": result.title.strip(),
            "abstract": result.summary.strip(),
            "authors": authors[:5],
            "published": result.published.strftime("%Y-%m-%d"),
            "categories": result.categories,
            "search_topic": topic,
            "url": result.entry_id,
        }
        papers.append(paper)

    logger.info(f"  → Found {len(papers)} papers")
    return papers


def fetch_all_papers() -> list[dict]:
    """Fetch papers for all topics and return a deduplicated list."""
    all_papers = []
    seen_ids = set()

    for i, topic in enumerate(SEARCH_TOPICS):
        logger.info(f"[{i+1}/{len(SEARCH_TOPICS)}] Topic: {topic}")
        papers = fetch_papers_for_topic(topic, PAPERS_PER_TOPIC)

        for paper in papers:
            arxiv_id = paper["arxiv_id"]
            if arxiv_id not in seen_ids:
                seen_ids.add(arxiv_id)
                all_papers.append(paper)

        time.sleep(API_DELAY_SECONDS)

    logger.info(f"\n✅ Total unique papers fetched: {len(all_papers)}")
    return all_papers


def save_raw_papers(papers: list[dict]) -> None:
    """Save raw paper metadata to data/raw/papers.json."""
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "downloaded_at": datetime.now().isoformat(),
            "total_papers": len(papers),
            "topics_searched": SEARCH_TOPICS,
            "papers_per_topic": PAPERS_PER_TOPIC,
        },
        "papers": papers,
    }

    with open(RAW_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 Raw data saved to: {RAW_DATA_PATH}")


def create_documents(papers: list[dict]) -> list[dict]:
    """
    Transform raw papers into LangChain-compatible document format.
    Each document contains page_content for embedding and metadata for citation.
    """
    documents = []

    for paper in papers:
        # Combine title and abstract for richer semantic embedding
        page_content = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"

        metadata = {
            "source": paper["arxiv_id"],
            "title": paper["title"],
            "authors": ", ".join(paper["authors"]),
            "published": paper["published"],
            "categories": ", ".join(paper["categories"]),
            "search_topic": paper["search_topic"],
            "url": paper["url"],
        }

        documents.append({"page_content": page_content, "metadata": metadata})

    return documents


def save_processed_documents(documents: list[dict]) -> None:
    """Save structured documents to data/processed/documents.json."""
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_documents": len(documents),
        },
        "documents": documents,
    }

    with open(PROCESSED_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 Processed documents saved to: {PROCESSED_DATA_PATH}")


def run_ingestion() -> None:
    """Main entry point — runs the full ingestion pipeline."""
    logger.info("=" * 60)
    logger.info("  LexiAssist — Data Ingestion Pipeline")
    logger.info("=" * 60)

    logger.info("\n📥 Step 1: Fetching papers from ArXiv...")
    papers = fetch_all_papers()

    logger.info("\n💾 Step 2: Saving raw data...")
    save_raw_papers(papers)

    logger.info("\n🔧 Step 3: Creating structured documents...")
    documents = create_documents(papers)

    logger.info("\n💾 Step 4: Saving processed documents...")
    save_processed_documents(documents)

    logger.info("\n" + "=" * 60)
    logger.info(f"✅ Ingestion complete! {len(documents)} documents ready.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_ingestion()
