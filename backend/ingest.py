"""
backend/ingest.py

This module is responsible for downloading AI/ML research papers from the
ArXiv API and saving them in a structured format for later processing.

ArXiv is a free, open-access repository of research papers. Its API lets
us programmatically search and download paper metadata and abstracts.
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import arxiv          # The official ArXiv Python client library
import json           # Built-in Python library for reading/writing JSON files
import os             # Built-in library for interacting with the file system
import time           # Built-in library for adding delays between API calls
import logging        # Built-in library for printing structured log messages

from datetime import datetime   # For adding timestamps to our saved data
from pathlib import Path        # Modern, object-oriented way to handle file paths

from dotenv import load_dotenv  # Reads our .env file and loads its values
                                # into environment variables

# ── Load Environment Variables ────────────────────────────────────────────────

# load_dotenv() reads the .env file in our project root and makes all the
# KEY=VALUE pairs available via os.getenv("KEY"). This must be called before
# we try to read any environment variables.
load_dotenv()

# ── Configure Logging ─────────────────────────────────────────────────────────

# logging is Python's built-in system for printing messages with context.
# It's better than print() for production code because it includes timestamps,
# severity levels (INFO, WARNING, ERROR), and can be redirected to files.
#
# basicConfig sets up the default behavior:
#   level=logging.INFO  → show messages at INFO level and above
#   format=...          → how each message looks when printed
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
    # %(asctime)s   → current date and time
    # %(levelname)s → INFO, WARNING, ERROR, etc.
    # %(message)s   → the actual message we write
)

# Create a logger object for this specific module.
# Using __name__ means the logger is named after this file (backend.ingest),
# which helps when you have multiple modules all logging at once.
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Path() creates a path object. Using Path(__file__) gives us the path to
# THIS file (ingest.py). .parent goes up one level (to backend/), .parent
# again goes up to the project root. This way the script works regardless
# of which directory you run it from.
PROJECT_ROOT = Path(__file__).parent.parent

# Define where we'll save our downloaded data.
# The / operator on Path objects joins path segments cleanly.
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "papers.json"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "documents.json"

# How many papers to fetch per search topic.
# 15 topics × 10 papers = up to 150 papers total.
PAPERS_PER_TOPIC = 10

# We add a small delay between API calls to be a polite API citizen —
# sending too many requests too fast can get your IP temporarily blocked.
API_DELAY_SECONDS = 1

# ── Search Topics ─────────────────────────────────────────────────────────────

# These are the topics we'll search ArXiv for.
# Each one will return up to PAPERS_PER_TOPIC papers.
# We chose these because they cover the core concepts of LLM engineering —
# exactly the knowledge base our chatbot should have.
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

# ── Functions ─────────────────────────────────────────────────────────────────

def fetch_papers_for_topic(topic: str, max_results: int) -> list[dict]:
    """
    Search ArXiv for papers matching a given topic and return their metadata.

    Parameters
    ----------
    topic : str
        The search query string (e.g., "retrieval augmented generation")
    max_results : int
        Maximum number of papers to return for this topic

    Returns
    -------
    list[dict]
        A list of dictionaries, each containing one paper's metadata.

    What a "dict" (dictionary) is:
        A Python data structure that stores key-value pairs, like:
        {"title": "Attention Is All You Need", "year": 2017}
        Dictionaries are how we represent structured data in Python
        before converting it to JSON.
    """

    logger.info(f"Fetching papers for topic: '{topic}'")
    # f"..." is an f-string — it lets you embed variables inside a string
    # using curly braces. {topic} gets replaced with the actual value of topic.

    # arxiv.Search creates a search query object.
    # query         → the search string
    # max_results   → how many results to return
    # sort_by       → we sort by relevance so we get the most on-topic papers
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    # papers is an empty list. A list in Python is an ordered collection
    # that we can add items to. We'll append one dict per paper.

    # arxiv.Client() creates a client that handles the API communication.
    # .results(search) sends the query to ArXiv and returns an iterator
    # (a stream of results that we process one at a time with the for loop).
    client = arxiv.Client()

    for result in client.results(search):
        # result is one paper returned by ArXiv.
        # result.title        → the paper's title (string)
        # result.summary      → the abstract (string)
        # result.authors      → list of Author objects
        # result.published    → datetime object of publication date
        # result.entry_id     → the ArXiv URL for this paper
        # result.categories   → list of category strings like ["cs.CL", "cs.AI"]

        # result.authors is a list of Author objects.
        # Each Author object has a .name attribute.
        # This list comprehension creates a list of name strings:
        # [author.name for author in result.authors]
        # reads as: "for each author in the authors list, get their name"
        authors = [author.name for author in result.authors]

        # Build a clean dictionary for this paper.
        paper = {
            "arxiv_id": result.entry_id,
            # entry_id looks like: http://arxiv.org/abs/2307.03172v1
            
            "title": result.title.strip(),
            # .strip() removes any leading/trailing whitespace or newlines
            
            "abstract": result.summary.strip(),
            # The abstract is the most important field — it's what we'll
            # embed and use for retrieval
            
            "authors": authors[:5],
            # We only keep the first 5 authors to save space.
            # List slicing: [start:end] — [:5] means items 0 through 4.
            
            "published": result.published.strftime("%Y-%m-%d"),
            # .strftime() converts a datetime object to a formatted string.
            # "%Y-%m-%d" produces dates like "2024-03-15"
            
            "categories": result.categories,
            # e.g., ["cs.CL", "cs.AI"] — cs.CL is Computation and Language
            
            "search_topic": topic,
            # We save which topic this paper was found under — useful later
            
            "url": result.entry_id,
        }

        papers.append(paper)
        # .append() adds the paper dict to the end of our papers list.

    logger.info(f"  → Found {len(papers)} papers")
    # len() returns the number of items in a list.

    return papers


def fetch_all_papers() -> list[dict]:
    """
    Fetch papers for all topics and return a deduplicated combined list.

    Returns
    -------
    list[dict]
        All papers across all topics, with duplicates removed.
    """

    all_papers = []
    # A set to track ArXiv IDs we've already seen, to avoid duplicates.
    # A set is like a list but automatically ignores duplicate values.
    seen_ids = set()

    for i, topic in enumerate(SEARCH_TOPICS):
        # enumerate() gives us both the index (i) and the value (topic)
        # as we loop through the list. So i=0, topic="retrieval augmented..."
        # then i=1, topic="large language model..." etc.

        logger.info(f"[{i+1}/{len(SEARCH_TOPICS)}] Topic: {topic}")

        # Fetch papers for this topic
        papers = fetch_papers_for_topic(topic, PAPERS_PER_TOPIC)

        for paper in papers:
            arxiv_id = paper["arxiv_id"]
            # Access a dictionary value using its key in square brackets.

            if arxiv_id not in seen_ids:
                # Only add this paper if we haven't seen this ID before.
                # This removes duplicates — the same paper can appear in
                # multiple topic searches.
                seen_ids.add(arxiv_id)
                all_papers.append(paper)

        # Pause between topics to avoid overwhelming the ArXiv API.
        # time.sleep() pauses execution for the given number of seconds.
        time.sleep(API_DELAY_SECONDS)

    logger.info(f"\n✅ Total unique papers fetched: {len(all_papers)}")
    return all_papers


def save_raw_papers(papers: list[dict]) -> None:
    """
    Save the raw paper list to data/raw/papers.json.

    Parameters
    ----------
    papers : list[dict]
        The list of paper dictionaries to save.

    Returns
    -------
    None
        This function doesn't return a value — it just saves a file.
        In Python, functions that don't return anything use -> None.
    """

    # Create the directory if it doesn't already exist.
    # parents=True → create any missing parent directories too
    # exist_ok=True → don't raise an error if the directory already exists
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Build a wrapper object that includes metadata about when the data
    # was downloaded. This is good practice for data provenance —
    # knowing where data came from and when.
    output = {
        "metadata": {
            "downloaded_at": datetime.now().isoformat(),
            # .isoformat() converts a datetime to a string like:
            # "2024-11-15T14:32:01.123456"
            "total_papers": len(papers),
            "topics_searched": SEARCH_TOPICS,
            "papers_per_topic": PAPERS_PER_TOPIC,
        },
        "papers": papers
    }

    # Open the file for writing ("w" = write mode).
    # "with open(...) as f:" is a context manager — it automatically
    # closes the file when the block finishes, even if an error occurs.
    # encoding="utf-8" ensures special characters (accents, symbols) are
    # handled correctly.
    with open(RAW_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        # json.dump() converts a Python dict/list to JSON and writes it to f.
        # indent=2 → pretty-print with 2-space indentation (human-readable)
        # ensure_ascii=False → allow non-ASCII chars like accented letters

    logger.info(f"💾 Raw data saved to: {RAW_DATA_PATH}")


def create_documents(papers: list[dict]) -> list[dict]:
    """
    Transform raw papers into clean documents ready for the RAG pipeline.

    Each document will have:
    - page_content: the text that gets embedded and retrieved
    - metadata: extra info attached to the chunk (not embedded)

    This structure matches what LangChain expects when we load documents
    in Phase 3.

    Parameters
    ----------
    papers : list[dict]
        Raw papers from ArXiv

    Returns
    -------
    list[dict]
        Cleaned, structured documents
    """

    documents = []

    for paper in papers:
        # We combine the title and abstract into one block of text.
        # This is the text that will be embedded (converted to a vector)
        # and used for semantic search.
        #
        # Why combine title + abstract?
        # The title adds crucial context. "Attention Is All You Need" +
        # the abstract together give better embedding quality than the
        # abstract alone.
        page_content = f"""Title: {paper['title']}

Abstract: {paper['abstract']}"""

        # Metadata is stored alongside the vector but NOT embedded.
        # It's returned with search results so we can show citations.
        # Think of it as labels attached to each document chunk.
        metadata = {
            "source": paper["arxiv_id"],
            "title": paper["title"],
            "authors": ", ".join(paper["authors"]),
            # ", ".join(list) → joins list items into a single string
            # e.g., ["Alice", "Bob", "Charlie"] → "Alice, Bob, Charlie"
            "published": paper["published"],
            "categories": ", ".join(paper["categories"]),
            "search_topic": paper["search_topic"],
            "url": paper["url"],
        }

        document = {
            "page_content": page_content,
            "metadata": metadata
        }

        documents.append(document)

    return documents


def save_processed_documents(documents: list[dict]) -> None:
    """
    Save the processed documents to data/processed/documents.json.
    """

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_documents": len(documents),
        },
        "documents": documents
    }

    with open(PROCESSED_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 Processed documents saved to: {PROCESSED_DATA_PATH}")


def run_ingestion() -> None:
    """
    Main entry point for the ingestion pipeline.
    Runs all steps in order: fetch → save raw → process → save processed.
    """

    logger.info("=" * 60)
    logger.info("  LexiAssist — Data Ingestion Pipeline")
    logger.info("=" * 60)

    # Step 1: Fetch all papers from ArXiv
    logger.info("\n📥 Step 1: Fetching papers from ArXiv...")
    papers = fetch_all_papers()

    # Step 2: Save raw data
    logger.info("\n💾 Step 2: Saving raw data...")
    save_raw_papers(papers)

    # Step 3: Create structured documents
    logger.info("\n🔧 Step 3: Creating structured documents...")
    documents = create_documents(papers)

    # Step 4: Save processed documents
    logger.info("\n💾 Step 4: Saving processed documents...")
    save_processed_documents(documents)

    logger.info("\n" + "=" * 60)
    logger.info(f"✅ Ingestion complete! {len(documents)} documents ready.")
    logger.info("=" * 60)


# ── Entry Point ───────────────────────────────────────────────────────────────

# This block only runs when you execute THIS file directly:
#   python backend/ingest.py
#
# It does NOT run when this file is imported by another module.
# This is a Python convention — always wrap your "run this script" code
# in if __name__ == "__main__":
if __name__ == "__main__":
    run_ingestion()