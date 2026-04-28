"""
build_vectorstore.py

One-time script to chunk all documents, generate embeddings,
and populate the ChromaDB vector store.

Run this from the project root:
    python build_vectorstore.py
"""

import logging
from backend.embeddings import load_documents, chunk_documents, build_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("  LexiAssist — Building Vector Store")
    logger.info("=" * 60)

    logger.info("\n📄 Step 1: Loading documents...")
    documents = load_documents()

    logger.info("\n✂️  Step 2: Chunking documents...")
    chunks = chunk_documents(documents)

    logger.info("\n🔢 Step 3: Embedding and storing in ChromaDB...")
    vector_store = build_vector_store(chunks)

    logger.info("\n" + "=" * 60)
    logger.info("✅ Vector store ready. You can now run the app.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()