"""
backend/embeddings.py

Initializes the embedding model and ChromaDB vector store.
Provides functions to build the vector store from documents
and retrieve a retriever object for the RAG chain.
"""

import json
import logging
import os

from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "documents.json"
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "vectorstore")

# How many relevant chunks to retrieve per query
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))


def get_embedding_model() -> OpenAIEmbeddings:
    """
    Returns an OpenAI embedding model instance.
    text-embedding-3-small is fast, cheap, and high quality —
    the standard choice for RAG applications.
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY")
    )


def load_documents() -> list[Document]:
    """
    Loads processed documents from disk and converts them into
    LangChain Document objects ready for chunking and embedding.
    """
    with open(PROCESSED_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for doc in data["documents"]:
        # LangChain's Document class wraps text content and metadata
        # into a single object that the rest of the pipeline expects
        langchain_doc = Document(
            page_content=doc["page_content"], metadata=doc["metadata"]
        )
        documents.append(langchain_doc)

    logger.info(f"Loaded {len(documents)} documents from disk")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Splits documents into smaller overlapping chunks for more
    precise embedding and retrieval.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
        # Tells the splitter to try breaking at paragraphs first,
        # then sentences, then words, then characters — in that order
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


def build_vector_store(chunks: list[Document]) -> Chroma:
    """
    Embeds all chunks and stores them in ChromaDB.
    This only needs to run once — ChromaDB persists to disk.
    """
    logger.info("Building vector store — this may take a minute...")

    embedding_model = get_embedding_model()

    # Chroma.from_documents() does three things in one call:
    # 1. Sends each chunk's text to OpenAI to get its embedding vector
    # 2. Stores the vectors in ChromaDB
    # 3. Persists everything to disk at CHROMA_PERSIST_DIR
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    logger.info(f"✅ Vector store built with {len(chunks)} chunks")
    logger.info(f"💾 Persisted to: {CHROMA_PERSIST_DIR}")
    return vector_store


def load_vector_store() -> Chroma:
    """
    Loads an existing ChromaDB vector store from disk.
    Used at app startup instead of rebuilding every time.
    """
    embedding_model = get_embedding_model()

    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIR, embedding_function=embedding_model
    )

    return vector_store


def get_retriever():
    """
    Returns a LangChain retriever that searches ChromaDB by
    semantic similarity. This is what the RAG chain calls
    to find relevant chunks for a given question.
    """
    vector_store = load_vector_store()

    # as_retriever() wraps the vector store in LangChain's
    # retriever interface. search_kwargs={"k": TOP_K_RESULTS}
    # means "return the top 5 most similar chunks per query"
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": TOP_K_RESULTS}
    )

    return retriever
