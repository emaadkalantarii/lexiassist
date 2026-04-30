"""
backend/main.py

FastAPI application for LexiAssist.
Exposes the RAG chain as a REST API with endpoints for
chat, health checking, and document ingestion.
"""

import logging
import os
import time

from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Pydantic Models ───────────────────────────────────────────────────────────
# Pydantic models define the exact shape of data coming in and going out
# of each endpoint. FastAPI uses them to automatically validate requests
# and generate API documentation. If a request doesn't match the model,
# FastAPI returns a clear 422 error before your code even runs.


class ChatMessage(BaseModel):
    """A single message in the conversation history."""

    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str = Field(..., description="The message text")


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""

    question: str = Field(..., min_length=1, description="The user's question")
    chat_history: Optional[list[ChatMessage]] = Field(
        default=[], description="Previous messages in the conversation"
    )


class Source(BaseModel):
    """A single source paper used to generate the answer."""

    title: str
    authors: str
    published: str
    url: str


class ChatResponse(BaseModel):
    """Response body returned by the /chat endpoint."""

    answer: str
    sources: list[Source]
    processing_time_ms: int


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str
    version: str
    environment: str


class IngestResponse(BaseModel):
    """Response body for the /ingest endpoint."""

    status: str
    documents_processed: int
    chunks_stored: int


# ── App Lifespan ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs startup and shutdown logic for the FastAPI app.
    Code before 'yield' runs on startup, code after runs on shutdown.
    We use this to pre-load the RAG chain so the first request
    isn't slow due to initialization.
    """
    logger.info("Starting LexiAssist API...")
    # Import here to trigger module-level chain initialization
    from backend.rag_chain import ask

    logger.info("RAG chain loaded and ready")
    yield
    logger.info("Shutting down LexiAssist API")


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="LexiAssist API",
    description="RAG-powered AI research assistant over 150+ ML/AI papers",
    version=os.getenv("APP_VERSION", "1.0.0"),
    lifespan=lifespan,
)

# CORS middleware allows the Streamlit frontend (running on a different port)
# to make requests to this API. Without this, browsers block cross-origin
# requests for security reasons.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns 200 OK if the server is running correctly.
    Used by Docker, AWS, and load balancers to verify the app is alive.
    """
    return HealthResponse(
        status="healthy",
        version=os.getenv("APP_VERSION", "1.0.0"),
        environment=os.getenv("ENVIRONMENT", "development"),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Accepts a question and optional chat history, returns an
    AI-generated answer with source citations.
    """
    logger.info(f"Received question: {request.question[:80]}...")

    try:
        start_time = time.time()

        from backend.rag_chain import ask

        # Convert Pydantic ChatMessage objects to plain dicts
        # that our rag_chain.ask() function expects
        history = [
            {"role": msg.role, "content": msg.content} for msg in request.chat_history
        ]

        result = ask(question=request.question, chat_history=history)

        processing_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Answered in {processing_time_ms}ms using {len(result['sources'])} sources"
        )

        return ChatResponse(
            answer=result["answer"],
            sources=[Source(**s) for s in result["sources"]],
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process question: {str(e)}"
        )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents():
    """
    Triggers the full ingestion pipeline programmatically:
    fetches papers from ArXiv, chunks them, embeds them,
    and stores them in ChromaDB.
    """
    logger.info("Ingestion triggered via API")

    try:
        from backend.ingest import run_ingestion
        from backend.embeddings import (
            load_documents,
            chunk_documents,
            build_vector_store,
        )

        run_ingestion()

        documents = load_documents()
        chunks = chunk_documents(documents)
        build_vector_store(chunks)

        return IngestResponse(
            status="success",
            documents_processed=len(documents),
            chunks_stored=len(chunks),
        )

    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
