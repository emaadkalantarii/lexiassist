# LexiAssist 🤖

> A production-ready RAG chatbot answering questions over 150+ AI/ML research papers — with source citations, LLM-as-judge evaluation, CI/CD automation, and full containerized deployment.

[![CI/CD Pipeline](https://github.com/emaadkalantarii/lexiassist/actions/workflows/ci.yml/badge.svg)](https://github.com/emaadkalantarii/lexiassist/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-latest-1C3C3C)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white)

---

## Overview

LexiAssist is an end-to-end LLM application demonstrating production-level AI engineering across the full stack. It ingests AI/ML research papers from ArXiv, stores them in a vector database, and uses Retrieval-Augmented Generation (RAG) to answer natural-language questions with grounded, cited responses.

The project was built to cover every layer of a real AI product: automated data ingestion, semantic embeddings, RAG pipeline with conversational memory, a REST API backend, an interactive chat UI, pipeline quality evaluation, Docker containerization, and GitHub Actions CI/CD.

**Key project stats:**
- 145 papers ingested across 15 AI/ML topics
- 394 indexed chunks in ChromaDB
- 6 automated API tests
- 4 evaluation metrics across 20 hand-curated questions
- Full CI/CD pipeline: lint → test → Docker push on every commit

---

## Demo

> The app runs locally and is not publicly hosted to avoid uncontrolled OpenAI API usage —
> a deliberate engineering decision to manage costs responsibly.
> Screenshots below show the full application running locally.

**Question 1 — Asking about RAG with cited sources expanded:**

![Chat Demo](docs/screenshots/chat_demo.png)

**Question 2 — Follow-up demonstrating conversational memory:**

![Memory Demo](docs/screenshots/memory_demo.png)

> The second question ("How does it compare to fine-tuning?") contains no context on its own.
> The chatbot answers it correctly only because it retains the previous conversation — demonstrating
> that the conversation memory pipeline is working.

**Session stats and knowledge base info visible in the sidebar:**

![Sidebar Demo](docs/screenshots/sidebar_demo.png)

**CI/CD pipeline — all three jobs passing on every push:**

![CI/CD Pipeline](docs/screenshots/cicd_pipeline.png)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       User Interface                        │
│                   Streamlit Chat App                        │
│                    (localhost:8501)                         │
└─────────────────────┬───────────────────────────────────────┘
                      │  HTTP POST /chat
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│        Pydantic validation · Conversation memory            │
│          Logging · Error handling · /docs UI                │
│                    (localhost:8000)                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  LangChain RAG Chain                        │
│                                                             │
│  User Question                                              │
│       │                                                     │
│       ▼                                                     │
│  OpenAI Embeddings (text-embedding-3-small)                 │
│       │                                                     │
│       ▼                                                     │
│  ChromaDB Vector Store ──► Top-5 Relevant Chunks            │
│  (394 chunks · 145 papers)        │                         │
│                                   ▼                         │
│                    Prompt Template                          │
│              (System + Context + Chat History)              │
│                                   │                         │
│                                   ▼                         │
│                      GPT-4o-mini Generation                 │
│                                   │                         │
│                                   ▼                         │
│                   Answer + Source Citations                 │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 Offline Data Pipeline                       │
│                                                             │
│  ArXiv API ──► 145 Papers ──► Text Chunking (394 chunks)    │
│       ──► OpenAI Embeddings ──► ChromaDB Persistence        │
└─────────────────────────────────────────────────────────────┘
```

---

## CI/CD Pipeline

Every push to `main` automatically triggers a three-stage pipeline via GitHub Actions:

```
Push to main
     │
     ▼
┌─────────────────┐     ┌───────────────────────┐     ┌────────────────────────┐
│  Code Quality   │────►│     Run Tests         │────►│  Build & Push Docker   │
│  Check          │     │                       │     │  Image                 │
│                 │     │  · Fixture test data  │     │                        │
│  · flake8 lint  │     │  · Build vectorstore  │     │  · docker buildx       │
│  · black format │     │  · 6 pytest API tests │     │  · Push to Docker Hub  │
└─────────────────┘     └───────────────────────┘     └────────────────────────┘
```

The test job uses a small fixture dataset instead of live ArXiv calls — keeping CI fast,
free, and independent of external APIs.

Docker image: [hub.docker.com/r/emaadkalantarii/lexiassist](https://hub.docker.com/r/emaadkalantarii/lexiassist)

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| LLM | OpenAI GPT-4o-mini | Answer generation |
| Embeddings | text-embedding-3-small | Semantic vector encoding |
| RAG Framework | LangChain | Pipeline orchestration |
| Vector Database | ChromaDB | Semantic similarity search |
| Backend API | FastAPI + Uvicorn | REST endpoints, validation |
| Frontend | Streamlit | Interactive chat UI |
| Evaluation | Custom LLM-as-judge | Pipeline quality measurement |
| Containerization | Docker + Docker Compose | Environment packaging |
| CI/CD | GitHub Actions | Automated lint, test, build |
| Image Registry | Docker Hub | Container distribution |
| Data Source | ArXiv API | Research paper ingestion |

---

## Evaluation Results

The RAG pipeline quality was measured using a custom **LLM-as-judge** framework built from scratch — GPT-4o-mini scores each metric independently across 20 hand-curated question/answer pairs. This approach mirrors what production evaluation frameworks like RAGAs do internally.

| Metric | Score | Rating | What It Measures |
|--------|-------|--------|-----------------|
| Answer Relevancy | **0.96** | ✅ Excellent | Does the answer address the question asked? |
| Faithfulness | **0.70** | 🟡 Good | Is the answer grounded in the retrieved documents? |
| Context Precision | **0.56** | 🟡 Fair | Are the retrieved chunks relevant to the query? |
| Context Recall | **0.54** | 🟡 Fair | Does the context contain what's needed to answer? |

> **Why context scores are lower:** The knowledge base uses paper abstracts only, not full text.
> Ingesting complete PDFs is the most direct path to improving context precision and recall.
> Answer relevancy at 0.96 confirms the generation step is working very well.

---

## Project Structure

```
lexiassist/
├── backend/
│   ├── main.py                  # FastAPI app — /chat, /health, /ingest endpoints
│   ├── rag_chain.py             # LangChain RAG pipeline with conversation memory
│   ├── embeddings.py            # ChromaDB vector store management
│   └── ingest.py                # ArXiv data ingestion (145 papers, 15 topics)
├── frontend/
│   └── app.py                   # Streamlit chat UI connected to FastAPI
├── streamlit_app.py             # Self-contained version (no separate API needed)
├── build_vectorstore.py         # One-time script to embed and index all documents
├── evaluation/
│   ├── evaluate.py              # LLM-as-judge evaluation pipeline
│   ├── eval_dataset.json        # 20 hand-curated Q&A evaluation pairs
│   └── eval_results.json        # Full per-sample evaluation results
├── tests/
│   ├── test_api.py              # 6 pytest tests for all FastAPI endpoints
│   └── fixtures/
│       └── sample_documents.json  # Lightweight fixture data for CI testing
├── data/
│   └── processed/
│       └── documents.json       # 145 structured ArXiv documents
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions — lint → test → docker push
├── docs/
│   └── screenshots/             # App screenshots used in this README
├── Dockerfile                   # Container image definition
├── docker-compose.yml           # Multi-service local orchestration
├── .env.example                 # Environment variable reference
├── conftest.py                  # pytest path configuration
└── requirements.txt             # Python dependencies
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Docker Desktop (for containerized deployment)
- OpenAI API key ([get one here](https://platform.openai.com))

### Local Setup

```bash
# Clone the repository
git clone https://github.com/emaadkalantarii/lexiassist.git
cd lexiassist

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate      # macOS/Linux

# Install all dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Open .env and add your OPENAI_API_KEY

# Run the data ingestion pipeline (downloads 145 papers — takes ~2 min)
python backend/ingest.py

# Build the ChromaDB vector store (embeds all documents — takes ~1 min)
python build_vectorstore.py
```

### Run the Application

Open two terminals:

```bash
# Terminal 1 — Start the FastAPI backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Start the Streamlit frontend
streamlit run frontend/app.py
```

Open in your browser:
- **Chat UI:** `http://localhost:8501`
- **API docs (Swagger):** `http://localhost:8000/docs`
- **Health check:** `http://localhost:8000/health`

---

## Docker Deployment

```bash
# Build and start all services with one command
docker compose up

# Run in the background
docker compose up -d

# Stop all services
docker compose down
```

Services after startup:
- FastAPI backend: `http://localhost:8999`
- Streamlit frontend: `http://localhost:8501`

Or pull the pre-built image from Docker Hub:

```bash
docker pull emaadkalantarii/lexiassist:latest
```

---

## API Reference

### `POST /chat`
Send a question and receive a grounded answer with source citations.

**Request body:**
```json
{
  "question": "What is retrieval augmented generation?",
  "chat_history": [
    {"role": "user", "content": "Previous question"},
    {"role": "assistant", "content": "Previous answer"}
  ]
}
```

**Response:**
```json
{
  "answer": "Retrieval Augmented Generation (RAG) combines a retrieval system...",
  "sources": [
    {
      "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
      "authors": "Patrick Lewis et al.",
      "published": "2021-04-12",
      "url": "http://arxiv.org/abs/2005.11401"
    }
  ],
  "processing_time_ms": 2341
}
```

### `GET /health`
Returns API health status, version, and environment.

### `POST /ingest`
Triggers the full ArXiv ingestion and vectorstore rebuild pipeline programmatically.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

The test suite covers: health check endpoint, basic chat with a question, multi-turn
conversation history, empty input rejection (422), missing field rejection (422),
and source citation field validation.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key — required | — |
| `LLM_MODEL` | Generation model name | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Embedding model name | `text-embedding-3-small` |
| `CHUNK_SIZE` | Max characters per document chunk | `1000` |
| `CHUNK_OVERLAP` | Character overlap between chunks | `200` |
| `TOP_K_RESULTS` | Number of chunks retrieved per query | `5` |
| `API_BASE_URL` | FastAPI URL used by Streamlit | `http://localhost:8000` |
| `ENVIRONMENT` | Runtime environment label | `development` |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Emad Kalantari**
Master's in Information and Computer Sciences, University of Luxembourg

[LinkedIn](https://linkedin.com/in/emaadkalantarii) · [GitHub](https://github.com/emaadkalantarii) · [Website](https://emadkalantari.com)
