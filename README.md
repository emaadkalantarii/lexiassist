# LexiAssist 🤖

> A production-ready Retrieval-Augmented Generation (RAG) chatbot that answers questions over a corpus of 150+ AI/ML research papers — with citations, evaluation, and full deployment pipeline.

[![CI/CD](https://github.com/emaadkalantarii/lexiassist/actions/workflows/ci.yml/badge.svg)](https://github.com/emaadkalantarii/lexiassist/actions)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-red)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

## Overview

LexiAssist is an end-to-end LLM application built to demonstrate production-level AI engineering skills. It ingests AI/ML research papers from ArXiv, stores them in a vector database, and uses RAG to answer natural-language questions with grounded, cited responses.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM & Embeddings | OpenAI GPT-4o-mini, text-embedding-3-small |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Evaluation | RAGAs |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Cloud Deployment | AWS EC2 + nginx |

## Project Structure