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

lexiassist/
├── backend/          # FastAPI app and RAG pipeline
├── frontend/         # Streamlit chat UI
├── data/             # Raw and processed documents
├── vectorstore/      # ChromaDB storage (local, not committed)
├── evaluation/       # RAGAs evaluation scripts and datasets
├── tests/            # pytest test suite
└── .github/          # GitHub Actions CI/CD workflows

## Setup Instructions

*(Will be completed at end of project)*

## Evaluation Results

Evaluated using [RAGAs](https://github.com/explodinggradients/ragas) on a hand-curated dataset of 20 questions.

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Faithfulness | 0.70 | Answers stay grounded in retrieved documents |
| Answer Relevancy | 0.96 | Answers directly address the question asked |
| Context Precision | 0.56 | Retrieved chunks are relevant to the query |
| Context Recall | 0.53 | Retrieved chunks cover the needed information |

## Live Demo

*(Will be completed in Phase 10)*