"""
tests/test_api.py

Automated tests for the LexiAssist FastAPI endpoints.
Uses pytest and httpx's TestClient to make real HTTP requests
against the app without needing a running server.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

# TestClient wraps our FastAPI app and lets us make HTTP requests
# to it directly in tests — no server needs to be running
client = TestClient(app)


def test_health_check():
    """Health endpoint should return 200 with status healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "environment" in data


def test_chat_basic_question():
    """Chat endpoint should return an answer and sources for a valid question."""
    response = client.post(
        "/chat",
        json={
            "question": "What is retrieval augmented generation?",
            "chat_history": []
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 50
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert "processing_time_ms" in data


def test_chat_with_history():
    """Chat endpoint should handle conversation history correctly."""
    response = client.post(
        "/chat",
        json={
            "question": "Can you elaborate on that?",
            "chat_history": [
                {"role": "user", "content": "What is fine-tuning?"},
                {"role": "assistant", "content": "Fine-tuning is adapting a pre-trained model."}
            ]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 20


def test_chat_empty_question():
    """Chat endpoint should reject an empty question with 422."""
    response = client.post(
        "/chat",
        json={"question": "", "chat_history": []}
    )
    # 422 Unprocessable Entity — Pydantic validation catches min_length=1
    assert response.status_code == 422


def test_chat_missing_question():
    """Chat endpoint should reject a request with no question field."""
    response = client.post(
        "/chat",
        json={"chat_history": []}
    )
    assert response.status_code == 422


def test_sources_have_required_fields():
    """Each source in the response should have title, authors, published, url."""
    response = client.post(
        "/chat",
        json={"question": "What is a transformer model?", "chat_history": []}
    )
    assert response.status_code == 200
    sources = response.json()["sources"]
    for source in sources:
        assert "title" in source
        assert "authors" in source
        assert "published" in source
        assert "url" in source