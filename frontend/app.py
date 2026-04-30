"""
frontend/app.py

Streamlit chat interface for LexiAssist.
Communicates with the FastAPI backend via HTTP requests.

Run with:
    streamlit run frontend/app.py
"""

import streamlit as st
import requests
import logging

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# The URL of your FastAPI backend.
# When running locally both services run simultaneously —
# FastAPI on port 8000, Streamlit on port 8501.
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8999")

# ── Page Setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LexiAssist",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Styling ────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .source-card {
        background-color: #1e1e2e;
        border-left: 3px solid #6c63ff;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 4px;
        font-size: 0.85em;
    }
    .source-title {
        font-weight: bold;
        color: #cdd6f4;
    }
    .source-meta {
        color: #a6adc8;
        font-size: 0.8em;
    }
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ── Helper Functions ──────────────────────────────────────────────────────────


def check_api_health() -> bool:
    """
    Pings the FastAPI health endpoint to verify the backend is running.
    Returns True if healthy, False otherwise.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def send_chat_request(question: str, chat_history: list) -> dict | None:
    """
    Sends the user's question and conversation history to the
    FastAPI /chat endpoint and returns the response dict.
    Returns None if the request fails.
    """
    try:
        payload = {"question": question, "chat_history": chat_history}
        response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Make sure the FastAPI server is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. The server may be overloaded.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {str(e)}")
        return None


def render_source_card(source: dict) -> None:
    """Renders a single source paper as a styled card."""
    url = source.get("url", "#")
    title = source.get("title", "Unknown")
    authors = source.get("authors", "")
    published = source.get("published", "")

    st.markdown(
        f"""
        <div class="source-card">
            <div class="source-title">
                <a href="{url}" target="_blank" style="color:#89b4fa; text-decoration:none;">
                    📄 {title}
                </a>
            </div>
            <div class="source-meta">{authors} · {published}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )


# ── Session State Initialization ──────────────────────────────────────────────
# Streamlit reruns the entire script on every user interaction.
# st.session_state persists data across reruns — it's how we maintain
# the conversation history throughout the session.

if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🤖 LexiAssist")
    st.markdown("*AI Research Paper Assistant*")
    st.divider()

    # API health indicator
    st.markdown("### API Status")
    if check_api_health():
        st.success("Backend connected", icon="✅")
    else:
        st.error("Backend offline", icon="🔴")
        st.info(
            "Start the backend with:\n```\nuvicorn backend.main:app --port 8000\n```"
        )

    st.divider()

    # Knowledge base info
    st.markdown("### Knowledge Base")
    st.info(
        "📚 150+ AI/ML research papers covering RAG, transformers, fine-tuning, agents, RLHF, and more."
    )

    st.divider()

    # Session stats
    st.markdown("### Session Stats")
    st.metric("Questions Asked", st.session_state.total_queries)

    st.divider()

    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()

    st.divider()
    st.markdown(
        "<div style='text-align:center; color:#a6adc8; font-size:0.8em;'>"
        "Built with LangChain · ChromaDB<br>FastAPI · Streamlit · OpenAI"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Main Chat Interface ───────────────────────────────────────────────────────

st.markdown(
    "<div class='main-header'><h1>🤖 LexiAssist</h1>"
    "<p style='color:#a6adc8'>Ask me anything about AI and ML research</p></div>",
    unsafe_allow_html=True,
)

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources if this was an assistant message with sources
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(
                f"📚 Sources ({len(message['sources'])} papers)", expanded=False
            ):
                for source in message["sources"]:
                    render_source_card(source)

# Chat input — st.chat_input renders a fixed input box at the bottom
# of the page. It returns the typed text when the user hits Enter,
# and returns None when the input is empty.
if prompt := st.chat_input("Ask a question about AI/ML research..."):

    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build the chat history format the API expects
    # We exclude the message we just added (last item) since
    # that's the current question, not history
    api_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    # Call the API and display the response
    with st.chat_message("assistant"):
        with st.spinner("Searching research papers..."):
            result = send_chat_request(prompt, api_history)

        if result:
            answer = result.get("answer", "Sorry, I could not generate an answer.")
            sources = result.get("sources", [])
            processing_time = result.get("processing_time_ms", 0)

            st.markdown(answer)

            # Show processing time as a subtle caption
            st.caption(
                f"⚡ Answered in {processing_time}ms using {len(sources)} sources"
            )

            # Expandable sources section
            if sources:
                with st.expander(f"📚 Sources ({len(sources)} papers)", expanded=False):
                    for source in sources:
                        render_source_card(source)

            # Save assistant message to session state
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )

            st.session_state.total_queries += 1

        else:
            error_msg = "I encountered an error. Please try again."
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg, "sources": []}
            )
