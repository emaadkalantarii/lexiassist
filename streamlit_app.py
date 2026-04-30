"""
streamlit_app.py

Self-contained Streamlit app for cloud deployment.
Runs the RAG chain directly without requiring a separate FastAPI server.
"""

import streamlit as st
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="LexiAssist",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
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
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_chain():
    """
    Loads the RAG chain once and caches it for the session.
    st.cache_resource means this runs only once per app instance,
    not on every user interaction — essential for performance.
    """
    from backend.rag_chain import ask
    return ask


@st.cache_resource
def load_vectorstore():
    """
    Builds or loads the vector store once and caches it.
    On first run it downloads papers and builds the index.
    """
    import json
    from pathlib import Path
    from backend.embeddings import load_documents, chunk_documents, build_vector_store, load_vector_store

    vectorstore_path = Path("vectorstore")
    processed_path = Path("data/processed/documents.json")

    # If vectorstore already exists, load it
    if vectorstore_path.exists() and any(vectorstore_path.iterdir()):
        return load_vector_store()

    # Otherwise build it from processed documents
    if processed_path.exists():
        documents = load_documents()
        chunks = chunk_documents(documents)
        return build_vector_store(chunks)

    # If no data exists at all, run ingestion first
    from backend.ingest import run_ingestion
    run_ingestion()
    documents = load_documents()
    chunks = chunk_documents(documents)
    return build_vector_store(chunks)


def render_source_card(source: dict) -> None:
    """Renders a single source paper as a styled card."""
    url = source.get("url", "#")
    title = source.get("title", "Unknown")
    authors = source.get("authors", "")
    published = source.get("published", "")

    st.markdown(f"""
        <div class="source-card">
            <div class="source-title">
                <a href="{url}" target="_blank" style="color:#89b4fa; text-decoration:none;">
                    📄 {title}
                </a>
            </div>
            <div class="source-meta">{authors} · {published}</div>
        </div>
    """, unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# ── Sidebar ───────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🤖 LexiAssist")
    st.markdown("*AI Research Paper Assistant*")
    st.divider()

    st.markdown("### Knowledge Base")
    st.info("📚 150+ AI/ML research papers covering RAG, transformers, fine-tuning, agents, RLHF, and more.")

    st.divider()

    st.markdown("### Session Stats")
    st.metric("Questions Asked", st.session_state.total_queries)

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()

    st.divider()
    st.markdown(
        "<div style='text-align:center; color:#a6adc8; font-size:0.8em;'>"
        "Built with LangChain · ChromaDB<br>FastAPI · Streamlit · OpenAI"
        "</div>",
        unsafe_allow_html=True
    )

# ── Main Interface ────────────────────────────────────────────

st.markdown(
    "<div style='text-align:center; padding: 1rem 0;'>"
    "<h1>🤖 LexiAssist</h1>"
    "<p style='color:#a6adc8'>Ask me anything about AI and ML research</p>"
    "</div>",
    unsafe_allow_html=True
)

# Load the RAG chain — cached after first load
ask = load_rag_chain()

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📚 Sources ({len(message['sources'])} papers)", expanded=False):
                for source in message["sources"]:
                    render_source_card(source)

# Chat input
if prompt := st.chat_input("Ask a question about AI/ML research..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    api_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Searching research papers..."):
            try:
                result = ask(question=prompt, chat_history=api_history)
                answer = result.get("answer", "Sorry, I could not generate an answer.")
                sources = result.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander(f"📚 Sources ({len(sources)} papers)", expanded=False):
                        for source in sources:
                            render_source_card(source)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                st.session_state.total_queries += 1

            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })