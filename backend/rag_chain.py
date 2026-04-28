"""
backend/rag_chain.py

Builds and exposes the RAG chain that powers LexiAssist.
Handles retrieval from ChromaDB, prompt construction,
LLM generation, and conversation memory.
"""

import logging
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

from backend.embeddings import get_retriever

load_dotenv()

logger = logging.getLogger(__name__)

# ── Prompt Template ───────────────────────────────────────────────────────────

# This is the system prompt that shapes how the LLM behaves.
# {context} will be replaced with retrieved document chunks.
# {chat_history} will be replaced with previous messages.
# {question} will be replaced with the user's current question.
SYSTEM_PROMPT = """You are LexiAssist, an expert AI research assistant with \
deep knowledge of machine learning, NLP, and AI systems.

Answer the user's question based on the following research paper excerpts. \
Be precise, informative, and cite the papers you reference by their title.

If the context does not contain enough information to answer the question \
confidently, say so clearly — do not fabricate information.

Context from research papers:
{context}

When citing sources, use the format: [Paper Title] by [Authors]"""


def format_documents(docs: list[Document]) -> str:
    """
    Formats a list of retrieved Document objects into a single
    context string to inject into the prompt.
    """
    formatted = []
    for i, doc in enumerate(docs):
        title = doc.metadata.get("title", "Unknown")
        authors = doc.metadata.get("authors", "Unknown")
        content = doc.page_content

        formatted.append(
            f"[Source {i+1}] {title} by {authors}\n{content}"
        )

    return "\n\n---\n\n".join(formatted)


def format_chat_history(chat_history: list[dict]) -> list:
    """
    Converts our internal chat history format (list of dicts)
    into LangChain message objects that the prompt template expects.
    """
    messages = []
    for message in chat_history:
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"]))
    return messages


def get_llm() -> ChatOpenAI:
    """
    Returns a ChatOpenAI instance using gpt-4o-mini.
    gpt-4o-mini gives excellent quality at very low cost —
    ideal for a RAG application with many queries.
    """
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True
    )


def build_rag_chain():
    """
    Builds and returns the full RAG chain.

    The chain flow is:
    user question → retrieve docs → format context →
    build prompt → LLM → parse output string
    """
    retriever = get_retriever()
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # StrOutputParser extracts the plain text string from the LLM's
    # response object, so callers receive a clean string answer
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    return chain, retriever


# Module-level chain instance — built once when the module is imported
# so we don't rebuild it on every request
_chain, _retriever = build_rag_chain()


def ask(question: str, chat_history: list[dict] = None) -> dict:
    """
    Main entry point for querying the RAG pipeline.

    Parameters
    ----------
    question : str
        The user's question
    chat_history : list[dict]
        Previous messages in format [{"role": "user", "content": "..."}]

    Returns
    -------
    dict with keys:
        answer   : str  — the LLM's response
        sources  : list — the papers used to generate the answer
    """
    if chat_history is None:
        chat_history = []

    # Retrieve relevant document chunks for this question
    docs = _retriever.invoke(question)

    # Format documents into a context string for the prompt
    context = format_documents(docs)

    # Convert chat history to LangChain message objects
    formatted_history = format_chat_history(chat_history)

    # Run the chain — this calls the LLM and returns the answer string
    answer = _chain.invoke({
        "context": context,
        "chat_history": formatted_history,
        "question": question,
    })

    # Build a deduplicated list of source papers for citation display
    seen_titles = set()
    sources = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        if title not in seen_titles:
            seen_titles.add(title)
            sources.append({
                "title": title,
                "authors": doc.metadata.get("authors", ""),
                "published": doc.metadata.get("published", ""),
                "url": doc.metadata.get("url", ""),
            })

    logger.info(f"Question answered using {len(sources)} unique sources")

    return {
        "answer": answer,
        "sources": sources,
    }