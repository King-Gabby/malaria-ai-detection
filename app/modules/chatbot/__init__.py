"""
Chatbot Module — Medical Assistant with RAG
Uses LangGraph for orchestration, local vector store for medical knowledge.
"""

try:
    from .rag_pipeline import RAGPipeline
except ImportError:  # pragma: no cover - optional runtime extras may be absent
    RAGPipeline = None

try:
    from .langgraph_orchestrator import LangGraphOrchestrator
except ImportError:  # pragma: no cover - optional runtime extras may be absent
    LangGraphOrchestrator = None

try:
    from .ui import render_chatbot_ui
except ImportError:  # pragma: no cover - optional runtime extras may be absent
    def render_chatbot_ui() -> None:
        import streamlit as st
        st.error("The medical chatbot requires optional dependencies. Install project requirements with: pip install -r requirements.txt")
        st.caption("The rest of the app remains available without the chatbot feature.")

__all__ = ["RAGPipeline", "LangGraphOrchestrator", "render_chatbot_ui"]