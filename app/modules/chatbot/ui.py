"""
Chatbot UI — Medical Assistant Interface
Streamlit components for the RAG-powered chatbot.
"""

import streamlit as st
from typing import List, Dict

from .rag_pipeline import get_rag_pipeline
from .langgraph_orchestrator import get_orchestrator


def render_chatbot_ui() -> None:
    """Render the medical assistant chatbot UI."""
    st.markdown("### 🤖 Medical Assistant (RAG + LangGraph)")
    st.caption("Grounded in WHO/NCDC guidelines. Responses include citations. Not for standalone diagnosis.")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "rag_initialized" not in st.session_state:
        with st.spinner("Loading medical knowledge base..."):
            rag = get_rag_pipeline()
            chunks_loaded = rag.load_medical_knowledge()
            st.session_state.rag_initialized = True
            if chunks_loaded > 0:
                st.success(f"Loaded {chunks_loaded} knowledge chunks from guidelines.")
            else:
                st.warning("No medical knowledge files found in data/medical_knowledge/. Add .md or .txt files for RAG.")

    # Initialize orchestrator
    orchestrator = get_orchestrator()

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("📚 Sources"):
                    for cit in msg["citations"]:
                        st.markdown(f"**{cit['metadata'].get('source', 'unknown')}** (distance: {cit['distance']:.3f})")
                        st.caption(cit["content"][:300] + "...")

    # Chat input
    if prompt := st.chat_input("Ask a medical question (e.g., 'WHO malaria treatment guidelines for severe cases')"):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Retrieving guidelines and generating response..."):
                result = orchestrator.process_query(prompt)

            st.markdown(result["response"])

            # Show confidence indicator
            conf = result["confidence"]
            if conf >= 0.8:
                st.success(f"Confidence: {conf:.0%} ✓ Well-grounded")
            elif conf >= 0.5:
                st.warning(f"Confidence: {conf:.0%} ⚠ Partial grounding")
            else:
                st.error(f"Confidence: {conf:.0%} 🔴 Low confidence - escalation recommended")

            if result["citations"]:
                with st.expander("📚 Sources"):
                    for cit in result["citations"]:
                        st.markdown(f"**{cit['metadata'].get('source', 'unknown')}** (distance: {cit['distance']:.3f})")
                        st.caption(cit["content"][:300] + "...")

        # Add assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["response"],
            "citations": result["citations"],
            "confidence": result["confidence"],
        })

    # Sidebar controls
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🛠️ Chatbot Controls")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

        if st.button("🔄 Reload Knowledge Base"):
            st.session_state.rag_initialized = False
            st.rerun()

        st.markdown("---")
        st.markdown("### 💡 Example Queries")
        examples = [
            "WHO severe malaria treatment protocol",
            "Sickle cell crisis management guidelines",
            "ALL induction chemotherapy regimen",
            "Iron deficiency anemia diagnostic criteria",
            "Pneumonia vs TB differential on chest X-ray",
            "Brain tumor MRI classification (WHO CNS5)",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": ex})
                with st.spinner("Processing..."):
                    result = orchestrator.process_query(ex)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["response"],
                    "citations": result["citations"],
                    "confidence": result["confidence"],
                })
                st.rerun()