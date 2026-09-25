"""
LangGraph Orchestrator — Multi-agent coordination for medical queries
Routes queries, retrieves context, generates grounded responses.
"""

from typing import Dict, List, Optional, TypedDict

import streamlit as st

try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_community.llms import Ollama
    from langchain_core.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional runtime extras may be absent
    StateGraph = None
    END = None
    HumanMessage = AIMessage = SystemMessage = None
    Ollama = None
    ChatPromptTemplate = None
    LANGCHAIN_AVAILABLE = False

from .rag_pipeline import RAGPipeline, get_rag_pipeline


class MedicalState(TypedDict):
    """State for the medical assistant graph."""
    query: str
    intent: str
    context: str
    response: str
    citations: List[Dict]
    needs_escalation: bool
    confidence: float


class LangGraphOrchestrator:
    """Orchestrates medical query processing with RAG and guardrails."""

    def __init__(
        self,
        rag_pipeline: RAGPipeline = None,
        llm_model: str = "llama3.2:3b",
        ollama_base_url: str = "http://localhost:11434",
    ):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "Chatbot dependencies are missing. Install the project requirements with: "
                "pip install -r requirements.txt"
            )
        self.rag = rag_pipeline or get_rag_pipeline()
        self.llm = Ollama(model=llm_model, base_url=ollama_base_url, temperature=0.1)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(MedicalState)

        # Add nodes
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("verify_grounding", self._verify_grounding)
        workflow.add_node("escalate", self._escalate)

        # Define edges
        workflow.set_entry_point("classify_intent")
        workflow.add_edge("classify_intent", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", "verify_grounding")
        workflow.add_conditional_edges(
            "verify_grounding",
            self._should_escalate,
            {
                "escalate": "escalate",
                "complete": END,
            }
        )
        workflow.add_edge("escalate", END)

        return workflow.compile()

    def _classify_intent(self, state: MedicalState) -> MedicalState:
        """Classify the medical query intent."""
        query = state["query"].lower()

        intent_keywords = {
            "diagnosis": ["diagnos", "what is", "identify", "detect", "symptoms"],
            "treatment": ["treat", "therapy", "medication", "drug", "dosage", "protocol"],
            "differential": ["differential", "ddx", "rule out", "versus", "vs"],
            "lab_interpretation": ["lab", "blood", "cbc", "result", "value", "interpret"],
            "guideline": ["guideline", "who", "protocol", "recommendation", "standard"],
            "drug_interaction": ["interact", "contraindicat", "side effect", "adverse"],
        }

        intent = "general"
        for key, keywords in intent_keywords.items():
            if any(kw in query for kw in keywords):
                intent = key
                break

        state["intent"] = intent
        return state

    def _retrieve_context(self, state: MedicalState) -> MedicalState:
        """Retrieve relevant medical knowledge."""
        # Enhance query with intent context
        enhanced_query = f"[{state['intent']}] {state['query']}"
        context = self.rag.get_context_string(enhanced_query, n_results=5)
        state["context"] = context
        state["citations"] = self.rag.query(enhanced_query, n_results=5)
        return state

    def _generate_response(self, state: MedicalState) -> MedicalState:
        """Generate grounded medical response."""
        system_prompt = """You are a medical AI assistant for clinical decision support.
Your responses must:
1. Be grounded ONLY in the provided context from medical guidelines
2. Cite sources using [Source: filename] format
3. Express uncertainty when context is insufficient
4. Never provide definitive diagnoses - only decision support
5. Include appropriate disclaimers for clinical use

Context:
{context}

Query: {query}
Intent: {intent}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}"),
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "context": state["context"],
            "query": state["query"],
            "intent": state["intent"],
        })

        state["response"] = response
        return state

    def _verify_grounding(self, state: MedicalState) -> MedicalState:
        """Verify response is grounded in retrieved context."""
        response = state["response"].lower()
        context = state["context"].lower()

        # Simple grounding check: key medical terms in response should appear in context
        medical_terms = [
            "parasitemia", "malaria", "sickle", "leukemia", "anemia",
            "pneumonia", "tuberculosis", "cardiomegaly", "tumor", "edema",
            "treatment", "dosage", "who", "guideline", "protocol",
        ]

        ungrounded_terms = []
        for term in medical_terms:
            if term in response and term not in context:
                ungrounded_terms.append(term)

        # Calculate confidence based on grounding
        if ungrounded_terms:
            state["confidence"] = 0.5
            state["needs_escalation"] = True
            state["response"] += f"\n\n⚠️ **Uncertainty Note**: Some terms in this response ({', '.join(ungrounded_terms)}) were not found in the retrieved guidelines. Please verify with official sources."
        else:
            state["confidence"] = 0.9
            state["needs_escalation"] = False

        return state

    def _should_escalate(self, state: MedicalState) -> str:
        """Determine if escalation is needed."""
        if state["needs_escalation"] or state["confidence"] < 0.6:
            return "escalate"
        return "complete"

    def _escalate(self, state: MedicalState) -> MedicalState:
        """Add escalation notice to response."""
        state["response"] += (
            "\n\n🔴 **ESCALATION RECOMMENDED**: This query requires specialist review. "
            "The AI response has low confidence or contains ungrounded information. "
            "Please consult a qualified specialist or refer to official clinical guidelines."
        )
        return state

    def process_query(self, query: str) -> Dict:
        """Process a medical query through the orchestrator."""
        initial_state = MedicalState(
            query=query,
            intent="",
            context="",
            response="",
            citations=[],
            needs_escalation=False,
            confidence=0.0,
        )

        result = self.graph.invoke(initial_state)
        return {
            "query": query,
            "intent": result["intent"],
            "response": result["response"],
            "citations": result["citations"],
            "confidence": result["confidence"],
            "needs_escalation": result["needs_escalation"],
        }


@st.cache_resource
def get_orchestrator(
    rag_pipeline: RAGPipeline = None,
    llm_model: str = "llama3.2:3b",
) -> LangGraphOrchestrator:
    """Get cached orchestrator instance."""
    return LangGraphOrchestrator(rag_pipeline, llm_model)