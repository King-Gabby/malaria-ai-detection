"""
RAG Pipeline — Medical Knowledge Retrieval
Uses local embeddings and vector store for grounded responses.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
import streamlit as st

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for medical knowledge."""

    def __init__(
        self,
        persist_dir: str = "data/vector_index",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "medical_knowledge",
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name

        # Initialize embedding model (cached)
        self.embedder = self._load_embedder()
        # Initialize vector store
        self.client = self._init_client()
        self.collection = self._get_or_create_collection()

    @st.cache_resource
    def _load_embedder(_self) -> SentenceTransformer:
        """Load embedding model (cached)."""
        return SentenceTransformer(_self.embedding_model_name, device="cpu")

    def _init_client(self) -> chromadb.Client:
        """Initialize ChromaDB client."""
        return chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

    def _get_or_create_collection(self):
        """Get or create the medical knowledge collection."""
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict] = None,
        ids: List[str] = None,
    ) -> None:
        """Add documents to the vector store."""
        if not documents:
            return

        embeddings = self.embedder.encode(documents, show_progress_bar=False).tolist()

        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in documents]

        if metadatas is None:
            metadatas = [{"source": "manual"} for _ in documents]

        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def load_medical_knowledge(self, knowledge_dir: str = "data/medical_knowledge") -> int:
        """Load medical guidelines from markdown/text files."""
        knowledge_path = Path(knowledge_dir)
        if not knowledge_path.exists():
            return 0

        files = list(knowledge_path.glob("*.md")) + list(knowledge_path.glob("*.txt"))
        if not files:
            return 0

        all_chunks = []
        all_metadatas = []
        all_ids = []

        for file_path in files:
            content = file_path.read_text(encoding="utf-8")
            # Simple chunking by paragraphs
            chunks = [c.strip() for c in content.split("\n\n") if len(c.strip()) > 50]
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": file_path.name,
                    "chunk_id": i,
                    "file_path": str(file_path),
                })
                all_ids.append(f"{file_path.stem}_{i}")

        if all_chunks:
            self.add_documents(all_chunks, all_metadatas, all_ids)

        return len(all_chunks)

    def query(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: Dict = None,
    ) -> List[Dict]:
        """Query the vector store for relevant context."""
        query_embedding = self.embedder.encode([query]).tolist()[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict,
        )

        formatted = []
        for i in range(len(results["documents"][0])):
            formatted.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return formatted

    def get_context_string(self, query: str, n_results: int = 5) -> str:
        """Get formatted context string for LLM prompt."""
        results = self.query(query, n_results)
        if not results:
            return "No relevant medical guidelines found."

        context_parts = []
        for r in results:
            source = r["metadata"].get("source", "unknown")
            context_parts.append(f"[Source: {source}]\n{r['content']}")

        return "\n\n---\n\n".join(context_parts)


@st.cache_resource
def get_rag_pipeline(
    persist_dir: str = "data/vector_index",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> RAGPipeline:
    """Get cached RAG pipeline instance."""
    return RAGPipeline(persist_dir, embedding_model)