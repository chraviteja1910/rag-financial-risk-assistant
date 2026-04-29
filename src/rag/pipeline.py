"""
RAG Pipeline — LangGraph-based agentic RAG orchestrator
Author: Ravi Teja Chittaluri
"""

from __future__ import annotations
import time
from typing import Any
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
import mlflow

from .retriever import HybridRetriever
from .evaluation.safety import SafetyLayer
from .evaluation.mlflow_logger import EvalLogger


# ─── State ───────────────────────────────────────────────────
@dataclass
class RAGState:
    query: str = ""
    retrieved_docs: list[dict] = field(default_factory=list)
    reranked_docs: list[dict] = field(default_factory=list)
    raw_response: str = ""
    final_response: str = ""
    citations: list[str] = field(default_factory=list)
    safety_passed: bool = False
    faithfulness_score: float = 0.0
    hallucination_detected: bool = False
    latency_ms: float = 0.0
    error: str | None = None


# ─── RAG Pipeline ────────────────────────────────────────────
class RAGPipeline:
    """
    Production-grade RAG pipeline with:
    - Hybrid FAISS + BM25 retrieval
    - Cross-encoder reranking
    - LLM generation with citation tracking
    - NLI-based safety validation
    - MLflow evaluation logging
    """

    SYSTEM_PROMPT = """You are a financial risk analysis assistant.
Answer questions using ONLY the provided context documents.
Always cite your sources using [Doc N] notation.
If the context does not contain sufficient information, say so clearly.
Never fabricate information not present in the context."""

    def __init__(
        self,
        faiss_index_path: str,
        llm_model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-large",
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5,
        safety_threshold: float = 0.85,
    ):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.0, max_tokens=1500)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.retriever = HybridRetriever(
            faiss_index_path=faiss_index_path,
            embeddings=self.embeddings,
            top_k=top_k_retrieve,
            rerank_top_k=top_k_rerank,
        )
        self.safety = SafetyLayer(threshold=safety_threshold)
        self.logger = EvalLogger()
        self.graph = self._build_graph()

    # ── Graph nodes ──────────────────────────────────────────

    def _retrieve(self, state: RAGState) -> RAGState:
        """Hybrid retrieval: FAISS dense + BM25 sparse."""
        docs = self.retriever.retrieve(state.query)
        state.retrieved_docs = docs
        return state

    def _rerank(self, state: RAGState) -> RAGState:
        """Cross-encoder reranking for precision."""
        reranked = self.retriever.rerank(state.query, state.retrieved_docs)
        state.reranked_docs = reranked
        return state

    def _generate(self, state: RAGState) -> RAGState:
        """LLM generation with context window management."""
        context = self._build_context(state.reranked_docs)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state.query}")
        ])
        response = self.llm.invoke(prompt.format_messages())
        state.raw_response = response.content
        state.citations = self._extract_citations(response.content)
        return state

    def _validate_safety(self, state: RAGState) -> RAGState:
        """NLI-based hallucination detection + faithfulness scoring."""
        result = self.safety.validate(
            query=state.query,
            response=state.raw_response,
            context_docs=state.reranked_docs,
        )
        state.safety_passed = result.passed
        state.faithfulness_score = result.faithfulness_score
        state.hallucination_detected = result.hallucination_detected
        state.final_response = (
            state.raw_response if result.passed
            else "I cannot provide a reliable answer based on the available documents. Please consult a risk officer."
        )
        return state

    def _log_metrics(self, state: RAGState) -> RAGState:
        """Log all evaluation metrics to MLflow."""
        self.logger.log({
            "query": state.query,
            "faithfulness_score": state.faithfulness_score,
            "hallucination_detected": int(state.hallucination_detected),
            "safety_passed": int(state.safety_passed),
            "num_docs_retrieved": len(state.retrieved_docs),
            "num_docs_reranked": len(state.reranked_docs),
            "latency_ms": state.latency_ms,
        })
        return state

    # ── Graph builder ─────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(RAGState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("rerank", self._rerank)
        graph.add_node("generate", self._generate)
        graph.add_node("validate", self._validate_safety)
        graph.add_node("log", self._log_metrics)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "generate")
        graph.add_edge("generate", "validate")
        graph.add_edge("validate", "log")
        graph.add_edge("log", END)
        return graph.compile()

    # ── Public interface ──────────────────────────────────────

    async def arun(self, query: str) -> dict[str, Any]:
        """Run the full RAG pipeline for a query."""
        start = time.perf_counter()
        state = RAGState(query=query)
        final_state = await self.graph.ainvoke(state)
        final_state.latency_ms = (time.perf_counter() - start) * 1000
        return {
            "answer": final_state.final_response,
            "citations": final_state.citations,
            "faithfulness_score": final_state.faithfulness_score,
            "safety_passed": final_state.safety_passed,
            "latency_ms": round(final_state.latency_ms, 2),
        }

    # ── Helpers ───────────────────────────────────────────────

    def _build_context(self, docs: list[dict], max_tokens: int = 6000) -> str:
        """Build context string with token budget management."""
        context_parts = []
        total_chars = 0
        char_budget = max_tokens * 4  # ~4 chars per token
        for i, doc in enumerate(docs):
            chunk = f"[Doc {i+1}] {doc['source']}\n{doc['content']}\n"
            if total_chars + len(chunk) > char_budget:
                break
            context_parts.append(chunk)
            total_chars += len(chunk)
        return "\n".join(context_parts)

    def _extract_citations(self, response: str) -> list[str]:
        """Extract [Doc N] citation references from response."""
        import re
        return list(set(re.findall(r'\[Doc \d+\]', response)))
