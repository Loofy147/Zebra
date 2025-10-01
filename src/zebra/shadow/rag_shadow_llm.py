# src/zebra/shadow/rag_shadow_llm.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
import logging
from datetime import datetime

from src.zebra.shadow.enhanced_shadow_llm import EnhancedShadowLLM
from src.zebra.shadow.llm_adapter import BaseLLMAdapter
from src.zebra_telemetry.opentelemetry_setup import ZebraObservability

logger = logging.getLogger("zebra.shadow.rag")

# Minimal VectorStore interface (adapter pattern)
class BaseVectorStore:
    def add(self, embedding: List[float], document: str, metadata: Dict[str, Any]) -> str:
        raise NotImplementedError

    def search(self, embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError

class RAGEnhancedShadowLLM:
    def __init__(
        self,
        llm_adapter: BaseLLMAdapter,
        vector_store: BaseVectorStore,
        embedder: Any,
        observability: Optional[ZebraObservability] = None,
    ):
        self.llm = EnhancedShadowLLM(llm_adapter, observability=observability)
        self.vs = vector_store
        self.embedder = embedder
        self.obs = observability

    def store_historical_analysis(self, analysis: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        doc = json.dumps(analysis, default=str)
        embedding = self.embedder.embed(doc)
        doc_id = self.vs.add(embedding=embedding, document=doc, metadata=metadata)
        return doc_id

    def analyze_with_context(self, current_telemetry: Dict[str, Any], causal_graph: Any, k: int = 5) -> Dict[str, Any]:
        query_emb = self.embedder.embed(json.dumps(current_telemetry, default=str))
        neighbors = self.vs.search(query_emb, k=k)
        # compose context
        context_docs = [n.get("document") for n in neighbors]
        prompt_context = {
            "telemetry": current_telemetry,
            "past_analyses": context_docs,
        }
        # call underlying llm with additional context (we pass causal_graph too)
        result = self.llm.analyze_with_chain_of_thought(prompt_context, causal_graph)
        # store result
        try:
            self.store_historical_analysis(result, {"timestamp": datetime.utcnow().isoformat(), "success": result.get("success", False)})
        except Exception:
            logger.exception("store_historical_analysis failed")
        return result