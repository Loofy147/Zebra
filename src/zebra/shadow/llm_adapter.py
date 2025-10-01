# src/zebra/shadow/llm_adapter.py
from __future__ import annotations
from typing import Dict, Any, Optional
import time
import logging
import json

logger = logging.getLogger("zebra.shadow.llm_adapter")

class BaseLLMAdapter:
    """
    واجهة موحّدة لمزوّدي LLM:
    - implement call(prompt, **kwargs) -> raw string
    - implement call_structured(prompt, schema, **kwargs) -> dict
    """

    def call(self, prompt: str, temperature: float = 0.0, **kwargs) -> str:
        raise NotImplementedError

    def call_structured(self, prompt: str, schema: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        يحاول أن يُرجع JSON من النموذج. الافتراضي يحاول parse JSON من النص.
        """
        raw = self.call(prompt, **kwargs)
        try:
            return json.loads(raw)
        except Exception:
            logger.warning("LLM returned non-JSON or parse failed; returning raw text in 'raw' field")
            return {"raw": raw}

# --- Example local/mock adapter for unit tests / offline PoC
class MockLLMAdapter(BaseLLMAdapter):
    def __init__(self, delay: float = 0.01):
        self.delay = delay

    def call(self, prompt: str, temperature: float = 0.0, **kwargs) -> str:
        time.sleep(self.delay)
        # very small heuristic PoC response
        return '{"insights": ["mock-insight"], "success": True}'