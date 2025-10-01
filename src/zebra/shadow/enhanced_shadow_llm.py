# src/zebra/shadow/enhanced_shadow_llm.py
from __future__ import annotations
from typing import Dict, Any, Optional
import json
from datetime import datetime
import logging

from src.zebra.shadow.llm_adapter import BaseLLMAdapter
from src.zebra_telemetry.opentelemetry_setup import ZebraObservability  # assumes earlier module

logger = logging.getLogger("zebra.shadow.enhanced")

PROMPT_TEMPLATE_STEP_BY_STEP = """
أنت محلل سببي خبير — قُم بتحليل البيانات خطوة بخطوة وابدأ بذكر الفرضيات ثم الدلائل:
### البيانات التشغيلية
{telemetry_json}

### الرسم السببي (مختصر)
{causal_graph_brief}

### المطلوب
1) تحديد الأنماط غير العادية
2) اقتراح علاقات سببية محتملة (مع تقدير ثقة 0..1)
3) اقتراح تجارب تحقق قصيرة المدى
4) تقديم توصيات قابلة للتنفيذ (صنفها Informational/Suggested/Actionable)

أعد الإجابة بصيغة JSON تحتوي الحقول: insights, causal_hypotheses (list of {{cause, effect, confidence}}), experiments, recommendations
"""

class EnhancedShadowLLM:
    def __init__(self, llm_adapter: BaseLLMAdapter, observability: Optional[ZebraObservability] = None):
        self.llm = llm_adapter
        self.obs = observability
        # memory minimal: keep recent analyses (in-memory), production => use persistent store
        self._recent_memory = []

    def analyze_with_chain_of_thought(self, telemetry_data: Dict[str, Any], causal_graph: Any, temperature: float = 0.2) -> Dict[str, Any]:
        # prepare prompt
        telemetry_json = json.dumps(telemetry_data, default=str, indent=2)
        causal_brief = getattr(causal_graph, "to_natural_language", lambda: str(causal_graph))()
        prompt = PROMPT_TEMPLATE_STEP_BY_STEP.format(telemetry_json=telemetry_json, causal_graph_brief=causal_brief)

        # telemetry span
        if self.obs:
            from opentelemetry import trace
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span("shadow_llm.analyze"):
                resp = self.llm.call(prompt, temperature=temperature)
        else:
            resp = self.llm.call(prompt, temperature=temperature)

        # parse structured or fallback
        try:
            parsed = json.loads(resp)
        except Exception:
            parsed = {"raw": resp}

        # save minimal memory artifact (id + timestamp + success flag)
        artifact = {"timestamp": datetime.utcnow().isoformat(), "telemetry": telemetry_data, "result": parsed}
        self._recent_memory.append(artifact)
        # keep small
        self._recent_memory = self._recent_memory[-20:]

        # observability: emit metric/event
        if self.obs:
            try:
                self.obs.record_request(path="shadow.analyze", method="LLM", status_code=200, latency_ms=0)  # latency best-effort
            except Exception:
                logger.exception("observability record failed")

        return parsed

    def generate_counterfactual_scenarios(self, current_state: Dict[str, Any], n: int = 5) -> Dict[str, Any]:
        prompt = f"""الحالة الحالية:\n{json.dumps(current_state, indent=2)}\n\nاقترح {n} سيناريوهات ماذا لو؟ استجب بJSON صارم."""
        resp = self.llm.call_structured(prompt, temperature=0.3)
        return resp

    def explain_causal_relationship(self, cause: str, effect: str, strength: float) -> str:
        prompt = f"""اشرح بسرعة: السبب: {cause} - الأثر: {effect} - القوة: {strength}. اجعل الشرح 1-3 جمل مفيدة لغير المتخصصين."""
        return self.llm.call(prompt, temperature=0.6)