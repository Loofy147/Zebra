# src/zebra/governance/policy_checker.py
from __future__ import annotations
from typing import Dict, Any, List
import logging

logger = logging.getLogger("zebra.policy")

class PolicyChecker:
    """
    محرك قواعد بسيط: يمكن استبداله بمحرك قواعد حقيقي (OPA, Cerbos, custom rules).
    - check_* methods ترجع True/False
    - assess_reversibility: تقدير مبدئي لقابلية التراجع
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}

    def check_safety(self, decision: Dict[str, Any]) -> bool:
        # مثال: لا نسمح بتغييرات تؤدي إلى انخفاض أداءٍ أكبر من threshold
        worst_case = float(decision.get('expected_impact', {}).get('performance_worst_case', 0.0))
        if worst_case < -0.2:
            logger.debug("PolicyChecker: safety fail due to worst_case %s", worst_case)
            return False
        return True

    def check_ethics(self, decision: Dict[str, Any]) -> bool:
        # PoC: حظر اتخاذ قرارات تعتمد على فئات حساسة بدون مراجعة بشرية
        if decision.get('uses_sensitive_feature', False) and decision.get('human_review') is None:
            return False
        return True

    def check_legal(self, decision: Dict[str, Any]) -> bool:
        # PoC: تحقق بسيط من قيود قانونية في metadata
        legal_flags = decision.get('legal_flags', {})
        if legal_flags.get('may_violate_gdpr'):
            return False
        return True

    def check_business_rules(self, decision: Dict[str, Any]) -> bool:
        # PoC: التأكد أن التكلفة المتوقعة ضمن حدود
        cost = float(decision.get('expected_impact', {}).get('cost_change', 0.0))
        max_cost = float(self.config.get('max_cost_change', 0.5))
        return abs(cost) <= max_cost

    def assess_reversibility(self, decision: Dict[str, Any]) -> str:
        # يقدّر إمكانية التراجع (high/medium/low)
        if decision.get('reversible', True) is False:
            return "low"
        if abs(float(decision.get('expected_impact', {}).get('cost_change', 0.0))) > 0.15:
            return "medium"
        return "high"