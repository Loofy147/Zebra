# src/zebra/governance/governance.py
from __future__ import annotations
from datetime import datetime
import logging
from typing import Dict, Any, List, Optional

from src.zebra.governance.policy_checker import PolicyChecker
from src.zebra.governance.audit_log import AuditLog
from src.zebra.governance.human_oversight import HumanOversightInterface

logger = logging.getLogger("zebra.governance")


class GovernanceSystem:
    """
    نظام الحوكمة المركزي: يجمع الفحوصات، السجل، ومداخل المراجعة البشرية.
    PoC: PolicyChecker/AuditLog/HumanOversight يمكن استبدالهم بتجهيزات إنتاجية.
    """

    def __init__(self, audit_store: Optional[AuditLog] = None,
                 policy_checker: Optional[PolicyChecker] = None,
                 human_interface: Optional[HumanOversightInterface] = None):
        self.policy_checker = policy_checker or PolicyChecker()
        self.audit_log = audit_store or AuditLog()
        self.human_oversight = human_interface or HumanOversightInterface()

    def check_decision_compliance(self, decision: Dict[str, Any]) -> bool:
        """
        يطبّق سلسلة من الفحوصات على القرار.
        - decision: dict يتضمن الحقول المتوقعة (id, inputs, outputs, expected_impact, risk_level, ...)
        """
        checks = {
            'safety': self.policy_checker.check_safety(decision),
            'ethics': self.policy_checker.check_ethics(decision),
            'legal': self.policy_checker.check_legal(decision),
            'business': self.policy_checker.check_business_rules(decision)
        }

        all_ok = all(checks.values())

        if not all_ok:
            failed = [k for k, v in checks.items() if not v]
            logger.warning("Decision %s failed compliance checks: %s", decision.get("id"), failed)
            self.audit_log.log_violation(decision, failed)

            if 'safety' in failed or 'legal' in failed:
                # high priority human review
                ticket = self.human_oversight.request_review(
                    decision,
                    priority='high',
                    reason=failed
                )
                logger.info("Human review requested: %s", ticket)

            return False

        # record approval event
        self.audit_log.log_approval(decision, checks)
        return True

    def require_human_approval(self, decision: Dict[str, Any]) -> bool:
        """
        قواعد لتحديد حاجه القرار لموافقة بشرية (PoC rules).
        """
        if decision.get('risk_level') == 'high':
            return True

        confidence = float(decision.get('confidence', 1.0))
        if confidence < 0.7:
            return True

        impact = decision.get('expected_impact', {})
        cost_change = abs(float(impact.get('cost_change', 0.0)))
        if cost_change > 0.2:
            return True

        # uncommon decision types or new policy flags
        if decision.get('type') in ('destructive', 'infra-change'):
            return True

        return False

    def explainability_for_auditing(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        يولّد تقرير تدقيق قابل للمراجعة، ويخزّنه في سجل المراجعة.
        يتوقّع القرار حقولاً: id, inputs, outputs, causal_chain, confidence_scores, alternatives, selection_rationale, human_review (opt).
        """
        report = {
            'decision_id': decision.get('id'),
            'timestamp': datetime.utcnow().isoformat(),
            'decision_maker': decision.get('decision_maker', 'zebra_ai'),
            'inputs': decision.get('inputs'),
            'outputs': decision.get('outputs'),
            'reasoning': {
                'causal_chain': decision.get('causal_chain'),
                'confidence_scores': decision.get('confidence_scores'),
                'alternatives_considered': decision.get('alternatives'),
                'why_chosen': decision.get('selection_rationale')
            },
            'compliance_checks': {
                'safety': self.policy_checker.check_safety(decision),
                'ethics': self.policy_checker.check_ethics(decision),
                'legal': self.policy_checker.check_legal(decision),
                'business': self.policy_checker.check_business_rules(decision)
            },
            'human_oversight': decision.get('human_review'),
            'reversibility': self.policy_checker.assess_reversibility(decision)
        }
        self.audit_log.store(report)
        return report