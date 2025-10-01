# src/zebra/governance/human_oversight.py
from __future__ import annotations
from typing import Dict, Any, Optional, List
import uuid
import logging
from datetime import datetime

logger = logging.getLogger("zebra.human_oversight")

class HumanOversightInterface:
    """
    PoC interface for human oversight:
    - request_review returns a review ticket id
    - Integrators should implement real ticketing/email/Slack/issue system integration
    """

    def request_review(self, decision: Dict[str, Any], priority: str = "normal", reason: Optional[List[str]] = None) -> Dict[str, Any]:
        ticket_id = str(uuid.uuid4())
        ticket = {
            "ticket_id": ticket_id,
            "decision_id": decision.get("id"),
            "priority": priority,
            "reason": reason or [],
            "created_at": datetime.utcnow().isoformat(),
            "status": "open"
        }
        # In prod: push to ticketing system (Jira/ServiceNow), notify oncall, attach audit artifacts.
        logger.info("HumanOversight: opened ticket %s for decision %s (priority=%s)", ticket_id, decision.get("id"), priority)
        return ticket