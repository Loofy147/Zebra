# tests/test_governance.py
import pytest
from src.zebra.governance.governance import GovernanceSystem
from src.zebra.governance.audit_log import AuditLog
from src.zebra.governance.human_oversight import HumanOversightInterface
from src.zebra.governance.policy_checker import PolicyChecker

@pytest.fixture
def governance_system(tmp_path):
    """Fixture to create a GovernanceSystem instance with a temporary audit log."""
    audit_db_path = str(tmp_path / "audit.db")
    audit_log = AuditLog(db_path=audit_db_path)
    policy_checker = PolicyChecker()
    human_oversight = HumanOversightInterface()
    return GovernanceSystem(audit_store=audit_log, policy_checker=policy_checker, human_interface=human_oversight)

def test_require_human_approval_high_risk(governance_system):
    """Test that a high-risk decision requires human approval."""
    decision = {"risk_level": "high"}
    assert governance_system.require_human_approval(decision) is True

def test_require_human_approval_low_confidence(governance_system):
    """Test that a low-confidence decision requires human approval."""
    decision = {"risk_level": "low", "confidence": 0.6}
    assert governance_system.require_human_approval(decision) is True

def test_no_human_approval_needed(governance_system):
    """Test that a low-risk, high-confidence decision does not require human approval."""
    decision = {"risk_level": "low", "confidence": 0.9, "expected_impact": {"cost_change": 0.0}}
    assert governance_system.require_human_approval(decision) is False

def test_check_decision_compliance_pass(governance_system):
    """Test a decision that should pass all compliance checks."""
    decision = {
        "id": "decision-pass-123",
        "expected_impact": {"performance_worst_case": 0.0, "cost_change": 0.0}
    }
    assert governance_system.check_decision_compliance(decision) is True

def test_check_decision_compliance_fail_safety(governance_system):
    """Test a decision that should fail the safety check."""
    decision = {
        "id": "decision-fail-456",
        "expected_impact": {"performance_worst_case": -0.5}
    }
    assert governance_system.check_decision_compliance(decision) is False