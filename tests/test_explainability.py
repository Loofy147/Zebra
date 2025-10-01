# tests/test_explainability.py
import pytest
import pandas as pd
from src.zebra.explainability.decision_explainer import DecisionExplainer, Decision

@pytest.fixture
def decision_explainer():
    """Fixture to create a DecisionExplainer instance."""
    return DecisionExplainer()

@pytest.fixture
def sample_decision():
    """Fixture to create a sample Decision object."""
    return Decision(
        id="t1",
        action="test_action",
        target_variable="y",
        input_variables={"x1": 1.0, "x2": 2.0},
        current_value={"x1": 1.0, "x2": 2.0},
        model=lambda d: 1,  # A simple lambda as a mock model
        confidence=0.9,
        expected_outcome={"description": "improve performance", "timeframe": "1h"}
    )

@pytest.fixture
def sample_causal_graph():
    """Fixture to create a sample causal graph."""
    try:
        import networkx as nx
        G = nx.DiGraph()
        G.add_edge("x1", "y", weight=0.8, p_value=0.01)
        G.add_edge("x2", "y", weight=0.3, p_value=0.2)
        return G
    except ImportError:
        return {"nodes": ["x1", "x2", "y"], "edges": [("x1", "y"), ("x2", "y")]}

@pytest.fixture
def sample_data():
    """Fixture to create sample data for testing."""
    return pd.DataFrame([
        {"x1": 1.0, "x2": 2.0, "y": 1.0},
        {"x1": 1.2, "x2": 2.1, "y": 1.1},
    ])

def test_explainer_instantiation(decision_explainer):
    """Test that DecisionExplainer can be instantiated."""
    assert decision_explainer is not None

def test_explain_causal_decision(decision_explainer, sample_decision, sample_causal_graph, sample_data):
    """Test the main explanation generation method."""
    explanation = decision_explainer.explain_causal_decision(
        decision=sample_decision,
        causal_graph=sample_causal_graph,
        data=sample_data
    )

    assert "decision" in explanation
    assert "causal_chain" in explanation
    assert "feature_importance" in explanation
    assert "counterfactuals" in explanation
    assert "natural_language" in explanation
    assert "visualization" in explanation
    assert isinstance(explanation["natural_language"], str)