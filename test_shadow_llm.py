import pytest
from src.zebra.shadow.llm_adapter import MockLLMAdapter
from src.zebra.shadow.enhanced_shadow_llm import EnhancedShadowLLM

@pytest.fixture
def mock_llm_adapter():
    """Fixture to create a mock LLM adapter."""
    return MockLLMAdapter()

@pytest.fixture
def enhanced_shadow_llm(mock_llm_adapter):
    """Fixture to create an instance of EnhancedShadowLLM."""
    return EnhancedShadowLLM(mock_llm_adapter)

def test_enhanced_llm_instantiation(enhanced_shadow_llm):
    """Test that EnhancedShadowLLM can be instantiated."""
    assert enhanced_shadow_llm is not None

def test_analyze_with_chain_of_thought(enhanced_shadow_llm):
    """Test the basic analysis method."""
    telemetry_data = {"cpu_usage": 0.8, "error_rate": 0.05}

    # A mock causal graph object with a to_natural_language method
    class MockCausalGraph:
        def to_natural_language(self):
            return "cpu_usage -> error_rate"

    causal_graph = MockCausalGraph()

    result = enhanced_shadow_llm.analyze_with_chain_of_thought(telemetry_data, causal_graph)

    assert "insights" in result
    assert result["insights"] == ["mock-insight"]
    assert result["success"] is True

def test_generate_counterfactual_scenarios(enhanced_shadow_llm):
    """Test the counterfactual generation method."""
    current_state = {"cpu_usage": 0.8}
    result = enhanced_shadow_llm.generate_counterfactual_scenarios(current_state)
    assert "insights" in result

def test_explain_causal_relationship(enhanced_shadow_llm):
    """Test the causal explanation method."""
    response = enhanced_shadow_llm.explain_causal_relationship("cpu_usage", "latency", 0.7)
    assert isinstance(response, str)
    assert "insights" in response or "raw" in response # Mock returns a dict, so we check for keys