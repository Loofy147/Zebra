import pytest
from src.zebra_telemetry.opentelemetry_setup import ZebraObservability

@pytest.fixture(scope="module")
def observability_setup():
    """Fixture to initialize ZebraObservability for testing."""
    # Use a mock OTLP endpoint for testing purposes
    # In a real integration test, this would point to a test collector.
    obs = ZebraObservability(service_name="test_service", otlp_endpoint="http://localhost:4317")
    return obs

def test_instantiation(observability_setup):
    """Test that ZebraObservability can be instantiated without errors."""
    assert observability_setup is not None
    assert observability_setup.meter is not None
    assert observability_setup.request_counter is not None

def test_record_request(observability_setup):
    """Test that recording a request does not raise an exception."""
    try:
        observability_setup.record_request(
            path="/test",
            method="GET",
            status_code=200,
            latency_ms=50.5
        )
    except Exception as e:
        pytest.fail(f"record_request raised an exception: {e}")

def test_span_creation(observability_setup):
    """Test that a span can be created without errors."""
    from opentelemetry import trace
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("test-span") as span:
        span.set_attribute("test.attribute", "test.value")
        assert span.is_recording()

def test_propagate_causal_context(observability_setup):
    """Test that the causal context propagation helper works."""
    from opentelemetry import baggage

    def dummy_service_call():
        # Check that the baggage is set within the context of the call
        graph_id = baggage.get_baggage("zebra.causal.graph_id")
        assert graph_id == "test_graph_123"

    try:
        observability_setup.propagate_causal_context(
            graph_id="test_graph_123",
            next_service_call=dummy_service_call
        )
    except Exception as e:
        pytest.fail(f"propagate_causal_context raised an exception: {e}")