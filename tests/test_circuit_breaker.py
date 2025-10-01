# tests/test_circuit_breaker.py
import pytest
from src.zebra.circuit_breaker.circuit_breaker import CircuitBreaker, PerformanceBreaker, ErrorRateBreaker

def test_circuit_breaker_initial_state():
    """Test that the circuit breaker starts in a closed state."""
    cb = CircuitBreaker()
    assert cb.state == 'closed'

def test_performance_breaker_triggers(mocker):
    """Test that the performance breaker opens the circuit on a significant performance drop."""
    mocker.patch('threading.Timer')
    cb = CircuitBreaker()
    metrics = {"performance_delta": -0.2, "error_rate": 0.01}
    triggered = cb.monitor_and_break(metrics)
    assert triggered is True
    assert cb.state == 'open'
    assert "performance_degradation" in cb.current_reason

def test_error_rate_breaker_triggers(mocker):
    """Test that the error rate breaker opens the circuit when the error rate is too high."""
    mocker.patch('threading.Timer')
    cb = CircuitBreaker()
    metrics = {"performance_delta": 0.0, "error_rate": 0.1}
    triggered = cb.monitor_and_break(metrics)
    assert triggered is True
    assert cb.state == 'open'
    assert "error_rate" in cb.current_reason

def test_no_break_when_metrics_are_good():
    """Test that the circuit remains closed when metrics are within acceptable limits."""
    cb = CircuitBreaker()
    metrics = {"performance_delta": 0.0, "error_rate": 0.01, "cpu_usage": 0.5, "memory_usage": 0.5}
    triggered = cb.monitor_and_break(metrics)
    assert triggered is False
    assert cb.state == 'closed'

def test_alert_fn_is_called_on_break(mocker):
    """Test that the alert function is called when the circuit opens."""
    mocker.patch('threading.Timer')
    alert_called = False
    breaker_name_alert = None
    reason_alert = None

    def mock_alert_fn(breaker_name, reason):
        nonlocal alert_called, breaker_name_alert, reason_alert
        alert_called = True
        breaker_name_alert = breaker_name
        reason_alert = reason

    cb = CircuitBreaker(alert_fn=mock_alert_fn)
    metrics = {"error_rate": 0.1}
    cb.monitor_and_break(metrics)

    assert alert_called is True
    assert breaker_name_alert == "error_rate"
    assert "error_rate > 0.05" in reason_alert