import numpy as np
import pytest
from src.zebra.stability.multitest_stability import MultiTestStabilityAnalyzer
from src.zebra.stability.multivariate_stability import MultivariateStabilityAnalyzer

@pytest.fixture
def stationary_ar1_series():
    """Generate a stationary AR(1) series."""
    n = 200
    phi = 0.2
    e = np.random.normal(scale=1.0, size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t-1] + e[t]
    return x

@pytest.fixture
def random_walk_series():
    """Generate a non-stationary random walk series."""
    n = 200
    e = np.random.normal(size=n)
    return np.cumsum(e)

@pytest.fixture
def multivariate_stable_series():
    """Generate a stable multivariate series."""
    n = 300
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.2 * x[t-1] + np.random.normal()
        y[t] = 0.1 * y[t-1] + 0.3 * x[t] + np.random.normal()
    return np.vstack([x, y]).T

def test_univariate_stationary_series(stationary_ar1_series):
    """Test that the analyzer identifies a stationary series."""
    analyzer = MultiTestStabilityAnalyzer()
    result = analyzer.comprehensive_stability_test(stationary_ar1_series)
    assert result["is_stationary"] is True
    assert result["confidence"] > 0.5

def test_univariate_non_stationary_series(random_walk_series):
    """Test that the analyzer identifies a non-stationary series."""
    analyzer = MultiTestStabilityAnalyzer()
    result = analyzer.comprehensive_stability_test(random_walk_series)
    assert result["is_stationary"] is False
    assert result["confidence"] < 0.5

def test_multivariate_var_stability(multivariate_stable_series):
    """Test the VAR stability check on a stable series."""
    analyzer = MultivariateStabilityAnalyzer()
    result = analyzer.vector_autoregression_stability(multivariate_stable_series)
    assert result["is_stable"] is True

def test_multivariate_cointegration(multivariate_stable_series):
    """Test the Johansen cointegration test."""
    analyzer = MultivariateStabilityAnalyzer()
    # This is more of a smoke test as interpreting the rank is complex.
    # We just want to ensure it runs without errors.
    try:
        result = analyzer.johansen_cointegration_test(multivariate_stable_series)
        assert "cointegration_rank" in result
    except Exception as e:
        pytest.fail(f"Johansen cointegration test failed: {e}")