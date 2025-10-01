# tests/test_experiments.py
import numpy as np
import pandas as pd
import pytest
from src.zebra.experiments.causal_ab_testing import CausalABTesting
from src.zebra.experiments.sequential_testing import SequentialTesting

@pytest.fixture
def gen_ab_df():
    """Fixture to generate a sample A/B test dataframe."""
    def _gen(n=200):
        # simple toy data: variant A has slightly higher mean
        a = np.random.normal(loc=0.1, scale=1.0, size=n)
        b = np.random.normal(loc=0.0, scale=1.0, size=n)
        df = pd.DataFrame({"variant": ["A"]*n + ["B"]*n, "outcome": np.concatenate([a,b])})
        return df
    return _gen

def test_causal_ab_basic(gen_ab_df):
    """Test basic functionality of CausalABTesting."""
    tester = CausalABTesting()
    exp = tester.design_experiment("test", ["A","B"])
    df = gen_ab_df(500) # Use a larger sample for more stable tests
    tester.ingest_experiment_data(exp["id"], df)

    try:
        res = tester.analyze_experiment_causally(exp["id"])
        assert hasattr(res, "traditional")
        assert hasattr(res, "causal")
    except ImportError as e:
        pytest.skip(f"Skipping causal test due to missing dependency: {e}")
    except Exception as e:
        pytest.fail(f"Causal analysis failed with an unexpected error: {e}")


def test_sprt_and_bayesian():
    """Test basic functionality of SequentialTesting."""
    seq = SequentialTesting(alpha=0.05, power=0.8)
    # simple stream: 1s then 0s...
    stream = ({"success": int(x)} for x in [1,1,0,1,1,1,0,1,1,1])
    try:
        sprt_res = seq.sequential_probability_ratio_test(stream, p0=0.4, p1=0.8)
        assert "decision" in sprt_res
    except ImportError as e:
        pytest.skip(f"Skipping SPRT test due to missing dependency: {e}")

    # bayesian stopping simple
    control = {"successes": 30, "trials": 100}
    treatment = {"successes": 40, "trials": 100}
    try:
        bayes = seq.bayesian_adaptive_stopping(control, treatment, trials=2000, threshold=0.9)
        assert "probability" in bayes
    except ImportError as e:
        pytest.skip(f"Skipping Bayesian test due to missing dependency: {e}")