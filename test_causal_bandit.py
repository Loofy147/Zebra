# tests/test_causal_bandit.py
import pytest
from src.zebra.decision.causal_bandit import CausalBandit

def test_bandit_basic_initialization():
    arms = ["A", "B", "C"]
    cb = CausalBandit(arms)
    assert set(cb.arms) == set(arms)
    assert all(count == 0 for count in cb.counts.values())

def test_thompson_sampling():
    arms = ["A", "B", "C"]
    cb = CausalBandit(arms)
    selected_arm = cb.thompson_sampling()
    assert selected_arm in arms

def test_update_with_reward():
    arms = ["A", "B", "C"]
    cb = CausalBandit(arms)

    # Update arm "A" with a reward
    cb.update_with_counterfactual("A", reward=1.0)

    assert cb.counts["A"] == 1
    assert cb.counts["B"] == 0
    assert cb.values["A"] == 1.0
    assert cb.sum_rewards["A"] == 1.0
    assert cb.squared_sums["A"] == 1.0

def test_contextual_bandit_with_causality():
    arms = ["A", "B", "C"]
    cb = CausalBandit(arms)

    def mock_causal_estimator(arm, context):
        if arm == "A":
            return 0.8
        elif arm == "B":
            return 0.5
        return 0.2

    context = {"feature1": 10}
    selected_arm, effect = cb.contextual_bandit_with_causality(context, mock_causal_estimator)

    # The arm with the highest causal estimate should be selected initially
    assert selected_arm == "A"
    assert effect == 0.8