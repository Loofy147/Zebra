# tests/test_safe_policy.py
import numpy as np
import pytest
from src.zebra.decision.safe_policy import SafePolicyLearning, SimulationResult, SimulationEnvInterface
from dataclasses import dataclass

class MockEnv(SimulationEnvInterface):
    def __init__(self, noise=1.0):
        self.noise = noise

    def simulate(self, policy):
        # policy is expected to be callable(state) -> action; for PoC ignore
        perf = np.random.normal(loc=1.0, scale=self.noise)  # positive perf usually
        success = perf > 0
        return SimulationResult(success=bool(success), performance=float(perf), metadata={})

def test_evaluate_policy_safety():
    env = MockEnv(noise=0.5)
    spl = SafePolicyLearning(safety_threshold=0.6)
    # policy can be any object for PoC
    res = spl.evaluate_policy_safety(policy=lambda s: 0, simulation_env=env, n_sim=200)
    assert "is_safe" in res
    assert "success_rate" in res

def test_gradual_rollout_simple():
    env = MockEnv(noise=0.3)
    spl = SafePolicyLearning(safety_threshold=0.5)
    # apply_policy_fn: simulate by returning N results proportionate to traffic
    def apply_policy(policy, traffic):
        n = max(1, int(200 * traffic))
        return [env.simulate(policy) for _ in range(n)]
    ok = spl.gradual_rollout(new_policy=None, apply_policy_fn=apply_policy, current_policy=None, stages=[0.01, 0.05])
    assert isinstance(ok, bool)