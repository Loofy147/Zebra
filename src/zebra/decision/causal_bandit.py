# src/zebra/decision/causal_bandit.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Callable, Optional
import numpy as np
import logging

logger = logging.getLogger("zebra.decision.causal_bandit")

class CausalBandit:
    """
    Multi-armed bandit that incorporates causal estimates:
    - arms: list of arm ids / objects
    - counts: number of times each arm pulled
    - values: running mean reward per arm
    - causal_effects: optional prior causal estimates per arm (callable or numeric)
    """

    def __init__(self, arms: List[str], prior_means: Optional[Dict[str, float]] = None):
        self.arms = list(arms)
        self.counts = {a: 0 for a in self.arms}
        self.values = {a: 0.0 for a in self.arms}
        self.sum_rewards = {a: 0.0 for a in self.arms}
        self.causal_effects = prior_means or {a: 0.0 for a in self.arms}
        # simple variance estimate per arm
        self.squared_sums = {a: 0.0 for a in self.arms}

    def thompson_sampling(self) -> str:
        """
        Simple Thompson sampling using gaussian posterior approx:
        sample ~ Normal(mean, sigma / sqrt(n+1))
        - works for approx continuous rewards; for Bernoulli a Beta posterior is better.
        """
        samples = {}
        for a in self.arms:
            n = self.counts[a]
            mean = self.values[a] if n > 0 else self.causal_effects.get(a, 0.0)
            # estimate std
            if n > 1:
                var = max(1e-6, (self.squared_sums[a] / n - mean * mean))
                std = np.sqrt(var)
            else:
                std = 1.0  # prior uncertainty
            # sample
            samples[a] = np.random.normal(loc=mean, scale=std / np.sqrt(n + 1))
        selected = max(samples, key=samples.get)
        return selected

    def contextual_bandit_with_causality(self, context: Dict[str, Any], causal_estimator: Callable[[str, Dict[str, Any]], float]) -> Tuple[str, float]:
        """
        Returns selected_arm and its estimated causal effect.
        - causal_estimator(arm, context) -> estimated_effect (float)
        - We compute exploration bonus UCB-like and pick max.
        """
        total_counts = sum(self.counts.values()) + 1
        causal_estimates = {}
        for arm in self.arms:
            try:
                causal_estimates[arm] = float(causal_estimator(arm, context))
            except Exception:
                causal_estimates[arm] = float(self.causal_effects.get(arm, 0.0))

        exploration_bonus = {
            arm: np.sqrt(2 * np.log(total_counts) / (self.counts[arm] + 1))
            for arm in self.arms
        }
        ucb_scores = {arm: causal_estimates[arm] + exploration_bonus[arm] for arm in self.arms}
        selected_arm = max(ucb_scores, key=ucb_scores.get)
        return selected_arm, causal_estimates[selected_arm]

    def update_with_counterfactual(self, chosen_arm: str, reward: float, context: Optional[Dict[str, Any]] = None,
                                  counterfactual_fn: Optional[Callable[[str, str, Dict[str, Any], float], float]] = None):
        """
        Update chosen arm and optionally update other arms using counterfactual estimates.
        - counterfactual_fn(other_arm, chosen_arm, context, observed_reward) -> estimated_reward
        """
        # update chosen arm statistics
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        self.sum_rewards[chosen_arm] += reward
        self.values[chosen_arm] = self.sum_rewards[chosen_arm] / n
        self.squared_sums[chosen_arm] += reward * reward

        if counterfactual_fn is None:
            return

        # update others with a small weight from estimated counterfactuals
        for other in self.arms:
            if other == chosen_arm:
                continue
            try:
                cf = float(counterfactual_fn(other, chosen_arm, context or {}, reward))
                # exponential smoothing update with low weight
                self.values[other] = 0.9 * self.values[other] + 0.1 * cf
            except Exception:
                logger.exception("counterfactual_fn failed for %s", other)