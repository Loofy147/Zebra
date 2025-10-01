# src/zebra/experiments/sequential_testing.py
"""
Sequential testing utilities (SPRT + Bayesian adaptive stopping).
"""
from __future__ import annotations
from typing import Dict, Any, Iterable, Optional
import numpy as np
import logging

logger = logging.getLogger("zebra.experiments.sequential")

try:
    from scipy import stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

class SequentialTesting:
    def __init__(self, alpha: float = 0.05, power: float = 0.8):
        self.alpha = alpha
        self.power = power

    def sequential_probability_ratio_test(self, data_stream: Iterable[Dict[str, Any]], p0: float = 0.5, p1: float = 0.6) -> Dict[str, Any]:
        """
        SPRT for Bernoulli observations (PoC).
        data_stream yields dicts with {'success': 0/1}
        p0: null hypothesis success prob; p1: alt hypothesis success prob.
        """
        A = (1 - self.power) / self.alpha  # accept H0 if LR <= A
        B = self.power / (1 - self.alpha)  # accept H1 if LR >= B
        lr = 1.0
        n = 0
        for obs in data_stream:
            n += 1
            k = int(obs.get("success", 0))
            # likelihood ratio for Bernoulli: (p1^k (1-p1)^(1-k)) / (p0^k (1-p0)^(1-k))
            # Add a small epsilon to avoid division by zero or log(0)
            p1_safe = np.clip(p1, 1e-9, 1 - 1e-9)
            p0_safe = np.clip(p0, 1e-9, 1 - 1e-9)

            log_lr_numerator = k * np.log(p1_safe) + (1 - k) * np.log(1 - p1_safe)
            log_lr_denominator = k * np.log(p0_safe) + (1 - k) * np.log(1 - p0_safe)

            lr += log_lr_numerator - log_lr_denominator # Using log-likelihood ratio for numerical stability

            if np.exp(lr) >= B:
                return {"decision": "reject_null", "stopped_early": True, "samples_used": n}
            if np.exp(lr) <= A:
                return {"decision": "accept_null", "stopped_early": True, "samples_used": n}
        return {"decision": "continue", "stopped_early": False, "samples_used": n}

    def bayesian_adaptive_stopping(self, control: Dict[str, int], treatment: Dict[str, int], trials: int = 10000, threshold: float = 0.95) -> Dict[str, Any]:
        """
        control / treatment: dicts with {'successes': int, 'trials': int}
        returns posterior probability treatment > control estimated via MC.
        """
        if not _HAS_SCIPY:
            raise NotImplementedError("scipy is required for Bayesian adaptive stopping.")

        a_c = 1 + control["successes"]
        b_c = 1 + control["trials"] - control["successes"]
        a_t = 1 + treatment["successes"]
        b_t = 1 + treatment["trials"] - treatment["successes"]

        posterior_control = stats.beta(a_c, b_c)
        posterior_treatment = stats.beta(a_t, b_t)
        ctrl_samps = posterior_control.rvs(trials)
        treat_samps = posterior_treatment.rvs(trials)
        prob_treatment_better = float((treat_samps > ctrl_samps).mean())
        expected_lift = float((treat_samps - ctrl_samps).mean())

        decision = "continue"
        if prob_treatment_better >= threshold:
            decision = "stop_and_implement"
        elif prob_treatment_better <= (1 - threshold):
            decision = "stop_and_discard"

        return {"decision": decision, "probability": prob_treatment_better, "expected_lift": expected_lift}