import logging

logging.basicConfig(level=logging.INFO)

# --- SIMULATED KNOWLEDGE GRAPH / CAUSAL MODEL ---
# In a real system, this graph would be dynamically generated from telemetry
# and machine learning models. Here, it is hardcoded for demonstration.
CAUSAL_MODEL = {
    "intervention_1_increase_roll_speed": {
        "is_causal_for": "high_roll_rate",
        "p_value": 0.05,  # Statistically significant
        "effect_size": 0.45, # Strong effect
    },
    "traffic_spike": {
        "is_causal_for": "latency_p95_spike",
        "p_value": 0.01,
        "effect_size": 0.60,
    },
    # The default state: high rolls are mostly random variance
    "default_variance": {
        "is_causal_for": "high_roll_rate",
        "p_value": 0.90, # Not significant
        "effect_size": 0.05,
    }
}

class CausalInferenceEngine:
    """
    Simulates root cause analysis based on system state and observed anomaly.

    In reality, this would run a Causal ML model (like a Causal Forest or G-formula)
    over a controlled, randomized data sample.
    """

    def __init__(self):
        # Current state tracker. In a full system, this would be updated
        # by the Orchestrator on every deploy.
        self.current_intervention = "default_variance"

    def set_intervention(self, intervention_name: str):
        """Used to simulate a deployment or system change."""
        self.current_intervention = intervention_name
        logging.warning(f"CIE: System state changed! Active intervention: {self.current_intervention}")

    def analyze_anomaly(self, anomaly_type: str) -> dict:
        """
        Determines the root cause of an observed anomaly.

        Args:
            anomaly_type: The anomaly observed (e.g., 'high_roll_rate').

        Returns:
            A dictionary containing the root cause and confidence.
        """

        logging.info(f"CIE: Running root cause analysis for '{anomaly_type}'...")

        # 1. Check if the current, active intervention explains the anomaly
        active_intervention_key = self.current_intervention

        if (active_intervention_key in CAUSAL_MODEL and
            CAUSAL_MODEL[active_intervention_key].get("is_causal_for") == anomaly_type):

            # The active intervention is designed to be the cause.
            model_output = CAUSAL_MODEL[active_intervention_key]

            if model_output["p_value"] < 0.1: # Threshold for significance
                logging.critical(
                    f"CIE RESULT: Cause found! The observed anomaly ('{anomaly_type}') is CAUSALLY LINKED "
                    f"to the current intervention ('{active_intervention_key}'). "
                    f"P-value: {model_output['p_value']:.2f}. Effect Size: {model_output['effect_size']:.2f}"
                )
                return {
                    "root_cause": active_intervention_key,
                    "confidence": 1.0 - model_output['p_value']  # High confidence
                }

        # 2. If no direct link is found, return the baseline/variance result
        model_output = CAUSAL_MODEL["default_variance"]
        logging.info(f"CIE RESULT: No causal link to active intervention. Attributing to baseline variance.")

        return {
            "root_cause": "uncontrolled_variance",
            "confidence": 0.5 # Neutral confidence
        }

# Global instance for easy import
CIE = CausalInferenceEngine()