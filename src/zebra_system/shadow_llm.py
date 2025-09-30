import logging

logging.basicConfig(level=logging.INFO)

# --- SIMULATED LLM FOR CODE GENERATION ---
# This simulates a fine-tuned LLM that can suggest code optimizations.
# In a real system, this would involve a complex RAG (Retrieval-Augmented Generation)
# pipeline and a powerful LLM. Here, we use a simple dictionary lookup.

OPTIMIZATION_PROPOSALS = {
    "intervention_1_increase_roll_speed": {
        "description": "The 'roll_dice_logic' function in 'sample_service.app' was identified as the root cause of high_roll_rate. The proposed optimization uses a more efficient random number generation method.",
        "original_code": """
def roll_dice():
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("roll_dice_logic") as span:
        result = random.randint(1, 6)
        span.set_attribute("dice.roll.value", result)
        logging.info(f"Dice roll result: {result}")
        return jsonify({"roll": result})
""",
        "optimized_code": """
import numpy as np

def roll_dice():
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("roll_dice_logic_optimized") as span:
        # Using numpy's random generator is often more performant in high-throughput scenarios.
        result = np.random.randint(1, 7) # numpy is exclusive on the high end
        span.set_attribute("dice.roll.value", int(result))
        logging.info(f"Dice roll result: {result}")
        return jsonify({"roll": int(result)})
"""
    }
}

class ShadowCodeGenerator:
    """
    Simulates an LLM that generates optimized code based on a bottleneck.
    """
    def generate_optimization(self, bottleneck_identifier: str) -> dict | None:
        """
        Looks up a pre-defined optimization proposal.

        Args:
            bottleneck_identifier: The key identifying the problem area.

        Returns:
            A dictionary containing the optimization proposal, or None if not found.
        """
        logging.info(f"SHADOW LLM: Received request to optimize bottleneck '{bottleneck_identifier}'.")

        proposal = OPTIMIZATION_PROPOSALS.get(bottleneck_identifier)

        if proposal:
            logging.critical("SHADOW LLM: Found a potential optimization. Generating proposal...")
            return proposal
        else:
            logging.warning(f"SHADOW LLM: No optimization found for bottleneck '{bottleneck_identifier}'.")
            return None

# Global instance for easy import
LLM = ShadowCodeGenerator()