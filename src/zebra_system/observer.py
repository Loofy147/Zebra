import logging
import json
from flask import Flask, request, jsonify
from .causal_engine import CIE # IMPORT the CIE
from .shadow_llm import LLM      # IMPORT the Shadow LLM

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Global flag to demonstrate changing the intervention state externally
@app.route("/control/intervention/<name>", methods=["POST"])
def set_intervention(name: str):
    """Allows the Orchestrator (us, manually) to set the system's state."""
    CIE.set_intervention(name)
    return jsonify({"status": "OK", "new_intervention": name}), 200

# The core telemetry receiving endpoint
@app.route("/v1/traces", methods=["POST"])
def receive_traces():
    # The data is sent as OTLP JSON. Flask's `request.json` can be too strict.
    # We'll parse the raw data manually for more robustness.
    raw_data = request.get_data(as_text=True)

    # Use a try-except block for the most robust parsing.
    # This handles empty payloads and any other malformed JSON.
    try:
        trace_data = json.loads(raw_data)
    except json.JSONDecodeError:
        logging.info("OBSERVER: Received empty or malformed trace payload.")
        return "Empty or malformed payload received", 200
    anomaly_detected = False

    # 1. Detect Anomaly
    for resource_span in trace_data.get("resourceSpans", []):
        for scope_span in resource_span.get("scopeSpans", []):
            for span in scope_span.get("spans", []):

                # We only analyze the results of our 'roll_dice_logic'
                if span.get("name") == "roll_dice_logic":
                    dice_roll_value = None
                    for attribute in span.get("attributes", []):
                        if attribute.get("key") == "dice.roll.value":
                            dice_roll_value = attribute["value"]["intValue"]
                            break

                    if dice_roll_value and int(dice_roll_value) > 4:
                        logging.warning(f"OBSERVER: Anomaly detected! High roll ({dice_roll_value})")
                        anomaly_detected = True

    # 2. Analyze Cause (Only if Anomaly was detected)
    if anomaly_detected:
        analysis_result = CIE.analyze_anomaly(anomaly_type="high_roll_rate")

        # Crucial next step: Log the strategic insight, not just the raw data.
        logging.critical(
            f"OBSERVER INSIGHT: Root Cause Analysis Completed. "
            f"Root Cause: {analysis_result['root_cause']} "
            f"| Confidence: {analysis_result['confidence']:.2f}"
        )

        # 3. Suggest Optimization (Only if a specific cause was found)
        root_cause = analysis_result.get("root_cause")
        if root_cause and root_cause != "uncontrolled_variance":
            proposal = LLM.generate_optimization(bottleneck_identifier=root_cause)
            if proposal:
                # In a real system, this proposal would go to a PR bot or dashboard.
                # For now, we log it for human review.
                logging.warning("--- Optimization Proposal for Human Review ---")
                logging.warning(f"Description: {proposal['description']}")
                logging.warning("Original Code:\n" + proposal['original_code'])
                logging.warning("Optimized Code:\n" + proposal['optimized_code'])
                logging.warning("--- End of Proposal ---")

    return "Traces received and analyzed", 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9090)