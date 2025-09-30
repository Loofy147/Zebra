import logging
import json
from flask import Flask, request, jsonify
from .causal_engine import CIE # IMPORT the CIE

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

    # Handle empty request bodies gracefully, which the collector can send.
    if not raw_data.strip():
        logging.info("OBSERVER: Received empty trace payload.")
        return "Empty payload received", 200

    trace_data = json.loads(raw_data)
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

    return "Traces received and analyzed", 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9090)