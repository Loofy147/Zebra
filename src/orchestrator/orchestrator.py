import logging
import json
import os
import requests
from flask import Flask, request, jsonify

# This is the main entry point for the new Ecosystem Orchestrator.
# It replaces the old observer and coordinates the new intelligent services.

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Service Discovery ---
# In a real-world scenario, service addresses would come from a discovery
# service or environment variables.
META_LEARNER_URL = os.environ.get("META_LEARNER_URL", "http://zebra-meta-learner:8000")
CODE_GENERATOR_URL = os.environ.get("CODE_GENERATOR_URL", "http://zebra-codegen:8000")


class EcosystemOrchestrator:
    """
    Coordinates the different services in the Zebra ecosystem to observe,
    analyze, and propose changes.
    """
    def __init__(self):
        logging.info("EcosystemOrchestrator initialized.")

    def _call_service(self, url: str, method: str = "get", json_data: dict = None) -> dict | None:
        """Helper function to call other services."""
        try:
            if method.lower() == "post":
                response = requests.post(url, json=json_data, timeout=10)
            else:
                response = requests.get(url, timeout=10)

            response.raise_for_status() # Raise an exception for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to call service at {url}: {e}")
            return None

    async def handle_telemetry(self, trace_data: dict):
        """
        The main pipeline for processing incoming telemetry data.
        Observe -> Analyze -> Generate
        """
        # 1. Observe: Detect an anomaly in the telemetry data.
        # (Using the same simple anomaly detection as the original observer)
        anomaly_detected = False
        for resource_span in trace_data.get("resourceSpans", []):
            for scope_span in resource_span.get("scopeSpans", []):
                for span in scope_span.get("spans", []):
                    if span.get("name") == "roll_dice_logic":
                        for attr in span.get("attributes", []):
                            if attr.get("key") == "dice.roll.value" and int(attr["value"]["intValue"]) > 4:
                                logging.warning("ORCHESTRATOR: Anomaly detected! High dice roll.")
                                anomaly_detected = True
                                break

        if not anomaly_detected:
            return

        # 2. Analyze & Learn: Call the Meta-Learning Engine
        logging.info("ORCHESTRATOR: Anomaly detected. Triggering Meta-Learning Engine...")
        analysis_result = self._call_service(f"{META_LEARNER_URL}/learn", method="post")

        if not analysis_result or analysis_result.get("rules_generated", 0) == 0:
            logging.error("ORCHESTRATOR: Meta-learning did not yield actionable insights.")
            return

        logging.critical(f"ORCHESTRATOR: Meta-learning complete. Insights: {analysis_result}")

        # 3. Generate & Propose: Call the Autonomous Code Generator
        # We'll create a mock analysis report to send to the codegen service.
        codegen_payload = {
            "bottleneck": "high_roll_rate",
            "description": "The Meta-Learning Engine identified a recurring pattern of high dice rolls. Generate an optimization to mitigate this."
        }

        logging.info("ORCHESTRATOR: Triggering Autonomous Code Generator...")
        proposal = self._call_service(f"{CODE_GENERATOR_URL}/generate", method="post", json_data=codegen_payload)

        if not proposal:
            logging.error("ORCHESTRATOR: Code generation failed or produced no valid variants.")
            return

        # 4. Act (Log): For now, we just log the final proposed change.
        # In a future phase, this would trigger a deployment pipeline.
        logging.critical(f"ORCHESTRATOR: Autonomous Code Generation successful! Proposed optimization: {proposal}")


# --- Flask App ---
ORCHESTRATOR = EcosystemOrchestrator()

@app.route("/v1/traces", methods=["POST"])
async def receive_traces():
    """
    Receives telemetry data from the OTel collector and initiates the pipeline.
    """
    try:
        trace_data = request.get_json(force=True)
    except json.JSONDecodeError:
        logging.info("ORCHESTRATOR: Received empty or malformed trace payload.")
        return "Empty or malformed payload", 200

    await ORCHESTRATOR.handle_telemetry(trace_data)

    return "Traces received and processed", 200

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

# The Dockerfile's CMD uses gunicorn to run this 'app' object.
# The following is for local development only.
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)