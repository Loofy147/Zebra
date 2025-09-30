import logging
from flask import Flask, request, jsonify
from .autonomous_code_generator import CODEGEN
# The PR Bot might be used later, so we keep the import path ready
# from .pull_request_bot import PR_BOT

# This file is the main entry point for the Code Generator service.
# It exposes the AutonomousCodeGenerator's capabilities via a simple REST API.

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route("/generate", methods=["POST"])
async def generate():
    """
    Triggers the autonomous code generation process based on an analysis report.
    """
    analysis_report = request.get_json()
    if not analysis_report:
        return jsonify({"error": "Invalid analysis report provided"}), 400

    logging.info(f"GENERATOR_API: Received request to /generate with report: {analysis_report}")

    generated_code = await CODEGEN.generate_optimization(analysis_report)

    if generated_code:
        # In a future step, this could trigger the PR_BOT
        # PR_BOT.create_pr(...)
        return jsonify(generated_code), 200
    else:
        return jsonify({"error": "Failed to generate a valid optimization"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """
    Simple health check endpoint.
    """
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    # Note: This is for local development. In production, Gunicorn is used.
    app.run(host='0.0.0.0', port=8000)