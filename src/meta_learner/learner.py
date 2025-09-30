import logging
from flask import Flask, jsonify
from .meta_learning_engine import MLE

# This file is the main entry point for the Meta-Learner service.
# It exposes the MetaLearningEngine's capabilities via a simple REST API.

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route("/learn", methods=["POST"])
async def learn():
    """
    Triggers the meta-learning process to learn from historical data.
    """
    logging.info("LEARNER_API: Received request to /learn")
    insights = await MLE.learn_from_history()
    return jsonify(insights), 200

@app.route("/predict", methods=["GET"])
async def predict():
    """
    Asks the engine to predict potential future issues.
    """
    logging.info("LEARNER_API: Received request to /predict")
    predictions = await MLE.predict_future_issues()
    return jsonify(predictions), 200

@app.route("/health", methods=["GET"])
def health_check():
    """
    Simple health check endpoint.
    """
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    # Note: This is for local development. In production, Gunicorn is used.
    app.run(host='0.0.0.0', port=8000)