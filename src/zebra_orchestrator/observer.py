import logging
import json
from flask import Flask, request, jsonify
from .causal_engine import CIE
from .shadow_llm import LLM
from .pr_bot import PR_BOT

try:
    from .deep_learning_causal_engine import deep_cie_engine
    from .anomaly_detector import anomaly_detector
    from .reinforcement_learning import rl_agent
    from .code_understanding import code_analyzer
    from .supabase_storage import supabase_storage
    from .continuous_learning import continuous_learning
    DEEP_LEARNING_ENABLED = True
except ImportError as e:
    logging.warning(f"Deep learning modules not available: {e}")
    DEEP_LEARNING_ENABLED = False

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

PIPELINE_MODE = 'deep_learning' if DEEP_LEARNING_ENABLED else 'classic'

# Global flag to demonstrate changing the intervention state externally
@app.route("/control/intervention/<name>", methods=["POST"])
def set_intervention(name: str):
    """Allows the Orchestrator (us, manually) to set the system's state."""
    CIE.set_intervention(name)
    return jsonify({"status": "OK", "new_intervention": name}), 200

@app.route("/v1/traces", methods=["POST"])
def receive_traces():
    try:
        raw_data = request.get_data(as_text=True)

        try:
            trace_data = json.loads(raw_data)
        except json.JSONDecodeError:
            logging.info("OBSERVER: Received empty or malformed trace payload.")
            return "Empty or malformed payload received", 200

        if not isinstance(trace_data, dict):
            logging.warning("OBSERVER: Invalid trace data format")
            return "Invalid data format", 400

        anomaly_detected = False
        telemetry_metrics = _extract_telemetry_metrics(trace_data)

        for resource_span in trace_data.get("resourceSpans", []):
            for scope_span in resource_span.get("scopeSpans", []):
                for span in scope_span.get("spans", []):
                    if span.get("name") == "roll_dice_logic":
                        dice_roll_value = None
                        for attribute in span.get("attributes", []):
                            if attribute.get("key") == "dice.roll.value":
                                dice_roll_value = attribute["value"]["intValue"]
                                break

                        if dice_roll_value and int(dice_roll_value) > 4:
                            logging.warning(f"OBSERVER: Anomaly detected! High roll ({dice_roll_value})")
                            anomaly_detected = True

        if DEEP_LEARNING_ENABLED and telemetry_metrics:
            deep_anomaly_result = anomaly_detector.detect_anomaly(telemetry_metrics)
            if deep_anomaly_result.get('is_anomaly', False):
                anomaly_detected = True
                logging.warning(f"DEEP LEARNING: Anomaly detected with score {deep_anomaly_result['anomaly_score']:.3f}")

                supabase_storage.store_anomaly({
                    **deep_anomaly_result,
                    'anomaly_type': 'neural_network_detection',
                    'telemetry_snapshot': telemetry_metrics
                })

        if anomaly_detected:
            if PIPELINE_MODE == 'deep_learning' and telemetry_metrics:
                analysis_result = deep_cie_engine.analyze_anomaly_deep(
                    telemetry_metrics,
                    anomaly_type="performance_degradation"
                )
            else:
                analysis_result = CIE.analyze_anomaly(anomaly_type="high_roll_rate")

            logging.critical(
                f"OBSERVER INSIGHT: Root Cause Analysis Completed. "
                f"Root Cause: {analysis_result['root_cause']} "
                f"| Confidence: {analysis_result['confidence']:.2f}"
            )

            root_cause = analysis_result.get("root_cause")
            if root_cause and root_cause != "uncontrolled_variance" and root_cause != "random_variance":
                proposal = LLM.generate_optimization(bottleneck_identifier=root_cause)

                if proposal and DEEP_LEARNING_ENABLED:
                    code_analysis = code_analyzer.analyze_bottleneck(
                        proposal.get('original_code', ''),
                        root_cause
                    )
                    proposal['code_analysis'] = code_analysis

                if proposal:
                    PR_BOT.create_pr(
                        bottleneck_identifier=root_cause,
                        proposal=proposal,
                        analysis=analysis_result
                    )

                    if DEEP_LEARNING_ENABLED:
                        supabase_storage.store_intervention({
                            'bottleneck_identifier': root_cause,
                            'root_cause': analysis_result.get('root_cause'),
                            'confidence': analysis_result.get('confidence'),
                            'causal_effect_size': analysis_result.get('causal_effect_size'),
                            'proposal_description': proposal.get('description'),
                            'original_code': proposal.get('original_code'),
                            'optimized_code': proposal.get('optimized_code'),
                            'analysis_method': analysis_result.get('analysis_method', 'classic')
                        })

                        continuous_learning.experience_collector.add_performance_data(telemetry_metrics)

        return "Traces received and analyzed", 200

    except Exception as e:
        logging.error(f"Error processing traces: {e}", exc_info=True)
        return "Internal server error", 500

def _extract_telemetry_metrics(trace_data: dict) -> dict:
    """Extract metrics from trace data for deep learning models."""
    metrics = {
        'latency_p50': 50.0,
        'latency_p95': 100.0,
        'latency_p99': 150.0,
        'request_rate': 100.0,
        'error_rate': 0.01,
        'cpu_usage': 0.3,
        'memory_usage': 0.5
    }

    try:
        for resource_span in trace_data.get("resourceSpans", []):
            for scope_span in resource_span.get("scopeSpans", []):
                for span in scope_span.get("spans", []):
                    start_time = span.get('startTimeUnixNano', 0)
                    end_time = span.get('endTimeUnixNano', 0)
                    if start_time and end_time:
                        latency_ms = (end_time - start_time) / 1_000_000
                        metrics['latency_p50'] = latency_ms
                        metrics['latency_p95'] = latency_ms * 1.5
                        metrics['latency_p99'] = latency_ms * 2.0
    except Exception as e:
        logging.warning(f"Error extracting telemetry metrics: {e}")

    return metrics

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'pipeline_mode': PIPELINE_MODE,
        'deep_learning_enabled': DEEP_LEARNING_ENABLED
    }), 200

@app.route("/stats", methods=["GET"])
def get_stats():
    """Get system statistics."""
    try:
        if DEEP_LEARNING_ENABLED:
            stats = continuous_learning.get_learning_statistics()
            return jsonify(stats), 200
        else:
            return jsonify({'message': 'Deep learning not enabled'}), 200
    except Exception as e:
        logging.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    logging.info(f"Starting Zebra Observer in {PIPELINE_MODE} mode")
    app.run(host='0.0.0.0', port=9090)