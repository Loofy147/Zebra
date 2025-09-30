import random
import logging
from flask import Flask, jsonify

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
# IMPORTANT: Import the OTLP Span Exporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.flask import FlaskInstrumentor

# --- Robust In-Code OpenTelemetry Configuration ---

logging.basicConfig(level=logging.INFO)

# 1. Define the service resource
resource = Resource(attributes={
    "service.name": "zebra-sample-service",
    "service.version": "1.0.0-final"
})

# 2. Configure Tracing
trace_provider = TracerProvider(resource=resource)
# Use OTLPSpanExporter to send data to the collector.
# The endpoint is configured by the OTEL_EXPORTER_OTLP_ENDPOINT env var
# which we set in docker-compose.yml.
trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(trace_provider)

# 3. Create and Instrument the Flask App
app = Flask(__name__)
FlaskInstrumentor().instrument_app(app)

# 4. Define Application Routes
@app.route("/")
def hello():
    return "Hello from the Zebra service!"

@app.route("/rolldice")
def roll_dice():
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("roll_dice_logic") as span:
        result = random.randint(1, 6)
        span.set_attribute("dice.roll.value", result)
        logging.info(f"Dice roll result: {result}")
        return jsonify({"roll": result})