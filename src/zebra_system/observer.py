import logging
from flask import Flask, request

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# This is the endpoint that the OTel Collector will send trace data to.
@app.route("/v1/traces", methods=["POST"])
def receive_traces():
    # The data is sent as OTLP JSON. We can inspect it.
    trace_data = request.json

    # This is the "Hello World" of our analysis engine.
    # We will iterate through the received spans and look for our dice roll.
    for resource_span in trace_data.get("resourceSpans", []):
        for scope_span in resource_span.get("scopeSpans", []):
            for span in scope_span.get("spans", []):
                # Check if this is the span we're interested in
                if span.get("name") == "roll_dice_logic":
                    dice_roll_value = None
                    # Find the attribute that holds our dice roll value
                    for attribute in span.get("attributes", []):
                        if attribute.get("key") == "dice.roll.value":
                            dice_roll_value = attribute["value"]["intValue"]
                            break

                    if dice_roll_value:
                        logging.info(f"OBSERVER: Received dice roll with value: {dice_roll_value}")
                        # Perform a rudimentary analysis
                        if int(dice_roll_value) > 4:
                            logging.warning(f"OBSERVER: High roll detected! ({dice_roll_value}). This could be an 'anomaly'!")
                        else:
                            logging.info(f"OBSERVER: Normal roll detected. ({dice_roll_value}).")

    return "Traces received", 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9090)