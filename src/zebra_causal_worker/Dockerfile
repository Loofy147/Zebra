# Dockerfile for the Zebra Causal Worker
FROM python:3.11-slim

WORKDIR /app

# Copy all requirements files
COPY requirements.txt .
COPY requirements-causal.txt .
COPY requirements-telemetry.txt .

# Install dependencies in a single layer to optimize image size
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-causal.txt && \
    pip install --no-cache-dir -r requirements-telemetry.txt

# Copy the entire src directory so Python can resolve packages
COPY src/ /app/src

# Set the entrypoint for the worker (placeholder command)
# This will be replaced with the actual worker script later.
CMD ["python", "-c", "import time; print('Causal worker started...'); time.sleep(3600)"]