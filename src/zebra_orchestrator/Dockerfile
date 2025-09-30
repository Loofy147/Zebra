FROM python:3.11-slim

WORKDIR /app

# Copy and install all dependencies from the shared requirements file.
# This ensures numpy and scipy are available for the causal engine.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire src directory
COPY src/ ./src/

# Run with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:9090", "--access-logfile", "-", "src.zebra_system.observer:app"]