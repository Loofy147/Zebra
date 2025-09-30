FROM python:3.11-slim

WORKDIR /app

# The observer needs Flask to run, and gunicorn to serve it.
# We'll add other analysis libraries here in the future.
RUN pip install --no-cache-dir Flask==3.0.3 gunicorn==22.0.0

# Copy the entire src directory
COPY src/ ./src/

# Run with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:9090", "--access-logfile", "-", "src.zebra_system.observer:app"]