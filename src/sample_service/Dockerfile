FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire src directory so Python can resolve packages
COPY src/ ./src/

# Run with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "--access-logfile", "-", "src.sample_service.app:app"]