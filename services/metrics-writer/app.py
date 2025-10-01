import os
import time
import json
import psycopg2
from psycopg2.extras import execute_values

# It's better to get the password from an environment variable
DB_PASSWORD = os.getenv("TIMESCALE_PASSWORD", "zebra_password")
DB_DSN = f"dbname=zebra_metrics user=zebra host=pgbouncer port=6432 password={DB_PASSWORD}"

def get_db_connection():
    """Establishes a connection to the database."""
    try:
        conn = psycopg2.connect(DB_DSN)
        return conn
    except psycopg2.OperationalError as e:
        print(f"Could not connect to the database: {e}")
        return None

def write_batch(metrics: list):
    """Writes a batch of metrics to the TimescaleDB."""
    conn = get_db_connection()
    if not conn:
        return

    try:
        with conn.cursor() as cur:
            records = [
                (
                    m['time'],
                    m['service_name'],
                    m['metric_name'],
                    m['value'],
                    json.dumps(m.get('tags', {}))
                )
                for m in metrics
            ]
            sql = "INSERT INTO metrics(time, service_name, metric_name, value, tags) VALUES %s"
            execute_values(cur, sql, records)
            conn.commit()
            print(f"Successfully wrote {len(records)} records to the database.")
    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == '__main__':
    # Example usage
    sample_metrics = [
        {
            "time": "2025-10-01T04:02:00Z",
            "service_name": "zebra-observer",
            "metric_name": "cpu_usage",
            "value": 0.75,
            "tags": {"host": "worker-1"}
        },
        {
            "time": "2025-10-01T04:02:05Z",
            "service_name": "zebra-causal-worker",
            "metric_name": "queue_length",
            "value": 150.0,
            "tags": {}
        }
    ]
    write_batch(sample_metrics)