-- init-timescale.sql
-- Enable the TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create a table for metrics
CREATE TABLE metrics (
    time TIMESTAMPTZ NOT NULL,
    service_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    tags JSONB
);

-- Create a hypertable from the metrics table, partitioned by time
SELECT create_hypertable('metrics', 'time');

-- Create indexes for faster queries
CREATE INDEX ON metrics (service_name, metric_name, time DESC);
CREATE INDEX ON metrics (time DESC);
-- Optional: Create an index on tags using GIN for faster JSONB queries
CREATE INDEX ON metrics USING GIN (tags);