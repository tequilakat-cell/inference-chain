-- 04_timeseries.sql  – nest_query_time, nest_time_bucket_agg

CREATE EXTENSION IF NOT EXISTS pg_nest;

CREATE TABLE readings (
    id        bigint PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    sensor    jsonb,
    measured  timestamptz
);

SELECT pg_nest.nest_create('readings', 'sensor', 'id', 'measured');

INSERT INTO readings (sensor, measured) VALUES
    ('{"type":"temp","value":21.5,"unit":"C"}', '2024-01-01 00:00:00+00'),
    ('{"type":"temp","value":22.1,"unit":"C"}', '2024-01-01 01:00:00+00'),
    ('{"type":"temp","value":23.0,"unit":"C"}', '2024-01-01 02:00:00+00'),
    ('{"type":"humidity","value":60,"unit":"%"}', '2024-01-01 00:30:00+00'),
    ('{"type":"temp","value":20.8,"unit":"C"}', '2024-01-02 00:00:00+00');

-- Time-bounded text query (uses (ts, path) INCLUDE index)
SELECT doc_id, val_text, ts
FROM pg_nest.nest_query_time(
    'readings',
    'type', 'temp',
    '2024-01-01 00:00:00+00',
    '2024-01-01 23:59:59+00'
)
ORDER BY ts;

-- Time-bucket aggregation over day 1 (1-hour buckets)
-- Uses date_bin() on plain PG; time_bucket() when TimescaleDB is present
SELECT bucket, count, vals
FROM pg_nest.nest_time_bucket_agg(
    'readings',
    'type',
    '1 hour',
    '2024-01-01 00:00:00+00',
    '2024-01-01 02:59:59+00'
)
ORDER BY bucket;

-- Numeric range over time
SELECT doc_id, val_num
FROM pg_nest.nest_query_num('readings', 'value', 20.0, 22.5)
ORDER BY doc_id;

-- nest_reindex preserves all data
SELECT pg_nest.nest_reindex('readings') AS path_rows;
SELECT doc_count FROM pg_nest.nest_stats('readings');

SELECT pg_nest.nest_drop('readings');
DROP TABLE readings;
