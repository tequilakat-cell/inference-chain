-- 02_nest_table.sql  – nest_create / nest_drop / trigger / reindex

CREATE EXTENSION IF NOT EXISTS pg_nest;

CREATE TABLE events (
    id      bigint PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    payload jsonb,
    created timestamptz DEFAULT now()
);

-- Register with time column
SELECT pg_nest.nest_create('events', 'payload', 'id', 'created');

-- Verify registry entry
SELECT nspname, relname, jsonb_col, id_col, time_col IS NOT NULL AS has_time
FROM pg_nest.nest_registry WHERE relname = 'events';

-- INSERT fires trigger, paths table is populated
INSERT INTO events (payload) VALUES
    ('{"type":"click","user":{"id":42,"name":"bob"},"meta":{"score":9.5}}');

SELECT path, val_text, val_num, path_depth
FROM public._nest_events_paths
ORDER BY path;

-- UPDATE replaces old paths
UPDATE events SET payload = '{"type":"view","user":{"id":42,"name":"bob"}}'
WHERE id = 1;

SELECT path, val_text
FROM public._nest_events_paths
ORDER BY path;

-- Trigger handles keys with special characters
INSERT INTO events (payload) VALUES
    ('{"key.with.dot":"yes","arr[0]":"bracket"}');

SELECT path, val_text
FROM public._nest_events_paths WHERE doc_id = 2
ORDER BY path;

-- nest_reindex
SELECT pg_nest.nest_reindex('events') > 0 AS reindexed;

-- Verify stats
SELECT doc_count, path_count > 0 AS has_paths, distinct_paths > 0 AS has_distinct
FROM pg_nest.nest_stats('events');

-- Path existence
SELECT pg_nest.nest_path_exists('events', 'type');
SELECT pg_nest.nest_path_exists('events', 'nonexistent.path');

-- nest_tables view shows registered table
SELECT schema, source_table, jsonb_col FROM pg_nest.nest_tables
WHERE source_table = 'events';

SELECT pg_nest.nest_drop('events');
DROP TABLE events;
