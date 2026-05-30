-- 01_decompose.sql  – nest_decompose() correctness tests

CREATE EXTENSION IF NOT EXISTS pg_nest;

-- Basic flat object
SELECT path, val_text, val_num, val_bool, val_null, depth
FROM pg_nest.nest_decompose('{"a":1,"b":"hello","c":true,"d":null}')
ORDER BY path;

-- Nested object – dot-notation paths
SELECT path, val_text, depth
FROM pg_nest.nest_decompose('{"x":{"y":{"z":"deep"}}}')
ORDER BY path;

-- Array – bracket notation
SELECT path, val_text
FROM pg_nest.nest_decompose('{"items":[10,20,30]}')
ORDER BY path;

-- Mixed nesting
SELECT path, val_text, val_bool, depth
FROM pg_nest.nest_decompose(
    '{"user":{"name":"alice","tags":["admin","staff"],"active":true}}'
)
ORDER BY path;

-- Numeric precision roundtrip
SELECT path, val_num
FROM pg_nest.nest_decompose('{"price":3.141592653589793}')
ORDER BY path;

-- Key with dot – must be double-quoted in path
SELECT path, val_text
FROM pg_nest.nest_decompose('{"outer":{"key.with.dot":"val"}}')
ORDER BY path;

-- Key with brackets
SELECT path, val_text
FROM pg_nest.nest_decompose('{"a[b]":"v"}')
ORDER BY path;

-- Null value
SELECT path, val_text, val_null
FROM pg_nest.nest_decompose('{"x":null}')
ORDER BY path;

-- Version
SELECT pg_nest.pg_nest_version();
