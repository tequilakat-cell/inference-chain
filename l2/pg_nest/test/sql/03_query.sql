-- 03_query.sql  – query functions and nest_where

CREATE EXTENSION IF NOT EXISTS pg_nest;

CREATE TABLE products (
    id    bigint PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    data  jsonb
);

SELECT pg_nest.nest_create('products', 'data', 'id');

INSERT INTO products (data) VALUES
    ('{"category":"electronics","brand":"acme","price":299.99,"stock":5,"tags":["sale"]}'),
    ('{"category":"books","brand":"pub","price":14.99,"stock":200}'),
    ('{"category":"electronics","brand":"zeta","price":49.99,"stock":0}'),
    ('{"category":"electronics","brand":"acme","price":799.00,"stock":3}');

-- nest_query: text equality
SELECT doc_id, val_text
FROM pg_nest.nest_query('products', 'category', 'electronics')
ORDER BY doc_id;

-- nest_query_num: numeric range [10, 100]
SELECT doc_id, val_num
FROM pg_nest.nest_query_num('products', 'price', 10.0, 100.0)
ORDER BY doc_id;

-- nest_path_list: schema discovery
SELECT path, freq, distinct_vals
FROM pg_nest.nest_path_list('products')
ORDER BY freq DESC, path;

-- nest_where: multi-condition AND intersection
-- documents matching category=electronics AND brand=acme
SELECT * FROM pg_nest.nest_where(
    'products',
    '{"category": "electronics", "brand": "acme"}'
)
ORDER BY 1;

-- nest_where: all three conditions
SELECT * FROM pg_nest.nest_where(
    'products',
    '{"category": "electronics", "brand": "acme", "stock": "5"}'
)
ORDER BY 1;

-- nest_path_exists
SELECT pg_nest.nest_path_exists('products', 'category');
SELECT pg_nest.nest_path_exists('products', 'tags[0]');
SELECT pg_nest.nest_path_exists('products', 'does.not.exist');

-- nest_stats
SELECT doc_count, distinct_paths FROM pg_nest.nest_stats('products');

SELECT pg_nest.nest_drop('products');
DROP TABLE products;
