MODULE_big = pg_inft

OBJS = src/pg_inft.o \
       src/inft_keccak.o \
       src/inft_proof.o \
       src/inft_search.o \
       src/inft_chunk.o

EXTENSION  = pg_inft
# Base script + full upgrade chain so `CREATE EXTENSION pg_inft` can reach
# default_version (1.3). Without the intermediate scripts postgres can only
# build version 1.0 and the registry/job/benchmark functions are missing.
DATA       = sql/pg_inft--1.0.sql \
             sql/pg_inft--1.0--1.1.sql \
             sql/pg_inft--1.1--1.2.sql \
             sql/pg_inft--1.2--1.3.sql

PG_CONFIG  ?= pg_config
PGXS       := $(shell $(PG_CONFIG) --pgxs)

PG_CPPFLAGS = -I$(srcdir)/src
SHLIB_LINK  =

ifdef DEBUG
PG_CPPFLAGS += -g -O0 -DDEBUG
else
PG_CPPFLAGS += -O2
endif

# Suppress warnings that fire in generated PG headers under -Wextra
PG_CPPFLAGS += -Wno-unused-parameter -Wno-sign-compare

include $(PGXS)
