EXTENSION   = pg_nest
EXTVERSION  = 1.0

MODULE_big  = pg_nest
OBJS        = src/pg_nest.o       \
              src/nest_decompose.o \
              src/nest_table.o     \
              src/nest_query.o     \
              src/nest_timeseries.o

DATA        = sql/pg_nest--1.0.sql

REGRESS      = 01_decompose 02_nest_table 03_query 04_timeseries
REGRESS_OPTS = --inputdir=test

# Link libm for numeric helpers; nothing else needed
SHLIB_LINK  = $(filter -lm, $(LIBS))

PG_CFLAGS   = -Wall -Wextra -Wno-unused-parameter -Wmissing-prototypes

PG_CONFIG  ?= pg_config
PGXS       := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
