"""Tests for schema validation in `src/ingestion/ingest.py`.

`validate_schema` is the boundary between "data someone handed us" and "data
we're willing to build anything on" -- these tests pin down exactly what it
accepts (an exact or superset match) and rejects (a missing column, a type
mismatch), since that boundary is precisely what would let a malformed
upstream export slip through silently if it regressed.
"""

from __future__ import annotations

import pytest
from pyspark.sql.types import StringType, StructField, StructType

from src.ingestion.ingest import SchemaValidationError, get_transaction_schema, validate_schema
from tests.factories import RAW_DEFAULT_ROW, RAW_COLUMN_ORDER

SAMPLE_ROW = tuple(RAW_DEFAULT_ROW[col] for col in RAW_COLUMN_ORDER)


def test_get_transaction_schema_has_expected_columns_in_order():
    schema = get_transaction_schema()
    assert [f.name for f in schema.fields] == RAW_COLUMN_ORDER


def test_validate_schema_passes_for_exact_match(spark):
    schema = get_transaction_schema()
    df = spark.createDataFrame([SAMPLE_ROW], schema)

    validate_schema(df, schema)  # must not raise


def test_validate_schema_raises_for_missing_column(spark):
    schema = get_transaction_schema()
    incomplete_schema = StructType([f for f in schema.fields if f.name != "customer_id"])
    incomplete_row = tuple(v for v, f in zip(SAMPLE_ROW, schema.fields) if f.name != "customer_id")
    df = spark.createDataFrame([incomplete_row], incomplete_schema)

    with pytest.raises(SchemaValidationError, match="customer_id"):
        validate_schema(df, schema)


def test_validate_schema_raises_for_type_mismatch(spark):
    schema = get_transaction_schema()
    mismatched_fields = [
        StructField("quantity", StringType(), True) if f.name == "quantity" else f for f in schema.fields
    ]
    mismatched_schema = StructType(mismatched_fields)
    row = list(SAMPLE_ROW)
    row[RAW_COLUMN_ORDER.index("quantity")] = "2.0"  # quantity as a string instead of a double
    df = spark.createDataFrame([tuple(row)], mismatched_schema)

    with pytest.raises(SchemaValidationError, match="quantity"):
        validate_schema(df, schema)


def test_validate_schema_tolerates_extra_columns(spark):
    schema = get_transaction_schema()
    extended_schema = StructType(schema.fields + [StructField("extra_col", StringType(), True)])
    df = spark.createDataFrame([SAMPLE_ROW + ("unexpected",)], extended_schema)

    validate_schema(df, schema)  # extra columns are a warning, not a failure -- must not raise
