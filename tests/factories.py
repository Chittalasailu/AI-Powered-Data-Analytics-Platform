"""Small helpers for building minimal, schema-correct test DataFrames.

Every function under test expects a specific full schema (the raw ingestion
schema, or the post-cleansing schema), so constructing rows by hand for
every test would mean repeating all 12 columns every time even when a test
only cares about one or two of them. These helpers let a test specify just
the columns its assertions depend on and get sensible defaults for the rest.
"""

from __future__ import annotations

from datetime import date

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DateType, DoubleType, IntegerType, StringType, StructField, StructType

from src.ingestion.ingest import get_transaction_schema

# --------------------------------------------------------------------------
# Raw ingestion shape (purchase_date as string, numerics as double) --
# for ingestion/cleansing tests, which operate on this schema.
# --------------------------------------------------------------------------

RAW_SCHEMA = get_transaction_schema()
RAW_COLUMN_ORDER = [f.name for f in RAW_SCHEMA.fields]

RAW_DEFAULT_ROW = {
    "transaction_id": "TXN0000001",
    "customer_id": "CUST00001",
    "product_id": "PROD0001",
    "category": "Electronics",
    "purchase_date": "2024-01-15",
    "quantity": 2.0,
    "unit_price": 19.99,
    "total_amount": 39.98,
    "region": "North",
    "payment_method": "Credit Card",
    "customer_age": 35.0,
    "customer_segment": "Regular",
}


def make_transactions_df(spark: SparkSession, rows: list[dict]) -> DataFrame:
    """Build a raw-ingestion-schema DataFrame from partial row dicts.

    Each dict is overlaid on `RAW_DEFAULT_ROW`; a test only needs to specify
    the columns its assertion actually depends on.
    """
    full_rows = [
        tuple({**RAW_DEFAULT_ROW, **row}[col] for col in RAW_COLUMN_ORDER) for row in rows
    ]
    return spark.createDataFrame(full_rows, RAW_SCHEMA)


# --------------------------------------------------------------------------
# Post-cleansing shape (purchase_date as a real date, quantity/customer_age
# as integers) -- for transformation and feature-engineering tests, which
# operate on `transactions_cleaned`'s schema, not the raw ingested one.
# --------------------------------------------------------------------------

CLEANSED_SCHEMA = StructType(
    [
        StructField("transaction_id", StringType(), True),
        StructField("customer_id", StringType(), True),
        StructField("product_id", StringType(), True),
        StructField("category", StringType(), True),
        StructField("purchase_date", DateType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("unit_price", DoubleType(), True),
        StructField("total_amount", DoubleType(), True),
        StructField("region", StringType(), True),
        StructField("payment_method", StringType(), True),
        StructField("customer_age", IntegerType(), True),
        StructField("customer_segment", StringType(), True),
    ]
)
CLEANSED_COLUMN_ORDER = [f.name for f in CLEANSED_SCHEMA.fields]

CLEANSED_DEFAULT_ROW = {
    "transaction_id": "TXN0000001",
    "customer_id": "CUST00001",
    "product_id": "PROD0001",
    "category": "Electronics",
    "purchase_date": date(2024, 1, 15),
    "quantity": 2,
    "unit_price": 19.99,
    "total_amount": 39.98,
    "region": "North",
    "payment_method": "Credit Card",
    "customer_age": 35,
    "customer_segment": "Regular",
}


def make_cleansed_transactions_df(spark: SparkSession, rows: list[dict]) -> DataFrame:
    """Build a post-cleansing-schema DataFrame from partial row dicts."""
    full_rows = [
        tuple({**CLEANSED_DEFAULT_ROW, **row}[col] for col in CLEANSED_COLUMN_ORDER) for row in rows
    ]
    return spark.createDataFrame(full_rows, CLEANSED_SCHEMA)


def make_rfm_df(spark: SparkSession, rows: list[tuple]) -> DataFrame:
    """Build a small `agg_customer_rfm`-shaped DataFrame.

    Schema is inferred rather than declared explicitly: unlike the two
    factories above, nothing under test here is sensitive to exact numeric
    width (Int vs Long), so inference keeps this one simple.
    """
    return spark.createDataFrame(
        rows, ["customer_id", "recency_days", "frequency", "monetary", "rfm_score", "customer_segment_label"]
    )
