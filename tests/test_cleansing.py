"""Tests for `src/transformation/cleanse.py`'s cleansing logic: deduplication,
outlier capping, range enforcement, and null imputation.

Every step here preserves row count except `deduplicate` -- outliers and
range violations are treated (capped or nulled) in place, never dropped
(see cleanse.py's module docstring) -- so several tests below assert that
invariant directly rather than just checking the "happy path" value change.
"""

from __future__ import annotations

import pytest
from pyspark.sql import functions as F

from src.transformation.cleanse import (
    CleansingConfig,
    DataCleanser,
    deduplicate,
    enforce_valid_ranges,
    impute_nulls,
    treat_outliers_iqr,
)
from tests.factories import make_transactions_df

# --------------------------------------------------------------------------
# Deduplication
# --------------------------------------------------------------------------


def test_deduplicate_removes_exact_duplicate_rows(spark):
    df = make_transactions_df(
        spark,
        [
            {"transaction_id": "TXN0001"},
            {"transaction_id": "TXN0001"},  # byte-for-byte duplicate of the row above
            {"transaction_id": "TXN0002"},
        ],
    )

    result_df, metrics = deduplicate(df)

    assert result_df.count() == 2
    assert metrics["exact_duplicates_removed"] == 1
    assert metrics["rows_after"] == 2


def test_deduplicate_keeps_the_more_complete_row_for_a_conflicting_transaction_id(spark):
    df = make_transactions_df(
        spark,
        [
            {"transaction_id": "TXN0001", "customer_id": None},  # less complete
            {"transaction_id": "TXN0001", "customer_id": "CUST00001"},  # more complete, same id
        ],
    )

    result_df, metrics = deduplicate(df)

    assert result_df.count() == 1
    assert metrics["duplicate_transaction_id_rows_removed"] == 1
    assert result_df.first()["customer_id"] == "CUST00001"


# --------------------------------------------------------------------------
# Outlier treatment
# --------------------------------------------------------------------------


def test_treat_outliers_iqr_caps_extreme_values_without_dropping_rows(spark):
    df = make_transactions_df(
        spark,
        [{"total_amount": v} for v in [50.0, 55.0, 60.0, 52.0, 58.0, 100_000.0]],
    )

    result_df, bounds = treat_outliers_iqr(df, columns=("total_amount",), multiplier=1.5)

    assert result_df.count() == df.count()  # capped, not dropped
    max_value = result_df.agg(F.max("total_amount")).first()[0]
    assert max_value == pytest.approx(bounds["total_amount"]["upper_bound"], rel=1e-6)
    assert max_value < 100_000.0
    assert bounds["total_amount"]["capped_high"] == 1


def test_treat_outliers_iqr_leaves_a_typical_dataset_untouched(spark):
    df = make_transactions_df(spark, [{"total_amount": v} for v in [40.0, 45.0, 50.0, 55.0, 60.0]])

    result_df, bounds = treat_outliers_iqr(df, columns=("total_amount",), multiplier=1.5)

    assert bounds["total_amount"]["capped_low"] == 0
    assert bounds["total_amount"]["capped_high"] == 0
    assert sorted(row["total_amount"] for row in result_df.collect()) == [40.0, 45.0, 50.0, 55.0, 60.0]


# --------------------------------------------------------------------------
# Valid-range enforcement
# --------------------------------------------------------------------------


def test_enforce_valid_ranges_nulls_out_impossible_values(spark):
    df = make_transactions_df(
        spark,
        [
            {"customer_age": 15.0},  # below age_min=18
            {"customer_age": 45.0},  # valid
            {"quantity": -2.0},  # a negative quantity is impossible, not just unusual
        ],
    )

    result_df, metrics = enforce_valid_ranges(df, age_min=18, age_max=100)

    assert result_df.count() == df.count()  # nulled, not dropped
    assert metrics["customer_age_out_of_range"] == 1
    assert metrics["quantity_out_of_range"] == 1
    ages = [row["customer_age"] for row in result_df.collect()]
    assert 15.0 not in ages
    assert 45.0 in ages


# --------------------------------------------------------------------------
# Null imputation
# --------------------------------------------------------------------------


def test_impute_nulls_fills_customer_id_with_sentinel(spark):
    df = make_transactions_df(spark, [{"customer_id": None}, {"customer_id": "CUST00002"}])

    result_df, metrics = impute_nulls(df)

    assert result_df.filter(F.col("customer_id").isNull()).count() == 0
    assert metrics["customer_id"]["nulls_imputed"] == 1
    assert metrics["customer_id"]["fill_value"] == "UNKNOWN_CUSTOMER"
    values = [row["customer_id"] for row in result_df.collect()]
    assert "UNKNOWN_CUSTOMER" in values


def test_impute_nulls_fills_quantity_with_the_median(spark):
    df = make_transactions_df(
        spark,
        [
            {"transaction_id": "TXN0001", "quantity": 1.0},
            {"transaction_id": "TXN0002", "quantity": 3.0},
            {"transaction_id": "TXN0003", "quantity": 5.0},
            {"transaction_id": "TXN0004", "quantity": None},
        ],
    )

    result_df, metrics = impute_nulls(df)

    assert result_df.filter(F.col("quantity").isNull()).count() == 0
    assert metrics["quantity"]["strategy"] == "median"
    assert metrics["quantity"]["nulls_imputed"] == 1
    imputed_row = result_df.filter(F.col("transaction_id") == "TXN0004").first()
    assert imputed_row["quantity"] == 3  # median of [1, 3, 5]


def test_impute_nulls_leaves_purchase_date_and_transaction_id_null(spark):
    df = make_transactions_df(spark, [{"purchase_date": None, "transaction_id": None}])

    result_df, metrics = impute_nulls(df)

    row = result_df.first()
    assert row["purchase_date"] is None
    assert row["transaction_id"] is None
    assert metrics["purchase_date"]["strategy"] == "not_imputed"
    assert metrics["purchase_date"]["nulls_remaining"] == 1


# --------------------------------------------------------------------------
# Full DataCleanser pipeline (integration of the pieces above)
# --------------------------------------------------------------------------


def test_data_cleanser_run_dedupes_and_imputes_end_to_end(spark):
    df = make_transactions_df(
        spark,
        [
            {"transaction_id": "TXN0001", "customer_id": "CUST00001"},
            {"transaction_id": "TXN0002", "customer_id": None, "quantity": None},
            {"transaction_id": "TXN0002", "customer_id": None, "quantity": None},  # exact duplicate
        ],
    )

    cleanser = DataCleanser(df, CleansingConfig())
    result_df = cleanser.run()

    assert result_df.count() == 2  # the exact duplicate is gone
    assert result_df.filter(F.col("quantity").isNull()).count() == 0  # nulls imputed
    assert "deduplicate" in cleanser.report
    assert "impute_nulls" in cleanser.report
    assert cleanser.report["deduplicate"]["exact_duplicates_removed"] == 1
