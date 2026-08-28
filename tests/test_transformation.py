"""Tests for `src/transformation/transform.py`'s aggregation functions.

These operate on the *post-cleansing* schema (real `DateType` dates, integer
quantity/age) rather than the raw ingestion schema `test_cleansing.py` uses --
that's the actual input `transform.py` is written against.
"""

from __future__ import annotations

from datetime import date

from src.transformation.transform import compute_customer_rfm, compute_daily_revenue, compute_top_products_by_category
from tests.factories import make_cleansed_transactions_df

# --------------------------------------------------------------------------
# compute_daily_revenue
# --------------------------------------------------------------------------


def test_compute_daily_revenue_aggregates_by_date_region_and_category(spark):
    df = make_cleansed_transactions_df(
        spark,
        [
            {
                "purchase_date": date(2024, 1, 1),
                "region": "North",
                "category": "Electronics",
                "total_amount": 100.0,
                "quantity": 2,
            },
            {
                "purchase_date": date(2024, 1, 1),
                "region": "North",
                "category": "Electronics",
                "total_amount": 50.0,
                "quantity": 1,
            },
            {
                "purchase_date": date(2024, 1, 1),
                "region": "South",
                "category": "Books",
                "total_amount": 20.0,
                "quantity": 1,
            },
        ],
    )

    result_df = compute_daily_revenue(df)
    rows = {(r["region"], r["category"]): r for r in result_df.collect()}

    assert result_df.count() == 2  # (North, Electronics) and (South, Books)
    north_electronics = rows[("North", "Electronics")]
    assert north_electronics["revenue"] == 150.0
    assert north_electronics["units_sold"] == 3
    assert north_electronics["num_transactions"] == 2
    assert north_electronics["avg_order_value"] == 75.0


def test_compute_daily_revenue_excludes_rows_with_null_purchase_date(spark):
    df = make_cleansed_transactions_df(
        spark,
        [
            {"purchase_date": date(2024, 1, 1), "total_amount": 100.0},
            {"purchase_date": None, "total_amount": 999.0},  # unparseable date, left null by cleansing
        ],
    )

    result_df = compute_daily_revenue(df)

    assert result_df.count() == 1
    assert result_df.first()["revenue"] == 100.0


# --------------------------------------------------------------------------
# compute_customer_rfm
# --------------------------------------------------------------------------


def test_compute_customer_rfm_computes_recency_frequency_monetary(spark):
    df = make_cleansed_transactions_df(
        spark,
        [
            {"customer_id": "CUST00001", "purchase_date": date(2024, 1, 1), "total_amount": 100.0},
            {"customer_id": "CUST00001", "purchase_date": date(2024, 1, 10), "total_amount": 50.0},
            {"customer_id": "CUST00002", "purchase_date": date(2024, 1, 5), "total_amount": 200.0},
        ],
    )

    result_df = compute_customer_rfm(df)
    rows = {r["customer_id"]: r for r in result_df.collect()}

    assert result_df.count() == 2
    # "today" is one day after the dataset's own latest date (2024-01-10 + 1)
    assert rows["CUST00001"]["recency_days"] == 1
    assert rows["CUST00001"]["frequency"] == 2
    assert rows["CUST00001"]["monetary"] == 150.0
    assert rows["CUST00002"]["recency_days"] == 6
    assert rows["CUST00002"]["frequency"] == 1
    assert rows["CUST00002"]["monetary"] == 200.0


def test_compute_customer_rfm_excludes_unknown_customer_and_null_dates(spark):
    df = make_cleansed_transactions_df(
        spark,
        [
            {"customer_id": "CUST00001", "purchase_date": date(2024, 1, 1)},
            {"customer_id": "UNKNOWN_CUSTOMER", "purchase_date": date(2024, 1, 1)},
            {"customer_id": "CUST00002", "purchase_date": None},
        ],
    )

    result_df = compute_customer_rfm(df)

    assert result_df.count() == 1
    assert result_df.first()["customer_id"] == "CUST00001"


# --------------------------------------------------------------------------
# compute_top_products_by_category
# --------------------------------------------------------------------------


def test_compute_top_products_by_category_ranks_and_limits_to_top_n(spark):
    df = make_cleansed_transactions_df(
        spark,
        [
            {"product_id": "PROD0001", "category": "Electronics", "total_amount": 300.0},
            {"product_id": "PROD0002", "category": "Electronics", "total_amount": 100.0},
            {"product_id": "PROD0003", "category": "Electronics", "total_amount": 200.0},
            {"product_id": "PROD0004", "category": "Books", "total_amount": 50.0},
        ],
    )

    result_df = compute_top_products_by_category(df, top_n=2)
    electronics_rows = [r for r in result_df.collect() if r["category"] == "Electronics"]

    assert result_df.count() == 3  # top 2 Electronics + the only Books product
    assert len(electronics_rows) == 2
    ranked = sorted(electronics_rows, key=lambda r: r["category_rank"])
    assert [r["product_id"] for r in ranked] == ["PROD0001", "PROD0003"]  # PROD0002 (lowest revenue) dropped
    assert ranked[0]["category_rank"] == 1
    assert ranked[0]["pct_of_category_revenue"] == 50.0  # 300 / (300+100+200) * 100
