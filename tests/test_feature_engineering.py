"""Tests for `src/ml/feature_engineering.py`'s two feature-table builders.

Focused on output shape/columns -- what a downstream `train_model.py` can
rely on existing -- plus the one thing that's easy to get quietly wrong in
each: `build_clustering_features` must exclude the `UNKNOWN_CUSTOMER`
sentinel, and `build_churn_features`'s `churned` label must actually reflect
activity after its cutoff date, not just be present as a column.
"""

from __future__ import annotations

from datetime import date

from src.ml.feature_engineering import build_churn_features, build_clustering_features
from tests.factories import make_cleansed_transactions_df, make_rfm_df

EXPECTED_CLUSTERING_COLUMNS = {
    "customer_id",
    "recency_days",
    "frequency",
    "monetary",
    "rfm_score",
    "customer_segment_label",
    "avg_order_value",
    "tenure_days",
    "distinct_payment_methods",
    "customer_age",
    "distinct_categories",
    "category_entropy",
    "region",
    "customer_segment",
}

EXPECTED_CHURN_COLUMNS = {
    "customer_id",
    "frequency",
    "monetary",
    "avg_order_value",
    "distinct_payment_methods",
    "customer_age",
    "tenure_days",
    "recency_days",
    "distinct_categories",
    "category_entropy",
    "region",
    "customer_segment",
    "churned",
}


# --------------------------------------------------------------------------
# build_clustering_features
# --------------------------------------------------------------------------


def test_build_clustering_features_has_expected_columns_and_row_count(spark):
    transactions_df = make_cleansed_transactions_df(
        spark,
        [
            {
                "customer_id": "CUST00001",
                "purchase_date": date(2024, 1, 1),
                "category": "Electronics",
                "payment_method": "Credit Card",
                "total_amount": 100.0,
            },
            {
                "customer_id": "CUST00001",
                "purchase_date": date(2024, 2, 1),
                "category": "Books",
                "payment_method": "PayPal",
                "total_amount": 50.0,
            },
            {
                "customer_id": "CUST00002",
                "purchase_date": date(2024, 1, 15),
                "category": "Electronics",
                "payment_method": "Credit Card",
                "total_amount": 200.0,
            },
        ],
    )
    rfm_df = make_rfm_df(
        spark,
        [
            ("CUST00001", 30, 2, 150.0, "342", "Loyal Customers"),
            ("CUST00002", 10, 1, 200.0, "441", "Champions"),
        ],
    )

    result_df = build_clustering_features(transactions_df, rfm_df)

    assert set(result_df.columns) == EXPECTED_CLUSTERING_COLUMNS
    assert result_df.count() == 2

    cust1 = result_df.filter(result_df.customer_id == "CUST00001").first()
    assert cust1["distinct_categories"] == 2  # Electronics + Books
    assert cust1["tenure_days"] == 31  # Jan 1 -> Feb 1
    assert cust1["avg_order_value"] == 75.0  # (100 + 50) / 2


def test_build_clustering_features_excludes_unknown_customer_sentinel(spark):
    transactions_df = make_cleansed_transactions_df(
        spark,
        [
            {"customer_id": "CUST00001", "purchase_date": date(2024, 1, 1)},
            {"customer_id": "UNKNOWN_CUSTOMER", "purchase_date": date(2024, 1, 1)},
        ],
    )
    rfm_df = make_rfm_df(spark, [("CUST00001", 30, 1, 39.98, "111", "At Risk")])

    result_df = build_clustering_features(transactions_df, rfm_df)

    assert result_df.count() == 1
    assert result_df.first()["customer_id"] == "CUST00001"


# --------------------------------------------------------------------------
# build_churn_features
# --------------------------------------------------------------------------


def test_build_churn_features_has_expected_columns(spark):
    transactions_df = make_cleansed_transactions_df(
        spark,
        [
            {"customer_id": "CUST00001", "purchase_date": date(2024, 1, 1), "total_amount": 100.0},
            {"customer_id": "CUST00002", "purchase_date": date(2024, 1, 1), "total_amount": 100.0},
            {"customer_id": "CUST00002", "purchase_date": date(2024, 3, 25), "total_amount": 50.0},
        ],
    )

    result_df = build_churn_features(transactions_df, churn_window_days=30)

    assert set(result_df.columns) == EXPECTED_CHURN_COLUMNS
    assert result_df.count() == 2


def test_build_churn_features_labels_churn_from_the_holdout_window(spark):
    # max(purchase_date) = 2024-03-25; churn_window_days=30 -> cutoff = 2024-02-24.
    # CUST00001's only transaction is before the cutoff and nothing after it -> churned.
    # CUST00002 has a transaction before AND after the cutoff -> retained.
    transactions_df = make_cleansed_transactions_df(
        spark,
        [
            {"customer_id": "CUST00001", "purchase_date": date(2024, 1, 1), "total_amount": 100.0},
            {"customer_id": "CUST00002", "purchase_date": date(2024, 1, 1), "total_amount": 100.0},
            {"customer_id": "CUST00002", "purchase_date": date(2024, 3, 25), "total_amount": 50.0},
        ],
    )

    result_df = build_churn_features(transactions_df, churn_window_days=30)
    labels = {r["customer_id"]: r["churned"] for r in result_df.collect()}

    assert labels["CUST00001"] == 1
    assert labels["CUST00002"] == 0


def test_build_churn_features_excludes_customers_with_no_pre_cutoff_activity(spark):
    # CUST00003's only transaction falls inside the holdout window itself,
    # so it has no pre-cutoff history to build features from at all.
    transactions_df = make_cleansed_transactions_df(
        spark,
        [
            {"customer_id": "CUST00001", "purchase_date": date(2024, 1, 1), "total_amount": 100.0},
            {"customer_id": "CUST00003", "purchase_date": date(2024, 3, 25), "total_amount": 75.0},
        ],
    )

    result_df = build_churn_features(transactions_df, churn_window_days=30)

    customer_ids = {r["customer_id"] for r in result_df.collect()}
    assert "CUST00003" not in customer_ids
