"""Build ML-ready, per-customer feature tables for the two models in
`train_model.py`.

Two feature tables come out of this module, because the two downstream
models have fundamentally different requirements for what "safe" features
look like:

1. **Clustering features** (`build_clustering_features`) -- full-history
   RFM + behavioral features per customer, for the unsupervised
   segmentation model. Reuses `agg_customer_rfm` (written by
   `transform.py`) for recency/frequency/monetary rather than recomputing
   them -- this is the "aggregated customer data" the segmentation model
   is meant to build on -- and enriches it with category diversity,
   payment-method diversity, and tenure computed here from
   `transactions_cleaned`.

2. **Churn features** (`build_churn_features`) -- a *temporally-safe*
   feature set for the supervised churn model. This deliberately does
   **not** reuse `agg_customer_rfm`: that table's `recency_days` is
   measured against the dataset's true final date, which is exactly the
   thing a churn label is trying to predict (a customer with a large
   `recency_days` has, almost by definition, already churned). Using it as
   a feature would leak the label straight into the inputs and produce a
   model that looks flawless and predicts nothing useful. Instead,
   features here are computed only from transactions before a
   `feature_cutoff_date`, and the label is computed from the held-out
   window after it -- the model is trained to predict the future from the
   past, the same way it will be used at inference time.

Both feature tables are written to the processed data layer as Spark
tables; `train_model.py` reads them and converts to pandas immediately
before scikit-learn needs them.

Usage:
    python src/ml/feature_engineering.py [--config config/config.yaml] [--env local|databricks]
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import load_yaml_config, resolve_environment  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402
from src.utils.spark_session import build_spark_session, get_dbutils, path_exists  # noqa: E402

DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Shared per-customer aggregation helpers
# --------------------------------------------------------------------------


def _mode_column(df: DataFrame, group_col: str, target_col: str, alias: str) -> DataFrame:
    """The most frequent non-null value of `target_col` per `group_col`.

    Used for `region` and `customer_segment`: both are recorded per
    *transaction* in the source data rather than per customer, so a single
    customer can have a handful of transactions disagreeing on either
    field (see cleanse.py's mode-imputation and the raw generator, which
    assigns `customer_segment` per row rather than per customer). Taking
    the mode collapses that back down to one representative value per
    customer for use as a single categorical feature.
    """
    counts = df.filter(F.col(target_col).isNotNull()).groupBy(group_col, target_col).count()
    ranking_window = Window.partitionBy(group_col).orderBy(F.desc("count"))
    return (
        counts.withColumn("rn", F.row_number().over(ranking_window))
        .filter(F.col("rn") == 1)
        .select(group_col, F.col(target_col).alias(alias))
    )


def compute_category_diversity(df: DataFrame) -> DataFrame:
    """Distinct category count and Shannon entropy of category share, per customer.

    `distinct_categories` alone treats a customer who bought 1 item each
    from 5 categories the same as one who bought 100 items from 4
    categories and 1 from a 5th -- both show `distinct_categories = 5`.
    `category_entropy` (`-sum(p * ln(p))` over each category's share of
    the customer's orders) captures the difference: it's 0 for a customer
    concentrated in a single category and highest (`ln(10)` for this
    dataset's 10 categories) for a customer spread evenly across all of
    them, so it distinguishes "shops everywhere a little" from "mostly
    loyal to one category, dabbles elsewhere."
    """
    category_counts = df.groupBy("customer_id", "category").agg(F.count(F.lit(1)).alias("category_orders"))
    totals = category_counts.groupBy("customer_id").agg(F.sum("category_orders").alias("total_orders"))

    share = F.col("category_orders") / F.col("total_orders")
    shares = category_counts.join(totals, "customer_id").withColumn(
        "p_log_p", -share * F.log(share)
    )

    return shares.groupBy("customer_id").agg(
        F.countDistinct("category").alias("distinct_categories"),
        F.round(F.sum("p_log_p"), 4).alias("category_entropy"),
    )


def compute_core_behavioral_features(df: DataFrame) -> DataFrame:
    """Frequency, monetary, tenure, payment diversity, and demographics per customer.

    Deliberately takes `df` as a parameter rather than reading a fixed
    table: the caller passes either the full transaction history (for
    clustering) or only the pre-cutoff slice (for churn), and this
    function doesn't need to know or care which -- it just aggregates
    whatever rows it's given.
    """
    behavioral = df.groupBy("customer_id").agg(
        F.count(F.lit(1)).alias("frequency"),
        F.round(F.sum("total_amount"), 2).alias("monetary"),
        F.round(F.avg("total_amount"), 2).alias("avg_order_value"),
        F.min("purchase_date").alias("first_purchase_date"),
        F.max("purchase_date").alias("last_purchase_date"),
        F.countDistinct("payment_method").alias("distinct_payment_methods"),
        F.round(F.avg("customer_age"), 1).alias("customer_age"),
    )
    return behavioral.withColumn(
        "tenure_days", F.datediff(F.col("last_purchase_date"), F.col("first_purchase_date"))
    )


def _known_customer_transactions(transactions: DataFrame) -> DataFrame:
    """Rows usable for customer-level features: a real customer, a real date.

    `UNKNOWN_CUSTOMER` (cleansing's sentinel for a missing customer_id) and
    a null `purchase_date` (left un-imputed by cleansing on purpose) both
    make a row unusable for per-customer, per-date feature engineering --
    there's no customer to attribute it to, or no date to place it in time.
    """
    return transactions.filter(
        (F.col("customer_id") != "UNKNOWN_CUSTOMER") & F.col("purchase_date").isNotNull()
    )


# --------------------------------------------------------------------------
# Clustering features (full history)
# --------------------------------------------------------------------------


def build_clustering_features(transactions: DataFrame, rfm: DataFrame) -> DataFrame:
    """Full-history features for the KMeans customer-segmentation model.

    `recency_days`, `frequency`, `monetary`, and `rfm_score` come straight
    from `agg_customer_rfm` -- there's no future outcome being predicted
    here (clustering is descriptive, not predictive), so there's no
    leakage risk in using the full-history view of "how has this customer
    behaved so far." `region` and `customer_segment` ride along as
    descriptive (not model-input) columns, useful later for profiling what
    each discovered cluster actually looks like.
    """
    known = _known_customer_transactions(transactions)

    behavioral = compute_core_behavioral_features(known).select(
        "customer_id", "avg_order_value", "tenure_days", "distinct_payment_methods", "customer_age"
    )
    diversity = compute_category_diversity(known)
    region = _mode_column(known, "customer_id", "region", "region")
    segment = _mode_column(known, "customer_id", "customer_segment", "customer_segment")

    return (
        rfm.select("customer_id", "recency_days", "frequency", "monetary", "rfm_score", "customer_segment_label")
        .join(behavioral, "customer_id", "inner")
        .join(diversity, "customer_id", "inner")
        .join(region, "customer_id", "left")
        .join(segment, "customer_id", "left")
    )


# --------------------------------------------------------------------------
# Churn features (temporal cutoff, leakage-safe)
# --------------------------------------------------------------------------


def build_churn_features(transactions: DataFrame, churn_window_days: int) -> DataFrame:
    """Leakage-safe features + label for the churn-prediction model.

    `feature_cutoff_date = max(purchase_date) - churn_window_days`. Every
    feature is computed from transactions on or before that date only.
    The label, `churned`, is 1 if the customer has *no* transaction in the
    `churn_window_days` after the cutoff (up to the dataset's true last
    date), else 0 -- i.e. "did this customer go quiet for the entire
    holdout window." A customer needs at least one pre-cutoff transaction
    to be included at all, since there'd be no features to compute
    otherwise (and no meaningful way to say they "churned" from a
    relationship that hadn't started yet).
    """
    known = _known_customer_transactions(transactions)

    max_date = known.agg(F.max("purchase_date").alias("max_date")).first()["max_date"]
    feature_cutoff_date = max_date - timedelta(days=churn_window_days)

    pre_cutoff = known.filter(F.col("purchase_date") <= F.lit(feature_cutoff_date))
    holdout_window = known.filter(F.col("purchase_date") > F.lit(feature_cutoff_date))

    behavioral = compute_core_behavioral_features(pre_cutoff)
    diversity = compute_category_diversity(pre_cutoff)
    region = _mode_column(pre_cutoff, "customer_id", "region", "region")
    segment = _mode_column(pre_cutoff, "customer_id", "customer_segment", "customer_segment")

    features = (
        behavioral
        # recency relative to the cutoff, not to the dataset's true last date --
        # the whole point of the cutoff is that the model can't see past it
        .withColumn("recency_days", F.datediff(F.lit(feature_cutoff_date), F.col("last_purchase_date")))
        .drop("first_purchase_date", "last_purchase_date")
        .join(diversity, "customer_id", "inner")
        .join(region, "customer_id", "left")
        .join(segment, "customer_id", "left")
    )

    retained_customer_ids = holdout_window.select("customer_id").distinct().withColumn("_retained", F.lit(1))
    return (
        features.join(retained_customer_ids, "customer_id", "left")
        .withColumn("churned", F.when(F.col("_retained").isNull(), F.lit(1)).otherwise(F.lit(0)))
        .drop("_retained")
    )


# --------------------------------------------------------------------------
# Run settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MlFeatureSettings:
    environment: str
    app_name: str
    transactions_path: str
    transactions_format: str
    rfm_path: str
    rfm_format: str
    churn_window_days: int
    clustering_output_path: str
    churn_output_path: str
    output_format: str
    write_mode: str
    log_level: str


def load_run_settings(config_path: Path, env_override: Optional[str] = None) -> MlFeatureSettings:
    raw_config = load_yaml_config(config_path)
    environment, _, processed_data_dir = resolve_environment(raw_config, env_override)
    processed_data_dir = processed_data_dir.rstrip("/")
    ml_cfg = raw_config["ml"]
    agg_cfg = raw_config["aggregation"]

    return MlFeatureSettings(
        environment=environment,
        app_name=f"{raw_config['spark']['app_name']}-ml-feature-engineering",
        transactions_path=f"{processed_data_dir}/{ml_cfg['input_table_name']}",
        transactions_format=raw_config["cleansing"]["output_format"],
        rfm_path=f"{processed_data_dir}/{agg_cfg['output_tables']['customer_rfm']}",
        rfm_format=agg_cfg["output_format"],
        churn_window_days=int(ml_cfg.get("churn_window_days", 90)),
        clustering_output_path=f"{processed_data_dir}/{ml_cfg['clustering_features_table']}",
        churn_output_path=f"{processed_data_dir}/{ml_cfg['churn_features_table']}",
        output_format=ml_cfg["output_format"],
        write_mode=ml_cfg["write_mode"],
        log_level=raw_config.get("logging", {}).get("level", "INFO"),
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--env",
        choices=["local", "databricks"],
        default=None,
        help="Override the 'environment' value from config.yaml",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    settings = load_run_settings(args.config, env_override=args.env)

    setup_logging(settings.log_level)

    use_delta = "delta" in (settings.transactions_format, settings.rfm_format, settings.output_format)
    spark = build_spark_session(settings.environment, settings.app_name, use_delta=use_delta)
    dbutils = get_dbutils(spark) if settings.environment == "databricks" else None

    try:
        logger.info(
            "feature_engineering_started",
            extra={"transactions_path": settings.transactions_path, "rfm_path": settings.rfm_path},
        )

        for path in (settings.transactions_path, settings.rfm_path):
            if not path_exists(path, settings.environment, dbutils):
                raise FileNotFoundError(f"Input path does not exist: {path}")

        # Cached: both build_clustering_features and build_churn_features
        # scan this same DataFrame independently below.
        transactions = spark.read.format(settings.transactions_format).load(settings.transactions_path).cache()
        row_count = transactions.count()
        logger.info("transactions_loaded", extra={"row_count": row_count})

        rfm = spark.read.format(settings.rfm_format).load(settings.rfm_path)

        clustering_features = build_clustering_features(transactions, rfm)
        clustering_row_count = clustering_features.count()
        (
            clustering_features.coalesce(1)
            .write.format(settings.output_format)
            .mode(settings.write_mode)
            .save(settings.clustering_output_path)
        )
        logger.info(
            "clustering_features_written",
            extra={"output_path": settings.clustering_output_path, "row_count": clustering_row_count},
        )

        churn_features = build_churn_features(transactions, settings.churn_window_days)
        churn_row_count = churn_features.count()
        churned_count = churn_features.filter(F.col("churned") == 1).count()
        (
            churn_features.coalesce(1)
            .write.format(settings.output_format)
            .mode(settings.write_mode)
            .save(settings.churn_output_path)
        )
        logger.info(
            "churn_features_written",
            extra={
                "output_path": settings.churn_output_path,
                "row_count": churn_row_count,
                "churned_count": churned_count,
                "churn_rate": round(churned_count / churn_row_count, 4) if churn_row_count else None,
                "churn_window_days": settings.churn_window_days,
            },
        )

        transactions.unpersist()
        logger.info("feature_engineering_completed")
    except Exception:
        logger.exception("feature_engineering_failed")
        raise
    finally:
        if settings.environment == "local":
            spark.stop()


if __name__ == "__main__":
    main()
