"""Build analytics-ready aggregated tables from the cleansed transactions dataset.

Reads the table written by `src/transformation/cleanse.py` and produces five
independent, BI-tool-ready output tables:

- Daily revenue by region and category
- Monthly revenue by region and category
- Customer RFM (Recency, Frequency, Monetary) segmentation
- Top-N products by category
- Payment method distribution

Every transformation here is written and commented with a specific
performance decision in mind, since this module doubles as a reference for
explaining *why* each choice was made (not just what it does):

- **Broadcast joins**: a tiny (5-row) region dimension table is broadcast
  onto the full fact table once (`prepare_base_dataset`); tiny per-category
  and grand-total aggregates are broadcast onto their larger sibling
  aggregates to compute percentage-of-total columns without a shuffle
  (`compute_top_products_by_category`, `compute_payment_method_distribution`).
- **Caching**: the prepared base DataFrame is cached because all 5 output
  tables are independent actions over it; the per-customer RFM aggregate is
  cached separately because it's read by 3 `approxQuantile` calls plus the
  final scoring pass.
- **Partitioning**: `repartition(..., "customer_id")` on the base dataset
  is chosen specifically so the RFM groupBy can skip its shuffle entirely;
  `coalesce()` (not `repartition()`) is used before every write, since
  narrowing partition count for a handful of small output tables doesn't
  need a full reshuffle; `.partitionBy(...)` on write enables partition
  pruning for the columns BI dashboards are expected to filter on.
- **Narrow over wide**: RFM's three metrics are computed in a single
  `groupBy().agg()` instead of three separate groupBys joined back
  together; RFM quartile scoring uses `approxQuantile` + `when/otherwise`
  bucketing instead of `ntile()` over an unpartitioned `Window`, which
  would collapse the whole dataset into one partition to compute a global
  order.

Usage:
    python src/transformation/transform.py [--config config/config.yaml] [--env local|databricks]
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType
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
# Reference (dimension) data -- broadcast-join candidates
# --------------------------------------------------------------------------


def build_region_dimension(spark: SparkSession) -> DataFrame:
    """A tiny, static region -> sales-territory lookup table (5 rows).

    This is the textbook broadcast-join shape: a small dimension joined
    against a large fact table. A regular (sort-merge) join would shuffle
    the *entire* transactions table across the network just to attach a
    lookup value that only has 5 distinct outcomes. `F.broadcast()` instead
    ships this 5-row table to every executor once, so the join happens
    locally with no fact-table shuffle at all.
    """
    schema = StructType(
        [
            StructField("region", StringType(), False),
            StructField("sales_territory", StringType(), False),
        ]
    )
    rows = [
        ("North", "Territory 1"),
        ("East", "Territory 1"),
        ("South", "Territory 2"),
        ("West", "Territory 2"),
        ("Central", "Territory 3"),
    ]
    return spark.createDataFrame(rows, schema)


# --------------------------------------------------------------------------
# Base dataset preparation
# --------------------------------------------------------------------------


def prepare_base_dataset(df: DataFrame, spark: SparkSession, shuffle_partitions: int) -> DataFrame:
    """Repartition, enrich, and cache the cleansed dataset once for all 5 outputs.

    Every output table below is a separate action (a `.write`) over this
    same DataFrame, so the work done here -- repartitioning, the broadcast
    join, date-part extraction -- must happen exactly once rather than once
    per output table. That's what makes `.cache()` worthwhile here even
    though it costs memory: without it, Spark's lazy evaluation would
    silently redo this entire lineage from the raw read for each of the 5
    writes.
    """
    # `spark.sql.shuffle.partitions` (set on the session before this call)
    # controls how many partitions `groupBy("customer_id")` in
    # `compute_customer_rfm` will require. Repartitioning here by the same
    # key and count means that groupBy's required distribution is already
    # satisfied by the time it runs, so Spark's planner can skip that
    # shuffle (`Exchange`) entirely instead of paying for it twice.
    # `customer_id` (not region/category/date) is the repartition key
    # because RFM is the most expensive of the 5 pipelines (3 extra
    # `approxQuantile` passes) and has the highest cardinality (~6,000
    # customers) of any grouping key used below, so it benefits the most
    # and spreads data the most evenly. The other 4 tables group by
    # different columns and will shuffle on their own keys regardless --
    # this repartition doesn't help or hurt them.
    df = df.repartition(shuffle_partitions, "customer_id")

    # `how="left"`, not `inner`: canonicalization in cleanse.py falls back
    # to a title-cased best guess for a region value it doesn't recognize,
    # so an inner join here could theoretically drop a transaction row
    # entirely just because its region didn't map to one of our 5
    # territories. A left join keeps every row, leaving `sales_territory`
    # null in that edge case instead of silently losing the transaction.
    df = df.join(F.broadcast(build_region_dimension(spark)), on="region", how="left")

    # Built-in Catalyst expressions (F.year/F.month/F.date_format), not a
    # Python UDF: these run inside the JVM under whole-stage codegen. A UDF
    # equivalent would serialize every row out to a Python process and back
    # for what is otherwise a cheap, vectorizable date computation.
    df = (
        df.withColumn("purchase_year", F.year("purchase_date"))
        .withColumn("purchase_month", F.month("purchase_date"))
        .withColumn("purchase_year_month", F.date_format("purchase_date", "yyyy-MM"))
    )

    df = df.cache()
    row_count = df.count()  # materializes the cache in one job, up front,
    # rather than letting whichever output table is written first pay for it
    logger.info("base_dataset_prepared", extra={"row_count": row_count})
    return df


# --------------------------------------------------------------------------
# Daily / monthly revenue by region and category
# --------------------------------------------------------------------------


def compute_daily_revenue(df: DataFrame) -> DataFrame:
    """Revenue, units, and order counts per (day, region, category).

    Rows with a null `purchase_date` (left un-imputed by cleansing, since a
    transaction date can't be statistically guessed) are excluded here --
    there's no day to bucket them into -- but they still count fully toward
    `compute_payment_method_distribution` and RFM's frequency/monetary,
    which don't require a date.
    """
    return (
        df.filter(F.col("purchase_date").isNotNull())
        .groupBy("purchase_date", "region", "category")
        .agg(
            F.round(F.sum("total_amount"), 2).alias("revenue"),
            F.sum("quantity").alias("units_sold"),
            F.count(F.lit(1)).alias("num_transactions"),
            F.round(F.avg("total_amount"), 2).alias("avg_order_value"),
        )
        .orderBy("purchase_date", "region", "category")
    )


def compute_monthly_revenue(df: DataFrame) -> DataFrame:
    """Same shape as `compute_daily_revenue`, bucketed by calendar month.

    Groups on the `purchase_year`/`purchase_month` columns computed once in
    `prepare_base_dataset` rather than re-deriving them here, so the date
    parsing isn't repeated per output table.
    """
    return (
        df.filter(F.col("purchase_date").isNotNull())
        .groupBy("purchase_year", "purchase_month", "purchase_year_month", "region", "category")
        .agg(
            F.round(F.sum("total_amount"), 2).alias("revenue"),
            F.sum("quantity").alias("units_sold"),
            F.count(F.lit(1)).alias("num_transactions"),
            F.round(F.avg("total_amount"), 2).alias("avg_order_value"),
        )
        .orderBy("purchase_year", "purchase_month", "region", "category")
    )


# --------------------------------------------------------------------------
# Customer RFM segmentation
# --------------------------------------------------------------------------


def _quartile_score_column(column: Column, cutpoints: list[float], ascending: bool) -> Column:
    """Bucket `column` into a 1-4 score using precomputed quartile cutpoints.

    Cutpoints are produced by `approxQuantile` -- a distributed,
    single-pass approximate-percentile algorithm -- rather than `ntile(4)`
    over a `Window.orderBy(...)` with no `partitionBy`. An unpartitioned
    window forces Spark to shuffle every single row into one partition to
    compute a global rank, serializing what should be parallel work onto a
    single task: a classic PySpark performance trap. Once the 3 cutpoints
    are plain Python floats, scoring every row against them is a narrow,
    embarrassingly parallel `when/otherwise` chain with no shuffle at all.
    """
    q1, q2, q3 = cutpoints
    if ascending:  # lower raw value = better score (recency: fewer days is better)
        return F.when(column <= q1, 4).when(column <= q2, 3).when(column <= q3, 2).otherwise(1)
    return F.when(column <= q1, 1).when(column <= q2, 2).when(column <= q3, 3).otherwise(4)


def compute_customer_rfm(df: DataFrame) -> DataFrame:
    """Recency (days since last purchase), Frequency, and Monetary value per customer.

    Excludes the `UNKNOWN_CUSTOMER` sentinel (cleansing's fill-in for a
    missing `customer_id`, not a real customer) and rows with a null
    `purchase_date` (recency can't be computed for them, and keeping them
    for frequency/monetary alone would weight some customers against an
    incomplete transaction history). "Today" for recency is defined as one
    day after the dataset's own latest transaction, not `current_date()`,
    since this dataset is historical (2023-2025) rather than live.
    """
    known = df.filter((F.col("customer_id") != "UNKNOWN_CUSTOMER") & F.col("purchase_date").isNotNull())

    max_date = known.agg(F.max("purchase_date").alias("max_date")).first()["max_date"]
    snapshot_date = max_date + timedelta(days=1)

    # Recency, Frequency, and Monetary in a *single* groupBy().agg() call:
    # all three share the same grouping key (customer_id), so computing
    # them together costs one shuffle. Computing them as three separate
    # groupBys and joining the results back together would cost three
    # shuffles plus two joins for the exact same numbers.
    rfm = known.groupBy("customer_id").agg(
        F.datediff(F.lit(snapshot_date), F.max("purchase_date")).alias("recency_days"),
        F.count(F.lit(1)).alias("frequency"),
        F.round(F.sum("total_amount"), 2).alias("monetary"),
    )
    # Cached: this small (~thousands-of-rows) aggregate is read again by 3
    # separate approxQuantile actions below plus the final scoring pass.
    # Without caching, each of those 4 actions would re-run the groupBy
    # shuffle above from scratch.
    rfm = rfm.cache()

    recency_cutpoints = rfm.approxQuantile("recency_days", [0.25, 0.5, 0.75], 0.01)
    frequency_cutpoints = rfm.approxQuantile("frequency", [0.25, 0.5, 0.75], 0.01)
    monetary_cutpoints = rfm.approxQuantile("monetary", [0.25, 0.5, 0.75], 0.01)

    scored = (
        rfm.withColumn("recency_score", _quartile_score_column(F.col("recency_days"), recency_cutpoints, ascending=True))
        .withColumn("frequency_score", _quartile_score_column(F.col("frequency"), frequency_cutpoints, ascending=False))
        .withColumn("monetary_score", _quartile_score_column(F.col("monetary"), monetary_cutpoints, ascending=False))
    )

    total_score = F.col("recency_score") + F.col("frequency_score") + F.col("monetary_score")
    result = scored.withColumn(
        "rfm_score",
        F.concat_ws(
            "",
            F.col("recency_score").cast("string"),
            F.col("frequency_score").cast("string"),
            F.col("monetary_score").cast("string"),
        ),
    ).withColumn(
        "customer_segment_label",
        F.when(total_score >= 10, "Champions")
        .when(total_score >= 8, "Loyal Customers")
        .when(total_score >= 6, "Potential Loyalists")
        .when(total_score >= 4, "At Risk")
        .otherwise("Lost / Hibernating"),
    )

    # `rfm` is deliberately left cached rather than `unpersist()`-ed here:
    # `result` is still a lazy transformation of it, not yet materialized,
    # since the caller writes it out later. Unpersisting now would evict
    # the cache before it's ever actually used, defeating the point of
    # caching it in the first place. It's small (a few thousand rows), so
    # leaving it cached for the rest of the Spark session is cheap.
    return result


# --------------------------------------------------------------------------
# Top products by category
# --------------------------------------------------------------------------


def compute_top_products_by_category(df: DataFrame, top_n: int) -> DataFrame:
    """The top `top_n` products by revenue within each category.

    `category_totals` has at most 10 rows (one per canonical category), so
    it's broadcast onto `product_agg` to compute each product's share of
    its category's revenue without a shuffle join. Ranking uses
    `Window.partitionBy("category")` -- unlike the unpartitioned window
    avoided in `_quartile_score_column`, partitioning by category here
    means the sort Spark performs for `row_number()` runs independently
    and in parallel per category, not collapsed onto a single task.
    """
    product_agg = df.groupBy("category", "product_id").agg(
        F.round(F.sum("total_amount"), 2).alias("revenue"),
        F.sum("quantity").alias("units_sold"),
        F.count(F.lit(1)).alias("num_orders"),
    )

    category_totals = product_agg.groupBy("category").agg(F.sum("revenue").alias("category_revenue"))

    ranking_window = Window.partitionBy("category").orderBy(F.desc("revenue"), F.asc("product_id"))

    return (
        product_agg.join(F.broadcast(category_totals), on="category", how="inner")
        .withColumn("pct_of_category_revenue", F.round(F.col("revenue") / F.col("category_revenue") * 100, 2))
        .withColumn("category_rank", F.row_number().over(ranking_window))
        .filter(F.col("category_rank") <= top_n)
        .drop("category_revenue")
        .orderBy("category", "category_rank")
    )


# --------------------------------------------------------------------------
# Payment method distribution
# --------------------------------------------------------------------------


def compute_payment_method_distribution(df: DataFrame) -> DataFrame:
    """Transaction count, revenue, and share-of-total per payment method.

    `grand_totals` is exactly one row, so it's attached with `crossJoin`
    (the correct, explicit API for joining against a single-row DataFrame
    with no shared key) rather than a keyed join. Broadcasting it keeps
    the percentage computation inside the Spark plan instead of collecting
    a scalar back to the driver with `.first()` and re-injecting it via
    `F.lit(...)` -- functionally similar for a single number, but this
    mirrors the same broadcast pattern used for `category_totals` above,
    which *does* need a real per-category value the driver can't collapse
    to one literal.
    """
    method_agg = df.groupBy("payment_method").agg(
        F.count(F.lit(1)).alias("num_transactions"),
        F.round(F.sum("total_amount"), 2).alias("revenue"),
        F.round(F.avg("total_amount"), 2).alias("avg_order_value"),
    )

    grand_totals = df.agg(
        F.count(F.lit(1)).alias("total_transactions"),
        F.sum("total_amount").alias("total_revenue"),
    )

    return (
        method_agg.crossJoin(F.broadcast(grand_totals))
        .withColumn(
            "pct_of_transactions", F.round(F.col("num_transactions") / F.col("total_transactions") * 100, 2)
        )
        .withColumn("pct_of_revenue", F.round(F.col("revenue") / F.col("total_revenue") * 100, 2))
        .drop("total_transactions", "total_revenue")
        .orderBy(F.desc("revenue"))
    )


# --------------------------------------------------------------------------
# Output writing
# --------------------------------------------------------------------------


def _write_table(
    df: DataFrame,
    output_path: str,
    output_format: str,
    write_mode: str,
    file_count: int,
    partition_by: Optional[str],
) -> None:
    """Write one aggregated output table.

    `coalesce(file_count)`, not `repartition(file_count)`: these are all
    small, already-aggregated tables (from a handful up to a few thousand
    rows), so all we want is fewer output files -- we don't need the data
    rebalanced. `coalesce` merges existing partitions locally without a
    shuffle; `repartition` would trigger one just to end up in the same
    place. `partition_by`, when given, is a low-cardinality column
    (region/category) that dashboards are expected to filter on, so a
    downstream reader can prune to the matching folder instead of scanning
    the whole table.
    """
    logger.info(
        "writing_table",
        extra={"output_path": output_path, "partition_by": partition_by, "file_count": file_count},
    )
    writer = df.coalesce(file_count).write.format(output_format).mode(write_mode)
    if partition_by:
        writer = writer.partitionBy(partition_by)
    writer.save(output_path)
    logger.info("table_written", extra={"output_path": output_path})


# --------------------------------------------------------------------------
# Run settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AggregationRunSettings:
    environment: str
    app_name: str
    input_path: str
    input_format: str
    output_format: str
    write_mode: str
    shuffle_partitions: int
    top_products_per_category: int
    output_paths: dict[str, str]
    log_level: str


def load_run_settings(config_path: Path, env_override: Optional[str] = None) -> AggregationRunSettings:
    raw_config = load_yaml_config(config_path)
    environment, _, processed_data_dir = resolve_environment(raw_config, env_override)
    agg_cfg = raw_config["aggregation"]
    processed_data_dir = processed_data_dir.rstrip("/")

    output_paths = {
        key: f"{processed_data_dir}/{table_name}" for key, table_name in agg_cfg["output_tables"].items()
    }

    return AggregationRunSettings(
        environment=environment,
        app_name=f"{raw_config['spark']['app_name']}-aggregation",
        input_path=f"{processed_data_dir}/{agg_cfg['input_table_name']}",
        input_format=raw_config["cleansing"]["output_format"],
        output_format=agg_cfg["output_format"],
        write_mode=agg_cfg["write_mode"],
        shuffle_partitions=int(agg_cfg.get("shuffle_partitions", 8)),
        top_products_per_category=int(agg_cfg.get("top_products_per_category", 10)),
        output_paths=output_paths,
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

    spark = build_spark_session(
        settings.environment,
        settings.app_name,
        use_delta=settings.input_format == "delta" or settings.output_format == "delta",
    )
    dbutils = get_dbutils(spark) if settings.environment == "databricks" else None

    # Default is 200, sized for large clusters. Left at the default here,
    # every aggregation below would split its output into 200 mostly-empty
    # partitions for a dataset of tens of thousands of rows -- tasks would
    # spend more time on scheduling overhead than on real work. Set once,
    # up front, so every groupBy/join shuffle in this module uses it.
    spark.conf.set("spark.sql.shuffle.partitions", settings.shuffle_partitions)

    try:
        logger.info(
            "aggregation_started",
            extra={"input_path": settings.input_path, "output_paths": settings.output_paths},
        )

        if not path_exists(settings.input_path, settings.environment, dbutils):
            raise FileNotFoundError(f"Input path does not exist: {settings.input_path}")

        raw_df = spark.read.format(settings.input_format).load(settings.input_path)
        df = prepare_base_dataset(raw_df, spark, settings.shuffle_partitions)

        _write_table(
            compute_daily_revenue(df),
            settings.output_paths["daily_revenue"],
            settings.output_format,
            settings.write_mode,
            file_count=4,
            partition_by="region",
        )
        _write_table(
            compute_monthly_revenue(df),
            settings.output_paths["monthly_revenue"],
            settings.output_format,
            settings.write_mode,
            file_count=4,
            partition_by="region",
        )
        _write_table(
            compute_customer_rfm(df),
            settings.output_paths["customer_rfm"],
            settings.output_format,
            settings.write_mode,
            file_count=1,
            partition_by=None,
        )
        _write_table(
            compute_top_products_by_category(df, settings.top_products_per_category),
            settings.output_paths["top_products"],
            settings.output_format,
            settings.write_mode,
            file_count=4,
            partition_by="category",
        )
        _write_table(
            compute_payment_method_distribution(df),
            settings.output_paths["payment_distribution"],
            settings.output_format,
            settings.write_mode,
            file_count=1,
            partition_by=None,
        )

        df.unpersist()
        logger.info("aggregation_completed")
    except Exception:
        logger.exception("aggregation_failed")
        raise
    finally:
        if settings.environment == "local":
            spark.stop()


if __name__ == "__main__":
    main()
