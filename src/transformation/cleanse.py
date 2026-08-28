"""Cleanse the ingested e-commerce transactions dataset.

Applies null imputation, deduplication, date standardization, IQR-based
outlier treatment, and data-type/range enforcement to the raw table written
by `src/ingestion/ingest.py`, then writes the result back to the processed
data layer under a separate table name.

Every step is a standalone, independently testable function of the shape
`(DataFrame, **params) -> (DataFrame, metrics_dict)`. `DataCleanser` just
threads a DataFrame and a cumulative report through calls to them so the
whole pipeline can be run and logged as one chainable unit:

    cleansed_df = (
        DataCleanser(raw_df)
        .deduplicate()
        .standardize_dates()
        .enforce_data_types()
        .enforce_valid_ranges()
        .treat_outliers()
        .impute_nulls()
        .df
    )

Design decisions worth knowing before extending this module:

- Every step preserves row count except `deduplicate` -- outliers and
  range violations are *treated* (capped or nulled) in place rather than
  dropped, so no transaction silently disappears from the dataset.
- `purchase_date`, `product_id`, and `transaction_id` are never imputed:
  there's no statistically sound way to guess a transaction's date or
  identity, so nulls there are left in place and only reported.

Usage:
    python src/transformation/cleanse.py [--config config/config.yaml] [--env local|databricks]
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.ingest import get_transaction_schema  # noqa: E402
from src.utils.config import load_yaml_config, resolve_environment  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402
from src.utils.spark_session import build_spark_session, get_dbutils, path_exists  # noqa: E402

DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"
SCHEMA_COLUMN_ORDER = [f.name for f in get_transaction_schema().fields]

logger = logging.getLogger(__name__)

DATE_PATTERNS = [
    "yyyy-MM-dd",
    "yyyy/MM/dd",
    "MM/dd/yyyy",
    "dd-MM-yyyy",
    "dd MMM yyyy",
    "MMMM dd, yyyy",
    "yyyy-MM-dd'T'HH:mm:ss",
    "MM-dd-yyyy HH:mm",
]

CATEGORICAL_COLUMNS = ["category", "region", "payment_method", "customer_segment"]
CANONICAL_VALUES = {
    "region": ["North", "South", "East", "West", "Central"],
    "payment_method": [
        "Credit Card",
        "Debit Card",
        "PayPal",
        "UPI",
        "Net Banking",
        "Cash on Delivery",
        "Gift Card",
    ],
    "customer_segment": ["New", "Regular", "Premium", "VIP"],
    "category": [
        "Electronics",
        "Clothing",
        "Home & Kitchen",
        "Books",
        "Beauty & Personal Care",
        "Sports & Outdoors",
        "Toys & Games",
        "Grocery",
        "Automotive",
        "Health & Wellness",
    ],
}
NUMERIC_MEDIAN_COLUMNS = ["quantity", "customer_age"]
CATEGORY_GROUPED_MEDIAN_COLUMNS = {"unit_price": "category", "total_amount": "category"}
MODE_COLUMNS = ["category", "region", "payment_method", "customer_segment"]
SENTINEL_COLUMNS = {"customer_id": "UNKNOWN_CUSTOMER"}
NOT_IMPUTED_COLUMNS = ["purchase_date", "product_id", "transaction_id"]


# --------------------------------------------------------------------------
# Deduplication
# --------------------------------------------------------------------------


def deduplicate(df: DataFrame) -> tuple[DataFrame, dict]:
    """Remove exact duplicate rows, then resolve duplicate `transaction_id`s.

    Two distinct duplicate patterns are handled: fully identical rows
    (`dropDuplicates`), and rows that share a `transaction_id` but disagree
    on other fields (e.g. a re-submitted transaction with a different
    quantity) -- for those, the most complete row (fewest nulls) is kept.
    """
    rows_before = df.count()

    df = df.dropDuplicates()
    rows_after_exact = df.count()
    exact_duplicates_removed = rows_before - rows_after_exact

    completeness = sum((F.when(F.col(c).isNotNull(), 1).otherwise(0) for c in df.columns), F.lit(0))
    ranked = df.withColumn("__completeness", completeness).withColumn(
        "__row_num",
        F.row_number().over(
            Window.partitionBy("transaction_id").orderBy(
                F.desc("__completeness"),
                F.desc("total_amount"),
                F.monotonically_increasing_id().desc(),
            )
        ),
    )
    df = ranked.filter(F.col("__row_num") == 1).drop("__completeness", "__row_num")
    rows_after = df.count()
    duplicate_transaction_id_rows_removed = rows_after_exact - rows_after

    metrics = {
        "rows_before": rows_before,
        "exact_duplicates_removed": exact_duplicates_removed,
        "duplicate_transaction_id_rows_removed": duplicate_transaction_id_rows_removed,
        "rows_after": rows_after,
    }
    logger.info("deduplication_completed", extra=metrics)
    return df, metrics


# --------------------------------------------------------------------------
# Date standardization
# --------------------------------------------------------------------------


def standardize_dates(df: DataFrame, column: str = "purchase_date") -> tuple[DataFrame, dict]:
    """Parse `column`'s mixed string date formats into a single DateType.

    Tries each pattern in `DATE_PATTERNS` in turn via
    `coalesce(try_to_date(...))`. `try_to_date` (rather than `to_date`) is
    required here because Spark runs in ANSI mode by default: plain
    `to_date` *raises* on a non-matching pattern instead of returning null,
    which would break the coalesce-over-candidates approach entirely.
    Values that don't fully match any known pattern (garbage placeholders
    like "N/A" or genuinely invalid dates like "31/02/2024") become null
    rather than raising, since ingestion already lets malformed dates
    through as plain strings.
    """
    original_null_count = df.filter(F.col(column).isNull()).count()

    trimmed = F.trim(F.col(column))
    blank_as_null = F.when(trimmed == "", None).otherwise(trimmed)
    parsed = F.coalesce(*[F.try_to_date(blank_as_null, pattern) for pattern in DATE_PATTERNS])

    df = df.withColumn(column, parsed)

    final_null_count = df.filter(F.col(column).isNull()).count()
    metrics = {
        "original_null_count": original_null_count,
        "unparseable_count": final_null_count - original_null_count,
        "final_null_count": final_null_count,
    }
    logger.info("date_standardization_completed", extra=metrics)
    return df, metrics


# --------------------------------------------------------------------------
# Data type enforcement
# --------------------------------------------------------------------------


def _canonicalization_map(canonical_values: list[str]) -> Column:
    """Build a `MapType` literal from a normalized lookup key to its canonical spelling.

    The lookup key strips whitespace/underscores and lowercases, so
    "credit_card", "CREDIT CARD", and "  Credit Card  " all resolve to the
    same key as canonical value "Credit Card".
    """
    pairs = [(re.sub(r"[\s_]+", "", value).lower(), value) for value in canonical_values]
    return F.create_map(*[F.lit(x) for pair in pairs for x in pair])


def _canonicalize_column(column: Column, canonical_values: list[str]) -> Column:
    """Standardize `column` against a known, fixed vocabulary.

    A pure case/whitespace/punctuation heuristic (e.g. blanket `initcap`)
    would mangle values that are already correct but not "Title Case" by
    convention -- an acronym like "UPI" becomes "Upi", a brand name like
    "PayPal" becomes "Paypal". Matching against the actual known-valid
    values instead avoids that: only genuine noise gets rewritten, and
    every recognized spelling maps back to itself unchanged. A value that
    matches none of them (a genuinely new/unexpected category) falls back
    to a title-cased best guess rather than being dropped.
    """
    normalized_key = F.lower(F.regexp_replace(F.trim(column), r"[\s_]+", ""))
    lookup = _canonicalization_map(canonical_values)
    fallback = F.initcap(F.regexp_replace(F.trim(column), "_", " "))
    return F.coalesce(lookup[normalized_key], fallback)


def enforce_data_types(df: DataFrame) -> tuple[DataFrame, dict]:
    """Cast numeric columns to their proper types and standardize text columns."""
    orig_aliases = {c: f"__orig_{c}" for c in CATEGORICAL_COLUMNS}
    df = df.select("*", *[F.col(c).alias(alias) for c, alias in orig_aliases.items()])

    for c in CATEGORICAL_COLUMNS:
        df = df.withColumn(c, _canonicalize_column(F.col(c), CANONICAL_VALUES[c]))

    change_counts = (
        df.agg(
            *[
                F.sum(F.when(~F.col(c).eqNullSafe(F.col(orig_aliases[c])), 1).otherwise(0)).alias(c)
                for c in CATEGORICAL_COLUMNS
            ]
        )
        .first()
        .asDict()
    )
    df = df.drop(*orig_aliases.values())

    df = (
        df.withColumn("quantity", F.round(F.col("quantity")).cast(IntegerType()))
        .withColumn("customer_age", F.round(F.col("customer_age")).cast(IntegerType()))
        .withColumn("unit_price", F.round(F.col("unit_price"), 2))
        .withColumn("total_amount", F.round(F.col("total_amount"), 2))
    )

    metrics = {f"{c}_values_normalized": int(change_counts[c] or 0) for c in CATEGORICAL_COLUMNS}
    logger.info("data_types_enforced", extra=metrics)
    return df, metrics


# --------------------------------------------------------------------------
# Referential / domain range checks
# --------------------------------------------------------------------------


def enforce_valid_ranges(
    df: DataFrame, age_min: int = 18, age_max: int = 100
) -> tuple[DataFrame, dict]:
    """Null out values that are technically well-typed but domain-invalid.

    E.g. a `customer_age` of -5 or 200 parses fine as an integer but can
    never be real; a `quantity` <= 0 or a negative price/amount indicates a
    data entry error rather than a legitimate transaction. These are
    distinct from statistical outliers (see `treat_outliers_iqr`): an
    impossible value is wrong at any frequency, whereas a statistical
    outlier is merely unusual.
    """
    invalid_age = (F.col("customer_age") < age_min) | (F.col("customer_age") > age_max)
    invalid_quantity = F.col("quantity") <= 0
    invalid_unit_price = F.col("unit_price") < 0
    invalid_total_amount = F.col("total_amount") < 0

    counts = (
        df.agg(
            F.sum(F.when(invalid_age, 1).otherwise(0)).alias("customer_age"),
            F.sum(F.when(invalid_quantity, 1).otherwise(0)).alias("quantity"),
            F.sum(F.when(invalid_unit_price, 1).otherwise(0)).alias("unit_price"),
            F.sum(F.when(invalid_total_amount, 1).otherwise(0)).alias("total_amount"),
        )
        .first()
        .asDict()
    )

    df = (
        df.withColumn("customer_age", F.when(invalid_age, None).otherwise(F.col("customer_age")))
        .withColumn("quantity", F.when(invalid_quantity, None).otherwise(F.col("quantity")))
        .withColumn("unit_price", F.when(invalid_unit_price, None).otherwise(F.col("unit_price")))
        .withColumn(
            "total_amount", F.when(invalid_total_amount, None).otherwise(F.col("total_amount"))
        )
    )

    metrics = {f"{k}_out_of_range": int(v or 0) for k, v in counts.items()}
    logger.info("valid_ranges_enforced", extra=metrics)
    return df, metrics


# --------------------------------------------------------------------------
# IQR outlier detection and treatment
# --------------------------------------------------------------------------


def treat_outliers_iqr(
    df: DataFrame, columns: tuple[str, ...] = ("total_amount", "quantity"), multiplier: float = 1.5
) -> tuple[DataFrame, dict]:
    """Cap statistically extreme values in `columns` to their IQR fences.

    For each column: `bounds = [Q1 - multiplier*IQR, Q3 + multiplier*IQR]`.
    Values outside the bounds are capped (winsorized) to the nearest fence
    rather than dropped, so a genuinely large-but-real order doesn't cost a
    whole row. Nulls pass through untouched -- they're handled separately by
    `impute_nulls`. Run this *after* `enforce_valid_ranges` so impossible
    values (e.g. negative quantities) don't skew the quartiles.
    """
    column_types = dict(df.dtypes)
    bounds: dict[str, dict] = {}

    for column in columns:
        q1, q3 = df.approxQuantile(column, [0.25, 0.75], 0.01)
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        if column_types[column] in ("int", "bigint"):
            lower, upper = int(round(lower)), int(round(upper))
        else:
            lower, upper = round(lower, 2), round(upper, 2)
        bounds[column] = {"q1": q1, "q3": q3, "lower_bound": lower, "upper_bound": upper}

    counts = (
        df.agg(
            *[
                agg
                for column in columns
                for agg in (
                    F.sum(F.when(F.col(column) < bounds[column]["lower_bound"], 1).otherwise(0)).alias(
                        f"{column}__low"
                    ),
                    F.sum(F.when(F.col(column) > bounds[column]["upper_bound"], 1).otherwise(0)).alias(
                        f"{column}__high"
                    ),
                )
            ]
        )
        .first()
        .asDict()
    )

    for column in columns:
        lower, upper = bounds[column]["lower_bound"], bounds[column]["upper_bound"]
        df = df.withColumn(
            column,
            F.when(F.col(column) < lower, F.lit(lower))
            .when(F.col(column) > upper, F.lit(upper))
            .otherwise(F.col(column)),
        )
        if column_types[column] in ("int", "bigint"):
            df = df.withColumn(column, F.col(column).cast(IntegerType()))
        bounds[column]["capped_low"] = int(counts[f"{column}__low"] or 0)
        bounds[column]["capped_high"] = int(counts[f"{column}__high"] or 0)

    logger.info("outliers_treated", extra={"columns": bounds})
    return df, bounds


# --------------------------------------------------------------------------
# Null imputation
# --------------------------------------------------------------------------


def impute_nulls(df: DataFrame) -> tuple[DataFrame, dict]:
    """Fill nulls using a strategy chosen per column's statistical role.

    - Discrete numeric (`quantity`, `customer_age`): global median.
    - Continuous numeric that varies by category (`unit_price`,
      `total_amount`): median within the same `category`, falling back to
      the global median for rows where `category` is also null.
    - Categorical (`category`, `region`, `payment_method`,
      `customer_segment`): mode (most frequent non-null value).
    - `customer_id`: not statistically imputable -- filled with a
      `UNKNOWN_CUSTOMER` sentinel so the row stays usable in aggregates.
    - `purchase_date`, `product_id`, `transaction_id`: left null and only
      reported; fabricating an identifier or a date would be misleading.
    """
    tracked_columns = (
        NUMERIC_MEDIAN_COLUMNS
        + list(CATEGORY_GROUPED_MEDIAN_COLUMNS)
        + MODE_COLUMNS
        + list(SENTINEL_COLUMNS)
        + NOT_IMPUTED_COLUMNS
    )
    null_counts = (
        df.agg(*[F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in tracked_columns])
        .first()
        .asDict()
    )
    null_counts = {k: int(v or 0) for k, v in null_counts.items()}

    metrics: dict[str, dict] = {}
    column_types = dict(df.dtypes)

    for column in NUMERIC_MEDIAN_COLUMNS:
        if null_counts[column]:
            median_value = df.approxQuantile(column, [0.5], 0.01)[0]
            fill_value = int(median_value) if column_types[column] in ("int", "bigint") else median_value
            df = df.fillna({column: fill_value})
        metrics[column] = {"strategy": "median", "nulls_imputed": null_counts[column]}

    for column, group_col in CATEGORY_GROUPED_MEDIAN_COLUMNS.items():
        if null_counts[column]:
            global_median = df.approxQuantile(column, [0.5], 0.01)[0]
            group_medians = (
                df.filter(F.col(column).isNotNull() & F.col(group_col).isNotNull())
                .groupBy(group_col)
                .agg(F.expr(f"percentile_approx({column}, 0.5)").alias("__group_median"))
            )
            df = df.join(group_medians, on=group_col, how="left")
            df = df.withColumn(
                column, F.coalesce(F.col(column), F.col("__group_median"), F.lit(global_median))
            ).drop("__group_median")
        metrics[column] = {"strategy": f"median_by_{group_col}", "nulls_imputed": null_counts[column]}

    for column in MODE_COLUMNS:
        if null_counts[column]:
            mode_row = (
                df.filter(F.col(column).isNotNull())
                .groupBy(column)
                .count()
                .orderBy(F.desc("count"))
                .first()
            )
            mode_value = mode_row[0] if mode_row else "Unknown"
            df = df.fillna({column: mode_value})
        metrics[column] = {"strategy": "mode", "nulls_imputed": null_counts[column]}

    for column, sentinel in SENTINEL_COLUMNS.items():
        if null_counts[column]:
            df = df.fillna({column: sentinel})
        metrics[column] = {
            "strategy": "sentinel",
            "nulls_imputed": null_counts[column],
            "fill_value": sentinel,
        }

    for column in NOT_IMPUTED_COLUMNS:
        metrics[column] = {"strategy": "not_imputed", "nulls_remaining": null_counts[column]}

    df = df.select(*SCHEMA_COLUMN_ORDER)

    logger.info("null_imputation_completed", extra={"columns": metrics})
    return df, metrics


# --------------------------------------------------------------------------
# Chainable pipeline
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CleansingConfig:
    customer_age_min: int = 18
    customer_age_max: int = 100
    iqr_multiplier: float = 1.5
    outlier_columns: tuple[str, ...] = field(default_factory=lambda: ("total_amount", "quantity"))


class DataCleanser:
    """Chainable PySpark cleansing pipeline for the transactions dataset.

    Wraps the module-level step functions, threading a DataFrame and a
    cumulative `report` dict through each call so a full run's per-step
    metrics are available afterwards (e.g. for a data-quality dashboard).
    """

    def __init__(self, df: DataFrame, config: Optional[CleansingConfig] = None) -> None:
        self.df = df
        self.config = config or CleansingConfig()
        self.report: dict[str, dict] = {}

    def deduplicate(self) -> "DataCleanser":
        self.df, self.report["deduplicate"] = deduplicate(self.df)
        return self

    def standardize_dates(self, column: str = "purchase_date") -> "DataCleanser":
        self.df, self.report["standardize_dates"] = standardize_dates(self.df, column)
        return self

    def enforce_data_types(self) -> "DataCleanser":
        self.df, self.report["enforce_data_types"] = enforce_data_types(self.df)
        return self

    def enforce_valid_ranges(self) -> "DataCleanser":
        self.df, self.report["enforce_valid_ranges"] = enforce_valid_ranges(
            self.df, self.config.customer_age_min, self.config.customer_age_max
        )
        return self

    def treat_outliers(self) -> "DataCleanser":
        self.df, self.report["treat_outliers"] = treat_outliers_iqr(
            self.df, self.config.outlier_columns, self.config.iqr_multiplier
        )
        return self

    def impute_nulls(self) -> "DataCleanser":
        self.df, self.report["impute_nulls"] = impute_nulls(self.df)
        return self

    def run(self) -> DataFrame:
        """Run every step in the recommended order and return the result.

        Order matters: dates/types are standardized before range checks so
        those checks see real numbers rather than string artifacts; range
        checks run before outlier treatment so impossible values don't
        distort the IQR quartiles; imputation runs last so it can fill
        nulls produced by every earlier step, not just the ones present in
        the raw input.
        """
        return (
            self.deduplicate()
            .standardize_dates()
            .enforce_data_types()
            .enforce_valid_ranges()
            .treat_outliers()
            .impute_nulls()
            .df
        )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CleansingRunSettings:
    environment: str
    app_name: str
    input_path: str
    input_format: str
    output_path: str
    output_format: str
    write_mode: str
    log_level: str
    cleansing_config: CleansingConfig


def load_run_settings(config_path: Path, env_override: Optional[str] = None) -> CleansingRunSettings:
    raw_config = load_yaml_config(config_path)
    environment, _, processed_data_dir = resolve_environment(raw_config, env_override)
    cleansing_cfg = raw_config["cleansing"]

    return CleansingRunSettings(
        environment=environment,
        app_name=f"{raw_config['spark']['app_name']}-cleansing",
        input_path=f"{processed_data_dir.rstrip('/')}/{cleansing_cfg['input_table_name']}",
        input_format=raw_config["ingestion"]["output_format"],
        output_path=f"{processed_data_dir.rstrip('/')}/{cleansing_cfg['output_table_name']}",
        output_format=cleansing_cfg["output_format"],
        write_mode=cleansing_cfg["write_mode"],
        log_level=raw_config.get("logging", {}).get("level", "INFO"),
        cleansing_config=CleansingConfig(
            customer_age_min=cleansing_cfg.get("customer_age_min", 18),
            customer_age_max=cleansing_cfg.get("customer_age_max", 100),
            iqr_multiplier=cleansing_cfg.get("iqr_multiplier", 1.5),
            outlier_columns=tuple(
                cleansing_cfg.get("outlier_columns", ["total_amount", "quantity"])
            ),
        ),
    )


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

    try:
        logger.info(
            "cleansing_started",
            extra={"input_path": settings.input_path, "output_path": settings.output_path},
        )

        if not path_exists(settings.input_path, settings.environment, dbutils):
            raise FileNotFoundError(f"Input path does not exist: {settings.input_path}")

        df = spark.read.format(settings.input_format).load(settings.input_path)
        rows_in = df.count()
        logger.info("input_loaded", extra={"row_count": rows_in})

        cleanser = DataCleanser(df, settings.cleansing_config)
        cleaned_df = cleanser.run()

        rows_out = cleaned_df.count()
        logger.info("cleansing_completed", extra={"rows_in": rows_in, "rows_out": rows_out})

        (
            cleaned_df.write.format(settings.output_format)
            .mode(settings.write_mode)
            .save(settings.output_path)
        )
        logger.info("write_completed", extra={"output_path": settings.output_path})
    except Exception:
        logger.exception("cleansing_failed")
        raise
    finally:
        if settings.environment == "local":
            spark.stop()


if __name__ == "__main__":
    main()
