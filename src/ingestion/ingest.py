"""Ingest raw e-commerce transaction CSV data into the processed data layer.

Reads a CSV file against an explicit schema, validates it, and writes the
result to `data/processed/` (or a Databricks mount/DBFS path) as Delta or
Parquet. Designed to run unchanged in two contexts:

- Locally, via `python src/ingestion/ingest.py`, using a local `local[*]`
  Spark session and the local filesystem.
- On Databricks (notebook, job, or Repos checkout), attaching to the
  cluster's existing Spark session and using `dbutils` for path checks.

Which context is used is controlled by `environment` in `config/config.yaml`
(or the `--env` CLI override) -- no code changes needed to move between them.

Usage:
    python src/ingestion/ingest.py [--config config/config.yaml] [--env local|databricks]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import load_yaml_config, resolve_environment  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402
from src.utils.spark_session import build_spark_session, get_dbutils, path_exists  # noqa: E402

DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Raised when ingested data does not conform to the expected schema."""


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestionSettings:
    """Resolved, environment-aware settings for a single ingestion run."""

    environment: str
    app_name: str
    raw_data_dir: str
    processed_data_dir: str
    input_filename: str
    output_table_name: str
    output_format: str
    write_mode: str
    log_level: str

    @property
    def input_path(self) -> str:
        return f"{self.raw_data_dir.rstrip('/')}/{self.input_filename}"

    @property
    def output_path(self) -> str:
        return f"{self.processed_data_dir.rstrip('/')}/{self.output_table_name}"


def load_settings(config_path: Path, env_override: Optional[str] = None) -> IngestionSettings:
    """Load `config.yaml` and resolve it into environment-specific settings."""
    raw_config = load_yaml_config(config_path)
    environment, raw_data_dir, processed_data_dir = resolve_environment(raw_config, env_override)
    ingestion_cfg = raw_config["ingestion"]

    return IngestionSettings(
        environment=environment,
        app_name=raw_config["spark"]["app_name"],
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        input_filename=ingestion_cfg["input_filename"],
        output_table_name=ingestion_cfg["output_table_name"],
        output_format=ingestion_cfg["output_format"],
        write_mode=ingestion_cfg["write_mode"],
        log_level=raw_config.get("logging", {}).get("level", "INFO"),
    )


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


def get_transaction_schema() -> StructType:
    """Explicit schema for raw transaction records (schema-on-read).

    Numeric fields prone to nulls, outliers, or malformed values in the raw
    data (`quantity`, `unit_price`, `total_amount`, `customer_age`) are typed
    as `DoubleType` rather than `IntegerType`/`LongType` so that ingestion
    never fails on a null or an out-of-range value -- cleansing of those
    values is the transformation layer's responsibility, not ingestion's.
    `purchase_date` is kept as `StringType` because the raw source mixes
    multiple date formats and unparseable placeholders.
    """
    return StructType(
        [
            StructField("transaction_id", StringType(), True),
            StructField("customer_id", StringType(), True),
            StructField("product_id", StringType(), True),
            StructField("category", StringType(), True),
            StructField("purchase_date", StringType(), True),
            StructField("quantity", DoubleType(), True),
            StructField("unit_price", DoubleType(), True),
            StructField("total_amount", DoubleType(), True),
            StructField("region", StringType(), True),
            StructField("payment_method", StringType(), True),
            StructField("customer_age", DoubleType(), True),
            StructField("customer_segment", StringType(), True),
        ]
    )


def validate_schema(df: DataFrame, expected_schema: StructType) -> None:
    """Validate `df`'s schema against `expected_schema` by name and type.

    Raises `SchemaValidationError` if an expected column is missing or a
    shared column's type doesn't match. Extra columns are tolerated but
    logged as a warning, since a superset schema is usually a harmless
    upstream addition rather than a breaking change.
    """
    actual_fields = {f.name: f.dataType for f in df.schema.fields}
    expected_fields = {f.name: f.dataType for f in expected_schema.fields}

    missing = sorted(set(expected_fields) - set(actual_fields))
    unexpected = sorted(set(actual_fields) - set(expected_fields))
    type_mismatches = {
        name: (str(expected_fields[name]), str(actual_fields[name]))
        for name in expected_fields.keys() & actual_fields.keys()
        if expected_fields[name] != actual_fields[name]
    }

    if missing or type_mismatches:
        logger.error(
            "schema_validation_failed",
            extra={"missing_columns": missing, "type_mismatches": type_mismatches},
        )
        raise SchemaValidationError(
            f"Schema validation failed: missing_columns={missing}, "
            f"type_mismatches={type_mismatches}"
        )

    if unexpected:
        logger.warning("unexpected_columns_found", extra={"columns": unexpected})

    logger.info("schema_validation_passed", extra={"column_count": len(expected_fields)})


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


class TransactionIngestionPipeline:
    """Reads, validates, and writes the raw transactions dataset."""

    def __init__(
        self,
        settings: IngestionSettings,
        spark: Optional[SparkSession] = None,
    ) -> None:
        self.settings = settings
        self.spark = spark or build_spark_session(
            settings.environment, settings.app_name, use_delta=settings.output_format == "delta"
        )
        self.dbutils = get_dbutils(self.spark) if settings.environment == "databricks" else None

    def run(self) -> DataFrame:
        """Execute the full ingest -> validate -> write pipeline."""
        logger.info(
            "ingestion_started",
            extra={
                "environment": self.settings.environment,
                "input_path": self.settings.input_path,
                "output_path": self.settings.output_path,
                "output_format": self.settings.output_format,
            },
        )

        if not path_exists(self.settings.input_path, self.settings.environment, self.dbutils):
            raise FileNotFoundError(f"Input path does not exist: {self.settings.input_path}")

        df = self._read_raw_csv()
        validate_schema(df, get_transaction_schema())

        row_count = df.count()
        logger.info("row_count_validated", extra={"row_count": row_count})
        if row_count == 0:
            raise SchemaValidationError("Ingested dataset contains zero rows.")

        self._write_output(df)

        logger.info("ingestion_completed", extra={"row_count": row_count})
        return df

    def _read_raw_csv(self) -> DataFrame:
        """Read the raw CSV against the explicit schema.

        Uses PERMISSIVE mode with a `_corrupt_record` column so that rows
        which don't conform to the schema (e.g. wrong field count) are
        captured and counted rather than silently dropped or failing the job.
        """
        schema_with_corrupt_col = StructType(
            get_transaction_schema().fields + [StructField("_corrupt_record", StringType(), True)]
        )

        df = (
            self.spark.read.option("header", True)
            .option("mode", "PERMISSIVE")
            .option("columnNameOfCorruptRecord", "_corrupt_record")
            .schema(schema_with_corrupt_col)
            .csv(self.settings.input_path)
            .cache()
        )

        # Spark disallows a query that touches *only* the corrupt-record column
        # unless the DataFrame is cached first (hence `.cache()` above).
        corrupt_count = df.filter(df["_corrupt_record"].isNotNull()).count()
        if corrupt_count > 0:
            logger.warning("corrupt_records_found", extra={"corrupt_record_count": corrupt_count})

        return df.drop("_corrupt_record")

    def _write_output(self, df: DataFrame) -> None:
        logger.info(
            "write_started",
            extra={
                "output_path": self.settings.output_path,
                "output_format": self.settings.output_format,
                "write_mode": self.settings.write_mode,
            },
        )
        start = time.monotonic()

        (
            df.write.format(self.settings.output_format)
            .mode(self.settings.write_mode)
            .save(self.settings.output_path)
        )

        elapsed_seconds = round(time.monotonic() - start, 2)
        logger.info(
            "write_completed",
            extra={"output_path": self.settings.output_path, "elapsed_seconds": elapsed_seconds},
        )

    def stop(self) -> None:
        """Stop the Spark session, but only when we own it (local mode).

        On Databricks the session is owned by the cluster/notebook and must
        not be stopped by job code.
        """
        if self.settings.environment == "local":
            self.spark.stop()


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
    settings = load_settings(args.config, env_override=args.env)

    setup_logging(settings.log_level)

    pipeline = TransactionIngestionPipeline(settings)
    try:
        pipeline.run()
    except Exception:
        logger.exception("ingestion_failed")
        raise
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
