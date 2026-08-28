"""Environment-aware SparkSession / dbutils construction.

Shared by every pipeline module so that "how do we get a Spark session and,
on Databricks, dbutils" is defined in exactly one place, whether the caller
is doing ingestion, transformation, or anything else added later.
"""

from __future__ import annotations

from typing import Any, Optional

from pyspark.sql import SparkSession


def build_spark_session(environment: str, app_name: str, use_delta: bool = False) -> SparkSession:
    """Return a SparkSession appropriate for `environment`.

    On Databricks, `SparkSession.builder.getOrCreate()` attaches to the
    cluster's already-running session (Delta support is built into the
    Databricks runtime, so no extra configuration is needed there). Locally,
    a `local[*]` session is built, and Delta's session extensions are wired
    up via `delta-spark` only when `use_delta` is True.
    """
    if environment == "databricks":
        return SparkSession.builder.appName(app_name).getOrCreate()

    builder = SparkSession.builder.appName(app_name).master("local[*]")

    if use_delta:
        builder = builder.config(
            "spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension"
        ).config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        try:
            from delta import configure_spark_with_delta_pip
        except ImportError as exc:
            raise ImportError(
                "Delta output requires the 'delta-spark' package locally. "
                "Install it with `pip install delta-spark` or use 'parquet' instead."
            ) from exc
        return configure_spark_with_delta_pip(builder).getOrCreate()

    return builder.getOrCreate()


def get_dbutils(spark: SparkSession) -> Optional[Any]:
    """Return a `DBUtils` instance when running on Databricks, else `None`."""
    try:
        from pyspark.dbutils import DBUtils

        return DBUtils(spark)
    except ImportError:
        return None


def path_exists(path: str, environment: str, dbutils: Optional[Any]) -> bool:
    """Check whether `path` exists, using dbutils on Databricks or pathlib locally."""
    if environment == "databricks" and dbutils is not None:
        try:
            dbutils.fs.ls(path)
            return True
        except Exception:  # noqa: BLE001 - dbutils raises a generic Py4JJavaError
            return False
    from pathlib import Path

    return Path(path).exists()
