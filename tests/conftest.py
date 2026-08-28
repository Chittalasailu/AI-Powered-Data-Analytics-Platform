"""Shared pytest fixtures for the whole test suite.

Windows note: a local SparkSession needs `HADOOP_HOME` pointing at a
directory containing `winutils.exe` even for purely local `local[*]` runs --
this is a Spark-on-Windows requirement, not specific to this project. See
`src/utils/spark_session.py`; the same setup needed to run the pipeline
locally on Windows is needed to run this test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.spark_session import build_spark_session  # noqa: E402


@pytest.fixture(scope="session")
def spark():
    """One local SparkSession shared by every test in the run.

    Session-scoped rather than per-test: starting a SparkSession costs real
    wall-clock time (JVM startup), and nothing here needs isolation between
    tests -- every test builds its own small DataFrame from scratch via
    `tests/factories.py`, so there's no mutable shared state one test could
    leak into another through a shared session. `use_delta=False` since
    these tests exercise transformation logic on in-memory DataFrames, never
    reading or writing an actual Delta table -- skipping Delta's session
    setup (which fetches the delta-spark package via Ivy on first use) keeps
    the suite fast to start.
    """
    session = build_spark_session("local", "pytest-suite", use_delta=False)
    session.sparkContext.setLogLevel("ERROR")
    # Every test DataFrame here is a handful of rows. Left at Spark's
    # default of 200, each shuffle (dedup's window, any groupBy) spawns 200
    # tiny tasks that each round-trip through the driver-JVM accumulator
    # socket -- on Windows this doesn't just waste time, it can wedge the
    # accumulator server badly enough to look like a hang between tests.
    session.conf.set("spark.sql.shuffle.partitions", "2")
    yield session
    session.stop()
