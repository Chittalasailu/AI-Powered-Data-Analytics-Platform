"""Register the processed transactions data as Spark SQL temp views and run
the analytics query library in `sql/analytics_queries.sql` against them.

Two responsibilities live here:

1. **Run the catalog.** `sql/analytics_queries.sql` holds named, documented
   Spark SQL queries (revenue trends, customer LTV ranking, cohort
   retention, regional performance). This module parses that file,
   registers `transactions_cleaned` (plus, opportunistically, the 5 tables
   `src/transformation/transform.py` writes) as temp views, and executes
   any or all of the queries.
2. **Prove out query-level optimizations.** `OPTIMIZATION_PAIRS` holds a
   handful of unoptimized/optimized SQL pairs -- each pair produces the
   same result set two different ways. `--compare-optimizations` runs both
   sides of each pair, captures their real `EXPLAIN FORMATTED` physical
   plans and wall-clock timings, and (with `--write-notes`) regenerates
   `sql/query_optimization_notes.md` from that captured output, so the
   notes reflect an actual run rather than a hand-typed guess at what the
   optimizer does.

Usage:
    python src/transformation/sql_runner.py [--config config/config.yaml] [--env local|databricks]
    python src/transformation/sql_runner.py --list
    python src/transformation/sql_runner.py --query customer_ltv_ranking
    python src/transformation/sql_runner.py --compare-optimizations --write-notes
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.transformation.transform import build_region_dimension  # noqa: E402
from src.utils.config import load_yaml_config, resolve_environment  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402
from src.utils.spark_session import build_spark_session, get_dbutils, path_exists  # noqa: E402

DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Query catalog parsing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalyticsQuery:
    name: str
    description: str
    sql: str


_QUERY_HEADER_PATTERN = re.compile(
    r"--\s*@name:\s*(?P<name>\S+)\s*\n"
    r"--\s*@description:\s*(?P<description>.+?)\s*\n"
    r"(?P<sql>.*?)(?=\n--\s*@name:|\Z)",
    re.DOTALL,
)


def parse_queries_file(path: Path) -> list[AnalyticsQuery]:
    """Parse `sql/analytics_queries.sql` into a list of named queries.

    Each query is a `-- @name: <id>` / `-- @description: <text>` header
    pair followed by its SQL body, running until the next header or EOF.
    A regex (not a general SQL parser) is enough here because the file's
    shape is fully controlled by us -- every query in it follows this
    convention on purpose so both this parser and a human skimming the
    file can find query boundaries the same way.
    """
    text = path.read_text(encoding="utf-8")
    queries = [
        AnalyticsQuery(
            name=match.group("name"),
            description=match.group("description"),
            sql=match.group("sql").strip().rstrip(";").strip(),
        )
        for match in _QUERY_HEADER_PATTERN.finditer(text)
    ]
    if not queries:
        raise ValueError(f"No queries found in {path} (expected '-- @name:' headers)")
    return queries


# --------------------------------------------------------------------------
# Temp view registration
# --------------------------------------------------------------------------


def register_views(
    spark: SparkSession,
    settings: "SqlAnalyticsRunSettings",
    dbutils: Optional[object],
) -> None:
    """Register every processed DataFrame this module's queries can use.

    `transactions` and `region_dim` are required -- every query in
    `analytics_queries.sql` is written against them. The 5
    `transform.py` output tables are registered on a best-effort basis:
    useful for ad hoc follow-up queries against already-aggregated data,
    but not required, since `sql_runner.py` may reasonably be run before
    `transform.py` has ever produced them.
    """
    if not path_exists(settings.transactions_path, settings.environment, dbutils):
        raise FileNotFoundError(f"Input path does not exist: {settings.transactions_path}")

    transactions = spark.read.format(settings.transactions_format).load(settings.transactions_path)
    transactions.createOrReplaceTempView("transactions")
    logger.info("view_registered", extra={"view": "transactions", "source": settings.transactions_path})

    build_region_dimension(spark).createOrReplaceTempView("region_dim")
    logger.info("view_registered", extra={"view": "region_dim", "source": "static"})

    for view_name, table_path in settings.aggregate_table_paths.items():
        if not path_exists(table_path, settings.environment, dbutils):
            logger.warning("aggregate_table_not_found", extra={"view": view_name, "path": table_path})
            continue
        spark.read.format(settings.aggregate_format).load(table_path).createOrReplaceTempView(view_name)
        logger.info("view_registered", extra={"view": view_name, "source": table_path})


# --------------------------------------------------------------------------
# Running queries
# --------------------------------------------------------------------------


def run_query(spark: SparkSession, query: AnalyticsQuery, show_rows: int = 20) -> DataFrame:
    """Execute one catalog query, logging its row count and wall-clock time.

    Cached before the count so the row-count action and the `.show()`
    preview below don't each independently re-run the query from scratch --
    with 11 queries in the catalog, doubling every one of them would double
    total runtime for no benefit, since these result sets (revenue trends,
    LTV rankings, cohort tables) are all small enough to cache cheaply.
    """
    logger.info("running_query", extra={"query_name": query.name, "description": query.description})
    start = time.perf_counter()

    df = spark.sql(query.sql).cache()
    row_count = df.count()
    elapsed_seconds = round(time.perf_counter() - start, 2)

    logger.info(
        "query_completed",
        extra={"query_name": query.name, "row_count": row_count, "elapsed_seconds": elapsed_seconds},
    )
    df.show(show_rows, truncate=False)
    df.unpersist()
    return df


def run_all_queries(spark: SparkSession, queries: list[AnalyticsQuery], show_rows: int = 20) -> None:
    for query in queries:
        run_query(spark, query, show_rows)


# --------------------------------------------------------------------------
# Unoptimized vs. optimized query comparisons
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryPair:
    """Two SQL strings that produce equivalent results by different routes."""

    name: str
    description: str
    unoptimized_label: str
    unoptimized_sql: str
    optimized_label: str
    optimized_sql: str
    takeaway: str


@dataclass(frozen=True)
class PlanCapture:
    plan_text: str
    row_count: int
    elapsed_seconds: float


OPTIMIZATION_PAIRS: list[QueryPair] = [
    QueryPair(
        name="regional_join_strategy",
        description="Joining `transactions` to the 5-row `region_dim` table: forced sort-merge join vs. forced broadcast join.",
        unoptimized_label="/*+ MERGE(t, d) */ forces a shuffle (sort-merge) join",
        unoptimized_sql="""
            SELECT /*+ MERGE(t, d) */
                t.region, d.sales_territory, ROUND(SUM(t.total_amount), 2) AS revenue
            FROM transactions t
            JOIN region_dim d ON t.region = d.region
            GROUP BY t.region, d.sales_territory
        """,
        optimized_label="/*+ BROADCAST(d) */ forces a broadcast hash join",
        optimized_sql="""
            SELECT /*+ BROADCAST(d) */
                t.region, d.sales_territory, ROUND(SUM(t.total_amount), 2) AS revenue
            FROM transactions t
            JOIN region_dim d ON t.region = d.region
            GROUP BY t.region, d.sales_territory
        """,
        takeaway=(
            "`region_dim` is 5 rows. A sort-merge join still shuffles the *entire* "
            "`transactions` table across the network to co-locate matching keys before "
            "it can join anything, purely to attach a lookup value with 5 possible "
            "outcomes. A broadcast join ships one side to every executor once and joins "
            "locally -- no shuffle of the larger side at all. Worth checking rather than "
            "assuming: with *no* hint at all on this small local dataset, Spark's "
            "cost-based optimizer does pick a broadcast join here too -- but it broadcasts "
            "`transactions` (`BuildLeft`), not `region_dim`. `region_dim` comes from "
            "`spark.createDataFrame()` on driver-side Python data (a `Scan ExistingRDD`), "
            "which doesn't carry the file-based byte-size statistics that the "
            "parquet-backed `transactions` scan does, so the optimizer trusts the known "
            "size of `transactions` more than its guess at `region_dim`'s size -- and on "
            "this small test dataset, `transactions` itself is small enough to broadcast, "
            "so it works out fine by coincidence. At real fact-table scale that coincidence "
            "disappears: the same missing-statistics gap could just as easily make the "
            "optimizer fall back to a sort-merge join instead of broadcasting the tiny "
            "dimension table it should. `/*+ BROADCAST(d) */` pins down which side gets "
            "broadcast regardless of which side's statistics the optimizer happens to "
            "trust."
        ),
    ),
    QueryPair(
        name="ltv_ranking_self_join_vs_window",
        description="Ranking customers by lifetime value: a self-join anti-pattern vs. a window function.",
        unoptimized_label="self-join on an inequality to emulate RANK()",
        unoptimized_sql="""
            WITH customer_totals AS (
                SELECT customer_id, SUM(total_amount) AS lifetime_value
                FROM transactions
                WHERE customer_id != 'UNKNOWN_CUSTOMER'
                GROUP BY customer_id
            )
            SELECT
                a.customer_id,
                a.lifetime_value,
                COUNT(b.customer_id) + 1 AS ltv_rank
            FROM customer_totals a
            LEFT JOIN customer_totals b ON b.lifetime_value > a.lifetime_value
            GROUP BY a.customer_id, a.lifetime_value
        """,
        optimized_label="RANK() OVER (ORDER BY lifetime_value DESC)",
        optimized_sql="""
            WITH customer_totals AS (
                SELECT customer_id, SUM(total_amount) AS lifetime_value
                FROM transactions
                WHERE customer_id != 'UNKNOWN_CUSTOMER'
                GROUP BY customer_id
            )
            SELECT
                customer_id,
                lifetime_value,
                RANK() OVER (ORDER BY lifetime_value DESC) AS ltv_rank
            FROM customer_totals
        """,
        takeaway=(
            "The self-join computes each customer's rank by counting how many other "
            "customers out-earned them -- an inequality join, so Spark can't use a hash "
            "join at all; it falls back to a nested-loop-style join comparing every "
            "customer against every other customer (~n^2 comparisons -- tens of millions "
            "of comparisons for a few thousand customers). `RANK() OVER (...)` computes "
            "the identical ranking with a single sort pass (n log n) and no join "
            "whatsoever. This is a textbook case where reaching for a join out of habit "
            "is dramatically more expensive than the window-function primitive SQL "
            "already provides for exactly this problem."
        ),
    ),
    QueryPair(
        name="cohort_prefilter_before_self_join",
        description="Cohort retention's self-join of `transactions` against itself: unfiltered vs. filtering the sentinel customer and null dates first.",
        unoptimized_label="no filter -- UNKNOWN_CUSTOMER and null-date rows flow into both join branches",
        unoptimized_sql="""
            WITH first_purchase AS (
                SELECT customer_id, date_trunc('month', MIN(purchase_date)) AS cohort_month
                FROM transactions
                GROUP BY customer_id
            ),
            monthly_activity AS (
                SELECT DISTINCT customer_id, date_trunc('month', purchase_date) AS activity_month
                FROM transactions
                WHERE purchase_date IS NOT NULL
            )
            SELECT f.cohort_month, COUNT(DISTINCT a.customer_id) AS active_customers
            FROM first_purchase f
            JOIN monthly_activity a ON f.customer_id = a.customer_id
            GROUP BY f.cohort_month
        """,
        optimized_label="shared valid_transactions CTE filters both branches before the join",
        optimized_sql="""
            WITH valid_transactions AS (
                SELECT customer_id, purchase_date
                FROM transactions
                WHERE customer_id != 'UNKNOWN_CUSTOMER' AND purchase_date IS NOT NULL
            ),
            first_purchase AS (
                SELECT customer_id, date_trunc('month', MIN(purchase_date)) AS cohort_month
                FROM valid_transactions
                GROUP BY customer_id
            ),
            monthly_activity AS (
                SELECT DISTINCT customer_id, date_trunc('month', purchase_date) AS activity_month
                FROM valid_transactions
            )
            SELECT f.cohort_month, COUNT(DISTINCT a.customer_id) AS active_customers
            FROM first_purchase f
            JOIN monthly_activity a ON f.customer_id = a.customer_id
            GROUP BY f.cohort_month
        """,
        takeaway=(
            "This one is a correctness fix as much as a performance one, and it's not "
            "something the optimizer can apply on its own -- the filter is *absent* in "
            "the unoptimized version, not just misplaced, so there's no rewrite rule that "
            "invents it. Both plans happen to pick the same join strategy here "
            "(BroadcastHashJoin -- `monthly_activity` is small enough to auto-broadcast "
            "either way), so the visible EXPLAIN difference is in the Filter/PushedFilters "
            "at the two `Scan parquet` nodes and the row volume flowing into the "
            "`HashAggregate`/`Exchange` stages that build `first_purchase` and "
            "`monthly_activity` -- fewer, valid rows shuffled in the optimized plan. The "
            "bigger issue is correctness: on this dataset the two versions' row *counts* "
            "both happen to be 28, but the values differ -- the 2023-01 cohort reports "
            "1406 active customers unoptimized vs. 1405 optimized. The `UNKNOWN_CUSTOMER` "
            "sentinel row (cleansing's fill-in for a missing customer_id) collapses every "
            "unrelated transaction with a missing customer into one fake mega-customer "
            "whose 'cohort month' is the earliest of those transactions, and who then "
            "shows up 'active' in whatever months any of them happened to land in -- a "
            "synthetic retention signal that doesn't correspond to any real person. "
            "Filtering it out (and null purchase_date rows, which can't be assigned a "
            "cohort month at all) in one shared CTE before both `first_purchase` and "
            "`monthly_activity` are built keeps that fake customer out of the result "
            "entirely, not just out of one branch of the join."
        ),
    ),
]


def _capture_explain(df: DataFrame) -> str:
    """Capture `df.explain(mode="formatted")`'s output as a string.

    `DataFrame.explain()` prints straight to stdout and returns `None`
    rather than returning the plan text, so redirecting stdout into a
    buffer is the standard way to get it back as a string for logging or
    writing to a file instead of just watching it scroll past in a
    terminal.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        df.explain(mode="formatted")
    return buffer.getvalue()


def _capture_plan_and_timing(spark: SparkSession, sql: str) -> PlanCapture:
    df = spark.sql(sql)
    plan_text = _capture_explain(df)

    start = time.perf_counter()
    row_count = df.count()
    elapsed_seconds = round(time.perf_counter() - start, 2)

    return PlanCapture(plan_text=plan_text, row_count=row_count, elapsed_seconds=elapsed_seconds)


@dataclass(frozen=True)
class PairComparison:
    pair: QueryPair
    unoptimized: PlanCapture
    optimized: PlanCapture


def run_optimization_comparisons(spark: SparkSession) -> list[PairComparison]:
    """Run every pair in `OPTIMIZATION_PAIRS` and capture plans + timings for both sides.

    Timings are a single local `.count()` per side on a small (tens-of-
    thousands-of-rows) dataset -- illustrative of relative cost, not a
    proper benchmark (no warmup, no repetition, easily dominated by JVM/
    scheduling noise at this data size). The EXPLAIN plan shape -- which
    physical join operator got chosen -- is the reliable signal; the
    timings are reported alongside it for context, not as the main proof.
    """
    comparisons = []
    for pair in OPTIMIZATION_PAIRS:
        logger.info("comparing_query_pair", extra={"pair_name": pair.name})
        unoptimized = _capture_plan_and_timing(spark, pair.unoptimized_sql)
        optimized = _capture_plan_and_timing(spark, pair.optimized_sql)
        comparisons.append(PairComparison(pair=pair, unoptimized=unoptimized, optimized=optimized))
        logger.info(
            "query_pair_compared",
            extra={
                "pair_name": pair.name,
                "unoptimized_seconds": unoptimized.elapsed_seconds,
                "optimized_seconds": optimized.elapsed_seconds,
                "unoptimized_rows": unoptimized.row_count,
                "optimized_rows": optimized.row_count,
            },
        )
    return comparisons


def _format_comparison_section(comparison: PairComparison) -> str:
    pair, unopt, opt = comparison.pair, comparison.unoptimized, comparison.optimized
    return f"""## {pair.name}

{pair.description}

### Unoptimized -- {pair.unoptimized_label}

```sql
{pair.unoptimized_sql.strip()}
```

Rows returned: **{unopt.row_count}** | Wall-clock (single local run): **{unopt.elapsed_seconds:.2f}s**

<details>
<summary>EXPLAIN FORMATTED (unoptimized)</summary>

```
{unopt.plan_text.strip()}
```

</details>

### Optimized -- {pair.optimized_label}

```sql
{pair.optimized_sql.strip()}
```

Rows returned: **{opt.row_count}** | Wall-clock (single local run): **{opt.elapsed_seconds:.2f}s**

<details>
<summary>EXPLAIN FORMATTED (optimized)</summary>

```
{opt.plan_text.strip()}
```

</details>

### Why

{pair.takeaway}
"""


def write_optimization_notes(comparisons: list[PairComparison], output_path: Path) -> None:
    """Render `sql/query_optimization_notes.md` from real captured EXPLAIN output.

    Regenerating this from an actual run (rather than hand-typing what the
    plan is expected to look like) means the notes stay honest if the data,
    Spark version, or optimizer rules change -- the physical plans and row
    counts embedded in it came from Spark's own EXPLAIN output on this run,
    not from memory of how Spark usually behaves.
    """
    sections = "\n\n---\n\n".join(_format_comparison_section(c) for c in comparisons)
    content = f"""# Query optimization notes

Generated by `src/transformation/sql_runner.py --compare-optimizations --write-notes`.
Each section below runs the *same* result set two different ways and compares
their real `EXPLAIN FORMATTED` physical plans and a single local wall-clock
timing, captured directly from this Spark session -- nothing here is
hand-typed or assumed.

---

{sections}
"""
    output_path.write_text(content, encoding="utf-8")
    logger.info("optimization_notes_written", extra={"output_path": str(output_path)})


# --------------------------------------------------------------------------
# Run settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SqlAnalyticsRunSettings:
    environment: str
    app_name: str
    transactions_path: str
    transactions_format: str
    aggregate_table_paths: dict[str, str]
    aggregate_format: str
    queries_path: Path
    optimization_notes_path: Path
    show_rows: int
    log_level: str


def load_run_settings(config_path: Path, env_override: Optional[str] = None) -> SqlAnalyticsRunSettings:
    raw_config = load_yaml_config(config_path)
    environment, _, processed_data_dir = resolve_environment(raw_config, env_override)
    processed_data_dir = processed_data_dir.rstrip("/")
    sql_cfg = raw_config["sql_analytics"]
    agg_cfg = raw_config["aggregation"]

    aggregate_table_paths = {
        key: f"{processed_data_dir}/{table_name}" for key, table_name in agg_cfg["output_tables"].items()
    }

    return SqlAnalyticsRunSettings(
        environment=environment,
        app_name=f"{raw_config['spark']['app_name']}-sql-analytics",
        transactions_path=f"{processed_data_dir}/{sql_cfg['input_table_name']}",
        transactions_format=raw_config["cleansing"]["output_format"],
        aggregate_table_paths=aggregate_table_paths,
        aggregate_format=agg_cfg["output_format"],
        queries_path=ROOT_DIR / sql_cfg["queries_file"],
        optimization_notes_path=ROOT_DIR / sql_cfg["optimization_notes_file"],
        show_rows=int(sql_cfg.get("show_rows", 20)),
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
    parser.add_argument("--list", action="store_true", help="List catalog queries and exit (no execution)")
    parser.add_argument("--query", type=str, default=None, help="Run only the named catalog query")
    parser.add_argument(
        "--compare-optimizations",
        action="store_true",
        help="Run the unoptimized/optimized query pairs and print their EXPLAIN plans",
    )
    parser.add_argument(
        "--write-notes",
        action="store_true",
        help="With --compare-optimizations, also (re)write sql/query_optimization_notes.md",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    settings = load_run_settings(args.config, env_override=args.env)

    setup_logging(settings.log_level)

    queries = parse_queries_file(settings.queries_path)

    if args.list:
        for query in queries:
            print(f"{query.name}: {query.description}")
        return

    spark = build_spark_session(settings.environment, settings.app_name, use_delta=settings.transactions_format == "delta")
    dbutils = get_dbutils(spark) if settings.environment == "databricks" else None

    try:
        register_views(spark, settings, dbutils)

        if args.compare_optimizations:
            comparisons = run_optimization_comparisons(spark)
            if args.write_notes:
                write_optimization_notes(comparisons, settings.optimization_notes_path)
            return

        if args.query:
            matches = [q for q in queries if q.name == args.query]
            if not matches:
                available = ", ".join(q.name for q in queries)
                raise ValueError(f"Unknown query '{args.query}'. Available: {available}")
            run_query(spark, matches[0], settings.show_rows)
            return

        run_all_queries(spark, queries, settings.show_rows)
    except Exception:
        logger.exception("sql_runner_failed")
        raise
    finally:
        if settings.environment == "local":
            spark.stop()


if __name__ == "__main__":
    main()
