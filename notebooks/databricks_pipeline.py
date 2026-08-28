# Databricks notebook source
# MAGIC %md
# MAGIC # AI-Powered Data Analytics Platform — Databricks Pipeline Walkthrough
# MAGIC
# MAGIC This notebook is the Databricks-native counterpart to the local CLI pipeline
# MAGIC (`src/pipeline.py`). It runs the exact same seven stages, in the exact same
# MAGIC order, against the exact same code in `src/` — this notebook does not
# MAGIC reimplement any business logic. It imports the already-tested functions and
# MAGIC classes from `src/ingestion`, `src/transformation`, and `src/ml`, and calls
# MAGIC them directly against this cluster's Spark session:
# MAGIC
# MAGIC 1. **Ingest** — raw transactions CSV → Delta
# MAGIC 2. **Cleanse** — dedupe, standardize, impute
# MAGIC 3. **Transform** — 5 analytics-ready aggregated tables
# MAGIC 4. **SQL Analytics** — the Spark SQL query catalog (revenue trends, LTV, cohorts, regional)
# MAGIC 5. **Feature Engineering** — clustering + leakage-safe churn feature tables
# MAGIC 6. **Train Model** — KMeans segmentation + RandomForest churn classifier
# MAGIC 7. **Evaluate** — silhouette / precision / recall / F1 / ROC-AUC + confusion matrix
# MAGIC
# MAGIC Each stage below has a markdown cell explaining *why* it exists and what to
# MAGIC look for in its output — this notebook is meant to be read top to bottom as a
# MAGIC portfolio walkthrough, not just executed. The very last section documents
# MAGIC exactly where you'd configure a Databricks Job and cluster to run this on a
# MAGIC schedule in production.

# COMMAND ----------

# MAGIC %md
# MAGIC ## How this differs from the local CLI pipeline
# MAGIC
# MAGIC `src/pipeline.py` runs each stage as its own **subprocess**, each building
# MAGIC and tearing down its own `SparkSession` — the right design for a laptop/cron
# MAGIC context, where isolating stages into separate OS processes protects the
# MAGIC orchestrator from any one stage's failure mode.
# MAGIC
# MAGIC A Databricks notebook works differently, and fighting that would be the
# MAGIC wrong move here: a notebook attaches to **one cluster** and gets **one**
# MAGIC shared `spark` (and `dbutils`) object, injected by the runtime before the
# MAGIC first cell even runs. So this notebook:
# MAGIC
# MAGIC - **Never calls `build_spark_session()`.** There's no session to build —
# MAGIC   `spark` already exists. Delta support is native to the Databricks Runtime
# MAGIC   too, so none of the local-only `configure_spark_with_delta_pip(...)`
# MAGIC   machinery in `src/utils/spark_session.py` is needed here either.
# MAGIC - **Never calls `get_dbutils()`.** `dbutils` is already a global.
# MAGIC - **Calls each stage's underlying functions directly** (e.g.
# MAGIC   `DataCleanser(df, config).run()`, `compute_customer_rfm(df)`,
# MAGIC   `train_segmentation_model(features_pdf, ...)`) instead of shelling out to
# MAGIC   `ingest.main()` / `cleanse.main()` / etc., since those `main()` functions
# MAGIC   are written to own their session's full lifecycle — exactly what we don't
# MAGIC   want inside a notebook that's supposed to keep one session alive
# MAGIC   throughout.
# MAGIC - **Reads every stage's input from Delta/DBFS, not from a Python variable
# MAGIC   left over by an earlier cell.** So any single cell here can be re-run on
# MAGIC   its own (say, just re-running Evaluate after tweaking a report path)
# MAGIC   without needing a full top-to-bottom re-run first.
# MAGIC - **Lets exceptions propagate.** Each stage is wrapped in `try/except` that
# MAGIC   logs full context and then **re-raises** — a Databricks Job needs an
# MAGIC   uncaught exception to mark the run as Failed and fire its notifications;
# MAGIC   quietly swallowing an error here would make a broken pipeline look green.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: install Python dependencies not in the base runtime
# MAGIC
# MAGIC If this cluster is running **Databricks Runtime for Machine Learning**,
# MAGIC scikit-learn / pandas / matplotlib / seaborn / joblib are already preinstalled
# MAGIC and this cell is a no-op you can skip. On the **Standard** runtime, run it
# MAGIC once per session (notebook-scoped `%pip install` — no cluster restart
# MAGIC required, the packages are importable in the very next cell).

# COMMAND ----------

# MAGIC %pip install scikit-learn matplotlib seaborn joblib pyyaml

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters (`dbutils.widgets`)
# MAGIC
# MAGIC Every value a Databricks Job might reasonably want to override per-run or
# MAGIC per-environment lives in a widget rather than being hardcoded, mirroring what
# MAGIC `--config`/`--env` do for the CLI. `environment` itself is **not** a widget —
# MAGIC this notebook only ever runs attached to a Databricks cluster, so it's fixed
# MAGIC to `"databricks"` below rather than pretending "local" is a meaningful choice
# MAGIC here.
# MAGIC
# MAGIC Widget values can be set interactively at the top of the notebook, or —
# MAGIC this is the part that matters for scheduling — as **Job parameters** when
# MAGIC this notebook is attached to a Databricks Job task (see the final section),
# MAGIC so the same notebook can run against, say, a staging config and a production
# MAGIC config without editing any code.

# COMMAND ----------

REPO_ROOT_PLACEHOLDER = "/Workspace/Repos/<your-username>/ai-data-analytics-platform"

dbutils.widgets.text(
    "repo_root",
    REPO_ROOT_PLACEHOLDER,
    "Repo root (Databricks Repos checkout path)",
)
dbutils.widgets.text(
    "config_path",
    f"{dbutils.widgets.get('repo_root')}/config/config.yaml",
    "Path to config.yaml",
)
dbutils.widgets.text("churn_window_days", "90", "Churn window (days)")
dbutils.widgets.text("top_products_per_category", "10", "Top products per category")
dbutils.widgets.text("kmeans_k_min", "3", "KMeans k (min)")
dbutils.widgets.text("kmeans_k_max", "8", "KMeans k (max)")
dbutils.widgets.text("rf_n_estimators", "300", "RandomForest n_estimators")
dbutils.widgets.text(
    "models_dir",
    "/dbfs/mnt/datalake/models",
    "Models output dir (DBFS FUSE path, for joblib)",
)
dbutils.widgets.text(
    "reports_dir",
    "/dbfs/mnt/datalake/reports",
    "Reports output dir (DBFS FUSE path, for joblib/matplotlib)",
)

# COMMAND ----------

# MAGIC %md
# MAGIC A quick, easy-to-miss Databricks detail baked into the two defaults above:
# MAGIC **Spark APIs and plain Python file I/O address DBFS differently.**
# MAGIC `spark.read`/`spark.write` understand a mount path like `/mnt/datalake/raw`
# MAGIC directly. Plain Python — `joblib.dump(...)`, `matplotlib`'s `fig.savefig(...)`,
# MAGIC `open(...)` — does **not** go through Spark, so it needs the FUSE-mounted
# MAGIC local view of DBFS instead, which is the same path with a `/dbfs` prefix:
# MAGIC `/dbfs/mnt/datalake/models`. `models_dir`/`reports_dir` use joblib and
# MAGIC matplotlib respectively, so they're `/dbfs/...` paths; every Spark table this
# MAGIC notebook reads or writes uses the plain `/mnt/...` paths already configured
# MAGIC in `config.yaml`'s `paths.databricks` block. (If this workspace uses Unity
# MAGIC Catalog Volumes instead of legacy DBFS mounts, swap these two widgets for
# MAGIC `/Volumes/<catalog>/<schema>/<volume>/models` style paths — Volumes are
# MAGIC POSIX-accessible directly, no `/dbfs` prefix needed.)

# COMMAND ----------

import dataclasses
import logging
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyspark.sql import functions as F

ENVIRONMENT = "databricks"  # fixed -- see the widgets markdown cell above

REPO_ROOT = Path(dbutils.widgets.get("repo_root"))
CONFIG_PATH = Path(dbutils.widgets.get("config_path"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ingestion.ingest import (  # noqa: E402
    TransactionIngestionPipeline,
    load_settings as load_ingestion_settings,
)
from src.transformation.cleanse import (  # noqa: E402
    DataCleanser,
    load_run_settings as load_cleansing_settings,
)
from src.transformation.transform import (  # noqa: E402
    compute_customer_rfm,
    compute_daily_revenue,
    compute_monthly_revenue,
    compute_payment_method_distribution,
    compute_top_products_by_category,
    load_run_settings as load_aggregation_settings,
    prepare_base_dataset,
)
from src.transformation.sql_runner import (  # noqa: E402
    load_run_settings as load_sql_settings,
    parse_queries_file,
    register_views,
)
from src.ml.feature_engineering import (  # noqa: E402
    build_churn_features,
    build_clustering_features,
    load_run_settings as load_feature_settings,
)
from src.ml.train_model import (  # noqa: E402
    CHURN_MODEL_FILENAME,
    CLUSTERING_MODEL_FILENAME,
    load_run_settings as load_train_settings,
    make_churn_train_test_split,
    save_model,
    train_churn_model,
    train_segmentation_model,
)
from src.ml.evaluate import (  # noqa: E402
    evaluate_churn_model,
    evaluate_segmentation_model,
    load_run_settings as load_evaluate_settings,
    save_csv,
    save_json,
)
from src.utils.config import load_yaml_config  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402

raw_config = load_yaml_config(CONFIG_PATH)
setup_logging(raw_config.get("logging", {}).get("level", "INFO"))
logger = logging.getLogger("databricks_pipeline")


def widget_int(name: str) -> int:
    return int(dbutils.widgets.get(name))


stage_timings: list[dict] = []


def start_stage(name: str) -> float:
    logger.info("stage_started", extra={"stage": name})
    return time.perf_counter()


def end_stage(name: str, start: float) -> None:
    elapsed_seconds = round(time.perf_counter() - start, 2)
    stage_timings.append({"stage": name, "elapsed_seconds": elapsed_seconds})
    logger.info("stage_completed", extra={"stage": name, "elapsed_seconds": elapsed_seconds})


print(f"repo_root:   {REPO_ROOT}")
print(f"config_path: {CONFIG_PATH}")
print(f"environment: {ENVIRONMENT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 1 — Ingestion
# MAGIC
# MAGIC Reads the raw transactions CSV against an explicit schema (`PERMISSIVE`
# MAGIC mode, so malformed rows are captured in a `_corrupt_record` column and
# MAGIC counted rather than silently dropped or failing the whole job), validates
# MAGIC it, and writes it to Delta as the immutable starting point everything else
# MAGIC in this pipeline builds on. Business framing: this is the boundary between
# MAGIC "data someone handed us" and "data we're willing to build a report or a
# MAGIC model on" — schema/row-count validation happens here specifically so a
# MAGIC malformed upstream export fails loudly on ingestion, not silently three
# MAGIC stages later as a confusing RFM number.

# COMMAND ----------

try:
    _t0 = start_stage("ingest")

    ingestion_settings = load_ingestion_settings(CONFIG_PATH, env_override=ENVIRONMENT)
    ingestion_pipeline = TransactionIngestionPipeline(ingestion_settings, spark=spark)
    raw_transactions_df = ingestion_pipeline.run()

    end_stage("ingest", _t0)
except Exception:
    logger.exception("stage_failed", extra={"stage": "ingest"})
    raise

display(raw_transactions_df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 2 — Cleansing
# MAGIC
# MAGIC Runs the `DataCleanser` pipeline: deduplication, date-format
# MAGIC standardization, type/range enforcement, IQR-based outlier capping, and
# MAGIC per-column null imputation (median/mode/sentinel, chosen per column's
# MAGIC statistical role — see `src/transformation/cleanse.py`'s module docstring
# MAGIC for the full reasoning). Every step preserves row count except
# MAGIC deduplication; nothing is silently dropped because it looked unusual.
# MAGIC The report table below is `cleanser.report` flattened for `display()` — the
# MAGIC same per-step metrics a data engineer would use to sanity-check a cleansing
# MAGIC run before trusting anything downstream of it.

# COMMAND ----------

try:
    _t0 = start_stage("cleanse")

    cleansing_settings = load_cleansing_settings(CONFIG_PATH, env_override=ENVIRONMENT)

    raw_for_cleansing_df = spark.read.format(cleansing_settings.input_format).load(cleansing_settings.input_path)
    cleanser = DataCleanser(raw_for_cleansing_df, cleansing_settings.cleansing_config)
    cleansed_df = cleanser.run()

    (
        cleansed_df.write.format(cleansing_settings.output_format)
        .mode(cleansing_settings.write_mode)
        .save(cleansing_settings.output_path)
    )

    end_stage("cleanse", _t0)
except Exception:
    logger.exception("stage_failed", extra={"stage": "cleanse"})
    raise

display(cleansed_df.limit(20))

cleansing_report_df = pd.json_normalize(cleanser.report, sep=".").T.reset_index()
cleansing_report_df.columns = ["metric", "value"]
display(cleansing_report_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 3 — Transformation (aggregated analytics tables)
# MAGIC
# MAGIC Builds the 5 analytics-ready tables: daily/monthly revenue by region and
# MAGIC category, customer RFM segmentation, top products per category, and payment
# MAGIC method distribution. This is also where the heavier Spark optimizations
# MAGIC live — a broadcast join for the 5-row region dimension, `repartition` by
# MAGIC `customer_id` chosen specifically so the RFM `groupBy` can skip its shuffle,
# MAGIC caching the base dataset since all 5 outputs read it independently. Full
# MAGIC reasoning for each choice is in `src/transformation/transform.py`'s module
# MAGIC and function docstrings — this notebook calls the same functions rather than
# MAGIC repeating that reasoning here.

# COMMAND ----------

try:
    _t0 = start_stage("transform")

    aggregation_settings = dataclasses.replace(
        load_aggregation_settings(CONFIG_PATH, env_override=ENVIRONMENT),
        top_products_per_category=widget_int("top_products_per_category"),
    )
    spark.conf.set("spark.sql.shuffle.partitions", aggregation_settings.shuffle_partitions)

    cleansed_for_agg_df = spark.read.format(aggregation_settings.input_format).load(aggregation_settings.input_path)
    base_df = prepare_base_dataset(cleansed_for_agg_df, spark, aggregation_settings.shuffle_partitions)

    daily_revenue_df = compute_daily_revenue(base_df)
    monthly_revenue_df = compute_monthly_revenue(base_df)
    customer_rfm_df = compute_customer_rfm(base_df)
    top_products_df = compute_top_products_by_category(base_df, aggregation_settings.top_products_per_category)
    payment_distribution_df = compute_payment_method_distribution(base_df)

    _aggregation_outputs = [
        ("daily_revenue", daily_revenue_df, "region", 4),
        ("monthly_revenue", monthly_revenue_df, "region", 4),
        ("customer_rfm", customer_rfm_df, None, 1),
        ("top_products", top_products_df, "category", 4),
        ("payment_distribution", payment_distribution_df, None, 1),
    ]
    for _name, _df, _partition_col, _file_count in _aggregation_outputs:
        _writer = (
            _df.coalesce(_file_count)
            .write.format(aggregation_settings.output_format)
            .mode(aggregation_settings.write_mode)
        )
        if _partition_col:
            _writer = _writer.partitionBy(_partition_col)
        _writer.save(aggregation_settings.output_paths[_name])

    base_df.unpersist()

    end_stage("transform", _t0)
except Exception:
    logger.exception("stage_failed", extra={"stage": "transform"})
    raise

display(daily_revenue_df.orderBy(F.desc("purchase_date")).limit(20))
display(customer_rfm_df.orderBy(F.desc("monetary")).limit(20))
display(top_products_df.filter(F.col("category_rank") <= 5).orderBy("category", "category_rank"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 4 — SQL Analytics
# MAGIC
# MAGIC Registers `transactions_cleaned` as the `transactions` temp view (plus a
# MAGIC small `region_dim` dimension table) and runs every named query in
# MAGIC `sql/analytics_queries.sql` — the same catalog covering revenue trends,
# MAGIC customer LTV ranking, cohort retention, and regional performance that
# MAGIC `sql_runner.py` runs on the CLI side. Each result below is captioned
# MAGIC directly from that query's `@name`/`@description` header, so the captions
# MAGIC can't drift out of sync with the query file — there's no separately
# MAGIC hand-maintained list of query descriptions to keep in sync by hand.

# COMMAND ----------

try:
    _t0 = start_stage("sql_analytics")

    sql_settings = load_sql_settings(CONFIG_PATH, env_override=ENVIRONMENT)
    register_views(spark, sql_settings, dbutils)
    catalog_queries = parse_queries_file(sql_settings.queries_path)

    end_stage("sql_analytics", _t0)
except Exception:
    logger.exception("stage_failed", extra={"stage": "sql_analytics"})
    raise

for _query in catalog_queries:
    displayHTML(f"<h4>{_query.name}</h4><p style='color:#666'>{_query.description}</p>")
    display(spark.sql(_query.sql))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 5 — Feature Engineering
# MAGIC
# MAGIC Builds two separate per-customer feature tables for two very different ML
# MAGIC problems (see `src/ml/feature_engineering.py`'s module docstring for the
# MAGIC full reasoning):
# MAGIC
# MAGIC - **Clustering features** reuse `agg_customer_rfm`'s recency/frequency/
# MAGIC   monetary as-is — there's no future outcome being predicted, so there's no
# MAGIC   leakage risk in using the full history.
# MAGIC - **Churn features** deliberately do *not* reuse that table. Its recency is
# MAGIC   measured against the dataset's true last date — exactly what a churn
# MAGIC   label is trying to predict. Instead, churn features are computed only
# MAGIC   from transactions before a cutoff date, and the label looks at the
# MAGIC   held-out window after it, so the model is validated the way it will
# MAGIC   actually be used: predicting the future from the past.

# COMMAND ----------

try:
    _t0 = start_stage("feature_engineering")

    ml_feature_settings = dataclasses.replace(
        load_feature_settings(CONFIG_PATH, env_override=ENVIRONMENT),
        churn_window_days=widget_int("churn_window_days"),
    )

    transactions_for_features_df = (
        spark.read.format(ml_feature_settings.transactions_format).load(ml_feature_settings.transactions_path).cache()
    )
    rfm_for_features_df = spark.read.format(ml_feature_settings.rfm_format).load(ml_feature_settings.rfm_path)

    clustering_features_df = build_clustering_features(transactions_for_features_df, rfm_for_features_df)
    (
        clustering_features_df.coalesce(1)
        .write.format(ml_feature_settings.output_format)
        .mode(ml_feature_settings.write_mode)
        .save(ml_feature_settings.clustering_output_path)
    )

    churn_features_df = build_churn_features(transactions_for_features_df, ml_feature_settings.churn_window_days)
    (
        churn_features_df.coalesce(1)
        .write.format(ml_feature_settings.output_format)
        .mode(ml_feature_settings.write_mode)
        .save(ml_feature_settings.churn_output_path)
    )

    transactions_for_features_df.unpersist()

    end_stage("feature_engineering", _t0)
except Exception:
    logger.exception("stage_failed", extra={"stage": "feature_engineering"})
    raise

display(clustering_features_df.limit(20))
display(churn_features_df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 6 — Model Training
# MAGIC
# MAGIC Trains both models as scikit-learn `Pipeline`s (preprocessing + estimator
# MAGIC bundled into one fitted, joblib-persistable object) on **pandas**
# MAGIC DataFrames — `.toPandas()` happens right here, once, on tables already
# MAGIC aggregated down to one row per customer (a few thousand rows), which is
# MAGIC exactly the point where it's cheap and safe to leave Spark behind.
# MAGIC
# MAGIC - **Customer segmentation — KMeans.** Business use case: group customers
# MAGIC   into behaviorally distinct segments so marketing can run *targeted*
# MAGIC   campaigns instead of one-size-fits-all messaging — a loyalty perk for a
# MAGIC   high-value, frequent, multi-category cluster; a reactivation discount for
# MAGIC   a low-frequency, single-category one. `k` is chosen automatically by
# MAGIC   sweeping a range and picking the silhouette-best value, not fixed by hand.
# MAGIC - **Churn prediction — RandomForestClassifier.** Business use case: flag
# MAGIC   customers at high risk of going inactive in the next `churn_window_days`
# MAGIC   *before* they're gone, so retention marketing can reach them with a
# MAGIC   win-back offer while there's still a relationship to save.
# MAGIC
# MAGIC Both models are saved via `joblib.dump` to the DBFS-mounted `models_dir`
# MAGIC widget path, so they persist independently of this cluster/session.

# COMMAND ----------

try:
    _t0 = start_stage("train_model")

    train_settings = dataclasses.replace(
        load_train_settings(CONFIG_PATH, env_override=ENVIRONMENT),
        kmeans_k_min=widget_int("kmeans_k_min"),
        kmeans_k_max=widget_int("kmeans_k_max"),
        rf_n_estimators=widget_int("rf_n_estimators"),
        models_dir=Path(dbutils.widgets.get("models_dir")),
    )

    clustering_pdf = spark.read.format(train_settings.features_format).load(train_settings.clustering_features_path).toPandas()
    churn_pdf = spark.read.format(train_settings.features_format).load(train_settings.churn_features_path).toPandas()

    segmentation_pipeline, best_k, k_scores = train_segmentation_model(
        clustering_pdf, train_settings.kmeans_k_min, train_settings.kmeans_k_max, train_settings.kmeans_random_state
    )
    save_model(segmentation_pipeline, train_settings.models_dir / CLUSTERING_MODEL_FILENAME)

    _X_train, _X_test, _y_train, _y_test = make_churn_train_test_split(
        churn_pdf, train_settings.rf_test_size, train_settings.rf_random_state
    )
    churn_pipeline = train_churn_model(
        _X_train, _y_train, train_settings.rf_n_estimators, train_settings.rf_max_depth, train_settings.rf_random_state
    )
    save_model(churn_pipeline, train_settings.models_dir / CHURN_MODEL_FILENAME)

    end_stage("train_model", _t0)
except Exception:
    logger.exception("stage_failed", extra={"stage": "train_model"})
    raise

print(f"Selected k = {best_k} (silhouette = {k_scores[best_k]:.4f})")
display(pd.DataFrame({"k": list(k_scores.keys()), "silhouette_score": list(k_scores.values())}))
print(f"Models written to: {train_settings.models_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 7 — Evaluation
# MAGIC
# MAGIC Reloads both models from the joblib files saved above — deliberately, not
# MAGIC reused from the training cell's Python variables — the same "train once,
# MAGIC evaluate any time, even in a fresh session" pattern `src/ml/evaluate.py`
# MAGIC uses on the CLI side, and the only way to actually prove persistence
# MAGIC works rather than just assuming it does.
# MAGIC
# MAGIC - **Segmentation**: silhouette score (cluster separation quality) plus a
# MAGIC   per-cluster feature-mean profile — a silhouette number alone doesn't tell
# MAGIC   a marketer *who* is in cluster 2, the profile does.
# MAGIC - **Churn**: precision/recall/F1 per class, macro-averaged, and ROC-AUC on
# MAGIC   the held-out test split (reproduced via the identical
# MAGIC   `make_churn_train_test_split` call train_model used, so this is
# MAGIC   guaranteed to be rows the model never trained on). Recall on the
# MAGIC   *churned* class is the number that matters most here: missing a customer
# MAGIC   who's about to leave costs a lost customer; a false positive just costs
# MAGIC   one unneeded retention email.

# COMMAND ----------

try:
    _t0 = start_stage("evaluate")

    eval_settings = dataclasses.replace(
        load_evaluate_settings(CONFIG_PATH, env_override=ENVIRONMENT),
        models_dir=Path(dbutils.widgets.get("models_dir")),
        reports_dir=Path(dbutils.widgets.get("reports_dir")),
    )

    eval_clustering_pdf = spark.read.format(eval_settings.features_format).load(eval_settings.clustering_features_path).toPandas()
    eval_churn_pdf = spark.read.format(eval_settings.features_format).load(eval_settings.churn_features_path).toPandas()

    reloaded_segmentation_pipeline = joblib.load(eval_settings.models_dir / CLUSTERING_MODEL_FILENAME)
    reloaded_churn_pipeline = joblib.load(eval_settings.models_dir / CHURN_MODEL_FILENAME)

    segmentation_metrics, cluster_profile_df = evaluate_segmentation_model(
        reloaded_segmentation_pipeline, eval_clustering_pdf
    )
    save_json(segmentation_metrics, eval_settings.reports_dir / "segmentation_metrics.json")
    save_csv(cluster_profile_df, eval_settings.reports_dir / "segmentation_cluster_profile.csv")

    _, eval_X_test, _, eval_y_test = make_churn_train_test_split(
        eval_churn_pdf, eval_settings.rf_test_size, eval_settings.rf_random_state
    )
    churn_metrics, confusion_matrix_array = evaluate_churn_model(reloaded_churn_pipeline, eval_X_test, eval_y_test)
    save_json(churn_metrics, eval_settings.reports_dir / "churn_metrics.json")

    end_stage("evaluate", _t0)
except Exception:
    logger.exception("stage_failed", extra={"stage": "evaluate"})
    raise

print("Segmentation metrics:")
display(pd.DataFrame([segmentation_metrics]))
display(cluster_profile_df.reset_index())

print("\nChurn metrics (test split):")
display(pd.DataFrame([{k: v for k, v in churn_metrics.items() if k != "classification_report"}]))
print(churn_metrics["classification_report"])

_fig, _ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    confusion_matrix_array,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["retained", "churned"],
    yticklabels=["retained", "churned"],
    ax=_ax,
    cbar=False,
)
_ax.set_xlabel("Predicted")
_ax.set_ylabel("Actual")
_ax.set_title("Churn Model -- Confusion Matrix (test split)")
_fig.tight_layout()

display(_fig)  # renders inline in the notebook output

_confusion_matrix_path = eval_settings.reports_dir / "churn_confusion_matrix.png"
_confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
_fig.savefig(_confusion_matrix_path, dpi=150)
plt.close(_fig)

print(f"\nReports written to: {eval_settings.reports_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline complete
# MAGIC
# MAGIC All 7 stages ran against this cluster's single shared Spark session, each
# MAGIC reading its input from Delta rather than from an in-memory variable handed
# MAGIC down by the previous cell. The timing table below is this notebook's
# MAGIC equivalent of `src/pipeline.py`'s printed summary table — same idea
# MAGIC (per-stage wall-clock time, so a slow stage is obvious at a glance), rendered
# MAGIC the Databricks-native way instead of as ASCII.

# COMMAND ----------

display(pd.DataFrame(stage_timings))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scheduling this as a Databricks Job
# MAGIC
# MAGIC Everything above runs interactively against whatever cluster this notebook
# MAGIC happens to be attached to. Turning it into a scheduled production pipeline
# MAGIC is entirely a **Workflows** configuration question — nothing in the notebook
# MAGIC itself needs to change. Here's exactly where each piece lives.
# MAGIC
# MAGIC ### 1. Create the Job
# MAGIC **Workflows → Jobs → Create Job.** Add one task:
# MAGIC - **Type**: Notebook
# MAGIC - **Source**: this repo (Databricks Repos / Git folder), path
# MAGIC   `notebooks/databricks_pipeline`
# MAGIC - **Parameters**: this is where the `dbutils.widgets` values get set for a
# MAGIC   scheduled run — e.g. a production Job can point `config_path` at a
# MAGIC   different `config.yaml`, or widen `churn_window_days`, without touching
# MAGIC   the notebook at all. This is the direct Job-level equivalent of passing
# MAGIC   `--config`/`--env` on the CLI.
# MAGIC
# MAGIC ### 2. Configure the cluster
# MAGIC Attach a **Job cluster**, not an all-purpose interactive one — it spins up
# MAGIC for the run and terminates immediately after, so there's no idle cost
# MAGIC between scheduled runs (the main cost lever for a job that might run nightly
# MAGIC but takes minutes to execute).
# MAGIC - **Databricks Runtime**: an **ML runtime** (e.g. `15.4 LTS ML`) ships
# MAGIC   scikit-learn/pandas/matplotlib/seaborn/joblib preinstalled, so the
# MAGIC   `%pip install` cell above becomes a no-op; on a Standard runtime, keep it.
# MAGIC - **Node type / autoscaling**: this dataset is tens of thousands of rows —
# MAGIC   a small single-node or 2-worker autoscaling cluster
# MAGIC   (e.g. `Standard_DS3_v2` / `i3.xlarge`-class) is plenty. Don't provision
# MAGIC   for a scale this pipeline isn't operating at.
# MAGIC - **Photon**: worth enabling for the Spark-heavy stages (ingest through SQL
# MAGIC   analytics); it doesn't do anything for the pure-Python ML stages.
# MAGIC - **Spark config**: none required beyond defaults — Delta is native to the
# MAGIC   runtime, and `spark.sql.shuffle.partitions` is already set in code
# MAGIC   (Stage 3) rather than needing a cluster-level override.
# MAGIC
# MAGIC ### 3. Schedule it
# MAGIC Job's **Schedule** tab → cron expression (e.g. daily at 02:00, after the
# MAGIC upstream system that produces `transactions.csv` is expected to have
# MAGIC finished writing it). Databricks also supports **file-arrival triggers** —
# MAGIC worth considering here specifically, since it would let a new file landing
# MAGIC in the `/mnt/datalake/raw` mount kick off the run instead of a fixed clock
# MAGIC time.
# MAGIC
# MAGIC ### 4. Reliability
# MAGIC In the Job's **Notifications**/**Retries** settings: configure a retry count
# MAGIC and interval (transient cluster-launch or cloud-provider issues are the
# MAGIC usual reason to retry), a timeout so a stuck run doesn't hold the cluster
# MAGIC indefinitely, and an email/Slack alert on failure. This is exactly why every
# MAGIC stage above re-raises on exception rather than catching and continuing —
# MAGIC without that, a Failed stage 4 could still leave the whole Job looking
# MAGIC Successful, and none of this retry/alerting machinery would ever fire.
# MAGIC
# MAGIC ### 5. Growth path: splitting into a multi-task Job
# MAGIC For now this is one notebook, one task, seven stages run in sequence — the
# MAGIC simplest thing that fully works. If this pipeline grows into something with
# MAGIC real per-stage SLAs (e.g. "ingestion must finish by 3am regardless of
# MAGIC whether training does"), the natural next step is a **multi-task Job**: one
# MAGIC task per stage (either separate notebooks, or this same notebook
# MAGIC parameterized by a `stage` widget, mirroring `src/pipeline.py --stage`),
# MAGIC wired together as a DAG in the Jobs UI. That trades one simple task for
# MAGIC per-stage retries, per-stage timing in the Jobs UI graph view, and the
# MAGIC ability to re-run just the failed stage instead of the whole pipeline — at
# MAGIC the cost of more Job-definition complexity than a single notebook needs
# MAGIC while the pipeline stays this size.
# MAGIC
# MAGIC ### 6. Production identity
# MAGIC Set the Job to **run as** a service principal rather than a personal account
# MAGIC — keeps the schedule working independent of any one person's access, and is
# MAGIC the standard recommendation for anything beyond a personal/demo workflow.
