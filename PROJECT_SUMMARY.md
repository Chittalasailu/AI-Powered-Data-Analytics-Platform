# AI-Powered E-Commerce Analytics Platform

An end-to-end data engineering + ML platform I built to mirror what a real
production analytics stack looks like: a PySpark/Delta Lake pipeline that
ingests, cleans, aggregates, and models ~50K e-commerce transactions, ships
two ways (a local CLI and a Databricks notebook), and serves the results
through an interactive Streamlit dashboard — all driven by one shared
config file.

**What it does**

- Ingests and validates raw transaction data against an explicit schema,
  then runs it through a cleansing pipeline: deduplication, IQR-based
  outlier capping, and per-column null imputation (median / mode /
  sentinel, chosen by each column's statistical role).
- Builds 5 analytics-ready Delta tables — daily/monthly revenue by region
  and category, customer RFM segmentation, top products, payment mix — with
  production-grade Spark optimizations (broadcast joins, a repartition
  chosen specifically to let a downstream shuffle be skipped entirely,
  caching with a stated reason, `approxQuantile` instead of an unpartitioned
  window for quartile scoring).
- Runs an 11-query Spark SQL analytics catalog (revenue trends, customer
  LTV ranking, cohort retention, regional performance), plus a set of
  documented before/after query optimizations backed by real captured
  `EXPLAIN` plans.
- Trains a KMeans customer-segmentation model (silhouette-driven k
  selection) and a temporally leakage-safe RandomForest churn classifier
  (features strictly before a cutoff date, labels strictly after it),
  evaluated with the appropriate metrics for each — silhouette for the
  former, precision/recall/F1/ROC-AUC + confusion matrix for the latter —
  and persisted as single `joblib` pipeline artifacts.
- Serves all of it through a 4-tab Streamlit dashboard (revenue trends,
  customer segments, top products, churn risk) with sidebar filters that
  scope every chart consistently, built against a validated
  accessibility-aware color methodology.
- Ships identically as a CLI pipeline (`src/pipeline.py --stage all`, with
  per-stage timing and fail-fast error handling) and a Databricks notebook
  (`dbutils.widgets` parameters, native `display()` output, documented Job/
  cluster scheduling configuration) — one `config.yaml`, two execution
  environments.
- Backed by a pytest suite (schema validation, cleansing logic, aggregation
  functions, ML feature-table shape) with a shared local `SparkSession`
  fixture.

**A finding I verified rather than assumed:** the churn model's ROC-AUC
lands at ~0.50. I confirmed this isn't a leakage bug or an implementation
error — it's the correct, honest result for a dataset whose purchase dates
are generated independently per transaction, so there's no genuine
persistent customer-level signal to find. A near-random AUC here is
actually reassuring: a leaky pipeline would have shown suspiciously *good*
performance instead. Same architecture, run against data with real temporal
customer behavior, would be expected to show real lift.

**Stack:** PySpark · Delta Lake · scikit-learn · Streamlit · Plotly ·
pandas · Databricks · pytest

**Repo:** _add your GitHub link here_
