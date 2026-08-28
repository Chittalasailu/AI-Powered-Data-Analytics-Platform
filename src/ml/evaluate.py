"""Evaluate the two persisted models from `train_model.py` and write their
metrics (and a confusion matrix plot) to `/reports`, for whoever has to sign
off on a model before it's used to build a marketing segment list or a
retention-outreach list.

- **Customer segmentation (KMeans).** A silhouette score alone tells you
  *whether* the clusters are well-separated, not *who* is in them, and a
  marketer deciding whether cluster 2 deserves a loyalty perk or a win-back
  discount needs the latter -- so this also writes a per-cluster feature
  profile (mean of every input feature, per cluster) alongside the score.

- **Churn prediction (RandomForestClassifier).** Reports precision,
  recall, and F1 per class plus macro-averages, and ROC-AUC, on the held-out
  test split -- the same split `train_model.py` created, reproduced here
  from the same `random_state` rather than re-derived by hand. For a churn
  model specifically, recall on the *churned* class is the number that
  matters most to the business: a false negative (missing a customer who's
  about to leave) is a lost customer, while a false positive (flagging a
  customer who was never at risk) just costs one unneeded retention email
  -- worth stating explicitly, since overall accuracy alone would hide that
  asymmetry.

Usage:
    python src/ml/evaluate.py [--config config/config.yaml] [--env local|databricks]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import matplotlib

matplotlib.use("Agg")  # write PNGs directly; no display/backend needed in a batch script
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ml.train_model import (  # noqa: E402
    CHURN_MODEL_FILENAME,
    CLUSTERING_FEATURE_COLUMNS,
    CLUSTERING_MODEL_FILENAME,
    make_churn_train_test_split,
)
from src.utils.config import load_yaml_config, resolve_environment  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402
from src.utils.spark_session import build_spark_session, get_dbutils, path_exists  # noqa: E402

DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

logger = logging.getLogger(__name__)

CHURN_CLASS_NAMES = ["retained", "churned"]


# --------------------------------------------------------------------------
# Customer segmentation evaluation
# --------------------------------------------------------------------------


def evaluate_segmentation_model(pipeline: Pipeline, features_pdf: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Silhouette score + a per-cluster mean-feature profile.

    Evaluated on the same full feature set the model was fit on -- unlike
    the churn classifier, there's no held-out split here, because
    clustering isn't predicting an outcome on unseen future data; it's
    describing structure in the customers the model has already seen, and
    that's exactly what a marketer using it wants described.
    """
    X = features_pdf[CLUSTERING_FEATURE_COLUMNS]
    cluster_labels = pipeline.predict(X)
    X_scaled = pipeline.named_steps["scaler"].transform(X)

    score = silhouette_score(X_scaled, cluster_labels)

    profiled = features_pdf.copy()
    profiled["cluster"] = cluster_labels
    cluster_sizes = profiled["cluster"].value_counts().sort_index()
    cluster_profile = profiled.groupby("cluster")[CLUSTERING_FEATURE_COLUMNS].mean().round(2)

    metrics = {
        "silhouette_score": round(float(score), 4),
        "n_clusters": int(pipeline.named_steps["kmeans"].n_clusters),
        "n_customers": int(len(features_pdf)),
        "cluster_sizes": {int(k): int(v) for k, v in cluster_sizes.items()},
    }
    return metrics, cluster_profile


# --------------------------------------------------------------------------
# Churn model evaluation
# --------------------------------------------------------------------------


def evaluate_churn_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, np.ndarray]:
    """Precision/recall/F1 (per class + macro) and ROC-AUC on the test split, plus a confusion matrix."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[0, 1], zero_division=0
    )

    metrics = {
        "test_rows": int(len(y_test)),
        "test_churn_rate": round(float(y_test.mean()), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "retained": {
            "precision": round(float(precision[0]), 4),
            "recall": round(float(recall[0]), 4),
            "f1": round(float(f1[0]), 4),
            "support": int(support[0]),
        },
        "churned": {
            "precision": round(float(precision[1]), 4),
            "recall": round(float(recall[1]), 4),
            "f1": round(float(f1[1]), 4),
            "support": int(support[1]),
        },
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "classification_report": classification_report(y_test, y_pred, target_names=CHURN_CLASS_NAMES, zero_division=0),
    }
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return metrics, cm


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CHURN_CLASS_NAMES,
        yticklabels=CHURN_CLASS_NAMES,
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Churn Model -- Confusion Matrix (test split)")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Report writing
# --------------------------------------------------------------------------


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("report_written", extra={"output_path": str(path)})


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    logger.info("report_written", extra={"output_path": str(path)})


# --------------------------------------------------------------------------
# Run settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationSettings:
    environment: str
    app_name: str
    clustering_features_path: str
    churn_features_path: str
    features_format: str
    models_dir: Path
    reports_dir: Path
    rf_test_size: float
    rf_random_state: int
    log_level: str


def load_run_settings(config_path: Path, env_override: Optional[str] = None) -> EvaluationSettings:
    raw_config = load_yaml_config(config_path)
    environment, _, processed_data_dir = resolve_environment(raw_config, env_override)
    processed_data_dir = processed_data_dir.rstrip("/")
    ml_cfg = raw_config["ml"]
    rf_cfg = ml_cfg["random_forest"]

    return EvaluationSettings(
        environment=environment,
        app_name=f"{raw_config['spark']['app_name']}-ml-evaluate",
        clustering_features_path=f"{processed_data_dir}/{ml_cfg['clustering_features_table']}",
        churn_features_path=f"{processed_data_dir}/{ml_cfg['churn_features_table']}",
        features_format=ml_cfg["output_format"],
        models_dir=ROOT_DIR / ml_cfg["models_dir"],
        reports_dir=ROOT_DIR / ml_cfg["reports_dir"],
        rf_test_size=float(rf_cfg.get("test_size", 0.2)),
        rf_random_state=int(rf_cfg.get("random_state", 42)),
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

    for model_filename in (CLUSTERING_MODEL_FILENAME, CHURN_MODEL_FILENAME):
        model_path = settings.models_dir / model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path} -- run train_model.py first")

    spark = build_spark_session(
        settings.environment, settings.app_name, use_delta=settings.features_format == "delta"
    )
    dbutils = get_dbutils(spark) if settings.environment == "databricks" else None

    try:
        for path in (settings.clustering_features_path, settings.churn_features_path):
            if not path_exists(path, settings.environment, dbutils):
                raise FileNotFoundError(f"Input path does not exist: {path}")

        clustering_pdf = spark.read.format(settings.features_format).load(settings.clustering_features_path).toPandas()
        churn_pdf = spark.read.format(settings.features_format).load(settings.churn_features_path).toPandas()
        logger.info(
            "features_loaded",
            extra={"clustering_rows": len(clustering_pdf), "churn_rows": len(churn_pdf)},
        )
    finally:
        if settings.environment == "local":
            spark.stop()

    # ---- Pure pandas / scikit-learn / matplotlib from here on ----

    try:
        segmentation_pipeline = joblib.load(settings.models_dir / CLUSTERING_MODEL_FILENAME)
        segmentation_metrics, cluster_profile = evaluate_segmentation_model(segmentation_pipeline, clustering_pdf)
        logger.info("segmentation_evaluated", extra=segmentation_metrics)
        save_json(segmentation_metrics, settings.reports_dir / "segmentation_metrics.json")
        save_csv(cluster_profile, settings.reports_dir / "segmentation_cluster_profile.csv")

        churn_pipeline = joblib.load(settings.models_dir / CHURN_MODEL_FILENAME)
        # Reproduces train_model.py's exact split (same function, same
        # test_size/random_state) so this evaluates strictly on rows the
        # model never trained on.
        _, X_test, _, y_test = make_churn_train_test_split(churn_pdf, settings.rf_test_size, settings.rf_random_state)
        churn_metrics, confusion = evaluate_churn_model(churn_pipeline, X_test, y_test)
        logger.info(
            "churn_evaluated",
            extra={k: v for k, v in churn_metrics.items() if k != "classification_report"},
        )
        save_json(churn_metrics, settings.reports_dir / "churn_metrics.json")
        plot_confusion_matrix(confusion, settings.reports_dir / "churn_confusion_matrix.png")

        logger.info("evaluation_completed", extra={"reports_dir": str(settings.reports_dir)})
    except Exception:
        logger.exception("evaluation_failed")
        raise


if __name__ == "__main__":
    main()
