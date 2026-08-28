"""Train the two customer-analytics models this project's marketing and
retention teams consume.

1. **Customer segmentation -- KMeans clustering (unsupervised).**
   Business use case: group customers into behaviorally distinct segments
   so marketing can run *targeted* campaigns instead of one-size-fits-all
   messaging -- a loyalty perk for a high-value, frequent, multi-category
   cluster; a reactivation discount for a low-frequency, single-category
   cluster; a welcome series for a low-tenure cluster. This complements
   `transform.py`'s rule-based RFM quartile labels (fixed thresholds on R,
   F, and M scored independently) with a *data-driven* grouping that finds
   its own boundaries across all the behavioral features at once, rather
   than three independent quartile cuts.

2. **Churn prediction -- RandomForestClassifier (supervised).**
   Business use case: flag customers at high risk of going inactive in the
   next `churn_window_days` (default 90) *before* they're gone, so
   retention marketing can reach them with a win-back offer while there's
   still a relationship to save, instead of noticing the loss 90 days
   later in a revenue report. Trained on features computed strictly before
   a cutoff date and labeled on activity strictly after it (see
   `feature_engineering.py`) -- the same information split the model would
   actually have at prediction time in production, so its offline
   performance means something.

Both models are scikit-learn `Pipeline`s (preprocessing + estimator bundled
into one fitted object) trained on pandas DataFrames. The feature tables
themselves are built in Spark (`feature_engineering.py`); this module reads
them and converts to pandas *immediately* -- both tables are already
aggregated down to one row per customer (a few thousand rows), so the
conversion is cheap and safe here specifically because it happens after
Spark has done the heavy row-level aggregation, not before. Everything from
that conversion onward (splitting, scaling, fitting, persistence) is plain
pandas/scikit-learn/joblib, with no further Spark involvement.

Usage:
    python src/ml/train_model.py [--config config/config.yaml] [--env local|databricks]
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import load_yaml_config, resolve_environment  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402
from src.utils.spark_session import build_spark_session, get_dbutils, path_exists  # noqa: E402

DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

logger = logging.getLogger(__name__)

# Numeric features the KMeans model clusters on. Deliberately *not* joined
# with one-hot-encoded region/customer_segment: mixing a handful of sparse
# 0/1 dimensions into a Euclidean-distance clustering alongside continuous
# RFM/behavioral features tends to dilute the geometry those continuous
# features actually carry, so the categorical columns ride along in the
# feature table as descriptive/profiling fields (see evaluate.py's cluster
# profile) rather than model inputs.
CLUSTERING_FEATURE_COLUMNS = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "distinct_categories",
    "category_entropy",
    "distinct_payment_methods",
    "tenure_days",
    "customer_age",
]

CHURN_NUMERIC_FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "distinct_categories",
    "category_entropy",
    "distinct_payment_methods",
    "tenure_days",
    "customer_age",
]
CHURN_CATEGORICAL_FEATURES = ["region", "customer_segment"]
CHURN_LABEL_COLUMN = "churned"

CLUSTERING_MODEL_FILENAME = "customer_segmentation_kmeans.joblib"
CHURN_MODEL_FILENAME = "churn_random_forest.joblib"


# --------------------------------------------------------------------------
# Customer segmentation (KMeans)
# --------------------------------------------------------------------------


def select_best_k(X_scaled, k_min: int, k_max: int, random_state: int) -> tuple[int, dict[int, float]]:
    """Pick the k in [k_min, k_max] with the highest silhouette score.

    Silhouette, not inertia/the elbow method: inertia decreases
    monotonically as k grows by construction (more clusters can only fit
    the data at least as well), so it never actually chooses a k on its
    own without a human eyeballing a bend in the curve. Silhouette
    directly scores how well-separated and internally cohesive the
    resulting clusters are, which is the property that actually matters
    for "are these customer segments meaningfully different from each
    other" -- so it can pick a winner automatically.
    """
    scores: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def train_segmentation_model(
    features_pdf: pd.DataFrame, k_min: int, k_max: int, random_state: int
) -> tuple[Pipeline, int, dict[int, float]]:
    """Fit the customer-segmentation KMeans pipeline.

    Returns the fitted `Pipeline` (scaler + kmeans, so a caller only ever
    needs to hand it raw feature rows), the chosen k, and the full
    k -> silhouette-score sweep (useful for `evaluate.py`/reporting to show
    the chosen k wasn't arbitrary).
    """
    X = features_pdf[CLUSTERING_FEATURE_COLUMNS]

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    best_k, k_scores = select_best_k(X_scaled, k_min, k_max, random_state)
    logger.info("kmeans_k_selected", extra={"best_k": best_k, "k_scores": {k: round(v, 4) for k, v in k_scores.items()}})

    kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10).fit(X_scaled)

    # Reuses the already-fitted scaler and kmeans rather than re-fitting a
    # fresh pipeline from scratch on the same data a second time.
    pipeline = Pipeline([("scaler", scaler), ("kmeans", kmeans)])
    return pipeline, best_k, k_scores


# --------------------------------------------------------------------------
# Churn prediction (RandomForest)
# --------------------------------------------------------------------------


def make_churn_train_test_split(features_pdf: pd.DataFrame, test_size: float, random_state: int):
    """Split churn features into train/test.

    Factored out (rather than inlined in `train_churn_model`) so
    `evaluate.py` can import this exact function and reproduce the
    identical split -- same columns, same `test_size`, same
    `random_state`, same `stratify` -- rather than re-implementing
    train/test splitting a second time in a way that could quietly drift
    out of sync and let evaluation see rows the model was trained on.

    Stratified on the label: churn is close to balanced in this dataset
    (~47/53), but stratifying costs nothing when the classes are near
    parity and prevents a train/test skew if that ever changes (e.g. after
    a churn_window_days config change makes the label rarer).
    """
    X = features_pdf[CHURN_NUMERIC_FEATURES + CHURN_CATEGORICAL_FEATURES]
    y = features_pdf[CHURN_LABEL_COLUMN]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_churn_model(
    X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int, max_depth: int, random_state: int
) -> Pipeline:
    """Fit the churn-prediction RandomForest pipeline.

    `class_weight="balanced"` re-weights each class inversely to its
    frequency during training. The label here is close to balanced so this
    matters less than it would on a typical churn dataset (real-world
    churn is often 5-10% positive), but it's included because a churn
    model is exactly the kind of model where silently defaulting to
    "predict the majority class" is a realistic failure mode worth
    guarding against by default, not something to bolt on only after
    noticing it in production.

    Preprocessing (scaling the numeric features, one-hot encoding
    region/customer_segment) and the classifier are one `Pipeline` so a
    single `joblib.dump` captures everything needed to score a new,
    unscaled, un-encoded row at inference time.
    """
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), CHURN_NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CHURN_CATEGORICAL_FEATURES),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("model_saved", extra={"output_path": str(path)})


# --------------------------------------------------------------------------
# Run settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainModelSettings:
    environment: str
    app_name: str
    clustering_features_path: str
    churn_features_path: str
    features_format: str
    models_dir: Path
    kmeans_k_min: int
    kmeans_k_max: int
    kmeans_random_state: int
    rf_n_estimators: int
    rf_max_depth: int
    rf_test_size: float
    rf_random_state: int
    log_level: str


def load_run_settings(config_path: Path, env_override: Optional[str] = None) -> TrainModelSettings:
    raw_config = load_yaml_config(config_path)
    environment, _, processed_data_dir = resolve_environment(raw_config, env_override)
    processed_data_dir = processed_data_dir.rstrip("/")
    ml_cfg = raw_config["ml"]
    kmeans_cfg = ml_cfg["kmeans"]
    rf_cfg = ml_cfg["random_forest"]

    return TrainModelSettings(
        environment=environment,
        app_name=f"{raw_config['spark']['app_name']}-ml-train",
        clustering_features_path=f"{processed_data_dir}/{ml_cfg['clustering_features_table']}",
        churn_features_path=f"{processed_data_dir}/{ml_cfg['churn_features_table']}",
        features_format=ml_cfg["output_format"],
        models_dir=ROOT_DIR / ml_cfg["models_dir"],
        kmeans_k_min=int(kmeans_cfg.get("k_min", 3)),
        kmeans_k_max=int(kmeans_cfg.get("k_max", 8)),
        kmeans_random_state=int(kmeans_cfg.get("random_state", 42)),
        rf_n_estimators=int(rf_cfg.get("n_estimators", 300)),
        rf_max_depth=int(rf_cfg.get("max_depth", 10)),
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

    spark = build_spark_session(
        settings.environment, settings.app_name, use_delta=settings.features_format == "delta"
    )
    dbutils = get_dbutils(spark) if settings.environment == "databricks" else None

    # Everything Spark needs to do -- read the two feature tables -- happens
    # in this block. `spark.stop()` right after means the rest of `main()`
    # (training, persistence) provably doesn't depend on Spark being alive.
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

    # ---- Pure pandas / scikit-learn / joblib from here on ----

    try:
        segmentation_pipeline, best_k, k_scores = train_segmentation_model(
            clustering_pdf, settings.kmeans_k_min, settings.kmeans_k_max, settings.kmeans_random_state
        )
        save_model(segmentation_pipeline, settings.models_dir / CLUSTERING_MODEL_FILENAME)
        logger.info("segmentation_model_trained", extra={"k": best_k, "silhouette": round(k_scores[best_k], 4)})

        X_train, X_test, y_train, y_test = make_churn_train_test_split(
            churn_pdf, settings.rf_test_size, settings.rf_random_state
        )
        churn_pipeline = train_churn_model(
            X_train, y_train, settings.rf_n_estimators, settings.rf_max_depth, settings.rf_random_state
        )
        save_model(churn_pipeline, settings.models_dir / CHURN_MODEL_FILENAME)
        logger.info(
            "churn_model_trained",
            extra={"train_rows": len(X_train), "test_rows": len(X_test), "train_churn_rate": round(y_train.mean(), 4)},
        )
    except Exception:
        logger.exception("training_failed")
        raise


if __name__ == "__main__":
    main()
