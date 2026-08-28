"""Interactive analytics dashboard for the e-commerce data platform.

Reads directly from this project's processed data layer and persisted
models -- nothing here recomputes anything Spark or scikit-learn already
did upstream:

- `data/processed/transactions_cleaned` (written by `src/transformation/cleanse.py`)
  for revenue trends and top products/categories -- read as the full fact
  table rather than a pre-aggregated one specifically so every chart here
  can be sliced by date range, region, and category at once, along
  whichever combination of dimensions the pre-aggregated Spark tables
  already collapsed away.
- `data/processed/ml_clustering_features` / `ml_churn_features`
  (`src/ml/feature_engineering.py`) plus `models/customer_segmentation_kmeans.joblib`
  / `models/churn_random_forest.joblib` (`src/ml/train_model.py`) for the
  customer segmentation and churn sections -- the same feature tables and
  models the CLI pipeline trains and evaluates, loaded here for interactive
  exploration rather than a one-shot batch report.

Run it with:
    streamlit run dashboards/app.py

Data/model files need to exist first -- run at least
`python src/pipeline.py --stage ingest cleanse transform feature_engineering train_model`
(or `--stage all`) before starting the dashboard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import load_yaml_config, resolve_environment  # noqa: E402

CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

# --------------------------------------------------------------------------
# Palette -- fixed categorical hues (never reassigned by filter state), a
# status pair reserved for the retained/churned outcome, and a sequential
# ramp for magnitude-only bars. See the dataviz skill for the full method;
# these are its validated default palette's values.
# --------------------------------------------------------------------------

CATEGORICAL_PALETTE = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
REGION_ORDER = ["North", "South", "East", "West", "Central"]
REGION_COLORS = dict(zip(REGION_ORDER, CATEGORICAL_PALETTE))
CLUSTER_MARKER_SYMBOLS = ["circle", "square", "diamond", "cross", "triangle-up", "star", "hexagon", "pentagon"]

STATUS_GOOD = "#0ca30c"      # retained
STATUS_CRITICAL = "#d03b3b"  # churned

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]  # light -> dark, one hue, for magnitude

CHART_SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
MUTED_TEXT = "#898781"

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
CHURN_NUMERIC_FEATURES = CLUSTERING_FEATURE_COLUMNS
CHURN_CATEGORICAL_FEATURES = ["region", "customer_segment"]


# --------------------------------------------------------------------------
# Config-driven paths -- same config.yaml every other stage reads, so this
# dashboard can't silently drift from where the pipeline actually writes.
# --------------------------------------------------------------------------

_raw_config = load_yaml_config(CONFIG_PATH)
_environment, _, _processed_data_dir = resolve_environment(_raw_config, env_override=None)
_processed_data_dir = _processed_data_dir.rstrip("/")

TRANSACTIONS_PATH = Path(f"{_processed_data_dir}/{_raw_config['cleansing']['output_table_name']}")
TRANSACTIONS_FORMAT = _raw_config["cleansing"]["output_format"]

CLUSTERING_FEATURES_PATH = Path(f"{_processed_data_dir}/{_raw_config['ml']['clustering_features_table']}")
CHURN_FEATURES_PATH = Path(f"{_processed_data_dir}/{_raw_config['ml']['churn_features_table']}")
ML_FORMAT = _raw_config["ml"]["output_format"]

MODELS_DIR = ROOT_DIR / _raw_config["ml"]["models_dir"]
REPORTS_DIR = ROOT_DIR / _raw_config["ml"]["reports_dir"]
SEGMENTATION_MODEL_PATH = MODELS_DIR / "customer_segmentation_kmeans.joblib"
CHURN_MODEL_PATH = MODELS_DIR / "churn_random_forest.joblib"


# --------------------------------------------------------------------------
# Data / model loading -- cached so filter changes don't re-read Delta or
# re-load a joblib file on every rerun. `st.cache_data` for DataFrames
# (Streamlit hashes/copies the return value so one tab's edits can't leak
# into another's), `st.cache_resource` for the sklearn Pipelines (loaded
# once and shared by reference -- re-loading a joblib file per rerun would
# be pure waste, and predict() doesn't mutate the pipeline anyway).
# --------------------------------------------------------------------------


def _read_table(path: Path, table_format: str) -> pd.DataFrame:
    """Read a processed table regardless of whether it's Delta or Parquet.

    A Delta table's data files are Parquet underneath, but reading the
    directory with a plain Parquet reader is unsafe in general -- overwritten
    versions can leave stale files on disk until VACUUMed, and a naive
    directory glob has no way to know which files the *current* Delta
    version actually points to. `deltalake` (delta-rs) reads the transaction
    log properly without needing a Spark/JVM session at all, which is the
    whole point of using it here in a pure-Python serving layer.
    """
    if table_format == "delta":
        from deltalake import DeltaTable

        return DeltaTable(str(path)).to_pandas()
    if table_format == "parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {table_format!r}")


@st.cache_data(show_spinner="Loading transactions...")
def load_transactions() -> pd.DataFrame:
    df = _read_table(TRANSACTIONS_PATH, TRANSACTIONS_FORMAT)
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    return df


@st.cache_data(show_spinner="Loading customer segmentation features...")
def load_clustering_features() -> pd.DataFrame:
    return _read_table(CLUSTERING_FEATURES_PATH, ML_FORMAT)


@st.cache_data(show_spinner="Loading churn features...")
def load_churn_features() -> pd.DataFrame:
    return _read_table(CHURN_FEATURES_PATH, ML_FORMAT)


@st.cache_resource(show_spinner="Loading segmentation model...")
def load_segmentation_model():
    return joblib.load(SEGMENTATION_MODEL_PATH)


@st.cache_resource(show_spinner="Loading churn model...")
def load_churn_model():
    return joblib.load(CHURN_MODEL_PATH)


def load_json_report(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------

st.set_page_config(page_title="E-Commerce Analytics", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] { font-size: 1.55rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("E-Commerce Analytics Dashboard")
st.caption(
    "Revenue, customer segmentation, product performance, and churn risk -- "
    "reading directly from this project's processed Delta tables and persisted models."
)

try:
    transactions_df = load_transactions()
    clustering_features_df = load_clustering_features()
    churn_features_df = load_churn_features()
    segmentation_pipeline = load_segmentation_model()
    churn_pipeline = load_churn_model()
except Exception as exc:  # noqa: BLE001 -- any load failure should show this same guidance
    st.error(
        "Couldn't load one of the processed data/model files this dashboard needs.\n\n"
        f"**Details:** {exc}\n\n"
        "Run the pipeline first, then reload this page:\n\n"
        "```\npython src/pipeline.py --stage ingest cleanse transform feature_engineering train_model\n```"
    )
    st.stop()


# --------------------------------------------------------------------------
# Sidebar filters -- scope everything below them. Date range applies to the
# transaction-level tabs (Revenue Trends, Top Products & Categories); region
# also applies to the customer-level tabs (Customer Segments, Churn
# Prediction), which don't carry a per-transaction date to filter on.
# --------------------------------------------------------------------------

st.sidebar.header("Filters")

min_date = transactions_df["purchase_date"].min().date()
max_date = transactions_df["purchase_date"].max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:  # user cleared one bound mid-edit -- fall back to the full range rather than erroring
    start_date, end_date = min_date, max_date

region_options = sorted(transactions_df["region"].dropna().unique().tolist())
selected_regions = st.sidebar.multiselect("Region", options=region_options, default=region_options)

category_options = sorted(transactions_df["category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect("Category", options=category_options, default=category_options)

st.sidebar.divider()
st.sidebar.caption(f"Data as of **{max_date:%b %d, %Y}**")
st.sidebar.caption(f"{len(transactions_df):,} transactions · {len(clustering_features_df):,} customers")

transactions_mask = (
    (transactions_df["purchase_date"].dt.date >= start_date)
    & (transactions_df["purchase_date"].dt.date <= end_date)
    & transactions_df["region"].isin(selected_regions)
    & transactions_df["category"].isin(selected_categories)
)
filtered_transactions_df = transactions_df.loc[transactions_mask]

filtered_clustering_df = clustering_features_df.loc[clustering_features_df["region"].isin(selected_regions)]
filtered_churn_df = churn_features_df.loc[churn_features_df["region"].isin(selected_regions)]


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

tab_revenue, tab_segments, tab_products, tab_churn = st.tabs(
    ["Revenue Trends", "Customer Segments", "Top Products & Categories", "Churn Prediction"]
)

# ---- Revenue Trends -------------------------------------------------------
with tab_revenue:
    if filtered_transactions_df.empty:
        st.warning("No transactions match the current filters.")
    else:
        total_revenue = filtered_transactions_df["total_amount"].sum()
        total_orders = len(filtered_transactions_df)
        avg_order_value = filtered_transactions_df["total_amount"].mean()
        known_customers = filtered_transactions_df.loc[
            filtered_transactions_df["customer_id"] != "UNKNOWN_CUSTOMER", "customer_id"
        ]

        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Total revenue", f"${total_revenue:,.0f}")
        kpi_cols[1].metric("Orders", f"{total_orders:,}")
        kpi_cols[2].metric("Avg order value", f"${avg_order_value:,.2f}")
        kpi_cols[3].metric("Unique customers", f"{known_customers.nunique():,}")

        granularity = st.radio("Granularity", ["Daily", "Weekly", "Monthly"], horizontal=True, index=2)
        resample_freq = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[granularity]

        # pd.Grouper (not groupby().resample()) -- resample-after-groupby is
        # fragile against a non-contiguous index, which is exactly what a
        # boolean-mask filter (the sidebar filters above) produces.
        revenue_trend_df = (
            filtered_transactions_df.groupby([pd.Grouper(key="purchase_date", freq=resample_freq), "region"])[
                "total_amount"
            ]
            .sum()
            .reset_index()
            .rename(columns={"total_amount": "revenue"})
        )

        # Region is the series identity here (5 lines, well within the
        # categorical soft cap) -- color is fixed per region via
        # REGION_COLORS, not re-assigned when the sidebar filter narrows
        # which regions are on screen.
        trend_fig = px.line(
            revenue_trend_df,
            x="purchase_date",
            y="revenue",
            color="region",
            color_discrete_map=REGION_COLORS,
            category_orders={"region": REGION_ORDER},
            labels={"purchase_date": "Date", "revenue": "Revenue ($)", "region": "Region"},
        )
        trend_fig.update_traces(line=dict(width=2), hovertemplate="$%{y:,.0f}")
        trend_fig.update_layout(
            hovermode="x unified",  # one tooltip lists every region's value at that date
            plot_bgcolor=CHART_SURFACE,
            paper_bgcolor=CHART_SURFACE,
            legend_title_text="Region",
            margin=dict(t=20),
        )
        trend_fig.update_xaxes(showgrid=False)
        trend_fig.update_yaxes(gridcolor=GRIDLINE, tickprefix="$")
        st.plotly_chart(trend_fig, use_container_width=True)

        with st.expander("View as table"):
            pivoted_trend_df = revenue_trend_df.pivot(index="purchase_date", columns="region", values="revenue").round(2)
            st.dataframe(pivoted_trend_df, use_container_width=True)

# ---- Customer Segments ------------------------------------------------------
with tab_segments:
    if filtered_clustering_df.empty:
        st.warning("No customers match the current region filter.")
    else:
        st.caption("Scoped by the sidebar's region filter (not the date range -- these are full-history customer features).")

        X_clustering = filtered_clustering_df[CLUSTERING_FEATURE_COLUMNS]
        cluster_labels = segmentation_pipeline.predict(X_clustering)

        plot_df = filtered_clustering_df.copy()
        plot_df["cluster"] = cluster_labels.astype(str)
        cluster_ids = sorted(plot_df["cluster"].unique(), key=int)
        cluster_color_map = {cid: CATEGORICAL_PALETTE[int(cid) % len(CATEGORICAL_PALETTE)] for cid in cluster_ids}
        cluster_symbol_map = {
            cid: CLUSTER_MARKER_SYMBOLS[int(cid) % len(CLUSTER_MARKER_SYMBOLS)] for cid in cluster_ids
        }

        cluster_sizes_df = plot_df["cluster"].value_counts().reindex(cluster_ids).reset_index()
        cluster_sizes_df.columns = ["cluster", "customers"]

        left_col, right_col = st.columns([1, 2])
        with left_col:
            st.subheader("Cluster sizes")
            size_fig = px.bar(
                cluster_sizes_df,
                x="cluster",
                y="customers",
                color="cluster",
                color_discrete_map=cluster_color_map,
                labels={"cluster": "Cluster", "customers": "Customers"},
            )
            size_fig.update_layout(
                showlegend=False, plot_bgcolor=CHART_SURFACE, paper_bgcolor=CHART_SURFACE, margin=dict(t=10)
            )
            size_fig.update_xaxes(showgrid=False, type="category")
            size_fig.update_yaxes(gridcolor=GRIDLINE)
            st.plotly_chart(size_fig, use_container_width=True)

        with right_col:
            st.subheader("Segment explorer")
            axis_col1, axis_col2 = st.columns(2)
            x_feature = axis_col1.selectbox(
                "X-axis", CLUSTERING_FEATURE_COLUMNS, index=CLUSTERING_FEATURE_COLUMNS.index("recency_days")
            )
            y_feature = axis_col2.selectbox(
                "Y-axis", CLUSTERING_FEATURE_COLUMNS, index=CLUSTERING_FEATURE_COLUMNS.index("monetary")
            )

            scatter_fig = px.scatter(
                plot_df,
                x=x_feature,
                y=y_feature,
                color="cluster",
                symbol="cluster",
                color_discrete_map=cluster_color_map,
                symbol_map=cluster_symbol_map,
                hover_data=["customer_id", "region", "customer_segment_label"],
                labels={x_feature: x_feature.replace("_", " ").title(), y_feature: y_feature.replace("_", " ").title()},
            )
            scatter_fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0)))
            scatter_fig.update_layout(
                plot_bgcolor=CHART_SURFACE, paper_bgcolor=CHART_SURFACE, legend_title_text="Cluster", margin=dict(t=10)
            )
            scatter_fig.update_xaxes(gridcolor=GRIDLINE)
            scatter_fig.update_yaxes(gridcolor=GRIDLINE)
            st.plotly_chart(scatter_fig, use_container_width=True)

        with st.expander("2D PCA projection (all 9 features at once, instead of picking 2)"):
            X_scaled = segmentation_pipeline.named_steps["scaler"].transform(X_clustering)
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_scaled)
            pca_df = plot_df[["customer_id", "cluster"]].copy()
            pca_df["pc1"], pca_df["pc2"] = coords[:, 0], coords[:, 1]

            pca_fig = px.scatter(
                pca_df,
                x="pc1",
                y="pc2",
                color="cluster",
                symbol="cluster",
                color_discrete_map=cluster_color_map,
                symbol_map=cluster_symbol_map,
                labels={
                    "pc1": f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)",
                    "pc2": f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)",
                },
            )
            pca_fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0)))
            pca_fig.update_layout(
                plot_bgcolor=CHART_SURFACE, paper_bgcolor=CHART_SURFACE, legend_title_text="Cluster"
            )
            pca_fig.update_xaxes(gridcolor=GRIDLINE)
            pca_fig.update_yaxes(gridcolor=GRIDLINE)
            st.plotly_chart(pca_fig, use_container_width=True)

        st.subheader("Cluster profile (average feature values)")
        st.caption("Computed from the customers currently in view, so it always matches the filter above.")
        profile_df = plot_df.groupby("cluster")[CLUSTERING_FEATURE_COLUMNS].mean().round(2)
        profile_df.insert(0, "customers", plot_df.groupby("cluster").size())
        st.dataframe(profile_df.reset_index(), use_container_width=True, hide_index=True)

        with st.expander("Model performance (from the last evaluation run)"):
            seg_metrics = load_json_report(REPORTS_DIR / "segmentation_metrics.json")
            if seg_metrics:
                metric_cols = st.columns(2)
                metric_cols[0].metric("Silhouette score", f"{seg_metrics['silhouette_score']:.3f}")
                metric_cols[1].metric("Clusters (k)", seg_metrics["n_clusters"])
            else:
                st.caption("Run `python src/ml/evaluate.py` to generate this report.")

# ---- Top Products & Categories --------------------------------------------
with tab_products:
    if filtered_transactions_df.empty:
        st.warning("No transactions match the current filters.")
    else:
        control_col1, control_col2 = st.columns([1, 2])
        view_mode = control_col1.radio("View by", ["Product", "Category"], horizontal=True)
        top_n = control_col2.slider("Show top N", min_value=5, max_value=25, value=10)

        if view_mode == "Product":
            ranked_df = (
                filtered_transactions_df.groupby(["product_id", "category"], as_index=False)["total_amount"]
                .sum()
                .rename(columns={"total_amount": "revenue"})
                .sort_values("revenue", ascending=False)
                .head(top_n)
            )
            axis_col, axis_label = "product_id", "Product"
        else:
            ranked_df = (
                filtered_transactions_df.groupby("category", as_index=False)["total_amount"]
                .sum()
                .rename(columns={"total_amount": "revenue"})
                .sort_values("revenue", ascending=False)
                .head(top_n)
            )
            axis_col, axis_label = "category", "Category"

        # Nominal bars ranked by a single magnitude (revenue) -- one
        # sequential hue keyed to the value itself, not a rainbow per bar
        # (each bar isn't its own "series"; there's exactly one measure here).
        bar_fig = px.bar(
            ranked_df.sort_values("revenue"),  # ascending so the largest bar renders at the top
            x="revenue",
            y=axis_col,
            orientation="h",
            color="revenue",
            color_continuous_scale=SEQUENTIAL_BLUE,
            labels={"revenue": "Revenue ($)", axis_col: axis_label},
        )
        bar_fig.update_layout(
            plot_bgcolor=CHART_SURFACE,
            paper_bgcolor=CHART_SURFACE,
            coloraxis_showscale=False,  # redundant with bar length; hover + table carry the exact value
            margin=dict(t=20),
        )
        bar_fig.update_xaxes(gridcolor=GRIDLINE, tickprefix="$")
        bar_fig.update_yaxes(showgrid=False)
        st.plotly_chart(bar_fig, use_container_width=True)

        st.dataframe(
            ranked_df.sort_values("revenue", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={"revenue": st.column_config.NumberColumn("Revenue", format="$%.2f")},
        )

# ---- Churn Prediction -------------------------------------------------------
with tab_churn:
    if filtered_churn_df.empty:
        st.warning("No customers match the current region filter.")
    else:
        st.caption("Scoped by the sidebar's region filter (not the date range -- churn features are already a fixed pre-cutoff window; see feature_engineering.py).")

        X_churn = filtered_churn_df[CHURN_NUMERIC_FEATURES + CHURN_CATEGORICAL_FEATURES]
        churn_probability = churn_pipeline.predict_proba(X_churn)[:, 1]

        predictions_df = filtered_churn_df.copy()
        predictions_df["churn_probability"] = churn_probability
        predictions_df["predicted_label"] = np.where(churn_probability >= 0.5, "Churned", "Retained")
        predictions_df["actual_label"] = np.where(predictions_df["churned"] == 1, "Churned", "Retained")

        kpi_cols = st.columns(3)
        kpi_cols[0].metric("Customers", f"{len(predictions_df):,}")
        kpi_cols[1].metric("Actual churn rate", f"{predictions_df['churned'].mean():.1%}")
        kpi_cols[2].metric("Predicted churn rate", f"{(churn_probability >= 0.5).mean():.1%}")

        st.subheader("Predicted churn probability distribution")
        st.caption("Split by actual outcome -- a model that separates these two well pushes them toward opposite ends.")
        # Retained/churned is a status (good/bad), not a generic "series 4" --
        # it gets the reserved status pair, not a categorical slot.
        hist_fig = px.histogram(
            predictions_df,
            x="churn_probability",
            color="actual_label",
            color_discrete_map={"Retained": STATUS_GOOD, "Churned": STATUS_CRITICAL},
            category_orders={"actual_label": ["Retained", "Churned"]},
            nbins=30,
            barmode="overlay",
            opacity=0.65,
            labels={"churn_probability": "Predicted churn probability", "actual_label": "Actual outcome"},
        )
        hist_fig.add_vline(x=0.5, line_dash="dash", line_color=MUTED_TEXT, annotation_text="0.5 threshold")
        hist_fig.update_layout(
            plot_bgcolor=CHART_SURFACE, paper_bgcolor=CHART_SURFACE, legend_title_text="Actual outcome", margin=dict(t=20)
        )
        hist_fig.update_xaxes(gridcolor=GRIDLINE, tickformat=".0%")
        hist_fig.update_yaxes(gridcolor=GRIDLINE, title="Customers")
        st.plotly_chart(hist_fig, use_container_width=True)

        st.subheader("At-risk customers")
        threshold = st.slider("Minimum predicted churn probability", 0.0, 1.0, 0.5, 0.05)
        table_columns = [
            "customer_id",
            "churn_probability",
            "predicted_label",
            "actual_label",
            "recency_days",
            "frequency",
            "monetary",
            "region",
            "customer_segment",
        ]
        at_risk_df = (
            predictions_df.loc[predictions_df["churn_probability"] >= threshold, table_columns]
            .sort_values("churn_probability", ascending=False)
        )
        st.dataframe(
            at_risk_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "churn_probability": st.column_config.ProgressColumn(
                    "Churn probability", min_value=0.0, max_value=1.0, format="%.0f%%"
                ),
                "monetary": st.column_config.NumberColumn("Lifetime value", format="$%.2f"),
            },
        )
        st.caption(f"{len(at_risk_df):,} of {len(predictions_df):,} customers at or above {threshold:.0%} predicted risk.")

        with st.expander("Model performance (from the last evaluation run)"):
            churn_metrics = load_json_report(REPORTS_DIR / "churn_metrics.json")
            if churn_metrics:
                metric_cols = st.columns(4)
                metric_cols[0].metric("Accuracy", f"{churn_metrics['accuracy']:.1%}")
                metric_cols[1].metric("ROC-AUC", f"{churn_metrics['roc_auc']:.3f}")
                metric_cols[2].metric("Recall (churned)", f"{churn_metrics['churned']['recall']:.1%}")
                metric_cols[3].metric("Precision (churned)", f"{churn_metrics['churned']['precision']:.1%}")
            else:
                st.caption("Run `python src/ml/evaluate.py` to generate this report.")

            confusion_matrix_path = REPORTS_DIR / "churn_confusion_matrix.png"
            if confusion_matrix_path.exists():
                st.image(str(confusion_matrix_path), caption="Confusion matrix (held-out test split)", width=380)
