"""Generate a synthetic e-commerce transactions dataset with realistic data quality issues.

Run: python src/utils/generate_sample_data.py [--rows 50000] [--seed 42]
Output: data/raw/transactions.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT_DIR / "data" / "raw" / "transactions.csv"

CATEGORY_PRICE_RANGES = {
    "Electronics": (15, 1200),
    "Clothing": (8, 150),
    "Home & Kitchen": (5, 500),
    "Books": (5, 60),
    "Beauty & Personal Care": (3, 120),
    "Sports & Outdoors": (10, 400),
    "Toys & Games": (5, 150),
    "Grocery": (1, 50),
    "Automotive": (10, 800),
    "Health & Wellness": (5, 200),
}
REGIONS = ["North", "South", "East", "West", "Central"]
PAYMENT_METHODS = [
    "Credit Card",
    "Debit Card",
    "PayPal",
    "UPI",
    "Net Banking",
    "Cash on Delivery",
    "Gift Card",
]
CUSTOMER_SEGMENTS = ["New", "Regular", "Premium", "VIP"]
CUSTOMER_SEGMENT_WEIGHTS = [0.25, 0.45, 0.20, 0.10]

N_CUSTOMERS = 6000
N_PRODUCTS = 800
DATE_START = pd.Timestamp("2023-01-01")
DATE_END = pd.Timestamp("2025-06-30")

DATE_FORMATS = [
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%d %b %Y",
    "%B %d, %Y",
    "%Y-%m-%dT%H:%M:%S",
    "%m-%d-%Y %H:%M",
]
DATE_FORMAT_WEIGHTS = [0.40, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04]
GARBAGE_DATE_VALUES = ["31/02/2024", "0000-00-00", "unknown", "N/A", "13/13/2023", ""]


def _build_product_catalog(rng: np.random.Generator) -> pd.DataFrame:
    categories = list(CATEGORY_PRICE_RANGES.keys())
    product_categories = rng.choice(categories, size=N_PRODUCTS)
    base_prices = np.array(
        [rng.uniform(*CATEGORY_PRICE_RANGES[cat]) for cat in product_categories]
    )
    return pd.DataFrame(
        {
            "product_id": [f"PROD{i:04d}" for i in range(1, N_PRODUCTS + 1)],
            "category": product_categories,
            "base_price": base_prices.round(2),
        }
    )


def _build_base_transactions(n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    customers = [f"CUST{i:05d}" for i in range(1, N_CUSTOMERS + 1)]
    products = _build_product_catalog(rng)

    product_idx = rng.integers(0, N_PRODUCTS, size=n_rows)
    chosen_products = products.iloc[product_idx].reset_index(drop=True)

    quantities = rng.choice(
        [1, 2, 3, 4, 5, 6, 7, 8, 10],
        size=n_rows,
        p=[0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01],
    )
    price_variation = rng.uniform(0.85, 1.15, size=n_rows)
    unit_prices = (chosen_products["base_price"].to_numpy() * price_variation).round(2)
    total_amounts = (quantities * unit_prices).round(2)

    date_offsets_days = rng.integers(0, (DATE_END - DATE_START).days + 1, size=n_rows)
    seconds_in_day = rng.integers(0, 86400, size=n_rows)
    purchase_datetimes = (
        DATE_START + pd.to_timedelta(date_offsets_days, unit="D") + pd.to_timedelta(seconds_in_day, unit="s")
    )

    ages = rng.normal(loc=40, scale=13, size=n_rows).clip(18, 80).round().astype(int)

    df = pd.DataFrame(
        {
            "transaction_id": [f"TXN{i:07d}" for i in range(1, n_rows + 1)],
            "customer_id": rng.choice(customers, size=n_rows),
            "product_id": chosen_products["product_id"].to_numpy(),
            "category": chosen_products["category"].to_numpy(),
            "purchase_datetime": purchase_datetimes,
            "quantity": quantities,
            "unit_price": unit_prices,
            "total_amount": total_amounts,
            "region": rng.choice(REGIONS, size=n_rows),
            "payment_method": rng.choice(PAYMENT_METHODS, size=n_rows),
            "customer_age": ages,
            "customer_segment": rng.choice(
                CUSTOMER_SEGMENTS, size=n_rows, p=CUSTOMER_SEGMENT_WEIGHTS
            ),
        }
    )
    return df


def _inject_duplicates(df: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    n_duplicates = int(len(df) * rate)
    dup_source_idx = rng.choice(df.index, size=n_duplicates, replace=False)
    exact_dupes = df.loc[dup_source_idx[: n_duplicates // 2]].copy()

    conflicting = df.loc[dup_source_idx[n_duplicates // 2 :]].copy()
    conflicting["quantity"] = np.maximum(
        1, conflicting["quantity"] + rng.integers(-2, 3, size=len(conflicting))
    )
    conflicting["total_amount"] = (conflicting["quantity"] * conflicting["unit_price"]).round(2)

    return pd.concat([df, exact_dupes, conflicting], ignore_index=True)


def _inject_nulls(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    null_rates = {
        "category": 0.02,
        "region": 0.03,
        "payment_method": 0.02,
        "customer_age": 0.04,
        "customer_segment": 0.02,
        "unit_price": 0.015,
        "quantity": 0.01,
        "total_amount": 0.02,
        "customer_id": 0.005,
    }
    n = len(df)
    for col, rate in null_rates.items():
        null_idx = rng.choice(n, size=int(n * rate), replace=False)
        df.loc[df.index[null_idx], col] = np.nan
    return df


def _inject_outliers(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)

    def disjoint_slices(*sizes: int) -> list[np.ndarray]:
        order = rng.permutation(n)
        slices, start = [], 0
        for size in sizes:
            slices.append(order[start : start + size])
            start += size
        return slices

    price_high_idx, price_negative_idx = disjoint_slices(int(n * 0.005), int(n * 0.003))
    df.loc[df.index[price_high_idx], "unit_price"] = rng.uniform(
        5000, 20000, size=len(price_high_idx)
    ).round(2)
    df.loc[df.index[price_negative_idx], "unit_price"] = -rng.uniform(
        5, 100, size=len(price_negative_idx)
    ).round(2)

    qty_high_idx, qty_invalid_idx = disjoint_slices(int(n * 0.005), int(n * 0.002))
    df.loc[df.index[qty_high_idx], "quantity"] = rng.integers(100, 1000, size=len(qty_high_idx))
    df.loc[df.index[qty_invalid_idx], "quantity"] = rng.integers(
        -5, 1, size=len(qty_invalid_idx)
    )

    age_outlier_idx = disjoint_slices(int(n * 0.005))[0]
    df.loc[df.index[age_outlier_idx], "customer_age"] = rng.choice(
        [-5, -1, 0, 130, 150, 200], size=len(age_outlier_idx)
    )

    mismatch_idx = disjoint_slices(int(n * 0.01))[0]
    factors = rng.choice([0.1, 0.25, 2.0, 3.0, 5.0], size=len(mismatch_idx))
    df.loc[df.index[mismatch_idx], "total_amount"] = (
        df.loc[df.index[mismatch_idx], "total_amount"] * factors
    ).round(2)

    return df


def _apply_casing_and_whitespace_noise(
    series: pd.Series, rng: np.random.Generator, rate: float
) -> pd.Series:
    series = series.copy()
    non_null_idx = series[series.notna()].index
    noisy_idx = rng.choice(non_null_idx, size=int(len(non_null_idx) * rate), replace=False)
    transforms = [
        str.upper,
        str.lower,
        lambda s: f"  {s}  ",
        lambda s: s.replace(" ", "_"),
    ]
    for idx in noisy_idx:
        transform = transforms[rng.integers(0, len(transforms))]
        series.loc[idx] = transform(series.loc[idx])
    return series


def _inject_formatting_inconsistencies(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    for col in ("region", "payment_method", "category"):
        df[col] = _apply_casing_and_whitespace_noise(df[col], rng, rate=0.03)
    return df


def _format_dates_inconsistently(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    formats = rng.choice(DATE_FORMATS, size=n, p=DATE_FORMAT_WEIGHTS)
    df["purchase_date"] = [
        dt.strftime(fmt) for dt, fmt in zip(df["purchase_datetime"], formats)
    ]

    garbage_idx = rng.choice(n, size=int(n * 0.01), replace=False)
    garbage_values = rng.choice(GARBAGE_DATE_VALUES, size=len(garbage_idx))
    df.loc[df.index[garbage_idx], "purchase_date"] = garbage_values

    missing_idx = rng.choice(n, size=int(n * 0.015), replace=False)
    df.loc[df.index[missing_idx], "purchase_date"] = np.nan

    return df.drop(columns=["purchase_datetime"])


def generate_dataset(n_rows: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    base_rows = n_rows - int(n_rows * 0.02)
    df = _build_base_transactions(base_rows, rng)
    df = _inject_duplicates(df, rate=0.02, rng=rng)
    df = _format_dates_inconsistently(df, rng)
    df = _inject_outliers(df, rng)
    df = _inject_nulls(df, rng)
    df = _inject_formatting_inconsistencies(df, rng)

    column_order = [
        "transaction_id",
        "customer_id",
        "product_id",
        "category",
        "purchase_date",
        "quantity",
        "unit_price",
        "total_amount",
        "region",
        "payment_method",
        "customer_age",
        "customer_segment",
    ]
    df = df[column_order].sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = generate_dataset(n_rows=args.rows, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Wrote {len(df):,} rows to {args.output}")
    print(f"  Duplicate transaction_ids: {df['transaction_id'].duplicated().sum():,}")
    print(f"  Fully duplicate rows: {df.duplicated().sum():,}")
    print(f"  Null cells: {df.isna().sum().sum():,}")


if __name__ == "__main__":
    main()
