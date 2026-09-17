"""CSV validation, cleaning, and historical summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("Date", "Product", "Sales")
OPTIONAL_COLUMNS = ("Inventory", "Price")

COLUMN_ALIASES = {
    "date": "Date",
    "ds": "Date",
    "product": "Product",
    "sku": "Product",
    "item": "Product",
    "sales": "Sales",
    "units": "Sales",
    "quantity": "Sales",
    "qty": "Sales",
    "demand": "Sales",
    "inventory": "Inventory",
    "stock": "Inventory",
    "price": "Price",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    used = set()
    for col in df.columns:
        key = str(col).strip().lower()
        canonical = COLUMN_ALIASES.get(key, str(col).strip())
        if canonical in used:
            continue
        renamed[col] = canonical
        used.add(canonical)
    return df.rename(columns=renamed)


def validate_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Validate and clean uploaded sales data.

    Returns (clean_df, warnings, errors). Errors mean the file cannot be used.
    """
    warnings: list[str] = []
    errors: list[str] = []

    if df is None or df.empty:
        return pd.DataFrame(), warnings, ["The uploaded file is empty. Please use a CSV with sales rows."]

    work = _normalize_columns(df.copy())
    missing = [c for c in REQUIRED_COLUMNS if c not in work.columns]
    if missing:
        errors.append(
            f"Missing required column(s): {', '.join(missing)}. "
            "Expected at least Date, Product, and Sales."
        )
        return pd.DataFrame(), warnings, errors

    original_rows = len(work)

    parsed_dates = pd.to_datetime(work["Date"], errors="coerce")
    invalid_dates = parsed_dates.isna()
    invalid_count = int(invalid_dates.sum())
    if invalid_count:
        warnings.append(f"{invalid_count} row(s) have invalid dates and were removed.")
        work = work.loc[~invalid_dates].copy()
        parsed_dates = parsed_dates.loc[~invalid_dates]
    work["Date"] = parsed_dates.dt.normalize()

    work["Product"] = work["Product"].astype(str).str.strip()
    blank_product = work["Product"].isin(["", "nan", "None", "NaT"])
    if blank_product.any():
        warnings.append(f"{int(blank_product.sum())} row(s) with missing product names were removed.")
        work = work.loc[~blank_product].copy()

    # Check inventory quality before rows get dropped for sales issues, so a
    # row missing inventory is still reported even if it's also removed for
    # having invalid sales.
    if "Inventory" in work.columns:
        inventory = pd.to_numeric(work["Inventory"], errors="coerce")
        missing_inv = inventory.isna()
        if missing_inv.any():
            warnings.append(
                f"{int(missing_inv.sum())} row(s) have missing inventory. "
                "The latest known stock per product will be used when possible."
            )
        work["Inventory"] = inventory
    else:
        warnings.append("Inventory column is missing. Inventory recommendations will assume current stock is 0.")
        work["Inventory"] = np.nan

    sales = pd.to_numeric(work["Sales"], errors="coerce")
    missing_sales = sales.isna()
    if missing_sales.any():
        warnings.append(f"{int(missing_sales.sum())} row(s) have missing or non-numeric sales and were removed.")
        work = work.loc[~missing_sales].copy()
        sales = sales.loc[~missing_sales]

    negative_sales = sales < 0
    if negative_sales.any():
        warnings.append(f"{int(negative_sales.sum())} row(s) have negative sales and were removed.")
        work = work.loc[~negative_sales].copy()
        sales = sales.loc[~negative_sales]
    work["Sales"] = sales.astype(float)

    if "Price" in work.columns:
        work["Price"] = pd.to_numeric(work["Price"], errors="coerce")
    else:
        work["Price"] = np.nan

    dup_mask = work.duplicated(subset=["Date", "Product"], keep=False)
    if dup_mask.any():
        n_dup_rows = int(dup_mask.sum())
        warnings.append(
            f"{n_dup_rows} duplicate Date+Product row(s) were found. "
            "Sales were summed and inventory/price kept from the last row."
        )
        work = (
            work.sort_values(["Product", "Date"])
            .groupby(["Date", "Product"], as_index=False)
            .agg(
                Sales=("Sales", "sum"),
                Inventory=("Inventory", "last"),
                Price=("Price", "last"),
            )
        )

    work = work.sort_values(["Product", "Date"]).reset_index(drop=True)
    if work.empty:
        errors.append("No valid sales rows remain after cleaning. Check dates and sales values.")
        return work, warnings, errors

    dropped = original_rows - len(work)
    if dropped > 0:
        warnings.append(f"Kept {len(work)} of {original_rows} original row(s) after cleaning.")

    return work, warnings, errors


def products(df: pd.DataFrame) -> list[str]:
    if df.empty or "Product" not in df.columns:
        return []
    return sorted(df["Product"].dropna().unique().tolist())


def filter_product(df: pd.DataFrame, product: str) -> pd.DataFrame:
    subset = df.loc[df["Product"] == product].copy()
    return subset.sort_values("Date").reset_index(drop=True)


def daily_series(df: pd.DataFrame) -> pd.Series:
    """Build a complete daily sales series (missing days filled with 0)."""
    if df.empty:
        return pd.Series(dtype=float)
    series = df.groupby("Date", as_index=True)["Sales"].sum().sort_index()
    full_idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(full_idx, fill_value=0.0)
    series.name = "Sales"
    return series


def latest_inventory(df: pd.DataFrame) -> float:
    if df.empty or "Inventory" not in df.columns:
        return 0.0
    inv = df["Inventory"].dropna()
    if inv.empty:
        return 0.0
    return float(inv.iloc[-1])


def historical_metrics(series: pd.Series) -> dict[str, Any]:
    if series.empty:
        return {"total_sales": 0.0, "average_sales": 0.0, "days": 0, "max_sales": 0.0, "min_sales": 0.0}
    return {
        "total_sales": float(series.sum()),
        "average_sales": float(series.mean()),
        "days": int(len(series)),
        "max_sales": float(series.max()),
        "min_sales": float(series.min()),
    }
