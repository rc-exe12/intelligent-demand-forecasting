"""Lightweight statistical anomaly detection (IQR / Z-score)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def detect_anomalies(series: pd.Series, method: str = "iqr", z_threshold: float = 3.0) -> dict[str, Any]:
    """Flag unusually high or low daily sales. Does not delete points."""
    if series is None or series.empty:
        return {
            "count": 0,
            "points": pd.DataFrame(columns=["Date", "Sales", "Type"]),
            "lower": 0.0,
            "upper": 0.0,
            "method": method,
            "normal_low": 0.0,
            "normal_high": 0.0,
        }

    values = series.astype(float)
    if method == "zscore" and values.std(ddof=0) > 0:
        z = (values - values.mean()) / values.std(ddof=0)
        mask = z.abs() > z_threshold
        lower = float(values.mean() - z_threshold * values.std(ddof=0))
        upper = float(values.mean() + z_threshold * values.std(ddof=0))
        used = "zscore"
    else:
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            # Values are almost constant; IQR gives no spread to work with, so
            # fall back to flagging points that deviate meaningfully from the
            # median instead of reporting "no anomalies" outright.
            median = float(values.median())
            threshold = max(abs(median) * 0.25, 1.0)
            lower = median - threshold
            upper = median + threshold
            mask = (values < lower) | (values > upper)
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (values < lower) | (values > upper)
        used = "iqr"

    flagged = values.loc[mask]
    types = np.where(flagged > upper, "High", "Low")
    points = pd.DataFrame({"Date": flagged.index, "Sales": flagged.values, "Type": types})

    normal = values.loc[~mask]
    if normal.empty:
        normal_low, normal_high = float(values.min()), float(values.max())
    else:
        normal_low, normal_high = float(normal.min()), float(normal.max())

    return {
        "count": int(len(points)),
        "points": points,
        "lower": float(lower),
        "upper": float(upper),
        "method": used,
        "normal_low": normal_low,
        "normal_high": normal_high,
    }


def winsorize_for_forecast(series: pd.Series, lower: float, upper: float) -> pd.Series:
    """Cap extreme values for model fitting without dropping history from charts."""
    if series.empty:
        return series
    return series.clip(lower=max(lower, 0.0), upper=upper)
