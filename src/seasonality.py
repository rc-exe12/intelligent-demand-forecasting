"""Weekly (and optional monthly) seasonality detection."""

from __future__ import annotations

from typing import Any

import pandas as pd

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def detect_seasonality(series: pd.Series) -> dict[str, Any]:
    empty = {
        "weekly_detected": False,
        "peak_day": None,
        "low_day": None,
        "weekday_averages": {},
        "monthly_detected": False,
        "peak_month": None,
        "explanation": "Not enough data to detect seasonality.",
        "strength": 0.0,
    }
    if series is None or series.empty or len(series) < 14:
        return empty

    frame = series.rename("Sales").to_frame()
    frame["weekday"] = frame.index.day_name()
    weekday_avg = frame.groupby("weekday")["Sales"].mean().reindex(WEEKDAY_ORDER)

    peak_day = str(weekday_avg.idxmax())
    low_day = str(weekday_avg.idxmin())
    mean_val = float(weekday_avg.mean()) if weekday_avg.mean() else 0.0
    spread = float(weekday_avg.max() - weekday_avg.min())
    strength = (spread / mean_val) if mean_val else 0.0
    weekly_detected = strength >= 0.08 and weekday_avg.notna().sum() >= 5

    monthly_detected = False
    peak_month = None
    monthly_note = ""
    unique_months = series.index.to_period("M").nunique()
    if unique_months >= 6:
        frame["month"] = frame.index.month_name()
        monthly_avg = frame.groupby("month")["Sales"].mean()
        if monthly_avg.max() > 0 and (monthly_avg.max() - monthly_avg.min()) / monthly_avg.mean() >= 0.15:
            monthly_detected = True
            peak_month = str(monthly_avg.idxmax())
            monthly_note = f" Monthly peak: {peak_month}."

    if weekly_detected:
        explanation = (
            f"Weekly seasonality detected. Peak day: {peak_day}. "
            f"Lowest day: {low_day}.{monthly_note}"
        )
    else:
        explanation = "No strong weekly seasonality was detected in weekday averages." + monthly_note

    return {
        "weekly_detected": weekly_detected,
        "peak_day": peak_day,
        "low_day": low_day,
        "weekday_averages": {k: float(v) for k, v in weekday_avg.dropna().items()},
        "monthly_detected": monthly_detected,
        "peak_month": peak_month,
        "explanation": explanation,
        "strength": float(strength),
    }
