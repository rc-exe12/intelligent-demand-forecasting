"""Lightweight demand forecasting with Holt-Winters, SES, and moving-average fallbacks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing


def _confidence_band(fitted_resid_std: float, horizon: int, mean_level: float) -> tuple[np.ndarray, np.ndarray]:
    """Simple expanding interval around the point forecast."""
    std = fitted_resid_std if fitted_resid_std and np.isfinite(fitted_resid_std) else max(mean_level * 0.1, 1.0)
    steps = np.arange(1, horizon + 1, dtype=float)
    width = 1.96 * std * np.sqrt(steps)
    return width, width


def _moving_average_forecast(series: pd.Series, horizon: int, window: int = 7) -> dict[str, Any]:
    window = min(window, max(1, len(series)))
    level = float(series.iloc[-window:].mean())
    future_idx = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    predicted = np.full(horizon, max(level, 0.0))
    width, _ = _confidence_band(float(series.iloc[-window:].std(ddof=0) or 0.0), horizon, level)
    lower = np.clip(predicted - width, 0, None)
    upper = predicted + width
    return {
        "method": "Moving Average",
        "horizon": horizon,
        "predicted": predicted,
        "lower": lower,
        "upper": upper,
        "index": future_idx,
        "total": float(predicted.sum()),
        "average": float(predicted.mean()),
    }


def _ses_forecast(series: pd.Series, horizon: int) -> dict[str, Any]:
    model = SimpleExpSmoothing(series, initialization_method="estimated")
    fit = model.fit(optimized=True)
    fcast = fit.forecast(horizon)
    predicted = np.clip(np.asarray(fcast, dtype=float), 0, None)
    resid_std = float(np.nanstd(fit.resid)) if getattr(fit, "resid", None) is not None else float(series.std(ddof=0) or 0)
    width, _ = _confidence_band(resid_std, horizon, float(predicted.mean() if len(predicted) else series.mean()))
    future_idx = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    return {
        "method": "Simple Exponential Smoothing",
        "horizon": horizon,
        "predicted": predicted,
        "lower": np.clip(predicted - width, 0, None),
        "upper": predicted + width,
        "index": future_idx,
        "total": float(predicted.sum()),
        "average": float(predicted.mean()),
    }


def _holt_winters_forecast(series: pd.Series, horizon: int) -> dict[str, Any]:
    seasonal_periods = 7
    use_seasonal = len(series) >= seasonal_periods * 2
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add" if use_seasonal else None,
        seasonal_periods=seasonal_periods if use_seasonal else None,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    fcast = fit.forecast(horizon)
    predicted = np.clip(np.asarray(fcast, dtype=float), 0, None)
    resid_std = float(np.nanstd(fit.resid)) if getattr(fit, "resid", None) is not None else float(series.std(ddof=0) or 0)
    width, _ = _confidence_band(resid_std, horizon, float(predicted.mean() if len(predicted) else series.mean()))
    future_idx = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    method = "Holt-Winters (additive seasonality)" if use_seasonal else "Holt-Winters (trend only)"
    return {
        "method": method,
        "horizon": horizon,
        "predicted": predicted,
        "lower": np.clip(predicted - width, 0, None),
        "upper": predicted + width,
        "index": future_idx,
        "total": float(predicted.sum()),
        "average": float(predicted.mean()),
    }


def forecast_demand(series: pd.Series, horizon: int = 14) -> dict[str, Any]:
    """Forecast daily demand. Falls back to simpler methods if the primary model fails."""
    horizon = int(horizon)
    if series is None or series.empty:
        future_idx = pd.date_range(pd.Timestamp.today().normalize(), periods=horizon, freq="D")
        zeros = np.zeros(horizon)
        return {
            "method": "None",
            "horizon": horizon,
            "predicted": zeros,
            "lower": zeros,
            "upper": zeros,
            "index": future_idx,
            "total": 0.0,
            "average": 0.0,
            "fallback_note": "No history available; forecast is zero.",
        }

    clean = series.astype(float).copy()
    clean = clean.asfreq("D", fill_value=0.0)
    fallback_note = None

    if len(clean) < 10:
        result = _moving_average_forecast(clean, horizon)
        result["fallback_note"] = "Limited history: using a moving-average forecast."
        result["frame"] = _to_frame(result)
        return result

    try:
        result = _holt_winters_forecast(clean, horizon)
    except Exception:
        fallback_note = "Holt-Winters failed; switched to Simple Exponential Smoothing."
        try:
            result = _ses_forecast(clean, horizon)
        except Exception:
            fallback_note = "Primary models failed; switched to a moving-average forecast."
            result = _moving_average_forecast(clean, horizon)

    if fallback_note:
        result["fallback_note"] = fallback_note
    result["frame"] = _to_frame(result)
    return result


def _to_frame(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": result["index"],
            "Predicted": result["predicted"],
            "Lower": result["lower"],
            "Upper": result["upper"],
        }
    )
