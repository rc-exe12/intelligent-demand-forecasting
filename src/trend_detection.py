"""Simple statistical trend detection from recent vs previous period averages."""

from __future__ import annotations

from typing import Any

import pandas as pd

STABLE_THRESHOLD = 0.05  # ±5% treated as stable


def detect_trend(series: pd.Series, window: int | None = None) -> dict[str, Any]:
    if series is None or series.empty:
        return {
            "label": "Unknown",
            "change_pct": 0.0,
            "recent_avg": 0.0,
            "previous_avg": 0.0,
            "explanation": "Not enough sales history to detect a trend.",
        }

    n = len(series)
    if window is None:
        window = min(14, max(7, n // 4))
    window = max(3, min(window, n // 2 if n >= 6 else n))

    if n < 6:
        return {
            "label": "Stable",
            "change_pct": 0.0,
            "recent_avg": float(series.mean()),
            "previous_avg": float(series.mean()),
            "explanation": "Limited history; demand is treated as stable until more days are available.",
        }

    recent = series.iloc[-window:]
    previous = series.iloc[-2 * window : -window]
    recent_avg = float(recent.mean())
    previous_avg = float(previous.mean()) if len(previous) else recent_avg

    if previous_avg == 0:
        change_pct = 100.0 if recent_avg > 0 else 0.0
    else:
        change_pct = ((recent_avg - previous_avg) / previous_avg) * 100.0

    if change_pct > STABLE_THRESHOLD * 100:
        label = "Increasing"
        explanation = (
            f"Recent demand is approximately {abs(change_pct):.0f}% higher than the previous period."
        )
    elif change_pct < -STABLE_THRESHOLD * 100:
        label = "Decreasing"
        explanation = (
            f"Recent demand is approximately {abs(change_pct):.0f}% lower than the previous period."
        )
    else:
        label = "Stable"
        explanation = (
            f"Recent demand is approximately unchanged ({change_pct:+.1f}%) versus the previous period."
        )

    return {
        "label": label,
        "change_pct": float(change_pct),
        "recent_avg": recent_avg,
        "previous_avg": previous_avg,
        "window_days": int(window),
        "explanation": explanation,
    }
