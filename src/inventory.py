"""Inventory recommendations from forecast demand and current stock."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def recommend_inventory(
    forecast_total: float,
    current_inventory: float,
    history: pd.Series | None = None,
    safety_pct: float = 0.15,
) -> dict[str, Any]:
    """Recommended Stock = Forecast Demand + Safety Stock.

    Order Quantity = max(0, Recommended Stock − Current Inventory)
    """
    forecast_total = max(float(forecast_total), 0.0)
    current_inventory = max(float(current_inventory), 0.0)

    variability_pct = safety_pct
    if history is not None and len(history) >= 7:
        mean = float(history.mean())
        std = float(history.std(ddof=0))
        if mean > 0:
            cv = std / mean
            variability_pct = float(np.clip(0.10 + cv * 0.25, 0.10, 0.25))

    safety_stock = forecast_total * variability_pct
    recommended_stock = forecast_total + safety_stock
    order_qty = max(0.0, recommended_stock - current_inventory)

    if current_inventory <= 0 and forecast_total > 0:
        warning = "Stockout risk: no current inventory against upcoming demand."
        status = "stockout"
    elif current_inventory < forecast_total:
        warning = "Stockout risk: current inventory is below forecast demand."
        status = "stockout"
    elif current_inventory > recommended_stock * 1.5:
        warning = "Overstock risk: current inventory is well above recommended stock."
        status = "overstock"
    else:
        warning = "Inventory level looks adequate relative to the forecast."
        status = "ok"

    return {
        "current_inventory": current_inventory,
        "forecast_demand": forecast_total,
        "safety_stock": float(safety_stock),
        "safety_pct": float(variability_pct),
        "recommended_stock": float(recommended_stock),
        "order_quantity": float(order_qty),
        "warning": warning,
        "status": status,
        "disclaimer": "Decision-support recommendation only — not an automated purchase order.",
    }
