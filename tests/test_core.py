"""Core calculation tests (stdlib unittest — no extra test runner required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anomaly_detection import detect_anomalies
from src.data_processing import daily_series, validate_and_clean
from src.forecasting import forecast_demand
from src.inventory import recommend_inventory
from src.seasonality import detect_seasonality
from src.trend_detection import detect_trend


class DataProcessingTests(unittest.TestCase):
    def test_missing_columns(self):
        df, warnings, errors = validate_and_clean(pd.DataFrame({"Date": ["2026-01-01"]}))
        self.assertTrue(df.empty)
        self.assertTrue(any("Missing required" in e for e in errors))

    def test_invalid_dates_negative_sales_duplicates(self):
        raw = pd.DataFrame(
            {
                "Date": ["2026-01-01", "not-a-date", "2026-01-01", "2026-01-02"],
                "Product": ["A", "A", "A", "A"],
                "Sales": [10, 5, 4, -3],
                "Inventory": [50, 50, 40, None],
            }
        )
        clean, warnings, errors = validate_and_clean(raw)
        self.assertFalse(errors)
        self.assertFalse(clean.empty)
        self.assertTrue(any("invalid dates" in w for w in warnings))
        self.assertTrue(any("negative sales" in w for w in warnings))
        self.assertTrue(any("duplicate" in w.lower() for w in warnings))
        self.assertTrue(any("missing inventory" in w.lower() for w in warnings))
        self.assertEqual(float(clean.loc[clean["Date"] == pd.Timestamp("2026-01-01"), "Sales"].iloc[0]), 14.0)


class TrendSeasonalityAnomalyTests(unittest.TestCase):
    def test_increasing_trend(self):
        idx = pd.date_range("2026-01-01", periods=28, freq="D")
        values = [10] * 14 + [20] * 14
        result = detect_trend(pd.Series(values, index=idx))
        self.assertEqual(result["label"], "Increasing")
        self.assertGreater(result["change_pct"], 50)

    def test_weekly_seasonality_peak_saturday(self):
        idx = pd.date_range("2026-01-05", periods=70, freq="D")  # starts Monday
        values = []
        for d in idx:
            values.append(170 if d.day_name() == "Saturday" else 100)
        result = detect_seasonality(pd.Series(values, index=idx))
        self.assertTrue(result["weekly_detected"])
        self.assertEqual(result["peak_day"], "Saturday")

    def test_iqr_anomaly(self):
        idx = pd.date_range("2026-01-01", periods=30, freq="D")
        values = [120] * 29 + [450]
        result = detect_anomalies(pd.Series(values, index=idx))
        self.assertGreaterEqual(result["count"], 1)
        self.assertGreaterEqual(float(result["points"]["Sales"].max()), 450)


class ForecastInventoryTests(unittest.TestCase):
    def test_forecast_horizon_and_non_negative(self):
        idx = pd.date_range("2026-01-01", periods=60, freq="D")
        series = pd.Series(100 + (idx.dayofweek == 5) * 40, index=idx, dtype=float)
        result = forecast_demand(series, horizon=14)
        self.assertEqual(result["horizon"], 14)
        self.assertEqual(len(result["predicted"]), 14)
        self.assertGreater(result["total"], 0)
        self.assertTrue((result["predicted"] >= 0).all())
        self.assertIn("frame", result)

    def test_inventory_order_quantity(self):
        rec = recommend_inventory(forecast_total=100, current_inventory=20, safety_pct=0.2)
        self.assertEqual(rec["safety_stock"], 20)
        self.assertEqual(rec["recommended_stock"], 120)
        self.assertEqual(rec["order_quantity"], 100)
        self.assertEqual(rec["status"], "stockout")

    def test_no_order_when_stocked(self):
        rec = recommend_inventory(forecast_total=50, current_inventory=200, safety_pct=0.1)
        self.assertEqual(rec["order_quantity"], 0)
        self.assertEqual(rec["status"], "overstock")

    def test_daily_series_fills_gaps(self):
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-01-01", "2026-01-03"]),
                "Product": ["A", "A"],
                "Sales": [5, 7],
            }
        )
        series = daily_series(df)
        self.assertEqual(len(series), 3)
        self.assertEqual(float(series.iloc[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
