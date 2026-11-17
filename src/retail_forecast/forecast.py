"""Illustrative time-series forecasting with transparent, testable baselines.

This module is a general-purpose educational implementation. It uses ordinary
weighted averages, damped trends, pooled seasonal indices and walk-forward
validation. It is not adapted to any employer-specific data warehouse or feed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastConfig:
    horizon_weeks: int = 12
    backtest_weeks: int = 8
    minimum_history_weeks: int = 8
    inactive_after_weeks: int = 8
    enable_seasonality: bool = True
    minimum_seasonal_weeks: int = 78
    seasonal_clip_low: float = 0.7
    seasonal_clip_high: float = 1.4
    trend_damping: float = 0.8

    def validate(self) -> None:
        if min(self.horizon_weeks, self.backtest_weeks, self.minimum_history_weeks) < 1:
            raise ValueError("Horizon, backtest window and minimum history must be positive.")
        if self.inactive_after_weeks < 0:
            raise ValueError("inactive_after_weeks must not be negative.")
        if not 0 < self.trend_damping < 1:
            raise ValueError("trend_damping must be between zero and one.")
        if not 0 < self.seasonal_clip_low <= 1 <= self.seasonal_clip_high:
            raise ValueError("Seasonal clipping must bracket one with positive values.")


def prepare_weekly(raw: pd.DataFrame, cutoff: str | None = None) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Validate item/week/units data, aggregate duplicates and fill item gaps.

    Input must contain complete week-ending-Friday observations. Missing *item*
    weeks after first recorded sale are assumed to be zero demand; a missing
    *global* week is instead treated as a likely data-feed problem and rejected.
    The latter check cannot detect partial outages affecting only some items.
    """
    required = {"item_id", "week_ending", "units"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if raw.empty:
        raise ValueError("The input file contains no records.")
    df = raw.loc[:, ["item_id", "week_ending", "units"]].copy()
    if df["item_id"].isna().any() or df["item_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Every row needs an item_id.")
    df["item_id"] = df["item_id"].astype(str).str.strip()
    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    if df["week_ending"].isna().any():
        raise ValueError("Invalid week_ending date: use YYYY-MM-DD.")
    if not df["week_ending"].dt.dayofweek.eq(4).all():
        raise ValueError("All week_ending dates must be Fridays (weekly data only).")
    df["units"] = pd.to_numeric(df["units"], errors="coerce")
    if df["units"].isna().any() or not np.isfinite(df["units"].to_numpy()).all():
        raise ValueError("units must be a finite numeric value.")
    if (df["units"] < 0).any():
        raise ValueError("units cannot be negative; pre-adjust returns separately.")
    df = df.groupby(["item_id", "week_ending"], as_index=False)["units"].sum()
    latest = df["week_ending"].max()
    anchor = pd.Timestamp(cutoff) if cutoff else latest
    if anchor.weekday() != 4 or anchor > latest or anchor < df["week_ending"].min():
        raise ValueError("Cutoff must be an observed Friday within the data range.")
    df = df[df["week_ending"] <= anchor].copy()
    if not (df["week_ending"] == anchor).any():
        raise ValueError("Cutoff has no observations; use an observed reporting week.")
    observed_weeks = pd.DatetimeIndex(df["week_ending"].unique()).sort_values()
    expected_weeks = pd.date_range(observed_weeks.min(), anchor, freq="W-FRI")
    missing_global_weeks = expected_weeks.difference(observed_weeks)
    if len(missing_global_weeks):
        raise ValueError(
            "Missing reporting weeks in source (possible data outage): "
            + ", ".join(w.date().isoformat() for w in missing_global_weeks[:5])
        )
    parts = []
    for item_id, item in df.groupby("item_id", sort=True):
        index = pd.date_range(item["week_ending"].min(), anchor, freq="W-FRI")
        complete = item.set_index("week_ending")["units"].reindex(index, fill_value=0.0)
        parts.append(pd.DataFrame({"item_id": item_id, "week_ending": index, "units": complete.to_numpy()}))
    return pd.concat(parts, ignore_index=True), anchor


def weighted_recent_mean(values: np.ndarray) -> float:
    """Simple weighted moving average over recent 3, previous 5 and prior 5 weeks."""
    if len(values) == 0:
        return 0.0
    blocks = ((3, 0.55), (5, 0.30), (5, 0.15))
    end = len(values)
    total, weight = 0.0, 0.0
    for length, factor in blocks:
        segment = values[max(0, end - length):end]
        if len(segment):
            total += float(np.mean(segment)) * factor
            weight += factor
        end = max(0, end - length)
        if end == 0:
            break
    return max(0.0, total / weight) if weight else 0.0


def pooled_seasonality(training: pd.DataFrame, cfg: ForecastConfig) -> dict[int, float]:
    """Estimate ISO-week factors from a pooled historic series, using only training rows."""
    if not cfg.enable_seasonality:
        return {}
    totals = training.groupby("week_ending")["units"].sum().sort_index()
    if len(totals) < cfg.minimum_seasonal_weeks:
        return {}
    # Smoothing is confined to historic observations at each forecast origin.
    baseline = totals.rolling(13, center=True, min_periods=7).mean()
    ratios = (totals / baseline.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    valid = ratios.dropna()
    if len(valid) < cfg.minimum_seasonal_weeks // 2:
        return {}
    factor = valid.groupby(valid.index.isocalendar().week.to_numpy()).median()
    if factor.empty or not np.isfinite(factor.mean()) or factor.mean() <= 0:
        return {}
    factor = (factor / factor.mean()).clip(cfg.seasonal_clip_low, cfg.seasonal_clip_high)
    # A partial-history ISO week falls back to a neutral index.
    return {int(week): float(value) for week, value in factor.items()}


def _seasonal_factor(week: pd.Timestamp, seasonal: dict[int, float]) -> float:
    return seasonal.get(int(week.isocalendar().week), 1.0)


def predict(
    history: pd.DataFrame, horizon: int, method: str,
    seasonal: dict[int, float], cfg: ForecastConfig,
) -> np.ndarray:
    """Predict a contiguous set of weeks after the last row of history."""
    if history.empty:
        raise ValueError("Cannot forecast an item without history.")
    values = history["units"].to_numpy(dtype=float)
    anchor = history["week_ending"].max()
    if method == "weighted_average":
        return np.repeat(weighted_recent_mean(values), horizon)
    if method != "damped_trend":
        raise ValueError(f"Unknown model: {method}")
    factors = np.array([_seasonal_factor(d, seasonal) for d in history["week_ending"]])
    adjusted = values / factors
    level = float(pd.Series(adjusted).ewm(span=min(len(values), 8), adjust=False).mean().iloc[-1])
    recent = adjusted[-min(len(adjusted), 10):]
    if len(recent) >= 4:
        slope = float(np.polyfit(np.arange(len(recent)), recent, 1)[0])
        # Cap extreme trend extrapolation in an educational, conservative model.
        slope = float(np.clip(slope, -0.2 * max(level, 1), 0.2 * max(level, 1)))
    else:
        slope = 0.0
    predictions = []
    for step in range(1, horizon + 1):
        future_week = anchor + pd.Timedelta(weeks=step)
        damped_sum = sum(cfg.trend_damping ** k for k in range(1, step + 1))
        base = max(0.0, level + slope * damped_sum)
        predictions.append(max(0.0, base * _seasonal_factor(future_week, seasonal)))
    return np.asarray(predictions, dtype=float)


def walk_forward_backtest(
    weekly: pd.DataFrame, cfg: ForecastConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window, one-week-ahead forecasts without future-data leakage.

    Model selection uses this same historical evaluation window and should not
    be mistaken for independent out-of-sample performance estimates.
    """
    max_week = weekly["week_ending"].max()
    dates = pd.date_range(end=max_week, periods=cfg.backtest_weeks, freq="W-FRI")
    by_item = {key: frame.sort_values("week_ending").reset_index(drop=True)
               for key, frame in weekly.groupby("item_id")}
    records: list[dict[str, Any]] = []
    for test_week in dates:
        train_all = weekly[weekly["week_ending"] < test_week]
        seasonal = pooled_seasonality(train_all, cfg)
        for item_id, item in by_item.items():
            hist = item[item["week_ending"] < test_week]
            actual_row = item[item["week_ending"] == test_week]
            if len(hist) < cfg.minimum_history_weeks or actual_row.empty:
                continue
            actual = float(actual_row.iloc[0]["units"])
            for method in ("weighted_average", "damped_trend"):
                p = float(predict(hist, 1, method, seasonal, cfg)[0])
                records.append({"item_id": item_id,
                                "week_ending": test_week,
                                "method": method,
                                "actual_units": actual,
                                "predicted_units": round(p, 4),
                                "absolute_error": abs(p - actual),
                                "signed_error": p - actual})
    detail = pd.DataFrame(records)
    if detail.empty:
        return detail, pd.DataFrame()
    summary = detail.groupby(["item_id", "method"], as_index=False).agg(
        actual_units=("actual_units", "sum"),
        absolute_error=("absolute_error", "sum"),
        signed_error=("signed_error", "sum"),
        evaluated_weeks=("actual_units", "size"),
    )
    summary["wape"] = np.where(
        summary["actual_units"] > 0,
        summary["absolute_error"] / summary["actual_units"],
        np.nan,
    )
    summary["bias"] = np.where(
        summary["actual_units"] > 0,
        summary["signed_error"] / summary["actual_units"],
        np.nan,
    )
    return detail, summary


def run_forecast(
    raw: pd.DataFrame, cfg: ForecastConfig | None = None, cutoff: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Produce aligned forecasts, backtest detail, model scorecard and run info."""
    cfg = cfg or ForecastConfig()
    cfg.validate()
    weekly, anchor = prepare_weekly(raw, cutoff=cutoff)
    detail, summary = walk_forward_backtest(weekly, cfg)
    seasonal = pooled_seasonality(weekly, cfg)
    forecasts = []
    skipped = []
    for item_id, item in weekly.groupby("item_id", sort=True):
        item = item.sort_values("week_ending")
        if cfg.inactive_after_weeks and item.tail(cfg.inactive_after_weeks)["units"].sum() == 0:
            skipped.append(str(item_id))
            continue
        scores = summary[summary["item_id"] == item_id] if not summary.empty else pd.DataFrame()
        # If all actuals are zero in a backtest window, compare mean absolute error.
        if scores.empty:
            chosen = "weighted_average"
        else:
            scores = scores.copy()
            scores["selection_score"] = np.where(
                scores["actual_units"] > 0,
                scores["wape"],
                scores["absolute_error"] / scores["evaluated_weeks"],
            )
            chosen = str(scores.sort_values(["selection_score", "method"]).iloc[0]["method"])
        predicted = predict(item, cfg.horizon_weeks, chosen, seasonal, cfg)
        for step, units in enumerate(predicted, start=1):
            forecasts.append({
                "item_id": item_id,
                "week_ending": (anchor + pd.Timedelta(weeks=step)).date().isoformat(),
                "forecast_week": step,
                "predicted_units": round(float(units), 2),
                "cumulative_units": round(float(predicted[:step].sum()), 2),
                "selected_method": chosen,
                "history_weeks": len(item),
            })
    if not forecasts:
        raise ValueError("No items remain to forecast; review the inactivity filter or source dates.")
    info = {"anchor_week_ending": anchor.date().isoformat(),
            "items_forecast": len(set(x["item_id"] for x in forecasts)),
            "items_skipped_inactive": len(skipped),
            "forecast_rows": len(forecasts),
            "backtest_rows": len(detail),
            "configuration": asdict(cfg),
            "limitations": [
                "Missing item-level weeks after first observation are treated as zero.",
                "Future promotions, lead times, stockouts and price changes are not modeled.",
                "Seasonal indices are pooled across items, not category-specific.",
                "Historical selection-window error is not an unbiased production accuracy estimate.",
            ]}
    return pd.DataFrame(forecasts), detail, summary, info