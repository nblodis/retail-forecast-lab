import numpy as np
import pandas as pd
import pytest

from retail_forecast import ForecastConfig, prepare_weekly, run_forecast
from retail_forecast.forecast import predict, weighted_recent_mean


def sample(weeks=36, items=("SKU-001", "SKU-002")):
    dates = pd.date_range("2025-01-03", periods=weeks, freq="W-FRI")
    return pd.DataFrame([
        {"item_id": item, "week_ending": week.date().isoformat(),
         "units": 10 + i // 3 + n}
        for n, item in enumerate(items) for i, week in enumerate(dates)
    ])


def test_same_anchor_every_item_and_nonnegative_forecasts():
    predicted, detail, score, info = run_forecast(sample(), ForecastConfig(horizon_weeks=5))
    assert len(predicted) == 10
    assert predicted.groupby("item_id")["week_ending"].nunique().eq(5).all()
    assert predicted["week_ending"].nunique() == 5
    assert predicted["predicted_units"].ge(0).all()
    assert not detail.empty and not score.empty
    assert info["items_forecast"] == 2


def test_missing_item_week_becomes_zero():
    raw = sample(15)
    row = (raw["item_id"] == "SKU-001") & (raw["week_ending"] == "2025-01-31")
    weekly, _ = prepare_weekly(raw.loc[~row])
    target = weekly[(weekly.item_id == "SKU-001") & (weekly.week_ending == pd.Timestamp("2025-01-31"))]
    assert target.iloc[0]["units"] == 0


def test_missing_global_reporting_week_rejected():
    raw = sample(15)
    raw = raw[raw.week_ending != "2025-01-31"]
    with pytest.raises(ValueError, match="Missing reporting weeks"):
        prepare_weekly(raw)


def test_wrong_weekday_and_negative_sales_rejected():
    raw = sample(15)
    raw.loc[0, "week_ending"] = "2025-01-04"
    with pytest.raises(ValueError, match="Fridays"):
        prepare_weekly(raw)
    raw = sample(15)
    raw.loc[0, "units"] = -1
    with pytest.raises(ValueError, match="negative"):
        prepare_weekly(raw)


def test_sparse_id_and_inactive_filter():
    raw = sample(15, items=("000123", "000321"))
    raw.loc[raw["item_id"] == "000321", "units"] = 0
    predicted, _, _, info = run_forecast(raw, ForecastConfig(horizon_weeks=2))
    assert predicted["item_id"].unique().tolist() == ["000123"]
    assert info["items_skipped_inactive"] == 1


def test_forecast_uses_only_history_before_prediction():
    hist = sample(25, items=("A",))
    hist["week_ending"] = pd.to_datetime(hist["week_ending"])
    cfg = ForecastConfig()
    first = predict(hist, 4, "damped_trend", {}, cfg)
    modified = hist.copy()
    # Predictor cannot peek at observations appended after the forecast anchor.
    future = pd.DataFrame({"item_id": ["A"],
                           "week_ending": [hist.week_ending.max() + pd.Timedelta(weeks=1)],
                           "units": [100000]})
    modified = pd.concat([modified, future], ignore_index=True)
    again = predict(modified[modified.week_ending <= hist.week_ending.max()], 4,
                    "damped_trend", {}, cfg)
    np.testing.assert_array_equal(first, again)


def test_short_series_and_valid_config():
    assert weighted_recent_mean(np.array([1., 2.])) >= 0
    with pytest.raises(ValueError):
        ForecastConfig(horizon_weeks=-1).validate()


def test_earlier_cutoff_truncates_data():
    raw = sample(20)
    weekly, anchor = prepare_weekly(raw, cutoff="2025-04-11")
    assert anchor == pd.Timestamp("2025-04-11")
    assert weekly.week_ending.max() == anchor
