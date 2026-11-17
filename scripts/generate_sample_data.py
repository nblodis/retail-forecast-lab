"""Generate deterministic, fictional retail demand history for demos/tests."""

from pathlib import Path

import numpy as np
import pandas as pd


def generate(destination: Path) -> None:
    rng = np.random.default_rng(2026)
    dates = pd.date_range("2024-01-05", periods=145, freq="W-FRI")
    rows = []
    for idx in range(24):
        item_id = f"DEMO-{idx + 1:03d}"
        base = rng.uniform(12, 85)
        seasonality_strength = rng.uniform(0.2, 0.85)
        slope = rng.uniform(-0.09, 0.12)
        first_week = 0 if idx < 20 else int(rng.integers(30, 65))
        for t, week in enumerate(dates[first_week:], start=first_week):
            holiday = np.exp(-0.5 * ((week.isocalendar().week - 48) / 5) ** 2)
            expected = max(0.4, (base + slope * t) * (1 + seasonality_strength * holiday))
            # Synthetic overdispersion and occasional legitimate zero sales.
            units = int(rng.poisson(expected * rng.lognormal(0, 0.12)))
            rows.append((item_id, week.date().isoformat(), units))
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["item_id", "week_ending", "units"]).to_csv(destination, index=False)
    print(f"Wrote fictional sample records to {destination}")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "data" / "synthetic_weekly_sales.csv")
