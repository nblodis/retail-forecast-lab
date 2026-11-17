"""Optional: render a forecast preview using fictional records and matplotlib."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def render() -> None:
    history = pd.read_csv(ROOT / "data" / "synthetic_weekly_sales.csv")
    forecasts = pd.read_csv(ROOT / "outputs" / "forecasts.csv")
    item = "DEMO-001"
    actual = history.loc[history.item_id == item].tail(26).copy()
    future = forecasts.loc[forecasts.item_id == item].copy()
    actual["week_ending"] = pd.to_datetime(actual["week_ending"])
    future["week_ending"] = pd.to_datetime(future["week_ending"])
    fig, ax = plt.subplots(figsize=(9, 3.9), layout="constrained")
    ax.plot(actual["week_ending"], actual["units"], marker="o", markersize=3,
            linewidth=1.5, label="Observed units (synthetic)")
    ax.plot(future["week_ending"], future["predicted_units"], marker="o",
            markersize=3, linewidth=2, label="Forecast units")
    ax.axvline(actual["week_ending"].max(), linestyle="--", linewidth=1,
               alpha=0.6, label="Forecast origin")
    ax.set(title=f"Illustrative weekly item demand: {item}", ylabel="Units", xlabel="Week ending")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, loc="upper left")
    fig.autofmt_xdate(rotation=20)
    output = ROOT / "assets" / "synthetic_forecast_preview.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    render()
