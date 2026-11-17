"""Command-line interface for reproducible CSV-to-CSV forecasting demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .forecast import ForecastConfig, run_forecast


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Weekly item demand forecasts from a three-column CSV."
    )
    parser.add_argument("--input", required=True, type=Path, help="CSV: item_id, week_ending, units")
    parser.add_argument("--output-dir", default=Path("outputs"), type=Path)
    parser.add_argument("--horizon", default=12, type=int, help="Forecast weeks (default 12)")
    parser.add_argument("--backtest-weeks", default=8, type=int)
    parser.add_argument("--cutoff", help="Optional YYYY-MM-DD reporting week (Friday)")
    parser.add_argument("--no-seasonality", action="store_true")
    parser.add_argument("--inactive-after", default=8, type=int,
                        help="Exclude items with no sales over this many recent weeks; 0 disables")
    args = parser.parse_args(argv)
    try:
        df = pd.read_csv(args.input, dtype={"item_id": "string"})
        cfg = ForecastConfig(horizon_weeks=args.horizon,
                             backtest_weeks=args.backtest_weeks,
                             enable_seasonality=not args.no_seasonality,
                             inactive_after_weeks=args.inactive_after)
        forecasts, detail, summary, info = run_forecast(df, cfg=cfg, cutoff=args.cutoff)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        forecasts.to_csv(args.output_dir / "forecasts.csv", index=False)
        detail.to_csv(args.output_dir / "backtest_detail.csv", index=False)
        summary.to_csv(args.output_dir / "backtest_scorecard.csv", index=False)
        (args.output_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    print(f"Forecasted {info['items_forecast']} items across {args.horizon} weeks.")
    print(f"Anchor: {info['anchor_week_ending']} | output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
