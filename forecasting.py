"""
forecasting.py
--------------
NOT ACTIVE YET -- this is scaffolding for a real forecasting layer, not a
working forecast. See metrics.py's module docstring for why: the current
CSV is a single snapshot (one row per track), and you cannot fit a
time-series model to a single point in time.

To turn this on:
  1. Start saving a dated copy of the cleaned export on a schedule
     (e.g. one CSV per day/week into data/history/YYYY-MM-DD.csv), or
     append rows with a `snapshot_date` column to one growing file.
  2. Once you have >= ~10-14 snapshots per track, `streams` (or `pos`)
     becomes an actual time series and the functions below stop being
     stubs -- fill in the TODOs.
  3. Wire `forecast_track()` into app.py's Future Outlook tab as an
     additional, clearly-labeled "statistical forecast" section
     alongside (not replacing) the existing momentum heuristic.

Suggested approach once you have history:
  - Prophet or a simple exponential-smoothing model per track for
    short-horizon stream forecasts (fast, works with short series).
  - XGBoost/LightGBM with lag features (streams_t-1, viral_score_t-1,
    days, genre, country as categoricals) if you want a single model
    across all tracks rather than one per track -- generally more
    robust once you have enough history to build lag features at all.
  - ARIMA only if you specifically want to check for autocorrelation
    structure per top track; it needs the longest series of the three
    options to fit reliably.
"""

import pandas as pd


def has_sufficient_history(history_df: pd.DataFrame, min_snapshots: int = 10) -> bool:
    """history_df expected columns: track_name, artist_name, snapshot_date, streams, ..."""
    if "snapshot_date" not in history_df.columns:
        return False
    counts = history_df.groupby(["track_name", "artist_name"])["snapshot_date"].nunique()
    return bool((counts >= min_snapshots).any())


def forecast_track(history_df: pd.DataFrame, track_name: str, artist_name: str, horizon_days: int = 14):
    """TODO once real history exists. Currently raises to avoid returning
    a fabricated number from a single-snapshot dataset."""
    raise NotImplementedError(
        "No time-series history available yet. This dataset is a single "
        "snapshot; collect repeated snapshots over time before forecasting. "
        "See this module's docstring."
    )
