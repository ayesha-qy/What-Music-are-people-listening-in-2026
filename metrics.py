"""

stage 2: rankings, composite scores, and aggregates.

Every score below is a is aggregated by formulas. None of it is a black box, and none of it is
handed to the LLM to compute, the LLM only ever narrates numbers that
are already final by the time it sees them.

"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Song-level metrics
# ---------------------------------------------------------------------------

LONGEVITY_STABILITY_WEIGHT = {"New": 0.0, "Stable Hit": 0.6, "Evergreen": 1.0}
TREND_BONUS = {"Rising": 1.0, "Falling": -1.0}


def compute_song_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["momentum_score"] = 0.5 * _zscore(df["stream_change"]) + 0.5 * _zscore(df["viral_score"])

    # future_potential_score (0-100): heuristic momentum/durability index,
    # NOT a predicted stream count. Weights: 40% current viral momentum,
    # 30% recent stream change, 20% trend direction, 10% longevity class.
    raw_potential = (
        0.40 * _zscore(df["viral_score"])
        + 0.30 * _zscore(df["stream_change"])
        + 0.20 * df["trend"].map(TREND_BONUS).fillna(0)
        + 0.10 * df["longevity"].map(LONGEVITY_STABILITY_WEIGHT).fillna(0)
    )
    df["future_potential_score"] = (_minmax(raw_potential) * 100).round(1)

    # long_term_popularity_score (0-100): rewards durability (Evergreen/
    # Stable Hit), sustained streams, and time already survived on chart.
    raw_ltp = (
        0.45 * df["longevity"].map(LONGEVITY_STABILITY_WEIGHT).fillna(0)
        + 0.35 * _minmax(df["streams"])
        + 0.20 * _minmax(np.log1p(df["days"]))
    )
    df["long_term_popularity_score"] = (_minmax(raw_ltp) * 100).round(1)

    return df


def top_songs(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    cols = ["track_name", "artist_name", "streams", "pos", "trend", "longevity"]
    return df.sort_values("streams", ascending=False).head(n)[cols]


def rising_songs(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    sub = df[df["trend"] == "Rising"].sort_values("momentum_score", ascending=False)
    cols = ["track_name", "artist_name", "stream_change", "viral_score", "momentum_score", "pos"]
    return sub.head(n)[cols]


def falling_songs(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    sub = df[df["trend"] == "Falling"].sort_values("momentum_score", ascending=True)
    cols = ["track_name", "artist_name", "stream_change", "viral_score", "momentum_score", "pos"]
    return sub.head(n)[cols]


def long_term_songs(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    sub = df[df["longevity"].isin(["Evergreen", "Stable Hit"])]
    cols = ["track_name", "artist_name", "longevity", "days", "streams", "long_term_popularity_score"]
    return sub.sort_values("long_term_popularity_score", ascending=False).head(n)[cols]


def future_potential_leaderboard(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    cols = ["track_name", "artist_name", "future_potential_score", "trend", "longevity", "viral_score"]
    return df.sort_values("future_potential_score", ascending=False).head(n)[cols]


# ---------------------------------------------------------------------------
# Artist-level metrics
# ---------------------------------------------------------------------------

def compute_artist_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate track-level rows into one row per artist and compute a
    transparent, decomposable artist_score so 'why is X ranked #1' can be
    answered by pointing at the four inputs below rather than asserted.
    """
    g = df.groupby("artist_name").agg(
        total_streams=("streams", "sum"),
        track_count=("track_name", "count"),
        avg_pos=("pos", "mean"),
        best_pos=("pos", "min"),
        avg_viral_score=("viral_score", "mean"),
        rising_tracks=("trend", lambda s: (s == "Rising").sum()),
        falling_tracks=("trend", lambda s: (s == "Falling").sum()),
        max_days_active=("days", "max"),
        evergreen_tracks=("longevity", lambda s: (s == "Evergreen").sum()),
    ).reset_index()

    g["pct_rising"] = (g["rising_tracks"] / g["track_count"] * 100).round(1)

    # artist_score (0-100): 35% total streams, 25% chart breadth
    # (how many simultaneous tracks charting -- a real dominance signal,
    # not noise), 25% average position (inverted: lower pos is better),
    # 15% average viral score.
    inv_avg_pos = -g["avg_pos"]  # invert so higher = better before scaling
    raw_score = (
        0.35 * _minmax(g["total_streams"])
        + 0.25 * _minmax(g["track_count"])
        + 0.25 * _minmax(inv_avg_pos)
        + 0.15 * _minmax(g["avg_viral_score"])
    )
    g["artist_score"] = (_minmax(raw_score) * 100).round(1)

    # tenure_proxy_days: approximation of "time near the top" using the
    # longest-running track this artist currently has charting. Explicitly
    # a proxy -- see module docstring.
    g = g.rename(columns={"max_days_active": "tenure_proxy_days"})

    return g.sort_values("artist_score", ascending=False).reset_index(drop=True)


def explain_artist_score_breakdown(artist_row: pd.Series, artist_metrics: pd.DataFrame) -> dict:
    """Return the normalized 0-1 component values behind one artist's
    score, for a radar/bar chart -- makes 'why #1' inspectable."""
    return {
        "Total streams": float(_minmax(artist_metrics["total_streams"]).loc[artist_row.name]),
        "Chart breadth": float(_minmax(artist_metrics["track_count"]).loc[artist_row.name]),
        "Avg position": float(_minmax(-artist_metrics["avg_pos"]).loc[artist_row.name]),
        "Avg viral score": float(_minmax(artist_metrics["avg_viral_score"]).loc[artist_row.name]),
    }


# ---------------------------------------------------------------------------
# Genre / country dominance
# ---------------------------------------------------------------------------

def genre_dominance(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("genre_clean").agg(
        total_streams=("streams", "sum"),
        track_count=("track_name", "count"),
        avg_viral_score=("viral_score", "mean"),
    ).reset_index().rename(columns={"genre_clean": "genre"})
    return g.sort_values("total_streams", ascending=False).reset_index(drop=True)


def country_dominance(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("country_clean").agg(
        total_streams=("streams", "sum"),
        track_count=("track_name", "count"),
        artist_count=("artist_name", "nunique"),
    ).reset_index().rename(columns={"country_clean": "country"})
    return g.sort_values("total_streams", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top-level summary bundle (this dict is what gets handed to the LLM layer)
# ---------------------------------------------------------------------------

def build_summary_context(df: pd.DataFrame, artist_metrics: pd.DataFrame) -> dict:
    """A compact, fully-computed JSON-able summary. This -- and only this
    -- is what the LLM layer is allowed to see and narrate. It never sees
    the raw dataframe, so it cannot 'compute' anything on its own.
    """
    top_artist = artist_metrics.iloc[0]
    top_song_row = df.sort_values("streams", ascending=False).iloc[0]

    return {
        "dataset_size": len(df),
        "top_artist": {
            "name": top_artist["artist_name"],
            "artist_score": float(top_artist["artist_score"]),
            "total_streams": int(top_artist["total_streams"]),
            "track_count": int(top_artist["track_count"]),
            "avg_pos": round(float(top_artist["avg_pos"]), 1),
            "avg_viral_score": round(float(top_artist["avg_viral_score"]), 0),
            "score_breakdown": explain_artist_score_breakdown(top_artist, artist_metrics),
        },
        "top_song": {
            "name": top_song_row["track_name"],
            "artist": top_song_row["artist_name"],
            "streams": int(top_song_row["streams"]),
            "pos": int(top_song_row["pos"]),
        },
        "rising_top5": rising_songs(df, 5)[["track_name", "artist_name", "momentum_score"]]
            .round(2).to_dict(orient="records"),
        "falling_top5": falling_songs(df, 5)[["track_name", "artist_name", "momentum_score"]]
            .round(2).to_dict(orient="records"),
        "long_term_top5": long_term_songs(df, 5)[["track_name", "artist_name", "longevity", "long_term_popularity_score"]]
            .to_dict(orient="records"),
        "future_potential_top5": future_potential_leaderboard(df, 5)[["track_name", "artist_name", "future_potential_score"]]
            .to_dict(orient="records"),
        "top_genres": genre_dominance(df).head(5).to_dict(orient="records"),
        "top_countries": country_dominance(df).head(5).to_dict(orient="records"),
        "methodology_caveat": (
            "Single-snapshot dataset: no true time-series forecast is computed. "
            "'future_potential_score' is a momentum/durability heuristic, and "
            "'tenure' is approximated from the days a track has been active, "
            "not a measured chart-history duration."
        ),
    }


if __name__ == "__main__":
    from data_processing import load_data, clean_data

    raw = load_data("data/spotify_global_trends_2026.csv")
    cleaned, report = clean_data(raw)
    songs = compute_song_metrics(cleaned)
    artists = compute_artist_metrics(songs)

    print("=== Top 5 artists by artist_score ===")
    print(artists[["artist_name", "artist_score", "total_streams", "track_count", "avg_pos"]].head())

    print("\n=== Top artist score breakdown ===")
    print(explain_artist_score_breakdown(artists.iloc[0], artists))

    print("\n=== Top 5 songs by streams ===")
    print(top_songs(songs, 5))

    print("\n=== Top 5 rising (by momentum) ===")
    print(rising_songs(songs, 5))

    print("\n=== Top 5 falling (by momentum) ===")
    print(falling_songs(songs, 5))

    print("\n=== Top 5 long-term popularity ===")
    print(long_term_songs(songs, 5))

    print("\n=== Top 5 future potential ===")
    print(future_potential_leaderboard(songs, 5))

    print("\n=== Genre dominance (top 5) ===")
    print(genre_dominance(songs).head())

    print("\n=== Country dominance (top 5) ===")
    print(country_dominance(songs).head())

    import json
    print("\n=== LLM context bundle ===")
    print(json.dumps(build_summary_context(songs, artists), indent=2, default=str))
