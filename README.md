# What-Music-are-people-listening-in-2026

# Who's Trending — 2026

A Streamlit dashboard over `spotify_global_trends_2026.csv` with a strict
separation between:

- **Data science layer** (`data_processing.py`, `metrics.py`) — every
  ranking, score, and statistic. Deterministic, inspectable, testable
  without any LLM.
- **LLM layer** (`llm_layer.py`) — takes the *already-computed* numbers
  and narrates them in plain English via a local Ollama model. It never
  computes a statistic itself.
- **Presentation layer** (`app.py`) — Streamlit UI. Calls the two layers
  above and displays the results; contains no calculations of its own.

## Setup

```bash
pip install -r requirements.txt

# Install Ollama (https://ollama.com) and pull a model, e.g.:
ollama pull llama3.1
ollama serve          # in a separate terminal, if not already running

streamlit run app.py
```

The dashboard works even without Ollama running — every tab falls back to
a deterministic, template-based summary and says so ("📋 Rule-based
summary (Ollama offline)"). This is not a fake AI response; it's plain
Python formatting of the same computed numbers, so the dashboard is never
blocked on the LLM.

If you use a different local model, set it in the sidebar text box, or
change `DEFAULT_MODEL` in `llm_layer.py`.

## Project structure

```
music_trend_intelligence/
├── app.py                # Streamlit dashboard (presentation only)
├── data_processing.py    # load + clean + validate raw CSV
├── metrics.py            # rankings, composite scores, aggregates
├── llm_layer.py          # Ollama narration + offline fallback
├── forecasting.py        # NOT active — scaffolding for real time-series
│                          # forecasting once you have repeated snapshots
├── requirements.txt
└── data/
    └── spotify_global_trends_2026.csv
```

## What each dashboard question maps to

| Question | Where it's computed | Where it's shown |
|---|---|---|
| Who is the top artist? | `metrics.compute_artist_metrics` → `artist_score` | Top Artist tab |
| Why is that artist ranked first? | `metrics.explain_artist_score_breakdown` (decomposes the 4 weighted inputs) | Top Artist tab, bar chart |
| Which songs are listened to most? | `metrics.top_songs` (sorted by `streams`) | Top Songs tab |
| Which artists/songs are rising or falling? | `metrics.rising_songs` / `falling_songs`, ranked by `momentum_score` | Rising & Falling tab |
| Which songs have long-term popularity? | `metrics.long_term_songs`, `long_term_popularity_score` | Long-Term Popularity tab |
| How long has an artist stayed near the top? | `tenure_proxy_days` in `compute_artist_metrics` | Top Artist tab |
| Which genres/countries dominate? | `metrics.genre_dominance` / `country_dominance` | Genre & Country tab |
| What does the data suggest about future popularity? | `metrics.future_potential_score` (heuristic) | Future Outlook tab |

## Honest limitations (please read before presenting this)

**This CSV is a single snapshot** — one row per track, not a repeated
time series. That has real consequences, surfaced directly in the
dashboard rather than glossed over:

1. **No true forecasting.** "Future potential" is a momentum/durability
   *heuristic* built from `viral_score`, `stream_change`, `trend`, and
   `longevity` — not a predicted future stream count. A real forecast
   needs a time series to fit, which this data doesn't have. This is
   stated on the Future Outlook tab itself, not just in this README.
   `forecasting.py` documents exactly what's needed to turn on real
   forecasting later (Prophet/XGBoost/ARIMA) once you're collecting
   dated snapshots.
2. **Tenure is a proxy, not a measurement.** "How long has an artist
   stayed near the top" is approximated from `days` on their
   longest-running currently-charting track. It is *not* a measured
   chart-history duration, because we don't have historical daily
   positions to measure from.
3. **Chart position (`pos`) does not simply track streams.** Only 23 of
   178 rows have `pos` matching a pure streams-descending rank — Spotify's
   actual chart methodology clearly weighs more than raw stream count.
   The `artist_score` and `future_potential_score` formulas are built from
   the columns actually in the CSV and are documented in `metrics.py`;
   they're a reasonable, transparent proxy, not a reproduction of
   Spotify's real algorithm.
4. **Two source columns had data-quality defects, found (not assumed) by
   inspection:**
   - `country` sometimes holds a city/region instead of a country
     (`Florida`, `England`, `Culiacán`, `Monterrey`) — corrected to ISO-2
     codes in `country_clean`, with every correction logged.
   - `genre` sometimes holds a chart name or audio format instead of a
     genre (`Billboard Hot 100`, `Offizielle Charts`, `Dolby Atmos`,
     `Special Purpose Artist`) — moved to `Unclassified` in `genre_clean`
     rather than guessed at, since there's no reliable signal to infer
     the true genre from the row. 14 of 178 rows (~8%) are affected.
   Both corrections, plus row counts, are shown live on the **Data
   Quality** tab so nothing is hidden from whoever's viewing the
   dashboard.

## Extending this

- **Real forecasting:** see `forecasting.py`'s docstring for exactly what
  data collection step unlocks it.
- **More data:** `data_processing.REQUIRED_COLUMNS` documents the schema
  contract; any new CSV with those columns drops in without code changes.
- **Swapping the LLM:** `llm_layer.py` only depends on Ollama's HTTP API
  (`localhost:11434`). Swapping models is a one-line change
  (`DEFAULT_MODEL` or the sidebar field); swapping to a different local
  runtime means changing `_call_ollama` only — nothing in `metrics.py` or
  `app.py` needs to change, by design.
