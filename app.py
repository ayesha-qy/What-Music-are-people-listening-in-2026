"""
app.py
------
Streamlit dashboard: Music Trend Intelligence for 2026.

Layer separation is enforced on purpose:
  - data_processing.py + metrics.py = data science layer (all numbers)
  - llm_layer.py                    = explanation layer (all narration)
  - app.py (this file)              = presentation only. It never computes
    a statistic itself; it calls metrics.py and displays what comes back.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

import data_processing as dp
import metrics as mx
import llm_layer as llm

st.set_page_config(page_title="What Music are people listening in 2026", page_icon="🎵", layout="wide")

DATA_PATH = "data/spotify_global_trends_2026.csv"


# ---------------------------------------------------------------------------
# Cached data + metrics pipeline
# ---------------------------------------------------------------------------

@st.cache_data
def load_and_process(path: str):
    raw = dp.load_data(path)
    cleaned, report = dp.clean_data(raw)
    songs = mx.compute_song_metrics(cleaned)
    artists = mx.compute_artist_metrics(songs)
    return songs, artists, report


songs_df, artists_df, quality_report = load_and_process(DATA_PATH)

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.title("🎛️ Filters")
genres = ["All"] + sorted(songs_df["genre_clean"].unique().tolist())
countries = ["All"] + sorted(songs_df["country_clean"].unique().tolist())
trends = ["All", "Rising", "Falling"]
longevities = ["All"] + sorted(songs_df["longevity"].unique().tolist())

sel_genre = st.sidebar.selectbox("Genre", genres)
sel_country = st.sidebar.selectbox("Country", countries)
sel_trend = st.sidebar.selectbox("Trend", trends)
sel_longevity = st.sidebar.selectbox("Longevity", longevities)

st.sidebar.markdown("---")
model_name = st.sidebar.text_input("Ollama model", value=llm.DEFAULT_MODEL,
                                    help="Must match a model you've already run `ollama pull` for.")
ollama_up = llm.is_ollama_available()
if ollama_up:
    st.sidebar.success("🟢 Ollama detected -- AI narration active")
else:
    st.sidebar.warning("🟡 Ollama not detected -- showing rule-based summaries.\nRun `ollama serve` to enable AI narration.")

filtered = songs_df.copy()
if sel_genre != "All":
    filtered = filtered[filtered["genre_clean"] == sel_genre]
if sel_country != "All":
    filtered = filtered[filtered["country_clean"] == sel_country]
if sel_trend != "All":
    filtered = filtered[filtered["trend"] == sel_trend]
if sel_longevity != "All":
    filtered = filtered[filtered["longevity"] == sel_longevity]

filtered_artists = mx.compute_artist_metrics(filtered) if len(filtered) else artists_df.iloc[0:0]


def narration_box(text: str, used_llm: bool):
    label = "🤖 AI narration (Ollama)" if used_llm else "📋 Rule-based summary (Ollama offline)"
    with st.container(border=True):
        st.caption(label)
        st.write(text)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🎵 What Music are people listening in 2026")
st.caption("Data science layer computes every number below. The LLM layer only explains them.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Tracks in view", len(filtered))
k2.metric("Total streams", f"{filtered['streams'].sum():,}" if len(filtered) else "0")
k3.metric("Distinct artists", filtered["artist_name"].nunique() if len(filtered) else 0)
k4.metric("Rising tracks", int((filtered["trend"] == "Rising").sum()) if len(filtered) else 0)

st.markdown("---")

tabs = st.tabs([
    "🏆 Top Artist", "🎧 Top Songs", "📈 Rising & Falling", "⏳ Long-Term Popularity",
    "🌍 Genre & Country", "🔮 Future Outlook", "💬 Ask the Data", "🧹 Data Quality",
])

# --- Top Artist -------------------------------------------------------------
with tabs[0]:
    st.subheader("Who is the top artist, and why?")
    if len(filtered_artists) == 0:
        st.info("No data matches the current filters.")
    else:
        top = filtered_artists.iloc[0]
        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.metric("Top artist", top["artist_name"], f"score {top['artist_score']}/100")
            st.write(f"**Total streams:** {int(top['total_streams']):,}")
            st.write(f"**Tracks charting:** {int(top['track_count'])}")
            st.write(f"**Average position:** {top['avg_pos']:.1f}")
            st.write(f"**Average viral score:** {top['avg_viral_score']:,.0f}")
            st.write(f"**Tenure proxy (days on longest-running charting track):** {int(top['tenure_proxy_days'])}")
        with c2:
            breakdown = mx.explain_artist_score_breakdown(top, filtered_artists)
            bd_df = pd.DataFrame({"factor": list(breakdown.keys()), "value": list(breakdown.values())})
            fig = px.bar(bd_df, x="value", y="factor", orientation="h",
                         title="Why this artist scores #1 (normalized 0-1 per factor)",
                         range_x=[0, 1])
            st.plotly_chart(fig, width="stretch")

        context = mx.build_summary_context(filtered, filtered_artists)
        text, used = llm.explain_top_artist(context, model_name)
        narration_box(text, used)

        st.markdown("##### Full artist leaderboard")
        st.dataframe(
            filtered_artists[["artist_name", "artist_score", "total_streams", "track_count",
                               "avg_pos", "avg_viral_score", "tenure_proxy_days"]],
            width="stretch", hide_index=True,
        )

# --- Top Songs ---------------------------------------------------------------
with tabs[1]:
    st.subheader("Which songs are listened to most often?")
    if len(filtered) == 0:
        st.info("No data matches the current filters.")
    else:
        top_songs = mx.top_songs(filtered, 15)
        fig = px.bar(top_songs.sort_values("streams"), x="streams", y="track_name",
                     orientation="h", color="artist_name", title="Top tracks by streams")
        st.plotly_chart(fig, width="stretch")

        text, used = llm.explain_songs(
            mx.top_songs(filtered, 10)[["track_name", "artist_name", "streams"]].to_dict(orient="records"),
            model_name,
        )
        narration_box(text, used)
        st.dataframe(top_songs, width="stretch", hide_index=True)

# --- Rising & Falling ---------------------------------------------------------
with tabs[2]:
    st.subheader("Which artists/songs are rising or falling?")
    if len(filtered) == 0:
        st.info("No data matches the current filters.")
    else:
        c1, c2 = st.columns(2)
        rising = mx.rising_songs(filtered, 10)
        falling = mx.falling_songs(filtered, 10)
        with c1:
            st.markdown("**🚀 Fastest rising**")
            fig_r = px.bar(rising.sort_values("momentum_score"), x="momentum_score", y="track_name",
                            orientation="h", color_discrete_sequence=["#2ecc71"])
            st.plotly_chart(fig_r, width="stretch")
        with c2:
            st.markdown("**📉 Fastest falling**")
            fig_f = px.bar(falling.sort_values("momentum_score", ascending=False), x="momentum_score",
                            y="track_name", orientation="h", color_discrete_sequence=["#e74c3c"])
            st.plotly_chart(fig_f, width="stretch")

        text, used = llm.explain_trends(
            rising[["track_name", "artist_name", "momentum_score"]].round(2).to_dict(orient="records"),
            falling[["track_name", "artist_name", "momentum_score"]].round(2).to_dict(orient="records"),
            model_name,
        )
        narration_box(text, used)

# --- Long-Term Popularity -----------------------------------------------------
with tabs[3]:
    st.subheader("Which songs have long-term popularity?")
    if len(filtered) == 0:
        st.info("No data matches the current filters.")
    else:
        lt = mx.long_term_songs(filtered, 15)
        if len(lt) == 0:
            st.info("No Evergreen / Stable Hit tracks match the current filters.")
        else:
            fig = px.scatter(filtered[filtered["longevity"].isin(["Evergreen", "Stable Hit"])],
                              x="days", y="streams", size="long_term_popularity_score",
                              color="longevity", hover_name="track_name",
                              title="Days active vs. streams (bubble = long-term popularity score)")
            st.plotly_chart(fig, width="stretch")

            text, used = llm.explain_long_term(lt.to_dict(orient="records"), model_name)
            narration_box(text, used)
            st.dataframe(lt, width="stretch", hide_index=True)

# --- Genre & Country -----------------------------------------------------------
with tabs[4]:
    st.subheader("Which genres and countries dominate?")
    if len(filtered) == 0:
        st.info("No data matches the current filters.")
    else:
        genre_dom = mx.genre_dominance(filtered)
        country_dom = mx.country_dominance(filtered)
        c1, c2 = st.columns(2)
        with c1:
            fig_g = px.treemap(genre_dom, path=["genre"], values="total_streams",
                                title="Genre dominance by total streams")
            st.plotly_chart(fig_g, width="stretch")
        with c2:
            fig_c = px.bar(country_dom.sort_values("total_streams"), x="total_streams", y="country",
                            orientation="h", title="Country dominance by total streams")
            st.plotly_chart(fig_c, width="stretch")

        text, used = llm.explain_genre_country(
            genre_dom.head(5).to_dict(orient="records"),
            country_dom.head(5).to_dict(orient="records"),
            model_name,
        )
        narration_box(text, used)
        if "Unclassified" in genre_dom["genre"].values:
            st.caption("Note: 'Unclassified' groups genre values that were actually chart names or "
                       "audio formats in the source data (see Data Quality tab), not a real genre.")

# --- Future Outlook --------------------------------------------------------------
with tabs[5]:
    st.subheader("What does the data suggest about future popularity?")
    st.warning(
        "⚠️ This dataset is a single snapshot with no repeated history per track, so a "
        "statistically valid time-series forecast (Prophet/ARIMA/XGBoost) isn't possible here. "
        "The scores below are a **momentum/durability heuristic**, not a predicted stream count. "
        "See the Data Quality tab and README for details."
    )
    if len(filtered) == 0:
        st.info("No data matches the current filters.")
    else:
        fp = mx.future_potential_leaderboard(filtered, 15)
        fig = px.bar(fp.sort_values("future_potential_score"), x="future_potential_score", y="track_name",
                     orientation="h", color="trend", title="Future potential score (heuristic, 0-100)")
        st.plotly_chart(fig, width="stretch")

        context = mx.build_summary_context(filtered, filtered_artists) if len(filtered_artists) else None
        if context:
            text, used = llm.explain_future_outlook(context, model_name)
            narration_box(text, used)
        st.dataframe(fp, width="stretch", hide_index=True)

# --- Ask the Data -----------------------------------------------------------------
with tabs[6]:
    st.subheader("Ask the data a question")
    st.caption("The model only ever sees the pre-computed summary bundle below -- it cannot invent numbers "
               "that aren't already calculated.")
    question = st.text_input("Your question", placeholder="e.g. Why is BTS ranked first?")
    if st.button("Ask", type="primary") and question:
        context = mx.build_summary_context(filtered if len(filtered) else songs_df,
                                            filtered_artists if len(filtered_artists) else artists_df)
        with st.spinner("Thinking..."):
            text, used = llm.answer_custom_question(question, context, model_name)
        narration_box(text, used)
    with st.expander("View the exact data the model can see"):
        context = mx.build_summary_context(filtered if len(filtered) else songs_df,
                                            filtered_artists if len(filtered_artists) else artists_df)
        st.json(context)

# --- Data Quality ---------------------------------------------------------------
with tabs[7]:
    st.subheader("Data cleaning & validation report")
    st.write(f"**Rows in source file:** {quality_report.total_rows}")
    st.write(f"**Duplicate rows removed:** {quality_report.duplicate_rows_removed}")
    st.write(f"**Rows dropped for invalid/negative numeric fields:** {quality_report.negative_or_invalid_numeric_rows}")
    st.write(f"**Total rows affected by any correction:** {quality_report.rows_affected_total}")

    st.markdown("##### Country field corrections")
    if quality_report.country_corrected:
        st.dataframe(pd.DataFrame(
            [{"raw_value": k, "corrected_to": v} for k, v in quality_report.country_corrected.items()]
        ), hide_index=True)
    else:
        st.caption("None needed.")
    if quality_report.country_unresolved:
        st.warning(f"Unresolved country values (kept as-is, flagged): {quality_report.country_unresolved}")

    st.markdown("##### Genre field reclassifications")
    st.caption("These raw values were chart names, audio formats, or labels -- not musical genres -- "
               "so they were moved to 'Unclassified' rather than guessed at.")
    if quality_report.genre_reclassified:
        st.dataframe(pd.DataFrame(
            [{"raw_value": k, "row_count": v} for k, v in quality_report.genre_reclassified.items()]
        ), hide_index=True)
    else:
        st.caption("None needed.")

    st.markdown("---")
    st.markdown("##### Methodology notes")
    st.caption(mx.build_summary_context(songs_df, artists_df)["methodology_caveat"])
