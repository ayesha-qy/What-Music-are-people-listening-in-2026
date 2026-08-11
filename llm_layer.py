"""
llm_layer.py
------------
LLM layer: turns already-computed numbers into natural-language
explanations. This module is not allowed to invent statistics -- every
prompt embeds a pre-computed JSON context (built by metrics.py) and
instructs the model to only narrate what's in it.

Two paths:
  1. Ollama (local model) if it's running -- real LLM narration.
  2. A deterministic, template-based fallback if Ollama is unreachable,
     so the dashboard is still fully usable without a local model running.
     This is not a lesser LLM call pretending to be one; it's plain
     Python string formatting, and the dashboard labels it as such.

Nothing here calls out to any external/cloud API -- per the project spec,
this only ever talks to http://localhost:11434 (Ollama's default).
"""

from __future__ import annotations

import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.1"  # change to whatever you've `ollama pull`-ed
REQUEST_TIMEOUT = 30

SYSTEM_INSTRUCTIONS = (
    "You are a music-analytics narrator for a dashboard called Music Trend "
    "Intelligence. You will be given a JSON object of ALREADY-COMPUTED "
    "statistics. Your only job is to explain those numbers clearly and "
    "concisely in plain English for a business audience. Rules:\n"
    "1. Never invent a number that is not in the JSON.\n"
    "2. Never recompute or 'correct' a number -- treat the JSON as ground truth.\n"
    "3. If the JSON contains a methodology_caveat, weave it in naturally "
    "when relevant (e.g. when discussing future potential or tenure) "
    "rather than ignoring it.\n"
    "4. Keep answers tight: 3-6 sentences unless asked for more detail.\n"
    "5. Do not use markdown headers; plain prose or short bullet points only."
)


def is_ollama_available(url: str = OLLAMA_URL) -> bool:
    try:
        base = url.replace("/api/generate", "/api/tags")
        r = requests.get(base, timeout=2)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    payload = {
        "model": model,
        "prompt": f"{SYSTEM_INSTRUCTIONS}\n\n{prompt}",
        "stream": False,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _safe_generate(prompt: str, fallback_fn, model: str = DEFAULT_MODEL) -> tuple[str, bool]:
    """Try Ollama; on any failure, use the deterministic fallback.
    Returns (text, used_llm)."""
    try:
        if is_ollama_available():
            return _call_ollama(prompt, model), True
    except Exception:
        pass
    return fallback_fn(), False


# ---------------------------------------------------------------------------
# Explanation functions -- one per dashboard question
# ---------------------------------------------------------------------------

def explain_top_artist(context: dict, model: str = DEFAULT_MODEL) -> tuple[str, bool]:
    a = context["top_artist"]
    prompt = (
        "Explain why this artist is ranked #1, referencing the score "
        "breakdown so the reasoning is transparent, not just asserted.\n\n"
        f"{json.dumps(a, indent=2)}"
    )

    def fallback():
        bd = a["score_breakdown"]
        top_factor = max(bd, key=bd.get)
        return (
            f"{a['name']} ranks #1 with an artist_score of {a['artist_score']}/100, "
            f"driven mainly by {top_factor.lower()}. They hold {a['track_count']} tracks "
            f"on the chart simultaneously, totaling {a['total_streams']:,} streams, at an "
            f"average chart position of {a['avg_pos']}. Charting this many tracks at once "
            "is itself a dominance signal, not just a single hit song."
        )

    return _safe_generate(prompt, fallback, model)


def explain_songs(top_songs_records: list[dict], model: str = DEFAULT_MODEL) -> tuple[str, bool]:
    prompt = (
        "Summarize which songs are listened to most often, based only on "
        f"this list (already sorted by streams):\n\n{json.dumps(top_songs_records, indent=2)}"
    )

    def fallback():
        lines = [f"{i+1}. \"{s['track_name']}\" by {s['artist_name']} ({s['streams']:,} streams)"
                  for i, s in enumerate(top_songs_records[:5])]
        return "Most-streamed tracks right now:\n" + "\n".join(lines)

    return _safe_generate(prompt, fallback, model)


def explain_trends(rising: list[dict], falling: list[dict], model: str = DEFAULT_MODEL) -> tuple[str, bool]:
    prompt = (
        "Explain which artists/songs are rising and which are falling, "
        "using momentum_score (positive = rising strength, negative = "
        f"falling strength).\n\nRising:\n{json.dumps(rising, indent=2)}\n\n"
        f"Falling:\n{json.dumps(falling, indent=2)}"
    )

    def fallback():
        r = ", ".join(f"\"{s['track_name']}\" ({s['artist_name']})" for s in rising[:3])
        f_ = ", ".join(f"\"{s['track_name']}\" ({s['artist_name']})" for s in falling[:3])
        return (
            f"Fastest-rising right now: {r}. Fastest-falling: {f_}. "
            "Momentum here reflects recent stream change and viral score, not total volume -- "
            "a song can be huge in absolute streams while still losing momentum."
        )

    return _safe_generate(prompt, fallback, model)


def explain_long_term(records: list[dict], model: str = DEFAULT_MODEL) -> tuple[str, bool]:
    prompt = (
        "Explain which songs show long-term popularity (Evergreen/Stable "
        f"Hit classification with sustained streams and days active):\n\n{json.dumps(records, indent=2)}"
    )

    def fallback():
        names = ", ".join(f"\"{r['track_name']}\" ({r['artist_name']})" for r in records[:5])
        return (
            f"Long-term popularity leaders: {names}. These are classified Evergreen or "
            "Stable Hit and have sustained streams over an extended active period, "
            "unlike New tracks that may be spiking on a recent release."
        )

    return _safe_generate(prompt, fallback, model)


def explain_genre_country(genres: list[dict], countries: list[dict], model: str = DEFAULT_MODEL) -> tuple[str, bool]:
    prompt = (
        "Explain which genres and countries dominate, based on total "
        f"streams:\n\nGenres:\n{json.dumps(genres, indent=2)}\n\n"
        f"Countries:\n{json.dumps(countries, indent=2)}"
    )

    def fallback():
        g0, c0 = genres[0], countries[0]
        return (
            f"{g0['genre']} leads by genre with {g0['total_streams']:,} total streams across "
            f"{g0['track_count']} tracks. {c0['country']} leads by country with "
            f"{c0['total_streams']:,} streams from {c0['artist_count']} distinct artists."
        )

    return _safe_generate(prompt, fallback, model)


def explain_future_outlook(context: dict, model: str = DEFAULT_MODEL) -> tuple[str, bool]:
    records = context["future_potential_top5"]
    caveat = context["methodology_caveat"]
    prompt = (
        "Explain what the data suggests about future popularity, using "
        "future_potential_score. IMPORTANT: this is a momentum/durability "
        "heuristic from a single data snapshot, not a statistical forecast "
        f"-- state that plainly. Caveat to include: {caveat}\n\n"
        f"{json.dumps(records, indent=2)}"
    )

    def fallback():
        names = ", ".join(f"\"{r['track_name']}\" ({r['artist_name']})" for r in records[:5])
        return (
            f"Based on current momentum and durability signals, {names} show the strongest "
            f"combination of viral score, positive stream change, and trend direction. "
            f"Note: {caveat}"
        )

    return _safe_generate(prompt, fallback, model)


def answer_custom_question(question: str, context: dict, model: str = DEFAULT_MODEL) -> tuple[str, bool]:
    prompt = (
        "Answer this question using ONLY the JSON data below. If the "
        "answer cannot be determined from this data, say so explicitly "
        "rather than guessing.\n\n"
        f"Question: {question}\n\nData:\n{json.dumps(context, indent=2, default=str)}"
    )

    def fallback():
        return (
            "Ollama isn't running, so free-form Q&A isn't available right now -- "
            "start Ollama (`ollama serve`) to enable it. In the meantime, the "
            "other tabs show the underlying computed metrics directly."
        )

    return _safe_generate(prompt, fallback, model)
