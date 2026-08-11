from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field


REQUIRED_COLUMNS = [
    "track_name", "artist_name", "streams", "stream_change", "7day",
    "genre", "country", "pos", "days", "viral_score", "trend",
    "popularity_category", "longevity",
]

# Added city/region - country corrections observed in this data.
#ran df.country.unique() and df.genre.value_counts() first to check all the unique and undefined values
#Hardcoding the bad ones below
COUNTRY_CORRECTIONS = {
    "Florida": "US",
    "England": "GB",
    "Culiacán": "MX",
    "Monterrey": "MX",
    "Toronto": "CA",
}

VALID_ISO2 = {
    "US", "GB", "KR", "CA", "PR", "MX", "AU", "CO", "SE", "JM", "IT",
    "JP", "FR", "IE", "NO", "DE", "ES", "BR", "NL", "SE", "PH", "NG",
}

#Attributed as Unclassified
NON_GENRE_VALUES = {
    "Billboard Hot 100", "Offizielle Charts", "Dolby Atmos",
    "Falcom", "Special Purpose Artist", "Toronto",
}

#structured object that runs all the way to the Data Quality tab in streamlit
@dataclass
class DataQualityReport:
    total_rows: int = 0
    duplicate_rows_removed: int = 0
    negative_or_invalid_numeric_rows: int = 0
    country_corrected: dict = field(default_factory=dict)   # original -> iso2
    country_unresolved: list = field(default_factory=list)  # rows still odd
    genre_reclassified: dict = field(default_factory=dict)  # value -> count
    rows_affected_total: int = 0

    def as_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "negative_or_invalid_numeric_rows": self.negative_or_invalid_numeric_rows,
            "country_corrected": self.country_corrected,
            "country_unresolved": self.country_unresolved,
            "genre_reclassified": self.genre_reclassified,
            "rows_affected_total": self.rows_affected_total,
        }


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    #Validate and clean the raw dataframe.
    report = DataQualityReport(total_rows=len(df))
    df = df.copy()

    #structural cleanup
    before = len(df)
    df = df.drop_duplicates(subset=["track_name", "artist_name"]).reset_index(drop=True)
    report.duplicate_rows_removed = before - len(df)

    numeric_cols = ["streams", "stream_change", "7day", "pos", "days", "viral_score"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce") #turning non-numeric junk into Nan isntead of crashing

    #Masking to drop invalid rows
    invalid_mask = (
        df["streams"].isna() | (df["streams"] < 0)
        | df["days"].isna() | (df["days"] < 0)
        | df["pos"].isna() | (df["pos"] < 1)
        | df["viral_score"].isna() | (df["viral_score"] < 0)
    )
    report.negative_or_invalid_numeric_rows = int(invalid_mask.sum())
    df = df.loc[~invalid_mask].reset_index(drop=True)

    #country normalization
    def fix_country(raw: str) -> tuple[str, bool]:
        raw = str(raw).strip()
        if raw in VALID_ISO2:
            return raw, False
        if raw in COUNTRY_CORRECTIONS:
            return COUNTRY_CORRECTIONS[raw], True
        return raw, True  # unresolved oddity, kept as-is but flagged

    fixed = df["country"].apply(fix_country)
    df["country_clean"] = fixed.apply(lambda t: t[0])
    df["country_flag"] = fixed.apply(lambda t: t[1])

    for raw_val in df.loc[df["country_flag"], "country"].unique():
        if raw_val in COUNTRY_CORRECTIONS:
            report.country_corrected[raw_val] = COUNTRY_CORRECTIONS[raw_val]
        else:
            report.country_unresolved.append(raw_val)

    #genre normalization, same logic as country but goes into unclassified instead of tagging per row
    def fix_genre(raw: str) -> tuple[str, bool]:
        raw = str(raw).strip()
        if raw in NON_GENRE_VALUES:
            return "Unclassified", True
        return raw, False

    fixed_g = df["genre"].apply(fix_genre)
    df["genre_clean"] = fixed_g.apply(lambda t: t[0])
    df["genre_flag"] = fixed_g.apply(lambda t: t[1])

    for raw_val, cnt in df.loc[df["genre_flag"], "genre"].value_counts().items():
        report.genre_reclassified[raw_val] = int(cnt)

    report.rows_affected_total = int((df["country_flag"] | df["genre_flag"]).sum())

    return df, report

#tally everything into the report #python3 run data_processing.py (2 data quality issues, genre and country)
if __name__ == "__main__":
    frame = load_data("data/spotify_global_trends_2026.csv")
    cleaned, rep = clean_data(frame)
    print(f"Rows in: {rep.total_rows}, rows out: {len(cleaned)}")
    print("Country corrections:", rep.country_corrected)
    print("Country unresolved:", rep.country_unresolved)
    print("Genre reclassified:", rep.genre_reclassified)
    print("Total rows affected by cleaning:", rep.rows_affected_total)
