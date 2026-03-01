"""Prepare raw CSVs into filtered JSON datasets for the benchmark."""

import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).parent.parent / "data" / "prepared"


def prepare_census_southeast():
    df = pd.read_csv(RAW_DIR / "cc-est2024-agesex-all.csv", encoding="latin-1")
    se_states = [
        "Florida", "Georgia", "Alabama", "Mississippi",
        "South Carolina", "North Carolina", "Tennessee", "Virginia",
    ]
    cols = [
        "STNAME", "CTYNAME", "POPESTIMATE", "POPEST_MALE", "POPEST_FEM",
        "UNDER5_TOT", "AGE513_TOT", "AGE1417_TOT", "AGE1824_TOT",
        "AGE2544_TOT", "AGE4564_TOT", "AGE65PLUS_TOT", "MEDIAN_AGE_TOT",
    ]
    filtered = df[(df["YEAR"] == 6) & (df["STNAME"].isin(se_states))][cols]
    records = filtered.to_dict(orient="records")
    out = OUT_DIR / "census_southeast.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"census_southeast: {len(records)} rows -> {out}")


def prepare_census_national():
    df = pd.read_csv(RAW_DIR / "cc-est2024-agesex-all.csv", encoding="latin-1")
    cols = ["STNAME", "CTYNAME", "POPESTIMATE", "POPEST_MALE", "POPEST_FEM"]
    filtered = df[df["YEAR"] == 6][cols]
    records = filtered.to_dict(orient="records")
    out = OUT_DIR / "census_national.json"
    out.write_text(json.dumps(records))  # compact -- no indent
    print(f"census_national: {len(records)} rows -> {out}")


def prepare_imdb_top():
    movies = pd.read_csv(RAW_DIR / "IMDb movies.csv", encoding="latin-1")
    ratings = pd.read_csv(RAW_DIR / "IMDb ratings.csv", encoding="latin-1")
    merged = movies.merge(ratings, on="imdb_title_id")
    merged = merged[merged["total_votes"] >= 200000]
    merged["year"] = pd.to_numeric(merged["year"], errors="coerce")
    merged = merged.dropna(subset=["year"])
    merged["year"] = merged["year"].astype(int)
    cols = [
        "title", "year", "genre", "duration", "country", "director",
        "avg_vote", "votes", "males_allages_avg_vote", "females_allages_avg_vote",
    ]
    records = merged[cols].to_dict(orient="records")
    out = OUT_DIR / "imdb_top.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"imdb_top: {len(records)} rows -> {out}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prepare_census_southeast()
    prepare_census_national()
    prepare_imdb_top()
    print("Done.")
