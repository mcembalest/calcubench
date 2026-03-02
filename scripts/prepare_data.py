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


def prepare_state_finance():
    df = pd.read_csv(RAW_DIR / "state-govt-finance.csv")
    # Remove the aggregate "UNITED STATES" row
    df = df[df["State"] != "UNITED STATES"].copy()
    # Select and rename columns for cleaner names
    col_map = {
        "State": "state",
        "Year": "year",
        "Totals.Revenue": "revenue",
        "Totals.Expenditure": "expenditure",
        "Totals.Tax": "tax",
        "Totals.General expenditure": "general_expenditure",
        "Totals.Intergovernmental": "intergovernmental_revenue",
        "Details.Education.Education Total": "education_spending",
        "Details.Health.Health Total Expenditure": "health_spending",
        "Details.Transportation.Highways.Highways Total Expenditure": "highway_spending",
        "Details.Correction.Correction Total": "correction_spending",
        "Details.Police protection": "police_spending",
        "Totals. Debt at end of fiscal year": "debt",
    }
    df = df[list(col_map.keys())].rename(columns=col_map)
    records = df.to_dict(orient="records")
    out = OUT_DIR / "state_finance.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"state_finance: {len(records)} rows -> {out}")


def prepare_pokemon():
    df = pd.read_csv(RAW_DIR / "pokemon.csv")
    cols = [
        "name", "type1", "type2", "generation", "is_legendary",
        "hp", "attack", "defense", "sp_attack", "sp_defense", "speed", "base_total",
        "height_m", "weight_kg",
        "capture_rate", "base_egg_steps", "base_happiness", "experience_growth",
    ]
    df = df[cols].copy()
    df = df.dropna(subset=["height_m", "weight_kg"])
    # Fix non-numeric capture_rate (e.g. "30 (Meteorite)255 (Core)" -> 30)
    df["capture_rate"] = df["capture_rate"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    records = df.to_dict(orient="records")
    out = OUT_DIR / "pokemon.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"pokemon: {len(records)} rows -> {out}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prepare_census_southeast()
    prepare_census_national()
    prepare_imdb_top()
    prepare_pokemon()
    prepare_state_finance()
    print("Done.")
