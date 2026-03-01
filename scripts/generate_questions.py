"""Generate benchmark questions with ground-truth answers computed via pandas."""

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "prepared"
Q_DIR = Path(__file__).parent.parent / "questions"


def load(name):
    return pd.DataFrame(json.loads((DATA_DIR / f"{name}.json").read_text()))


def census_southeast_questions(df):
    # Precompute age bracket sums per state for Q10
    age_cols = {
        "under5": "UNDER5_TOT",
        "age5_13": "AGE513_TOT",
        "age14_17": "AGE1417_TOT",
        "age18_24": "AGE1824_TOT",
        "age25_44": "AGE2544_TOT",
        "age45_64": "AGE4564_TOT",
        "age65plus": "AGE65PLUS_TOT",
    }

    return [
        # 001: Total population per state (8 rows)
        {
            "id": "census_se_001",
            "dataset": "census_southeast",
            "question": "What is the total population per state? Return a JSON object mapping each state name to its total population.",
            "answer": {k: int(v) for k, v in df.groupby("STNAME")["POPESTIMATE"].sum().to_dict().items()},
            "answer_type": "table",
            "value_types": "integer",
            "tolerance": 0,
        },
        # 002: Pct female per state (8 rows, float)
        {
            "id": "census_se_002",
            "dataset": "census_southeast",
            "question": "What is the percentage of the population that is female for each state? Round to 2 decimal places. Return a JSON object mapping each state name to its percentage female.",
            "answer": {
                k: round(v, 2)
                for k, v in (
                    df.groupby("STNAME")["POPEST_FEM"].sum()
                    / df.groupby("STNAME")["POPESTIMATE"].sum()
                    * 100
                ).to_dict().items()
            },
            "answer_type": "table",
            "value_types": "float",
            "tolerance": 0.01,
        },
        # 003: Avg median age per state (8 rows, float)
        {
            "id": "census_se_003",
            "dataset": "census_southeast",
            "question": "What is the average median age across counties for each state? Round to 2 decimal places. Return a JSON object mapping each state name to its average county median age.",
            "answer": {
                k: round(v, 2)
                for k, v in df.groupby("STNAME")["MEDIAN_AGE_TOT"].mean().to_dict().items()
            },
            "answer_type": "table",
            "value_types": "float",
            "tolerance": 0.01,
        },
        # 004: Total 65+ pop per state (8 rows)
        {
            "id": "census_se_004",
            "dataset": "census_southeast",
            "question": "What is the total population aged 65 and over per state? Return a JSON object mapping each state name to its total 65+ population.",
            "answer": {k: int(v) for k, v in df.groupby("STNAME")["AGE65PLUS_TOT"].sum().to_dict().items()},
            "answer_type": "table",
            "value_types": "integer",
            "tolerance": 0,
        },
        # 005: Every county with pop > 100K (~163 rows)
        {
            "id": "census_se_005",
            "dataset": "census_southeast",
            "question": "For every county with a total population greater than 100,000, return a JSON object mapping 'CountyName, StateName' to that county's population.",
            "answer": {
                f"{row['CTYNAME']}, {row['STNAME']}": int(row["POPESTIMATE"])
                for _, row in df[df["POPESTIMATE"] > 100_000].iterrows()
            },
            "answer_type": "table",
            "value_types": "integer",
            "tolerance": 0,
        },
        # 006: Pct under-5 per state (8 rows, float)
        {
            "id": "census_se_006",
            "dataset": "census_southeast",
            "question": "What percentage of each state's total population is under 5 years old? Round to 2 decimal places. Return a JSON object mapping each state name to its under-5 percentage.",
            "answer": {
                k: round(v, 2)
                for k, v in (
                    df.groupby("STNAME")["UNDER5_TOT"].sum()
                    / df.groupby("STNAME")["POPESTIMATE"].sum()
                    * 100
                ).to_dict().items()
            },
            "answer_type": "table",
            "value_types": "float",
            "tolerance": 0.01,
        },
        # 007: Most populous county per state (8 rows, multi-col)
        {
            "id": "census_se_007",
            "dataset": "census_southeast",
            "question": "What is the most populous county in each state? Return a JSON object mapping each state name to an object with 'county' (county name) and 'population' (its population).",
            "answer": (
                lambda: (
                    top := df.loc[df.groupby("STNAME")["POPESTIMATE"].idxmax()],
                    {
                        row["STNAME"]: {"county": row["CTYNAME"], "population": int(row["POPESTIMATE"])}
                        for _, row in top.iterrows()
                    },
                )[-1]
            )(),
            "answer_type": "table",
            "value_types": {"county": "string", "population": "integer"},
            "tolerance": 0,
        },
        # 008: County count + total pop per state (8 rows, multi-col)
        {
            "id": "census_se_008",
            "dataset": "census_southeast",
            "question": "For each state, how many counties are there and what is the total population? Return a JSON object mapping each state name to an object with 'county_count' and 'total_pop'.",
            "answer": (
                lambda: (
                    counts := df.groupby("STNAME").size(),
                    pops := df.groupby("STNAME")["POPESTIMATE"].sum(),
                    {
                        st: {"county_count": int(counts[st]), "total_pop": int(pops[st])}
                        for st in sorted(counts.index)
                    },
                )[-1]
            )(),
            "answer_type": "table",
            "value_types": {"county_count": "integer", "total_pop": "integer"},
            "tolerance": 0,
        },
        # 009: Every Georgia county with pop > 50K (multi-col)
        {
            "id": "census_se_009",
            "dataset": "census_southeast",
            "question": "For every county in Georgia with a population greater than 50,000, return a JSON object mapping the county name to an object with 'population' and 'median_age' (median age as-is from the data).",
            "answer": {
                row["CTYNAME"]: {"population": int(row["POPESTIMATE"]), "median_age": float(row["MEDIAN_AGE_TOT"])}
                for _, row in df[(df["STNAME"] == "Georgia") & (df["POPESTIMATE"] > 50_000)].iterrows()
            },
            "answer_type": "table",
            "value_types": {"population": "integer", "median_age": "float"},
            "tolerances": {"median_age": 0.1},
            "tolerance": 0,
        },
        # 010: Total pop in each age bracket per state (8 rows, 7 cols)
        {
            "id": "census_se_010",
            "dataset": "census_southeast",
            "question": "For each state, what is the total population in each age bracket? Return a JSON object mapping each state name to an object with keys 'under5', 'age5_13', 'age14_17', 'age18_24', 'age25_44', 'age45_64', 'age65plus'.",
            "answer": {
                st: {
                    label: int(df[df["STNAME"] == st][col].sum())
                    for label, col in age_cols.items()
                }
                for st in sorted(df["STNAME"].unique())
            },
            "answer_type": "table",
            "value_types": {label: "integer" for label in age_cols},
            "tolerance": 0,
        },
    ]


def census_national_questions(df):
    return [
        # 001: Total population per state (51 rows)
        {
            "id": "census_nat_001",
            "dataset": "census_national",
            "question": "What is the total population per state? Return a JSON object mapping each state name to its total population (all 51 states including DC).",
            "answer": {k: int(v) for k, v in df.groupby("STNAME")["POPESTIMATE"].sum().to_dict().items()},
            "answer_type": "table",
            "value_types": "integer",
            "tolerance": 0,
        },
        # 002: Total male pop per state (51 rows)
        {
            "id": "census_nat_002",
            "dataset": "census_national",
            "question": "What is the total male population per state? Return a JSON object mapping each state name to its total male population.",
            "answer": {k: int(v) for k, v in df.groupby("STNAME")["POPEST_MALE"].sum().to_dict().items()},
            "answer_type": "table",
            "value_types": "integer",
            "tolerance": 0,
        },
        # 003: Pct male per state (51 rows, float)
        {
            "id": "census_nat_003",
            "dataset": "census_national",
            "question": "What percentage of each state's population is male? Round to 2 decimal places. Return a JSON object mapping each state name to its male percentage.",
            "answer": {
                k: round(v, 2)
                for k, v in (
                    df.groupby("STNAME")["POPEST_MALE"].sum()
                    / df.groupby("STNAME")["POPESTIMATE"].sum()
                    * 100
                ).to_dict().items()
            },
            "answer_type": "table",
            "value_types": "float",
            "tolerance": 0.01,
        },
        # 004: County count per state (51 rows)
        {
            "id": "census_nat_004",
            "dataset": "census_national",
            "question": "How many counties does each state have? Return a JSON object mapping each state name to its county count.",
            "answer": {k: int(v) for k, v in df.groupby("STNAME").size().to_dict().items()},
            "answer_type": "table",
            "value_types": "integer",
            "tolerance": 0,
        },
        # 005: Avg county pop per state (51 rows, rounded int)
        {
            "id": "census_nat_005",
            "dataset": "census_national",
            "question": "What is the average county population per state, rounded to the nearest integer? Return a JSON object mapping each state name to its average county population.",
            "answer": {
                k: int(round(v))
                for k, v in df.groupby("STNAME")["POPESTIMATE"].mean().to_dict().items()
            },
            "answer_type": "table",
            "value_types": "integer",
            "tolerance": 0,
        },
        # 006: States with 100+ counties -> county count + total pop (multi-col)
        {
            "id": "census_nat_006",
            "dataset": "census_national",
            "question": "For states with 100 or more counties, return a JSON object mapping each state name to an object with 'county_count' (number of counties) and 'total_pop' (total population).",
            "answer": (
                lambda: (
                    counts := df.groupby("STNAME").size(),
                    pops := df.groupby("STNAME")["POPESTIMATE"].sum(),
                    big := counts[counts >= 100].index,
                    {
                        st: {"county_count": int(counts[st]), "total_pop": int(pops[st])}
                        for st in sorted(big)
                    },
                )[-1]
            )(),
            "answer_type": "table",
            "value_types": {"county_count": "integer", "total_pop": "integer"},
            "tolerance": 0,
        },
        # 007: All counties with pop > 500K -> county: pop + state (~148 rows, multi-col)
        {
            "id": "census_nat_007",
            "dataset": "census_national",
            "question": "For every county with a population greater than 500,000, return a JSON object mapping 'CountyName, StateName' to an object with 'population' and 'state'.",
            "answer": {
                f"{row['CTYNAME']}, {row['STNAME']}": {
                    "population": int(row["POPESTIMATE"]),
                    "state": row["STNAME"],
                }
                for _, row in df[df["POPESTIMATE"] > 500_000].iterrows()
            },
            "answer_type": "table",
            "value_types": {"population": "integer", "state": "string"},
            "tolerance": 0,
        },
        # 008: Most populous county per state (51 rows, multi-col)
        {
            "id": "census_nat_008",
            "dataset": "census_national",
            "question": "What is the most populous county in each state? Return a JSON object mapping each state name to an object with 'county' (county name) and 'population'.",
            "answer": (
                lambda: (
                    top := df.loc[df.groupby("STNAME")["POPESTIMATE"].idxmax()],
                    {
                        row["STNAME"]: {"county": row["CTYNAME"], "population": int(row["POPESTIMATE"])}
                        for _, row in top.iterrows()
                    },
                )[-1]
            )(),
            "answer_type": "table",
            "value_types": {"county": "string", "population": "integer"},
            "tolerance": 0,
        },
        # 009: Total pop of top-3 counties per state (51 rows)
        {
            "id": "census_nat_009",
            "dataset": "census_national",
            "question": "For each state, what is the combined population of the 3 most populous counties? Return a JSON object mapping each state name to the total population of its top 3 counties.",
            "answer": {
                st: int(grp.nlargest(3, "POPESTIMATE")["POPESTIMATE"].sum())
                for st, grp in df.groupby("STNAME")
            },
            "answer_type": "table",
            "value_types": "integer",
            "tolerance": 0,
        },
        # 010: Pop + pct female for states with pop > 10M (multi-col)
        {
            "id": "census_nat_010",
            "dataset": "census_national",
            "question": "For states with a total population greater than 10 million, return a JSON object mapping each state name to an object with 'population' (total pop) and 'pct_female' (percentage female, rounded to 2 decimal places).",
            "answer": (
                lambda: (
                    pops := df.groupby("STNAME")["POPESTIMATE"].sum(),
                    fem := df.groupby("STNAME")["POPEST_FEM"].sum(),
                    big := pops[pops > 10_000_000].index,
                    {
                        st: {
                            "population": int(pops[st]),
                            "pct_female": round(float(fem[st] / pops[st] * 100), 2),
                        }
                        for st in sorted(big)
                    },
                )[-1]
            )(),
            "answer_type": "table",
            "value_types": {"population": "integer", "pct_female": "float"},
            "tolerances": {"pct_female": 0.01},
            "tolerance": 0,
        },
    ]


def imdb_top_questions(df):
    # Precompute decade column
    df = df.copy()
    df["decade"] = (df["year"] // 10) * 10

    # Precompute primary genre (first listed genre)
    df["primary_genre"] = df["genre"].str.split(",").str[0].str.strip()

    return [
        # 001: Avg rating per decade (decades w/ 20+ movies, float)
        {
            "id": "imdb_001",
            "dataset": "imdb_top",
            "question": "What is the average rating per decade, for decades with 20 or more movies? Return a JSON object mapping each decade (as a string like '1970') to the average rating rounded to 2 decimal places.",
            "answer": (
                lambda: (
                    decade_counts := df.groupby("decade").size(),
                    valid := decade_counts[decade_counts >= 20].index,
                    {
                        str(d): round(float(df[df["decade"] == d]["avg_vote"].mean()), 2)
                        for d in sorted(valid)
                    },
                )[-1]
            )(),
            "answer_type": "table",
            "value_types": "float",
            "tolerance": 0.01,
        },
        # 002: Movie count per genre (genres w/ 10+ movies)
        {
            "id": "imdb_002",
            "dataset": "imdb_top",
            "question": "How many movies list each genre (a movie counts for a genre if the genre appears anywhere in its genre field)? Only include genres with 10 or more movies. The genres to check are: Drama, Comedy, Crime, Action, Adventure, Biography, Animation, Horror, Mystery, Thriller, Sci-Fi, Romance, Fantasy, War, Family, Western, Music, History. Return a JSON object mapping each genre name to its movie count.",
            "answer": (
                lambda: (
                    genres := ["Drama", "Comedy", "Crime", "Action", "Adventure", "Biography",
                               "Animation", "Horror", "Mystery", "Thriller", "Sci-Fi", "Romance",
                               "Fantasy", "War", "Family", "Western", "Music", "History"],
                    counts := {g: int(df["genre"].str.contains(g, na=False).sum()) for g in genres},
                    {g: c for g, c in counts.items() if c >= 10},
                )[-1]
            )(),
            "answer_type": "table",
            "value_types": "integer",
            "tolerance": 0,
        },
        # 003: Count of 8.0+ rated movies per decade
        {
            "id": "imdb_003",
            "dataset": "imdb_top",
            "question": "How many movies with a rating of 8.0 or higher are there in each decade? Only include decades that have at least one such movie. Return a JSON object mapping each decade (as a string like '1970') to the count.",
            "answer": (
                lambda: (
                    high := df[df["avg_vote"] >= 8.0],
                    counts := high.groupby("decade").size(),
                    {str(d): int(c) for d, c in counts.items()},
                )[-1]
            )(),
            "answer_type": "table",
            "value_types": "integer",
            "tolerance": 0,
        },
        # 004: Avg male vs female rating difference per genre (top 10 genres, float)
        {
            "id": "imdb_004",
            "dataset": "imdb_top",
            "question": "For the 10 most common primary genres (the first genre listed for each movie), what is the average difference between male and female ratings (male minus female)? Round to 2 decimal places. Return a JSON object mapping each genre to the average difference.",
            "answer": (
                lambda: (
                    top10 := df["primary_genre"].value_counts().head(10).index,
                    {
                        g: round(float(
                            (df[df["primary_genre"] == g]["males_allages_avg_vote"]
                             - df[df["primary_genre"] == g]["females_allages_avg_vote"]).mean()
                        ), 2)
                        for g in sorted(top10)
                    },
                )[-1]
            )(),
            "answer_type": "table",
            "value_types": "float",
            "tolerance": 0.01,
        },
        # 005: Avg duration per decade (decades w/ 20+ movies, float)
        {
            "id": "imdb_005",
            "dataset": "imdb_top",
            "question": "What is the average movie duration (in minutes) per decade, for decades with 20 or more movies? Round to 2 decimal places. Return a JSON object mapping each decade (as a string like '1970') to the average duration.",
            "answer": (
                lambda: (
                    decade_counts := df.groupby("decade").size(),
                    valid := decade_counts[decade_counts >= 20].index,
                    {
                        str(d): round(float(df[df["decade"] == d]["duration"].mean()), 2)
                        for d in sorted(valid)
                    },
                )[-1]
            )(),
            "answer_type": "table",
            "value_types": "float",
            "tolerance": 0.01,
        },
        # 006: Directors with 5+ movies -> count + avg rating (multi-col, ~38 rows)
        {
            "id": "imdb_006",
            "dataset": "imdb_top",
            "question": "For directors with 5 or more movies in the dataset, return a JSON object mapping each director's name to an object with 'movie_count' and 'avg_rating' (rounded to 2 decimal places).",
            "answer": (
                lambda: (
                    counts := df["director"].value_counts(),
                    big := counts[counts >= 5].index,
                    {
                        d: {
                            "movie_count": int(counts[d]),
                            "avg_rating": round(float(df[df["director"] == d]["avg_vote"].mean()), 2),
                        }
                        for d in sorted(big)
                    },
                )[-1]
            )(),
            "answer_type": "table",
            "value_types": {"movie_count": "integer", "avg_rating": "float"},
            "tolerances": {"avg_rating": 0.01},
            "tolerance": 0,
        },
        # 007: Total votes per director (directors w/ 5+ movies, ~38 rows)
        {
            "id": "imdb_007",
            "dataset": "imdb_top",
            "question": "For directors with 5 or more movies, what is the total number of votes across all their movies? Return a JSON object mapping each director's name to their total votes.",
            "answer": (
                lambda: (
                    counts := df["director"].value_counts(),
                    big := counts[counts >= 5].index,
                    {
                        d: int(df[df["director"] == d]["votes"].sum())
                        for d in sorted(big)
                    },
                )[-1]
            )(),
            "answer_type": "table",
            "value_types": "integer",
            "tolerance": 0,
        },
        # 008: Movie count + avg rating + avg duration per country (top 15 countries, multi-col)
        {
            "id": "imdb_008",
            "dataset": "imdb_top",
            "question": "For the 15 most common primary countries (the first country listed for each movie), return a JSON object mapping each country to an object with 'movie_count', 'avg_rating' (rounded to 2dp), and 'avg_duration' (rounded to 2dp).",
            "answer": (
                lambda: (
                    df.__setitem__("primary_country", df["country"].str.split(",").str[0].str.strip()),
                    top15 := df["primary_country"].value_counts().head(15).index,
                    {
                        c: {
                            "movie_count": int(df[df["primary_country"] == c].shape[0]),
                            "avg_rating": round(float(df[df["primary_country"] == c]["avg_vote"].mean()), 2),
                            "avg_duration": round(float(df[df["primary_country"] == c]["duration"].mean()), 2),
                        }
                        for c in sorted(top15)
                    },
                )[-1]
            )(),
            "answer_type": "table",
            "value_types": {"movie_count": "integer", "avg_rating": "float", "avg_duration": "float"},
            "tolerances": {"avg_rating": 0.01, "avg_duration": 0.01},
            "tolerance": 0,
        },
        # 009: Directors with 3+ movies -> count + total votes + avg rating (~107 rows, multi-col)
        {
            "id": "imdb_009",
            "dataset": "imdb_top",
            "question": "For directors with 3 or more movies in the dataset, return a JSON object mapping each director's name to an object with 'movie_count', 'total_votes', and 'avg_rating' (rounded to 2 decimal places).",
            "answer": (
                lambda: (
                    counts := df["director"].value_counts(),
                    big := counts[counts >= 3].index,
                    {
                        d: {
                            "movie_count": int(counts[d]),
                            "total_votes": int(df[df["director"] == d]["votes"].sum()),
                            "avg_rating": round(float(df[df["director"] == d]["avg_vote"].mean()), 2),
                        }
                        for d in sorted(big)
                    },
                )[-1]
            )(),
            "answer_type": "table",
            "value_types": {"movie_count": "integer", "total_votes": "integer", "avg_rating": "float"},
            "tolerances": {"avg_rating": 0.01},
            "tolerance": 0,
        },
        # 010: Every movie rated 8.5+ -> title: rating + votes (multi-col)
        {
            "id": "imdb_010",
            "dataset": "imdb_top",
            "question": "For every movie with a rating of 8.5 or higher, return a JSON object mapping the movie title to an object with 'rating' and 'votes'.",
            "answer": {
                row["title"]: {"rating": float(row["avg_vote"]), "votes": int(row["votes"])}
                for _, row in df[df["avg_vote"] >= 8.5].iterrows()
            },
            "answer_type": "table",
            "value_types": {"rating": "float", "votes": "integer"},
            "tolerances": {"rating": 0.1},
            "tolerance": 0,
        },
    ]


def pokemon_questions(df):
    return [
        # 001: Total base_total per generation (7 rows)
        {
            "id": "pokemon_001",
            "dataset": "pokemon",
            "question": "What is the total base_total per generation? Return a JSON object mapping each generation (as a string like '1') to the sum of base_total for all Pokemon in that generation.",
            "answer": {str(k): int(v) for k, v in df.groupby("generation")["base_total"].sum().to_dict().items()},
            "answer_type": "table",
            "value_types": "integer",
            "tolerance": 0,
        },
        # 002: Avg attack per type1 (18 rows, float)
        {
            "id": "pokemon_002",
            "dataset": "pokemon",
            "question": "What is the average attack stat per primary type (type1)? Round to 2 decimal places. Return a JSON object mapping each type1 to the average attack.",
            "answer": {
                k: round(v, 2)
                for k, v in df.groupby("type1")["attack"].mean().to_dict().items()
            },
            "answer_type": "table",
            "value_types": "float",
            "tolerance": 0.01,
        },
        # 003: Legendary vs non-legendary per generation (7 rows, multi-col)
        {
            "id": "pokemon_003",
            "dataset": "pokemon",
            "question": "For each generation, how many legendary and non-legendary Pokemon are there, and what is the average base_total for each group? Round averages to 2 decimal places. Return a JSON object mapping each generation (as a string) to an object with 'legendary_count', 'legendary_avg_base_total', 'non_legendary_count', 'non_legendary_avg_base_total'.",
            "answer": (
                lambda: (
                    gens := sorted(df["generation"].unique()),
                    {
                        str(g): {
                            "legendary_count": int((df[(df["generation"] == g) & (df["is_legendary"] == 1)].shape[0])),
                            "legendary_avg_base_total": round(float(df[(df["generation"] == g) & (df["is_legendary"] == 1)]["base_total"].mean()), 2) if df[(df["generation"] == g) & (df["is_legendary"] == 1)].shape[0] > 0 else 0,
                            "non_legendary_count": int(df[(df["generation"] == g) & (df["is_legendary"] == 0)].shape[0]),
                            "non_legendary_avg_base_total": round(float(df[(df["generation"] == g) & (df["is_legendary"] == 0)]["base_total"].mean()), 2),
                        }
                        for g in gens
                    },
                )[-1]
            )(),
            "answer_type": "table",
            "value_types": {"legendary_count": "integer", "legendary_avg_base_total": "float", "non_legendary_count": "integer", "non_legendary_avg_base_total": "float"},
            "tolerances": {"legendary_avg_base_total": 0.01, "non_legendary_avg_base_total": 0.01},
            "tolerance": 0,
        },
        # 004: Heaviest Pokemon per type1 (18 rows, multi-col)
        {
            "id": "pokemon_004",
            "dataset": "pokemon",
            "question": "What is the heaviest Pokemon for each primary type (type1)? Return a JSON object mapping each type1 to an object with 'name' and 'weight_kg'.",
            "answer": (
                lambda: (
                    top := df.loc[df.groupby("type1")["weight_kg"].idxmax()],
                    {
                        row["type1"]: {"name": row["name"], "weight_kg": float(row["weight_kg"])}
                        for _, row in top.iterrows()
                    },
                )[-1]
            )(),
            "answer_type": "table",
            "value_types": {"name": "string", "weight_kg": "float"},
            "tolerances": {"weight_kg": 0.1},
            "tolerance": 0,
        },
        # 005: Non-legendary with base_total >= 600 (multi-col)
        {
            "id": "pokemon_005",
            "dataset": "pokemon",
            "question": "Which non-legendary Pokemon have a base_total of 600 or higher? Return a JSON object mapping each Pokemon's name to an object with 'base_total' and 'type1'.",
            "answer": {
                row["name"]: {"base_total": int(row["base_total"]), "type1": row["type1"]}
                for _, row in df[(df["is_legendary"] == 0) & (df["base_total"] >= 600)].iterrows()
            },
            "answer_type": "table",
            "value_types": {"base_total": "integer", "type1": "string"},
            "tolerance": 0,
        },
        # 006: Avg of each base stat per generation (7 rows, 6 cols)
        {
            "id": "pokemon_006",
            "dataset": "pokemon",
            "question": "What is the average of each base stat (hp, attack, defense, sp_attack, sp_defense, speed) per generation? Round to 2 decimal places. Return a JSON object mapping each generation (as a string) to an object with keys 'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed'.",
            "answer": (
                lambda: (
                    stats := ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"],
                    {
                        str(g): {
                            s: round(float(grp[s].mean()), 2)
                            for s in stats
                        }
                        for g, grp in df.groupby("generation")
                    },
                )[-1]
            )(),
            "answer_type": "table",
            "value_types": {s: "float" for s in ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]},
            "tolerances": {s: 0.01 for s in ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]},
            "tolerance": 0,
        },
        # 007: Sum of weight_kg per type1 (18 rows, float)
        {
            "id": "pokemon_007",
            "dataset": "pokemon",
            "question": "What is the total weight (sum of weight_kg) per primary type (type1)? Round to 2 decimal places. Return a JSON object mapping each type1 to the total weight.",
            "answer": {
                k: round(v, 2)
                for k, v in df.groupby("type1")["weight_kg"].sum().to_dict().items()
            },
            "answer_type": "table",
            "value_types": "float",
            "tolerance": 0.01,
        },
        # 008: Per generation: count + avg height + avg weight + avg capture_rate (7 rows, multi-col)
        {
            "id": "pokemon_008",
            "dataset": "pokemon",
            "question": "For each generation, what is the count of Pokemon, average height_m, average weight_kg, and average capture_rate? Round averages to 2 decimal places. Return a JSON object mapping each generation (as a string) to an object with 'count', 'avg_height', 'avg_weight', 'avg_capture_rate'.",
            "answer": {
                str(g): {
                    "count": int(grp.shape[0]),
                    "avg_height": round(float(grp["height_m"].mean()), 2),
                    "avg_weight": round(float(grp["weight_kg"].mean()), 2),
                    "avg_capture_rate": round(float(grp["capture_rate"].mean()), 2),
                }
                for g, grp in df.groupby("generation")
            },
            "answer_type": "table",
            "value_types": {"count": "integer", "avg_height": "float", "avg_weight": "float", "avg_capture_rate": "float"},
            "tolerances": {"avg_height": 0.01, "avg_weight": 0.01, "avg_capture_rate": 0.01},
            "tolerance": 0,
        },
        # 009: For each type1 with speed>100 Pokemon: count + avg base_total (multi-col)
        {
            "id": "pokemon_009",
            "dataset": "pokemon",
            "question": "For each primary type (type1), how many Pokemon have a speed stat greater than 100, and what is their average base_total? Only include types that have at least one such Pokemon. Round average to 2 decimal places. Return a JSON object mapping each type1 to an object with 'fast_count' and 'avg_base_total'.",
            "answer": (
                lambda: (
                    fast := df[df["speed"] > 100],
                    {
                        t: {
                            "fast_count": int(grp.shape[0]),
                            "avg_base_total": round(float(grp["base_total"].mean()), 2),
                        }
                        for t, grp in fast.groupby("type1")
                    },
                )[-1]
            )(),
            "answer_type": "table",
            "value_types": {"fast_count": "integer", "avg_base_total": "float"},
            "tolerances": {"avg_base_total": 0.01},
            "tolerance": 0,
        },
        # 010: All Pokemon with hp >= 100 -> name: hp + defense + sp_defense (multi-col)
        {
            "id": "pokemon_010",
            "dataset": "pokemon",
            "question": "For every Pokemon with an hp stat of 100 or higher, return a JSON object mapping the Pokemon's name to an object with 'hp', 'defense', and 'sp_defense'.",
            "answer": {
                row["name"]: {"hp": int(row["hp"]), "defense": int(row["defense"]), "sp_defense": int(row["sp_defense"])}
                for _, row in df[df["hp"] >= 100].iterrows()
            },
            "answer_type": "table",
            "value_types": {"hp": "integer", "defense": "integer", "sp_defense": "integer"},
            "tolerance": 0,
        },
    ]


if __name__ == "__main__":
    Q_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "census_southeast": census_southeast_questions,
        "census_national": census_national_questions,
        "imdb_top": imdb_top_questions,
        "pokemon": pokemon_questions,
    }

    total = 0
    for name, gen_fn in datasets.items():
        df = load(name)
        questions = gen_fn(df)
        out = Q_DIR / f"{name}.json"
        out.write_text(json.dumps(questions, indent=2))
        total += len(questions)
        print(f"{name}: {len(questions)} questions")
        for q in questions:
            ans = q["answer"]
            if isinstance(ans, dict):
                print(f"  {q['id']}: {len(ans)} keys")
            else:
                print(f"  {q['id']}: answer = {ans}")

    print(f"\nTotal: {total} questions")
