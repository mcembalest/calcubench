"""Generate benchmark questions with ground-truth answers computed via pandas."""

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "prepared"
Q_DIR = Path(__file__).parent.parent / "questions"


def load(name):
    return pd.DataFrame(json.loads((DATA_DIR / f"{name}.json").read_text()))


def census_southeast_questions(df):
    return [
        {
            "id": "census_se_001",
            "dataset": "census_southeast",
            "question": "What is the total population of all counties in Florida?",
            "difficulty": "medium",
            "category": "sum_by_group",
            "answer": int(df[df["STNAME"] == "Florida"]["POPESTIMATE"].sum()),
            "answer_type": "integer",
            "tolerance": 0,
            "pandas_code": "df[df['STNAME']=='Florida']['POPESTIMATE'].sum()",
        },
        {
            "id": "census_se_002",
            "dataset": "census_southeast",
            "question": "What is the total 65+ population across all Georgia counties?",
            "difficulty": "medium",
            "category": "sum_by_group",
            "answer": int(df[df["STNAME"] == "Georgia"]["AGE65PLUS_TOT"].sum()),
            "answer_type": "integer",
            "tolerance": 0,
            "pandas_code": "df[df['STNAME']=='Georgia']['AGE65PLUS_TOT'].sum()",
        },
        {
            "id": "census_se_003",
            "dataset": "census_southeast",
            "question": "What is the average median age across Tennessee counties? Round to 2 decimal places.",
            "difficulty": "medium",
            "category": "average",
            "answer": round(float(df[df["STNAME"] == "Tennessee"]["MEDIAN_AGE_TOT"].mean()), 2),
            "answer_type": "float",
            "tolerance": 0.01,
            "pandas_code": "round(df[df['STNAME']=='Tennessee']['MEDIAN_AGE_TOT'].mean(), 2)",
        },
        {
            "id": "census_se_004",
            "dataset": "census_southeast",
            "question": "How many counties in this dataset have a total population over 500,000?",
            "difficulty": "easy",
            "category": "count",
            "answer": int((df["POPESTIMATE"] > 500_000).sum()),
            "answer_type": "integer",
            "tolerance": 0,
            "pandas_code": "(df['POPESTIMATE'] > 500000).sum()",
        },
        {
            "id": "census_se_005",
            "dataset": "census_southeast",
            "question": "What percentage of Virginia's total population is under 5 years old? Round to 2 decimal places.",
            "difficulty": "hard",
            "category": "multi_step",
            "answer": round(
                float(
                    df[df["STNAME"] == "Virginia"]["UNDER5_TOT"].sum()
                    / df[df["STNAME"] == "Virginia"]["POPESTIMATE"].sum()
                    * 100
                ),
                2,
            ),
            "answer_type": "float",
            "tolerance": 0.01,
            "pandas_code": "round(df[df['STNAME']=='Virginia']['UNDER5_TOT'].sum() / df[df['STNAME']=='Virginia']['POPESTIMATE'].sum() * 100, 2)",
        },
        {
            "id": "census_se_006",
            "dataset": "census_southeast",
            "question": "Which 3 states have the highest total population? List them from highest to lowest, separated by commas.",
            "difficulty": "hard",
            "category": "sort_aggregate",
            "answer": ", ".join(
                df.groupby("STNAME")["POPESTIMATE"]
                .sum()
                .sort_values(ascending=False)
                .head(3)
                .index.tolist()
            ),
            "answer_type": "string",
            "tolerance": 0,
            "pandas_code": "', '.join(df.groupby('STNAME')['POPESTIMATE'].sum().sort_values(ascending=False).head(3).index)",
        },
    ]


def census_national_questions(df):
    return [
        {
            "id": "census_nat_001",
            "dataset": "census_national",
            "question": "What is the total population of California?",
            "difficulty": "medium",
            "category": "sum_by_group",
            "answer": int(df[df["STNAME"] == "California"]["POPESTIMATE"].sum()),
            "answer_type": "integer",
            "tolerance": 0,
            "pandas_code": "df[df['STNAME']=='California']['POPESTIMATE'].sum()",
        },
        {
            "id": "census_nat_002",
            "dataset": "census_national",
            "question": "What is the total male population of Texas?",
            "difficulty": "medium",
            "category": "sum_by_group",
            "answer": int(df[df["STNAME"] == "Texas"]["POPEST_MALE"].sum()),
            "answer_type": "integer",
            "tolerance": 0,
            "pandas_code": "df[df['STNAME']=='Texas']['POPEST_MALE'].sum()",
        },
        {
            "id": "census_nat_003",
            "dataset": "census_national",
            "question": "How many US counties have a total population over 1 million?",
            "difficulty": "easy",
            "category": "count",
            "answer": int((df["POPESTIMATE"] > 1_000_000).sum()),
            "answer_type": "integer",
            "tolerance": 0,
            "pandas_code": "(df['POPESTIMATE'] > 1000000).sum()",
        },
        {
            "id": "census_nat_004",
            "dataset": "census_national",
            "question": "What is the average county population in Ohio? Round to the nearest integer.",
            "difficulty": "medium",
            "category": "average",
            "answer": int(round(df[df["STNAME"] == "Ohio"]["POPESTIMATE"].mean())),
            "answer_type": "integer",
            "tolerance": 0,
            "pandas_code": "round(df[df['STNAME']=='Ohio']['POPESTIMATE'].mean())",
        },
        {
            "id": "census_nat_005",
            "dataset": "census_national",
            "question": "What is the total population of the 10 most populous counties?",
            "difficulty": "hard",
            "category": "multi_step",
            "answer": int(df.nlargest(10, "POPESTIMATE")["POPESTIMATE"].sum()),
            "answer_type": "integer",
            "tolerance": 0,
            "pandas_code": "df.nlargest(10, 'POPESTIMATE')['POPESTIMATE'].sum()",
        },
        {
            "id": "census_nat_006",
            "dataset": "census_national",
            "question": "Which state has more total population: New York or Florida?",
            "difficulty": "medium",
            "category": "comparison",
            "answer": (
                "New York"
                if df[df["STNAME"] == "New York"]["POPESTIMATE"].sum()
                > df[df["STNAME"] == "Florida"]["POPESTIMATE"].sum()
                else "Florida"
            ),
            "answer_type": "string",
            "tolerance": 0,
            "pandas_code": "'New York' if df[df['STNAME']=='New York']['POPESTIMATE'].sum() > df[df['STNAME']=='Florida']['POPESTIMATE'].sum() else 'Florida'",
        },
    ]


def imdb_top_questions(df):
    return [
        {
            "id": "imdb_001",
            "dataset": "imdb_top",
            "question": "What is the total number of votes for all Christopher Nolan movies in the dataset?",
            "difficulty": "medium",
            "category": "sum_by_group",
            "answer": int(df[df["director"] == "Christopher Nolan"]["votes"].sum()),
            "answer_type": "integer",
            "tolerance": 0,
            "pandas_code": "df[df['director']=='Christopher Nolan']['votes'].sum()",
        },
        {
            "id": "imdb_002",
            "dataset": "imdb_top",
            "question": "What is the average rating of movies containing 'Drama' in their genre? Round to 2 decimal places.",
            "difficulty": "medium",
            "category": "average",
            "answer": round(
                float(df[df["genre"].str.contains("Drama", na=False)]["avg_vote"].mean()), 2
            ),
            "answer_type": "float",
            "tolerance": 0.01,
            "pandas_code": "round(df[df['genre'].str.contains('Drama')]['avg_vote'].mean(), 2)",
        },
        {
            "id": "imdb_003",
            "dataset": "imdb_top",
            "question": "How many movies in the dataset have a rating of 8.0 or above?",
            "difficulty": "easy",
            "category": "count",
            "answer": int((df["avg_vote"] >= 8.0).sum()),
            "answer_type": "integer",
            "tolerance": 0,
            "pandas_code": "(df['avg_vote'] >= 8.0).sum()",
        },
        {
            "id": "imdb_004",
            "dataset": "imdb_top",
            "question": "For movies with 'Action' in the genre, is the average male rating higher or lower than the average female rating? Answer 'higher' or 'lower'.",
            "difficulty": "hard",
            "category": "comparison",
            "answer": (
                "higher"
                if df[df["genre"].str.contains("Action", na=False)][
                    "males_allages_avg_vote"
                ].mean()
                > df[df["genre"].str.contains("Action", na=False)][
                    "females_allages_avg_vote"
                ].mean()
                else "lower"
            ),
            "answer_type": "string",
            "tolerance": 0,
            "pandas_code": "'higher' if action['males_allages_avg_vote'].mean() > action['females_allages_avg_vote'].mean() else 'lower'",
        },
        {
            "id": "imdb_005",
            "dataset": "imdb_top",
            "question": "What is the average duration (in minutes) of the 20 highest-rated movies? Round to 2 decimal places.",
            "difficulty": "hard",
            "category": "multi_step",
            "answer": round(float(df.nlargest(20, "avg_vote")["duration"].mean()), 2),
            "answer_type": "float",
            "tolerance": 0.01,
            "pandas_code": "round(df.nlargest(20, 'avg_vote')['duration'].mean(), 2)",
        },
        {
            "id": "imdb_006",
            "dataset": "imdb_top",
            "question": "Which director has the most movies in the dataset? Give the director's name and their average rating, separated by a comma, with the rating rounded to 2 decimal places.",
            "difficulty": "hard",
            "category": "sort_aggregate",
            "answer": (
                lambda: (
                    top_dir := df["director"].value_counts().idxmax(),
                    avg := round(float(df[df["director"] == top_dir]["avg_vote"].mean()), 2),
                    f"{top_dir}, {avg}",
                )[-1]
            )(),
            "answer_type": "string",
            "tolerance": 0,
            "pandas_code": "top = df['director'].value_counts().idxmax(); f\"{top}, {round(df[df['director']==top]['avg_vote'].mean(), 2)}\"",
        },
    ]


if __name__ == "__main__":
    Q_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "census_southeast": census_southeast_questions,
        "census_national": census_national_questions,
        "imdb_top": imdb_top_questions,
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
            print(f"  {q['id']}: answer = {q['answer']}")

    print(f"\nTotal: {total} questions")
