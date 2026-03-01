"""Run benchmark: send each question + dataset to each model via litellm."""

import argparse
import json
import os
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "prepared"
Q_DIR = BASE_DIR / "questions"
RESULTS_DIR = BASE_DIR / "results"

MODELS = {
    "claude": {
        "model": "anthropic/claude-opus-4-6",
        "extra": {"thinking": {"type": "adaptive"}},
    },
    "gpt5": {
        "model": "openai/responses/gpt-5.2",
        "extra": {"reasoning_effort": "xhigh"},
    },
    "gpt5pro": {
        "model": "openai/responses/gpt-5.2-pro",
        "extra": {"reasoning_effort": "xhigh"},
    },
    "gemini": {
        "model": "gemini/gemini-3.1-pro-preview",
        "extra": {"reasoning_effort": "high"},
    },
}

SYSTEM_PROMPT = (
    "Answer the user's question about the provided data. Do not use tools -- "
    "just examine the data directly and provide your answer.\n"
    "End your response with exactly:\n"
    "ANSWER: <your answer>\n"
    "For numbers, no commas or units. For text, exact match."
)

ALL_DATASETS = ["census_southeast", "census_national", "imdb_top"]


def load_completed(results_file):
    """Load already-completed (question_id, model) pairs from JSONL."""
    completed = set()
    if results_file.exists():
        for line in results_file.read_text().strip().split("\n"):
            if not line:
                continue
            rec = json.loads(line)
            completed.add((rec["question_id"], rec["model_name"]))
    return completed


def call_model(model_name, model_cfg, question_text, json_data, max_retries=3):
    """Call a model with exponential backoff retries. Returns None on total failure."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question: {question_text}\n\nHere is the dataset:\n{json_data}",
        },
    ]

    for attempt in range(max_retries):
        try:
            resp = litellm.completion(
                model=model_cfg["model"],
                messages=messages,
                timeout=2000,
                **model_cfg["extra"],
            )
            content = resp.choices[0].message.content
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }
            return content, usage
        except Exception as e:
            wait = 2 ** (attempt + 1)
            if attempt < max_retries - 1:
                print(f"\n    Error ({type(e).__name__}). Retrying in {wait}s...", end=" ", flush=True)
                time.sleep(wait)
            else:
                print(f"\n    FAILED after {max_retries} retries ({type(e).__name__}). Skipping.")
                return None


def extract_answer(response_text):
    """Extract the answer after 'ANSWER:' marker."""
    if "ANSWER:" in response_text:
        return response_text.split("ANSWER:")[-1].strip()
    return response_text.strip().split("\n")[-1].strip()


def run(models, datasets, question_ids=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "results.jsonl"
    completed = load_completed(results_file)

    for ds_name in datasets:
        data_path = DATA_DIR / f"{ds_name}.json"
        json_data = data_path.read_text()
        questions = json.loads((Q_DIR / f"{ds_name}.json").read_text())

        if question_ids:
            questions = [q for q in questions if q["id"] in question_ids]
            if not questions:
                continue

        print(f"\n=== Dataset: {ds_name} ({len(questions)} questions) ===")

        for q in questions:
            for model_name in models:
                key = (q["id"], model_name)
                if key in completed:
                    print(f"  SKIP {q['id']} / {model_name} (already done)")
                    continue

                model_cfg = MODELS[model_name]
                print(f"  Running {q['id']} / {model_name}...", end=" ", flush=True)
                t0 = time.time()

                result_pair = call_model(
                    model_name, model_cfg, q["question"], json_data
                )
                if result_pair is None:
                    continue

                response, usage = result_pair
                extracted = extract_answer(response)
                elapsed = round(time.time() - t0, 2)
                print(f"done ({elapsed}s) -> {extracted[:80]}")

                result = {
                    "question_id": q["id"],
                    "dataset": ds_name,
                    "model_name": model_name,
                    "model_id": model_cfg["model"],
                    "question": q["question"],
                    "expected_answer": q["answer"],
                    "answer_type": q["answer_type"],
                    "tolerance": q["tolerance"],
                    "raw_response": response,
                    "extracted_answer": extracted,
                    "usage": usage,
                    "elapsed_seconds": elapsed,
                }

                with open(results_file, "a") as f:
                    f.write(json.dumps(result) + "\n")

                completed.add(key)


def main():
    parser = argparse.ArgumentParser(description="Run Calcubench benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Models to test",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASETS,
        default=ALL_DATASETS,
        help="Datasets to test",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        default=None,
        help="Only run specific question IDs (e.g. imdb_002 census_nat_001)",
    )
    args = parser.parse_args()
    run(args.models, args.datasets, question_ids=args.questions)


if __name__ == "__main__":
    main()
