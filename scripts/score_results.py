"""Score benchmark results: parse JSONL, compare answers, produce summary."""

import json
import re
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def parse_numeric(s):
    """Try to parse a string as a number."""
    s = s.strip().replace(",", "")
    # Remove trailing periods
    s = s.rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def score_one(result):
    """Score a single result. Returns dict with exact, soft, details."""
    expected = result["expected_answer"]
    extracted = result["extracted_answer"]
    answer_type = result["answer_type"]
    tolerance = result["tolerance"]

    if answer_type == "string":
        # Case-insensitive exact match
        exact = extracted.strip().lower() == str(expected).strip().lower()
        return {"exact": exact, "soft": exact, "detail": f"expected='{expected}', got='{extracted}'"}

    # Numeric types
    got = parse_numeric(extracted)
    if got is None:
        # Try to find a number in the response
        nums = re.findall(r"-?[\d,]+\.?\d*", extracted.replace(",", ""))
        if nums:
            got = parse_numeric(nums[-1])

    if got is None:
        return {"exact": False, "soft": False, "detail": f"could not parse '{extracted}'"}

    exp = float(expected)

    # Exact match (within tolerance)
    if tolerance > 0:
        exact = abs(got - exp) <= tolerance
    else:
        exact = got == exp

    # Soft match: within 5% for large numbers, within 0.5 for small
    if exp != 0:
        pct_diff = abs(got - exp) / abs(exp)
        soft = pct_diff <= 0.05
    else:
        soft = abs(got) <= 0.5

    return {
        "exact": exact,
        "soft": soft,
        "detail": f"expected={exp}, got={got}, diff={abs(got-exp):.4f}",
    }


def main():
    results_file = RESULTS_DIR / "results.jsonl"
    if not results_file.exists():
        print("No results file found. Run the benchmark first.")
        return

    results = []
    for line in results_file.read_text().strip().split("\n"):
        if line:
            results.append(json.loads(line))

    if not results:
        print("No results to score.")
        return

    # Score each result
    scored = []
    for r in results:
        s = score_one(r)
        scored.append({**r, **s})

    # Aggregate by model
    by_model = defaultdict(list)
    by_model_dataset = defaultdict(list)
    by_model_category = defaultdict(list)
    by_model_difficulty = defaultdict(list)

    for s in scored:
        by_model[s["model_name"]].append(s)
        by_model_dataset[(s["model_name"], s["dataset"])].append(s)
        # Extract category from question_id pattern or use dataset
        q_id = s["question_id"]
        by_model_difficulty[(s["model_name"], "all")].append(s)

    # Print summary table
    print("\n" + "=" * 70)
    print("CALCUBENCH RESULTS SUMMARY")
    print("=" * 70)

    # Overall by model
    print(f"\n{'Model':<15} {'Exact':>8} {'Soft':>8} {'Total':>8} {'Exact%':>8} {'Soft%':>8}")
    print("-" * 55)
    for model in sorted(by_model.keys()):
        items = by_model[model]
        exact = sum(1 for s in items if s["exact"])
        soft = sum(1 for s in items if s["soft"])
        total = len(items)
        print(
            f"{model:<15} {exact:>8} {soft:>8} {total:>8} "
            f"{exact/total*100:>7.1f}% {soft/total*100:>7.1f}%"
        )

    # By model + dataset
    print(f"\n{'Model':<15} {'Dataset':<25} {'Exact':>6} {'Soft':>6} {'Total':>6}")
    print("-" * 65)
    for (model, ds) in sorted(by_model_dataset.keys()):
        items = by_model_dataset[(model, ds)]
        exact = sum(1 for s in items if s["exact"])
        soft = sum(1 for s in items if s["soft"])
        total = len(items)
        print(f"{model:<15} {ds:<25} {exact:>6} {soft:>6} {total:>6}")

    # Detailed results
    print(f"\n{'='*70}")
    print("DETAILED RESULTS")
    print(f"{'='*70}")
    for s in scored:
        status = "EXACT" if s["exact"] else ("SOFT" if s["soft"] else "MISS")
        print(f"[{status:>5}] {s['model_name']:<12} {s['question_id']:<20} {s['detail']}")

    # Save summary JSON
    summary = {
        "total_questions": len(scored),
        "models": {},
    }
    for model in sorted(by_model.keys()):
        items = by_model[model]
        exact = sum(1 for s in items if s["exact"])
        soft = sum(1 for s in items if s["soft"])
        total = len(items)
        summary["models"][model] = {
            "exact": exact,
            "soft": soft,
            "total": total,
            "exact_pct": round(exact / total * 100, 1),
            "soft_pct": round(soft / total * 100, 1),
        }

    summary_file = RESULTS_DIR / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
