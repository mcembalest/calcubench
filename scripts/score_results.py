"""Score benchmark results: parse JSONL, compare answers, produce summary."""

import json
import re
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def parse_numeric(s):
    """Try to parse a string as a number."""
    s = s.strip().replace(",", "")
    s = s.rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def _compare_values(expected, extracted, tolerance, value_type):
    """Compare a single expected vs extracted value. Returns (exact, soft) booleans."""
    if value_type == "integer":
        tolerance = tolerance or 0
    elif value_type == "float":
        tolerance = tolerance if tolerance else 0.01

    got = parse_numeric(str(extracted))
    if got is None:
        return False, False

    exp = float(expected)

    if tolerance > 0:
        exact = abs(got - exp) <= tolerance
    else:
        exact = got == exp

    if exp != 0:
        soft = abs(got - exp) / abs(exp) <= 0.05
    else:
        soft = abs(got) <= 0.5

    return exact, soft


def score_table(result):
    """Score a table result. Returns dict with exact, soft, details."""
    expected = result["expected_answer"]
    extracted_raw = result["extracted_answer"]
    value_types = result.get("value_types", "float")
    tolerances = result.get("tolerances", {})
    base_tolerance = result.get("tolerance", 0)

    # Parse JSON from extracted answer, stripping markdown code fences if present
    text = extracted_raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        got = json.loads(text)
    except json.JSONDecodeError:
        return {"exact": False, "soft": False, "detail": f"could not parse JSON from: {extracted_raw[:100]}"}

    if not isinstance(got, dict) or not isinstance(expected, dict):
        return {"exact": False, "soft": False, "detail": "expected dict, got non-dict"}

    # Check each expected key
    total_cells = 0
    correct_exact = 0
    correct_soft = 0
    mismatches = []

    for key in expected:
        if key not in got:
            exp_val = expected[key]
            if isinstance(exp_val, dict):
                total_cells += len(exp_val)
            else:
                total_cells += 1
            mismatches.append(f"missing key '{key}'")
            continue

        exp_val = expected[key]
        got_val = got[key]

        if isinstance(exp_val, dict):
            # Multi-column row
            for col, exp_cell in exp_val.items():
                total_cells += 1
                col_type = value_types.get(col, "float") if isinstance(value_types, dict) else value_types
                col_tol = tolerances.get(col, base_tolerance)
                got_cell = got_val.get(col) if isinstance(got_val, dict) else None
                if got_cell is None:
                    mismatches.append(f"'{key}'.'{col}' missing")
                    continue
                ex, sf = _compare_values(exp_cell, got_cell, col_tol, col_type)
                if ex:
                    correct_exact += 1
                if sf:
                    correct_soft += 1
                if not ex:
                    mismatches.append(f"'{key}'.'{col}': expected={exp_cell}, got={got_cell}")
        else:
            # Single-value row
            total_cells += 1
            vtype = value_types if isinstance(value_types, str) else "float"
            ex, sf = _compare_values(exp_val, got_val, base_tolerance, vtype)
            if ex:
                correct_exact += 1
            if sf:
                correct_soft += 1
            if not ex:
                mismatches.append(f"'{key}': expected={exp_val}, got={got_val}")

    has_extra = set(got.keys()) - set(expected.keys())
    all_exact = correct_exact == total_cells and not has_extra
    soft_pass = total_cells > 0 and correct_soft / total_cells >= 0.8

    detail_parts = [f"{correct_exact}/{total_cells} cells exact"]
    if has_extra:
        detail_parts.append(f"{len(has_extra)} extra keys")
    if mismatches:
        detail_parts.append(f"mismatches: {'; '.join(mismatches[:3])}")

    return {"exact": all_exact, "soft": soft_pass, "detail": ", ".join(detail_parts)}


def score_one(result):
    """Score a single result. Returns dict with exact, soft, details."""
    expected = result["expected_answer"]
    extracted = result["extracted_answer"]
    answer_type = result["answer_type"]
    tolerance = result["tolerance"]

    if answer_type == "table":
        return score_table(result)

    if answer_type == "string":
        exact = extracted.strip().lower() == str(expected).strip().lower()
        return {"exact": exact, "soft": exact, "detail": f"expected='{expected}', got='{extracted}'"}

    got = parse_numeric(extracted)
    if got is None:
        nums = re.findall(r"-?[\d,]+\.?\d*", extracted.replace(",", ""))
        if nums:
            got = parse_numeric(nums[-1])

    if got is None:
        return {"exact": False, "soft": False, "detail": f"could not parse '{extracted}'"}

    exp = float(expected)

    if tolerance > 0:
        exact = abs(got - exp) <= tolerance
    else:
        exact = got == exp

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


def print_table(header, rows, col_widths=None):
    """Print a formatted table."""
    if not col_widths:
        col_widths = [max(len(str(row[i])) for row in [header] + rows) + 2 for i in range(len(header))]
    fmt = "".join(f"{{:<{w}}}" if i < 2 else f"{{:>{w}}}" for i, w in enumerate(col_widths))
    print(fmt.format(*header))
    print("-" * sum(col_widths))
    for row in rows:
        print(fmt.format(*row))


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

    scored = []
    for r in results:
        s = score_one(r)
        scored.append({**r, **s})

    # Aggregate by model_name (which includes effort level, e.g. "claude-opus-4.6@high")
    by_model = defaultdict(list)
    by_model_dataset = defaultdict(list)

    for s in scored:
        by_model[s["model_name"]].append(s)
        by_model_dataset[(s["model_name"], s["dataset"])].append(s)

    # Overall by model
    print("\n" + "=" * 70)
    print("CALCUBENCH RESULTS SUMMARY")
    print("=" * 70)

    rows = []
    for model in sorted(by_model.keys()):
        items = by_model[model]
        exact = sum(1 for s in items if s["exact"])
        soft = sum(1 for s in items if s["soft"])
        total = len(items)
        avg_time = sum(s["elapsed_seconds"] for s in items) / total
        has_reasoning = sum(1 for s in items if s.get("reasoning_trace"))
        rows.append((
            model, str(total),
            f"{exact}/{total}", f"{exact/total*100:.0f}%",
            f"{soft}/{total}", f"{soft/total*100:.0f}%",
            f"{avg_time:.1f}s",
            f"{has_reasoning}/{total}",
        ))

    print()
    print_table(
        ("Model", "N", "Exact", "Exact%", "Soft", "Soft%", "Avg Time", "Reasoning"),
        rows,
    )

    # By model + dataset
    print(f"\n{'='*70}")
    print("BY DATASET")
    print(f"{'='*70}\n")

    rows = []
    for (model, ds) in sorted(by_model_dataset.keys()):
        items = by_model_dataset[(model, ds)]
        exact = sum(1 for s in items if s["exact"])
        soft = sum(1 for s in items if s["soft"])
        total = len(items)
        rows.append((model, ds, f"{exact}/{total}", f"{soft}/{total}"))

    print_table(("Model", "Dataset", "Exact", "Soft"), rows)

    # Effort ablation view: group by base model, compare efforts
    print(f"\n{'='*70}")
    print("EFFORT ABLATION")
    print(f"{'='*70}\n")

    base_models = defaultdict(dict)
    for model_name, items in by_model.items():
        if "@" in model_name:
            base, effort = model_name.rsplit("@", 1)
            exact = sum(1 for s in items if s["exact"])
            total = len(items)
            avg_time = sum(s["elapsed_seconds"] for s in items) / total
            base_models[base][effort] = (exact, total, avg_time)

    for base in sorted(base_models.keys()):
        print(f"  {base}:")
        for effort in ["low", "medium", "high"]:
            if effort in base_models[base]:
                exact, total, avg_time = base_models[base][effort]
                bar = "#" * exact + "." * (total - exact)
                print(f"    {effort:<8} {exact:>2}/{total}  [{bar}]  avg {avg_time:.1f}s")
        print()

    # Detailed results
    print(f"{'='*70}")
    print("DETAILED RESULTS")
    print(f"{'='*70}")
    for s in sorted(scored, key=lambda x: (x["question_id"], x["model_name"])):
        status = "EXACT" if s["exact"] else ("SOFT" if s["soft"] else "MISS")
        reasoning_len = len(s.get("reasoning_trace") or "")
        reasoning_info = f" [reasoning: {reasoning_len} chars]" if reasoning_len else ""
        print(f"[{status:>5}] {s['model_name']:<30} {s['question_id']:<20} {s['detail']}{reasoning_info}")

    # Save summary JSON
    summary = {
        "total_results": len(scored),
        "models": {},
    }
    for model in sorted(by_model.keys()):
        items = by_model[model]
        exact = sum(1 for s in items if s["exact"])
        soft = sum(1 for s in items if s["soft"])
        total = len(items)
        avg_time = sum(s["elapsed_seconds"] for s in items) / total
        summary["models"][model] = {
            "exact": exact,
            "soft": soft,
            "total": total,
            "exact_pct": round(exact / total * 100, 1),
            "soft_pct": round(soft / total * 100, 1),
            "avg_seconds": round(avg_time, 1),
        }

    summary_file = RESULTS_DIR / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
