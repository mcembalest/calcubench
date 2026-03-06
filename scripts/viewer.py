"""Calcubench Results Viewer — zero-dependency HTML dashboard."""

import http.server
import json
import re
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
QUESTIONS_DIR = ROOT / "questions"
PREPARED_DIR = ROOT / "data" / "prepared"

PORT = 8080
DATA_PREVIEW_ROWS = 15

# Per-token costs (from cost.md)
COSTS = {
    "claude": {"input": 5.00, "output": 25.00, "model_id": "claude-opus-4-6"},
    "gemini": {"input": 2.00, "output": 12.00, "model_id": "gemini-3.1-pro"},
    "gpt5":   {"input": 1.75, "output": 14.00, "model_id": "gpt-5.2"},
}

# Reasoning level per model (from run_benchmark.py MODELS config)
REASONING = {
    "claude": "adaptive (thinking)",
    "gpt5": "xhigh",
    "gemini": "high",
}


# --------------- scoring (copied from score_results.py) ---------------

def parse_numeric(s):
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

    text = extracted_raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        got = json.loads(text)
    except json.JSONDecodeError:
        return {"exact": False, "soft": False, "detail": f"could not parse JSON from: {extracted_raw[:100]}"}

    if not isinstance(got, dict) or not isinstance(expected, dict):
        return {"exact": False, "soft": False, "detail": "expected dict, got non-dict"}

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
        "detail": f"expected={exp}, got={got}, diff={abs(got - exp):.4f}",
    }


# --------------- data loading ---------------

def load_data():
    # Results
    results = []
    for line in (RESULTS_DIR / "results.jsonl").read_text().strip().split("\n"):
        if line:
            results.append(json.loads(line))

    # Score each result
    scored = []
    for r in results:
        s = score_one(r)
        scored.append({**r, **s})

    # Questions keyed by id
    questions = {}
    for qf in sorted(QUESTIONS_DIR.glob("*.json")):
        for q in json.loads(qf.read_text()):
            questions[q["id"]] = q

    # Summary
    summary = json.loads((RESULTS_DIR / "summary.json").read_text())

    # Data previews (first N rows + total count)
    data_previews = {}
    for pf in sorted(PREPARED_DIR.glob("*.json")):
        rows = json.loads(pf.read_text())
        data_previews[pf.stem] = {
            "columns": list(rows[0].keys()) if rows else [],
            "rows": rows[:DATA_PREVIEW_ROWS],
            "total_rows": len(rows),
        }

    # Attach per-result cost
    for r in scored:
        rates = COSTS.get(r["model_name"], {"input": 0, "output": 0})
        r["input_cost"] = round(r["usage"]["prompt_tokens"] * rates["input"] / 1_000_000, 4)
        r["output_cost"] = round(r["usage"]["completion_tokens"] * rates["output"] / 1_000_000, 4)
        r["total_cost"] = round(r["input_cost"] + r["output_cost"], 4)

    return {
        "scored": scored,
        "questions": questions,
        "summary": summary,
        "data_previews": data_previews,
        "cost_rates": {k: {"input": v["input"], "output": v["output"], "model_id": v["model_id"]} for k, v in COSTS.items()},
        "reasoning": REASONING,
    }


# --------------- HTML template (loaded from files) ---------------

VIEWER_DIR = Path(__file__).parent


def build_page(data):
    html = (VIEWER_DIR / "viewer.html").read_text()
    js = (VIEWER_DIR / "viewer.js").read_text()
    data_json = json.dumps(data, default=str)
    return html.replace("__INLINE_SCRIPT__", js).replace("__DATA_JSON__", data_json)


class Handler(http.server.BaseHTTPRequestHandler):
    html_bytes = b""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(self.html_bytes)

    def log_message(self, fmt, *args):
        pass  # quiet


def main():
    import sys

    print("Loading data...")
    data = load_data()
    print(f"  {len(data['scored'])} results, {len(data['questions'])} questions")

    print("Building HTML...")
    html = build_page(data)

    if "--build" in sys.argv:
        out = ROOT / "index.html"
        out.write_text(html, encoding="utf-8")
        print(f"Wrote {out} ({len(html):,} bytes)")
        return

    Handler.html_bytes = html.encode("utf-8")

    server = http.server.HTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"Serving at {url}")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
