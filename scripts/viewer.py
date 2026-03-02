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


def score_one(result):
    expected = result["expected_answer"]
    extracted = result["extracted_answer"]
    answer_type = result["answer_type"]
    tolerance = result["tolerance"]

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
