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

    # Cost info
    cost_info = []
    for model_name, rates in COSTS.items():
        model_results = [r for r in scored if r["model_name"] == model_name]
        prompt_tokens = sum(r["usage"]["prompt_tokens"] for r in model_results)
        completion_tokens = sum(r["usage"]["completion_tokens"] for r in model_results)
        input_cost = prompt_tokens * rates["input"] / 1_000_000
        output_cost = completion_tokens * rates["output"] / 1_000_000
        total_time = sum(r["elapsed_seconds"] for r in model_results)
        n = len(model_results)
        cost_info.append({
            "model": model_name,
            "model_id": rates["model_id"],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "input_cost": round(input_cost, 4),
            "output_cost": round(output_cost, 4),
            "total_cost": round(input_cost + output_cost, 4),
            "total_time": round(total_time, 2),
            "avg_time": round(total_time / n, 2) if n else 0,
            "num_questions": n,
        })

    return {
        "scored": scored,
        "questions": questions,
        "summary": summary,
        "data_previews": data_previews,
        "costs": cost_info,
    }


# --------------- HTML template ---------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Calcubench Results Viewer</title>
<style>
:root {
  --green: #06d6a0;
  --yellow: #ffd166;
  --red: #ef476f;
  --blue: #4361ee;
  --bg: #f8f9fa;
  --card: #fff;
  --text: #212529;
  --muted: #6c757d;
  --border: #dee2e6;
  --shadow: 0 1px 3px rgba(0,0,0,.08);
  --radius: 8px;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: var(--bg); color: var(--text); line-height: 1.5;
}
nav {
  position: sticky; top: 0; z-index: 100;
  display: flex; gap: 0; background: #fff;
  border-bottom: 2px solid var(--blue);
  box-shadow: var(--shadow);
}
nav button {
  flex: 1; padding: 14px 16px; border: none; background: none;
  font-size: 15px; font-weight: 500; cursor: pointer; color: var(--muted);
  transition: color .15s, border-bottom .15s;
  border-bottom: 3px solid transparent;
}
nav button:hover { color: var(--text); }
nav button.active { color: var(--blue); border-bottom-color: var(--blue); }
.container { max-width: 1200px; margin: 0 auto; padding: 24px 16px; }
.tab { display: none; }
.tab.active { display: block; }
h2 { font-size: 22px; margin-bottom: 16px; }
h3 { font-size: 17px; margin-bottom: 12px; }
.card {
  background: var(--card); border-radius: var(--radius);
  box-shadow: var(--shadow); padding: 20px; margin-bottom: 16px;
}

/* Dashboard */
.model-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-bottom: 24px; }
.model-card { text-align: center; }
.model-card h3 { font-size: 20px; color: var(--blue); margin-bottom: 8px; }
.model-card .scores { display: flex; justify-content: center; gap: 24px; margin: 12px 0; }
.ring-wrap { position: relative; width: 90px; height: 90px; }
.ring-wrap svg { transform: rotate(-90deg); }
.ring-wrap circle { fill: none; stroke-width: 7; }
.ring-bg { stroke: var(--border); }
.ring-fg { stroke-linecap: round; transition: stroke-dashoffset .6s ease; }
.ring-label { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; font-size: 18px; font-weight: 700; }
.ring-label small { font-size: 11px; font-weight: 400; color: var(--muted); }

.bar-chart { margin: 24px 0; }
.bar-row { display: flex; align-items: center; margin-bottom: 10px; }
.bar-row .label { width: 90px; font-weight: 500; font-size: 14px; }
.bar-row .track { flex: 1; height: 28px; background: var(--border); border-radius: 4px; position: relative; overflow: hidden; }
.bar-row .fill { height: 100%; border-radius: 4px; display: flex; align-items: center; padding-left: 8px; font-size: 13px; font-weight: 600; color: #fff; transition: width .5s ease; }
.bar-row .fill.exact { background: var(--green); }
.bar-row .fill.soft { background: var(--yellow); color: var(--text); }

.hardest-list { list-style: none; }
.hardest-list li { padding: 8px 0; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; }
.hardest-list li:last-child { border: none; }

/* Dataset tab */
.ds-section { margin-bottom: 24px; }
.ds-toggle { cursor: pointer; user-select: none; }
.ds-toggle h3 { display: inline; }
table { width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 8px; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }
th { background: var(--bg); font-weight: 600; position: sticky; top: 0; }
.data-preview-wrap { max-height: 400px; overflow: auto; margin-top: 8px; border: 1px solid var(--border); border-radius: var(--radius); }
.data-preview-wrap table { margin: 0; }
.data-preview-wrap th { top: 0; z-index: 1; background: #e9ecef; }
.row-count { font-size: 13px; color: var(--muted); margin-top: 4px; }

/* Question explorer */
.filter-bar { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
.filter-bar select { padding: 8px 12px; border: 1px solid var(--border); border-radius: 6px; font-size: 14px; }
.q-card { border-left: 4px solid var(--blue); }
.q-header { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 8px; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
.badge-easy { background: #d4edda; color: #155724; }
.badge-medium { background: #fff3cd; color: #856404; }
.badge-hard { background: #f8d7da; color: #721c24; }
.badge-cat { background: #d0d7ff; color: #1a237e; }
.expected { font-size: 14px; color: var(--muted); margin-bottom: 4px; }
code, .mono { font-family: "SF Mono", "Fira Code", "Cascadia Code", Consolas, monospace; font-size: 13px; }
.code-block { background: #f1f3f5; padding: 8px 12px; border-radius: 6px; overflow-x: auto; margin: 8px 0; }
.model-result { padding: 10px; border-radius: 6px; background: var(--bg); margin-top: 8px; }
.status-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
.dot-exact { background: var(--green); }
.dot-soft { background: var(--yellow); }
.dot-wrong { background: var(--red); }
.model-result .meta { font-size: 13px; color: var(--muted); }
details summary { cursor: pointer; font-size: 13px; color: var(--blue); margin-top: 4px; }
details pre { max-height: 400px; overflow: auto; background: #f1f3f5; padding: 12px; border-radius: 6px; white-space: pre-wrap; word-break: break-word; font-size: 12px; margin-top: 4px; }

/* Cost tab */
.cost-table th { white-space: nowrap; }
.cost-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
</style>
</head>
<body>

<nav>
  <button class="active" onclick="showTab('dashboard')">Dashboard</button>
  <button onclick="showTab('dataset')">By Dataset</button>
  <button onclick="showTab('explorer')">Question Explorer</button>
  <button onclick="showTab('cost')">Cost &amp; Performance</button>
</nav>

<div class="container">

<!-- ============ TAB 1: DASHBOARD ============ -->
<div id="tab-dashboard" class="tab active"></div>

<!-- ============ TAB 2: BY DATASET ============ -->
<div id="tab-dataset" class="tab"></div>

<!-- ============ TAB 3: QUESTION EXPLORER ============ -->
<div id="tab-explorer" class="tab"></div>

<!-- ============ TAB 4: COST ============ -->
<div id="tab-cost" class="tab"></div>

</div>

<script>
const DATA = __DATA_JSON__;

// ---- helpers ----
function $(sel, el) { return (el||document).querySelector(sel); }
function h(s) { const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
function fmt(n) { return n.toLocaleString(); }
function pct(n, d) { return d ? (n/d*100).toFixed(1) : '0.0'; }

function showTab(id) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  $('#tab-'+id).classList.add('active');
  document.querySelectorAll('nav button').forEach(b => {
    if (b.textContent.toLowerCase().replace(/[^a-z]/g,'').startsWith(id.slice(0,4))) b.classList.add('active');
  });
}

// ---- build tabs ----
function ring(pctVal, label, color) {
  const r = 36, c = 2*Math.PI*r, off = c*(1-pctVal/100);
  return `<div class="ring-wrap">
    <svg width="90" height="90" viewBox="0 0 90 90">
      <circle class="ring-bg" cx="45" cy="45" r="${r}"/>
      <circle class="ring-fg" cx="45" cy="45" r="${r}" stroke="${color}"
        stroke-dasharray="${c}" stroke-dashoffset="${off}"/>
    </svg>
    <div class="ring-label">${pctVal}%<small>${label}</small></div>
  </div>`;
}

function buildDashboard() {
  const models = DATA.summary.models;
  const scored = DATA.scored;
  let html = '<h2>Dashboard</h2><div class="model-cards">';
  for (const [name, m] of Object.entries(models)) {
    html += `<div class="card model-card">
      <h3>${h(name)}</h3>
      <div class="scores">
        ${ring(m.exact_pct, 'exact', 'var(--green)')}
        ${ring(m.soft_pct, 'soft', 'var(--blue)')}
      </div>
      <div>${m.exact}/${m.total} exact &middot; ${m.soft}/${m.total} soft</div>
    </div>`;
  }
  html += '</div>';

  // Bar chart
  html += '<div class="card"><h3>Model Comparison</h3><div class="bar-chart">';
  for (const [name, m] of Object.entries(models)) {
    html += `<div class="bar-row"><span class="label">${h(name)}</span>
      <div class="track">
        <div class="fill exact" style="width:${m.exact_pct}%">${m.exact_pct}% exact</div>
      </div></div>`;
    html += `<div class="bar-row"><span class="label"></span>
      <div class="track">
        <div class="fill soft" style="width:${m.soft_pct}%">${m.soft_pct}% soft</div>
      </div></div>`;
  }
  html += '</div></div>';

  // Hardest questions
  const qFails = {};
  for (const r of scored) {
    if (!qFails[r.question_id]) qFails[r.question_id] = {q: r.question, fails: 0, total: 0};
    qFails[r.question_id].total++;
    if (!r.exact) qFails[r.question_id].fails++;
  }
  const hardest = Object.entries(qFails).filter(([,v])=>v.fails>0).sort((a,b)=>b[1].fails-a[1].fails);
  if (hardest.length) {
    html += '<div class="card"><h3>Hardest Questions</h3><ul class="hardest-list">';
    for (const [id, v] of hardest) {
      html += `<li><span>${h(id)}: ${h(v.q)}</span><span style="color:var(--red);font-weight:600">${v.fails}/${v.total} failed</span></li>`;
    }
    html += '</ul></div>';
  }
  $('#tab-dashboard').innerHTML = html;
}

function buildDataset() {
  const scored = DATA.scored;
  const previews = DATA.data_previews;
  const datasets = [...new Set(scored.map(r => r.dataset))].sort();
  const models = [...new Set(scored.map(r => r.model_name))].sort();

  let html = '<h2>By Dataset</h2>';
  for (const ds of datasets) {
    html += `<div class="card ds-section">
      <div class="ds-toggle" onclick="this.parentElement.querySelector('.ds-body').classList.toggle('hidden')">
        <h3>${h(ds)}</h3>
      </div><div class="ds-body">`;

    // Score table
    html += '<table><tr><th>Model</th><th>Exact</th><th>Soft</th><th>Total</th><th>Exact %</th></tr>';
    for (const m of models) {
      const items = scored.filter(r => r.dataset === ds && r.model_name === m);
      const ex = items.filter(r=>r.exact).length;
      const sf = items.filter(r=>r.soft).length;
      html += `<tr><td>${h(m)}</td><td>${ex}</td><td>${sf}</td><td>${items.length}</td><td>${pct(ex,items.length)}%</td></tr>`;
    }
    html += '</table>';

    // Data preview
    const prev = previews[ds];
    if (prev) {
      html += `<details style="margin-top:12px"><summary>Data Preview</summary>
        <div class="row-count">Showing ${prev.rows.length} of ${fmt(prev.total_rows)} rows</div>
        <div class="data-preview-wrap"><table><tr>`;
      for (const col of prev.columns) html += `<th>${h(col)}</th>`;
      html += '</tr>';
      for (const row of prev.rows) {
        html += '<tr>';
        for (const col of prev.columns) html += `<td>${h(String(row[col]!=null?row[col]:''))}</td>`;
        html += '</tr>';
      }
      html += '</table></div></details>';
    }
    html += '</div></div>';
  }
  $('#tab-dataset').innerHTML = html;
}

function buildExplorer() {
  const scored = DATA.scored;
  const questions = DATA.questions;
  const datasets = ['all', ...new Set(scored.map(r => r.dataset))];
  const statuses = ['all', 'correct', 'wrong'];

  let html = `<h2>Question Explorer</h2>
    <div class="filter-bar">
      <select id="flt-ds">${datasets.map(d=>`<option value="${d}">${d}</option>`).join('')}</select>
      <select id="flt-status">${statuses.map(s=>`<option value="${s}">${s}</option>`).join('')}</select>
    </div><div id="q-list"></div>`;
  $('#tab-explorer').innerHTML = html;

  const render = () => {
    const ds = $('#flt-ds').value;
    const st = $('#flt-status').value;
    // Group scored by question_id
    const byQ = {};
    for (const r of scored) {
      if (ds !== 'all' && r.dataset !== ds) continue;
      if (!byQ[r.question_id]) byQ[r.question_id] = [];
      byQ[r.question_id].push(r);
    }
    let out = '';
    for (const [qid, results] of Object.entries(byQ)) {
      // Filter by status
      if (st === 'correct' && !results.every(r => r.exact)) continue;
      if (st === 'wrong' && !results.some(r => !r.exact)) continue;

      const q = questions[qid] || {};
      const diffClass = q.difficulty === 'easy' ? 'badge-easy' : q.difficulty === 'hard' ? 'badge-hard' : 'badge-medium';
      out += `<div class="card q-card">
        <div class="q-header">
          <strong>${h(qid)}</strong>
          <span class="badge ${diffClass}">${h(q.difficulty||'')}</span>
          <span class="badge badge-cat">${h(q.category||'')}</span>
        </div>
        <div style="margin-bottom:8px">${h(q.question||results[0].question)}</div>
        <div class="expected">Expected: <strong>${h(String(q.answer!=null?q.answer:results[0].expected_answer))}</strong></div>`;
      if (q.pandas_code) {
        out += `<div class="code-block"><code>${h(q.pandas_code)}</code></div>`;
      }
      for (const r of results) {
        const dotCls = r.exact ? 'dot-exact' : r.soft ? 'dot-soft' : 'dot-wrong';
        const statusText = r.exact ? 'Exact' : r.soft ? 'Soft' : 'Wrong';
        out += `<div class="model-result">
          <span class="status-dot ${dotCls}"></span>
          <strong>${h(r.model_name)}</strong> — ${statusText}
          <span class="meta" style="margin-left:8px">
            answer: ${h(r.extracted_answer)} &middot;
            ${r.elapsed_seconds.toFixed(1)}s &middot;
            ${fmt(r.usage.prompt_tokens)} in / ${fmt(r.usage.completion_tokens)} out
          </span>
          <details><summary>Raw response</summary><pre>${h(r.raw_response)}</pre></details>
        </div>`;
      }
      out += '</div>';
    }
    if (!out) out = '<div class="card" style="text-align:center;color:var(--muted)">No questions match filters.</div>';
    $('#q-list').innerHTML = out;
  };

  $('#flt-ds').addEventListener('change', render);
  $('#flt-status').addEventListener('change', render);
  render();
}

function buildCost() {
  const costs = DATA.costs;
  let html = `<h2>Cost &amp; Performance</h2><div class="card">
    <table class="cost-table"><tr>
      <th>Model</th><th>Model ID</th><th>Questions</th>
      <th>Prompt Tokens</th><th>Completion Tokens</th>
      <th>Input Cost</th><th>Output Cost</th><th>Total Cost</th>
      <th>Total Time</th><th>Avg Time/Q</th>
    </tr>`;
  let grandTotal = 0;
  for (const c of costs) {
    grandTotal += c.total_cost;
    html += `<tr>
      <td>${h(c.model)}</td><td class="mono">${h(c.model_id)}</td><td class="num">${c.num_questions}</td>
      <td class="num">${fmt(c.prompt_tokens)}</td><td class="num">${fmt(c.completion_tokens)}</td>
      <td class="num">$${c.input_cost.toFixed(2)}</td><td class="num">$${c.output_cost.toFixed(2)}</td>
      <td class="num"><strong>$${c.total_cost.toFixed(2)}</strong></td>
      <td class="num">${c.total_time.toFixed(1)}s</td><td class="num">${c.avg_time.toFixed(1)}s</td>
    </tr>`;
  }
  html += `<tr style="font-weight:700;border-top:2px solid var(--text)">
    <td colspan="7" style="text-align:right">Grand Total</td>
    <td class="num">$${grandTotal.toFixed(2)}</td><td colspan="2"></td></tr>`;
  html += '</table></div>';

  // Rates reference
  html += '<div class="card"><h3>Pricing Rates (per 1M tokens)</h3><table><tr><th>Model</th><th>Input</th><th>Output</th></tr>';
  const rates = {claude:'$5.00 / $25.00', gemini:'$2.00 / $12.00', gpt5:'$1.75 / $14.00'};
  for (const [m,r] of Object.entries(rates)) {
    const [i,o] = r.split(' / ');
    html += `<tr><td>${h(m)}</td><td>${i}</td><td>${o}</td></tr>`;
  }
  html += '</table></div>';
  $('#tab-cost').innerHTML = html;
}

// Init
buildDashboard();
buildDataset();
buildExplorer();
buildCost();
</script>
</body>
</html>"""


def build_page(data):
    data_json = json.dumps(data, default=str)
    return HTML_TEMPLATE.replace("__DATA_JSON__", data_json)


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
    print("Loading data...")
    data = load_data()
    print(f"  {len(data['scored'])} results, {len(data['questions'])} questions, {len(data['costs'])} models")

    print("Building HTML...")
    html = build_page(data)
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
