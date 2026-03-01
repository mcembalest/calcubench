const DATA = __DATA_JSON__;

function $(sel, el) { return (el||document).querySelector(sel); }
function $$(sel, el) { return (el||document).querySelectorAll(sel); }
function h(s) { const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
function fmt(n) { return n.toLocaleString(); }
function pct(n, d) { return d ? (n/d*100).toFixed(1) : '0.0'; }
function money(n) { return '$' + n.toFixed(2); }

document.querySelectorAll('nav button').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('nav button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    $('#tab-' + b.dataset.tab).classList.add('active');
  });
});

// ---- Dataset Tab ----
function buildDataset() {
  const scored = DATA.scored;
  const previews = DATA.data_previews;
  const datasets = [...new Set(scored.map(r => r.dataset))].sort();
  const models = [...new Set(scored.map(r => r.model_name))].sort();

  let html = '<h2>Datasets</h2>';
  for (const ds of datasets) {
    const dsResults = scored.filter(r => r.dataset === ds);
    html += `<div class="card ds-section"><h3>${h(ds)}</h3>`;

    // Combined score + cost table
    html += `<table>
      <tr><th>Model</th><th>Reasoning</th><th class="num">Exact</th><th class="num">Soft</th><th class="num">Total</th><th class="num">Exact %</th>
      <th class="num">Prompt Tok</th><th class="num">Compl Tok</th><th class="num">Cost</th><th class="num">Time</th><th class="num">Avg Time</th></tr>`;
    for (const m of models) {
      const items = dsResults.filter(r => r.model_name === m);
      const ex = items.filter(r=>r.exact).length;
      const sf = items.filter(r=>r.soft).length;
      const pt = items.reduce((s,r) => s + r.usage.prompt_tokens, 0);
      const ct = items.reduce((s,r) => s + r.usage.completion_tokens, 0);
      const cost = items.reduce((s,r) => s + r.total_cost, 0);
      const time = items.reduce((s,r) => s + r.elapsed_seconds, 0);
      const reasoning = (DATA.reasoning && DATA.reasoning[m]) || '';
      html += `<tr><td>${h(m)}</td><td class="mono" style="font-size:12px">${h(reasoning)}</td><td class="num">${ex}</td><td class="num">${sf}</td><td class="num">${items.length}</td><td class="num">${pct(ex,items.length)}%</td>
        <td class="num">${fmt(pt)}</td><td class="num">${fmt(ct)}</td><td class="num">${money(cost)}</td>
        <td class="num">${time.toFixed(1)}s</td><td class="num">${items.length ? (time/items.length).toFixed(1) : 0}s</td></tr>`;
    }
    html += '</table>';

    // Data preview
    const prev = previews[ds];
    if (prev) {
      html += `<details style="margin-top:12px"><summary>Data Preview (${prev.rows.length} of ${fmt(prev.total_rows)} rows)</summary>
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
    html += '</div>';
  }

  $('#tab-dataset').innerHTML = html;
}

// ---- Draggable post-it system ----
let postitZ = 1000;
function openPostit(title, text, anchorEl, meta) {
  const el = document.createElement('div');
  el.className = 'postit';
  el.style.zIndex = ++postitZ;
  let metaHtml = '';
  if (meta) {
    const parts = [];
    if (meta.modelId) parts.push(`Model: ${h(meta.modelId)}`);
    if (meta.thinking) parts.push(`Thinking: ${h(meta.thinking)}`);
    if (meta.cost != null) parts.push(`Cost: ${money(meta.cost)}`);
    if (meta.latency != null) parts.push(`Latency: ${meta.latency.toFixed(1)}s`);
    metaHtml = `<div style="padding:6px 12px;font-size:12px;color:#555;background:#fff8c4;border-bottom:1px solid #e0d88a;display:flex;gap:16px;flex-wrap:wrap">${parts.join('<span style="color:#ccc">|</span>')}</div>`;
  }
  el.innerHTML = `<div class="postit-header"><span>${h(title)}</span><button class="postit-close">&times;</button></div>${metaHtml}<div class="postit-body"></div>`;
  el.querySelector('.postit-body').textContent = text;
  document.body.appendChild(el);

  // Position near the clicked element (fixed positioning = viewport coords)
  const rect = anchorEl.getBoundingClientRect();
  let top = rect.top;
  let left = rect.right + 12;
  // Keep on screen
  if (left + 500 > window.innerWidth) left = Math.max(8, rect.left - 500);
  if (top + 350 > window.innerHeight) top = Math.max(8, window.innerHeight - 360);
  el.style.top = top + 'px';
  el.style.left = left + 'px';

  // Close button
  el.querySelector('.postit-close').addEventListener('click', () => el.remove());

  // Bring to front on click
  el.addEventListener('mousedown', () => { el.style.zIndex = ++postitZ; });

  // Drag
  const header = el.querySelector('.postit-header');
  let dragging = false, dx = 0, dy = 0;
  header.addEventListener('mousedown', e => {
    if (e.target.classList.contains('postit-close')) return;
    dragging = true; dx = e.clientX - el.offsetLeft; dy = e.clientY - el.offsetTop;
    el.style.zIndex = ++postitZ;
    e.preventDefault();
  });
  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    el.style.left = (e.clientX - dx) + 'px';
    el.style.top = (e.clientY - dy) + 'px';
  });
  document.addEventListener('mouseup', () => { dragging = false; });
}

// ---- Questions Tab ----
function buildExplorer() {
  const scored = DATA.scored;
  const questions = DATA.questions;
  const datasets = ['all', ...new Set(scored.map(r => r.dataset))];
  const statuses = ['all', 'correct', 'wrong'];
  const models = [...new Set(scored.map(r => r.model_name))].sort();

  let html = `<h2>Questions</h2>
    <div class="filter-bar">
      <select id="flt-ds">${datasets.map(d=>`<option value="${d}">${d}</option>`).join('')}</select>
      <select id="flt-status">${statuses.map(s=>`<option value="${s}">${s}</option>`).join('')}</select>
      <span class="count" id="q-count"></span>
    </div><div id="q-list"></div>`;
  $('#tab-explorer').innerHTML = html;

  const render = () => {
    const ds = $('#flt-ds').value;
    const st = $('#flt-status').value;

    const qIds = [];
    const byQ = {};
    for (const r of scored) {
      if (ds !== 'all' && r.dataset !== ds) continue;
      if (!byQ[r.question_id]) { byQ[r.question_id] = []; qIds.push(r.question_id); }
      byQ[r.question_id].push(r);
    }

    let out = '<div class="card" style="overflow-x:auto;padding:12px"><table class="q-table"><thead><tr>';
    out += '<th>Question</th><th>Answer</th><th>Model</th><th>Reasoning</th><th>Extracted Answer</th><th>Full Response</th><th class="num">Latency</th><th class="num">Cost</th>';
    out += '</tr></thead><tbody>';

    let count = 0;
    let totalResults = 0;
    for (const qid of qIds) {
      const results = byQ[qid];
      // Consistent subrow order: same model order as Dataset tab (alphabetical)
      results.sort((a, b) => models.indexOf(a.model_name) - models.indexOf(b.model_name));
      if (st === 'correct' && !results.every(r => r.exact)) continue;
      if (st === 'wrong' && !results.some(r => !r.exact)) continue;
      count++;
      totalResults += results.length;

      const q = questions[qid] || {};
      const n = results.length;
      const answer = String(q.answer != null ? q.answer : results[0].expected_answer);
      const groupCls = count % 2 === 0 ? 'q-group-even' : 'q-group-odd';

      for (let i = 0; i < results.length; i++) {
        const r = results[i];
        const dotCls = r.exact ? 'dot-exact' : r.soft ? 'dot-soft' : 'dot-wrong';
        const answerCls = r.exact ? 'exact-answer' : '';
        // Truncate response preview
        const preview = r.raw_response.length > 120 ? r.raw_response.slice(0, 120) + '...' : r.raw_response;

        out += `<tr class="${groupCls}">`;
        if (i === 0) {
          out += `<td rowspan="${n}" class="q-text" style="vertical-align:top;font-weight:500">${h(q.question || r.question)}</td>`;
          out += `<td rowspan="${n}" class="mono" style="vertical-align:top">${h(answer)}</td>`;
        }
        const reasoning = (DATA.reasoning && DATA.reasoning[r.model_name]) || '';
        out += `<td>${h(r.model_name)}</td>`;
        out += `<td class="mono" style="font-size:12px">${h(reasoning)}</td>`;
        out += `<td class="${answerCls}"><span class="status-dot ${dotCls}"></span>${h(r.extracted_answer)}</td>`;
        out += `<td><div class="response-preview" data-qid="${h(qid)}" data-model="${h(r.model_name)}">${h(preview)}</div></td>`;
        out += `<td class="num">${r.elapsed_seconds.toFixed(1)}s</td>`;
        out += `<td class="num">${money(r.total_cost)}</td>`;
        out += '</tr>';
      }
    }
    out += '</tbody></table></div>';
    if (count === 0) out = '<div class="card" style="text-align:center;color:var(--muted)">No questions match filters.</div>';
    $('#q-list').innerHTML = out;
    $('#q-count').textContent = `${count} questions, ${totalResults} results`;

    // Attach click handlers for post-it popups
    $$('.response-preview').forEach(el => {
      el.addEventListener('click', () => {
        const qid = el.dataset.qid;
        const model = el.dataset.model;
        const r = scored.find(x => x.question_id === qid && x.model_name === model);
        if (r) {
          const rates = (DATA.cost_rates && DATA.cost_rates[model]) || {};
          const reasoning = (DATA.reasoning && DATA.reasoning[model]) || '';
          openPostit(`${model} — ${qid}`, r.raw_response, el, {
            modelId: rates.model_id || r.model_id || model,
            thinking: reasoning,
            cost: r.total_cost,
            latency: r.elapsed_seconds
          });
        }
      });
    });
  };

  $('#flt-ds').addEventListener('change', render);
  $('#flt-status').addEventListener('change', render);
  render();
}

// Init
buildDataset();
buildExplorer();