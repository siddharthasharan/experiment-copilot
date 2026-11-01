let sessionId = null;
let currentScorecard = null;
let pollTimer = null;

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");

function showView(name) {
  const isChat = name === "chat" || name === "new" || name === "examples" || name === "explorer";
  $("messages").style.display = isChat ? "" : "none";
  $("inputbar").style.display = isChat ? "" : "none";
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  if (!isChat) $("view-" + name).classList.add("active");
  document.querySelectorAll(".navbtn").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  if (name === "saved") loadSaved();
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  return r.json();
}

function addBubble(text, who) {
  const d = document.createElement("div");
  d.className = "bubble " + who;
  d.textContent = text;
  $("messages").appendChild(d);
  $("messages").scrollTop = $("messages").scrollHeight;
}

function addChips(items) {
  const d = document.createElement("div");
  d.className = "chips";
  items.forEach((item) => {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = item;
    b.onclick = () => sendMessage(item);
    d.appendChild(b);
  });
  $("messages").appendChild(d);
  $("messages").scrollTop = $("messages").scrollHeight;
}

function renderBlocks(blocks) {
  blocks.forEach((b) => {
    if (b.type === "text") addBubble(b.text, "assistant");
    else if (b.type === "chips") addChips(b.items);
    else if (b.type === "industry_cards") {
      const wrap = document.createElement("div");
      wrap.className = "cards";
      b.industries.forEach((ind) => {
        const c = document.createElement("div");
        c.className = "card";
        c.innerHTML = `<h3>${esc(ind.title)}</h3><p>${esc(ind.examples)}</p><p>${esc(ind.focus)}</p>`;
        c.onclick = () => sendAction("industry", { industry_id: ind.id }, ind.title);
        wrap.appendChild(c);
      });
      $("messages").appendChild(wrap);
    } else if (b.type === "scenario_cards") {
      const wrap = document.createElement("div");
      wrap.className = "cards";
      b.cards.forEach((card) => {
        const c = document.createElement("div");
        c.className = "card" + (card.runnable ? "" : " disabled");
        c.innerHTML = `<h3>${esc(card.title)}</h3><p>${esc(card.subtitle)}</p>` +
          (card.runnable ? `<span class="tag">Runnable</span>` : `<span class="tag">Coming next</span>`);
        if (card.runnable) {
          // Example cards use the example action; flagships use scenario.
          const isExample = b.cards.length === 14;
          c.onclick = () => sendAction(isExample ? "example" : "scenario", { scenario_id: card.scenario_id }, card.title);
        }
        wrap.appendChild(c);
      });
      $("messages").appendChild(wrap);
    } else if (b.type === "config") {
      const cfg = b.config;
      const rows = [
        ["Experiment", cfg.name], ["Business hypothesis", cfg.hypothesis],
        ["Treatment", cfg.treatment], ["Control", cfg.control],
        ["Unit of randomization", cfg.unit], ["Population", cfg.population],
        ["Baseline period", cfg.baseline], ["Duration", cfg.duration],
        ["Primary metric", cfg.primary], ["Secondary metrics", (cfg.secondary || []).join(", ")],
        ["Guardrails", cfg.guardrails], ["Method", cfg.method], ["Seed", cfg.seed],
      ].map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("");
      const wrap = document.createElement("div");
      wrap.className = "config-card";
      wrap.innerHTML = `<h3>Recommended design</h3><table>${rows}</table><div style="margin-top:12px;display:flex;gap:8px">
        <button class="btn" data-a="run">Run Experiment</button>
        <button class="btn secondary" data-a="modify">Modify Design</button>
        <button class="btn secondary" data-a="explain">Explain Methodology</button></div>`;
      wrap.querySelector('[data-a="run"]').onclick = () => startRun();
      wrap.querySelector('[data-a="modify"]').onclick = () => sendAction("modify", {}, "Modify Design");
      wrap.querySelector('[data-a="explain"]').onclick = () => sendAction("explain", {}, "Explain Methodology");
      $("messages").appendChild(wrap);
    } else if (b.type === "example_scorecard") {
      // Panel is opened by the server response scorecard; nothing in chat.
    } else if (b.type === "scorecard_open") {
      openScorecard(b.scorecard);
    }
  });
  $("messages").scrollTop = $("messages").scrollHeight;
}

async function sendMessage(text) {
  if (text) addBubble(text, "user");
  const res = await api("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message: text }) });
  sessionId = res.session_id;
  renderBlocks(res.blocks || []);
  if (res.scorecard) openScorecard(res.scorecard);
}

async function sendAction(action, payload, label) {
  if (label) addBubble(label, "user");
  const res = await api("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, action, payload }) });
  sessionId = res.session_id;
  renderBlocks(res.blocks || []);
  if (res.scorecard) openScorecard(res.scorecard);
}

async function startRun() {
  const res = await api("/api/run", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }) });
  if (res.error) { addBubble("Couldn't start the run: " + res.error, "assistant"); return; }
  const prog = document.createElement("div");
  prog.className = "config-card";
  prog.innerHTML = `<h3>Running your experiment using realistic synthetic data</h3><div id="stages"></div>`;
  $("messages").appendChild(prog);
  $("messages").scrollTop = $("messages").scrollHeight;
  pollTimer = setInterval(async () => {
    const st = await api("/api/run-status?session_id=" + sessionId);
    const el = document.getElementById("stages");
    if (el) el.innerHTML = (st.stages || []).map((s) => `<div class="stage">${esc(s.stage)} — ${esc(s.detail)}</div>`).join("");
    if (st.done) {
      clearInterval(pollTimer);
      if (st.error) addBubble("The run failed: " + st.error, "assistant");
      else {
        renderBlocks(st.summary || []);
        if (st.scorecard) openScorecard(st.scorecard);
      }
    }
  }, 400);
}

// ------------------------------------------------------------ scorecard
function svgLine(series, color) {
  if (!series || !series.length) return "<p>No trend data for this precomputed example.</p>";
  const W = 560, H = 160, vals = series.map((r) => r.control).concat(series.map((r) => r.treatment));
  const min = Math.min(...vals), max = Math.max(...vals);
  const x = (i) => 10 + (i / (series.length - 1)) * (W - 20);
  const y = (v) => H - 10 - ((v - min) / (max - min || 1)) * (H - 20);
  const path = (key) => series.map((r, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(r[key]).toFixed(1)}`).join(" ");
  return `<svg width="100%" viewBox="0 0 ${W} ${H}"><path d="${path("control")}" stroke="#9ca3af" fill="none" stroke-width="2"/><path d="${path("treatment")}" stroke="${color}" fill="none" stroke-width="2"/></svg><p style="font-size:12px;color:#6b7280">Grey = control, purple = treatment.</p>`;
}

function openScorecard(sc) {
  currentScorecard = sc;
  document.getElementById("app").classList.add("results-open");
  $("results-title").textContent = sc.experiment_name;
  $("results-verdict").textContent = sc.recommendation;
  $("results-verdict").className = "verdict " + (sc.verdict || "");
  const kpis = (sc.kpis || []).map((k) => `<div class="kpi"><div class="label">${esc(k.label)}</div><div class="value">${esc(k.value)}</div></div>`).join("");
  const rows = (sc.metrics || []).map((m) => `<tr><td>${esc(m.name)}</td><td>${(m.control ?? "").toString().slice(0, 12)}</td><td>${(m.treatment ?? "").toString().slice(0, 12)}</td><td>${m.lift_pct != null ? m.lift_pct.toFixed(2) + "%" : "—"}</td><td>${m.ci_low_pct != null ? "[" + m.ci_low_pct.toFixed(2) + "%, " + m.ci_high_pct.toFixed(2) + "%]" : "—"}</td><td>${m.p_value != null ? Number(m.p_value).toPrecision(3) : "—"}</td><td>${m.guardrail ? "Guardrail" : "—"}</td></tr>`).join("");
  const segs = (sc.segments || []).map((s) => `<tr><td>${esc(s.segment)}</td><td>${s.lift_pct.toFixed(2)}%</td><td>[${s.ci_low_pct.toFixed(2)}%, ${s.ci_high_pct.toFixed(2)}%]</td><td>${Number(s.p_value).toPrecision(3)}</td></tr>`).join("");
  $("tab-scorecard").innerHTML = `
    <div class="view-section"><h3>Executive decision</h3>
      <p><b>${esc(sc.recommendation)}</b> · ${esc(sc.primary_metric || "")}
      ${sc.primary_lift_pct != null ? ` · lift ${sc.primary_lift_pct.toFixed(2)}% (95% CI [${sc.ci_low_pct.toFixed(2)}%, ${sc.ci_high_pct.toFixed(2)}%], p=${Number(sc.p_value).toPrecision(3)})` : ""}
      · Guardrails: ${esc(sc.guardrail_status)} · SRM: ${esc(sc.srm_status)} · Seed: ${esc(sc.seed)}
      ${sc.precomputed ? " · Precomputed demonstration (1M users, seed 42)." : ""}</p>
      ${sc.business_reason ? `<p>${esc(sc.business_reason)}</p>` : ""}</div>
    <div class="kpis">${kpis}</div>
    <div class="view-section"><h3>Treatment vs control trend</h3>${svgLine(sc.trend, "#6d28d9")}</div>
    ${segs ? `<div class="view-section"><h3>Segment effects</h3><table class="metrics"><tr><th>Segment</th><th>Lift</th><th>95% CI</th><th>P-value</th></tr>${segs}</table></div>` : ""}
    <div class="view-section"><h3>Metrics table</h3><table class="metrics"><tr><th>Metric</th><th>Control</th><th>Treatment</th><th>Lift</th><th>95% CI</th><th>P-value</th><th>Guardrail</th></tr>${rows}</table></div>`;
  const pm = sc.pm || {};
  $("tab-pm").innerHTML = `<div class="view-section"><h3>PM view</h3>
    <p><b>What happened?</b> ${esc(pm.what || "")}</p><p><b>Why does it matter?</b> ${esc(pm.why || "")}</p>
    <p><b>What is the business impact?</b> ${esc(pm.impact || "")}</p><p><b>Should we launch?</b> ${esc(pm.launch || "")}</p>
    <p><b>What should we do next?</b> ${esc(pm.next || "")}</p></div>`;
  const ds = sc.ds || {};
  $("tab-ds").innerHTML = `<div class="view-section"><h3>Data Scientist view</h3>` +
    Object.entries(ds).map(([k, v]) => `<p><b>${esc(k)}:</b> ${esc(v)}</p>`).join("") + `</div>`;
  $("tab-watch").innerHTML = `<div class="view-section"><h3>Watch-outs &amp; limitations</h3><ul>` +
    (sc.watchouts || []).map((w) => `<li>${esc(w)}</li>`).join("") + `</ul></div>` +
    (sc.trap_reveal ? `<div class="view-section"><h3>Trap reveal</h3><p>${esc(sc.trap_reveal.what_happened || "")}</p><p>${esc(sc.trap_reveal.why_it_matters || "")}</p></div>` : "");
  $("tab-explorer").innerHTML = `<div class="view-section"><p>Loading tables…</p></div>`;
  loadTables();
  showTab("scorecard");
}

function showTab(name) {
  document.querySelectorAll(".tab-pane").forEach((p) => (p.style.display = "none"));
  $("tab-" + name).style.display = "block";
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
}

async function loadTables() {
  const res = await api("/api/tables?session_id=" + sessionId);
  if (res.error) { $("tab-explorer").innerHTML = `<div class="view-section"><p>${esc(res.error)} — run a flagship experiment first. Precomputed examples do not expose raw tables in Phase 1.</p></div>`; return; }
  const html = res.tables.map((t) => `<div class="view-section"><h3>${esc(t.name)} (${t.rows.toLocaleString()} rows)</h3>
    <p>${t.columns.map((c) => `${esc(c.name)} (${esc(c.dtype)})`).join(" · ")}</p>
    <table class="metrics"><tr>${t.columns.map((c) => `<th>${esc(c.name)}</th>`).join("")}</tr>
    ${t.sample.map((r) => `<tr>${t.columns.map((c) => `<td>${esc(r[c.name] ?? "")}</td>`).join("")}</tr>`).join("")}</table></div>`).join("");
  $("tab-explorer").innerHTML = html +
    `<div class="view-section"><h3>Example queries (run as follow-ups in chat)</h3><ul>
    <li>“Show results for new customers” — segment recomputation on the stored data.</li>
    <li>“Exclude the five largest markets” — geo recomputation, not a canned answer.</li>
    <li>“Which publisher performed best?” — descriptive only, with the causal caveat.</li></ul></div>`;
}

function downloads() {
  const types = [["scorecard_json", "Scorecard JSON"], ["config_json", "Experiment config JSON"],
    ["metrics_csv", "Metrics CSV"], ["pdf", "Executive PDF"]];
  return types.map(([t, label]) => `<a href="/api/download?session_id=${sessionId}&type=${t}"><button class="btn secondary" style="margin:4px">${label}</button></a>`).join("");
}

async function loadSaved() {
  const res = await api("/api/saved");
  $("view-saved").innerHTML = `<h2>Saved Experiments</h2><p style="font-size:13px;color:#6b7280">Saved in server memory for this session; a server restart clears them. That is a Phase 1 tradeoff, stated here rather than hidden.</p>` +
    (res.saved.length ? res.saved.map((s) => `<div class="saved-item" data-id="${s.id}"><b>${esc(s.name)}</b> — ${esc(s.verdict)} · lift ${s.primary_lift_pct.toFixed(2)}%</div>`).join("")
      : "<p>No completed experiments yet. Run a flagship from the chat.</p>");
  document.querySelectorAll(".saved-item").forEach((el) => {
    el.onclick = async () => {
      const r = await api("/api/saved-load", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: el.dataset.id }) });
      if (r.error) { alert(r.error); return; }
      sessionId = r.session_id;
      $("messages").innerHTML = "";
      renderBlocks(r.blocks || []);
      if (r.scorecard) openScorecard(r.scorecard);
      showView("chat");
    };
  });
}

window.addEventListener("DOMContentLoaded", async () => {
  $("send").onclick = () => { const v = $("input").value.trim(); if (v) { $("input").value = ""; sendMessage(v); } };
  $("input").addEventListener("keydown", (e) => { if (e.key === "Enter") $("send").click(); });
  document.querySelectorAll(".navbtn").forEach((b) => {
    b.onclick = () => {
      const v = b.dataset.view;
      if (v === "examples") { showView("chat"); sendAction("example_list", {}, "Explore the 14 example scenarios"); }
      else if (v === "new") { showView("chat"); sendMessage("New experiment"); }
      else if (v === "explorer") { showView("chat"); if (currentScorecard) { document.getElementById("app").classList.add("results-open"); showTab("explorer"); } else addBubble("Run a flagship experiment first, then open Data Explorer.", "assistant"); }
      else if (v === "docs") { showView("docs"); }
      else showView(v);
    };
  });
  document.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
  $("close-results").onclick = () => document.getElementById("app").classList.remove("results-open");
  $("open-results").onclick = () => { if (currentScorecard) document.getElementById("app").classList.add("results-open"); };
  // First load: welcome blocks from the server.
  const res = await api("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) });
  sessionId = res.session_id;
  renderBlocks(res.blocks || []);
});
