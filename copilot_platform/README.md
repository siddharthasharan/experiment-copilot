# Experiment Copilot — Chat-first platform (Phase 1 + Phase 2 scenarios)

A live Python web service (stdlib HTTP only, no new dependencies) where
the chat is the product and the scorecard is an artifact the
conversation generates. The existing static showroom is untouched and
remains the "Example Scenarios" destination.

## Run locally

```bash
pip install -r requirements.txt
python -m copilot_platform.server --port 8000
# open http://localhost:8000
python copilot_platform/tests/test_platform.py   # 92 checks, no pytest needed
```

## Layout (from the spec, no mockup file was available in the workspace)

- Left nav, ~11%: New Experiment, Chat with Copilot, Saved Experiments,
  Example Scenarios, Data Explorer, Documentation.
- Chat initially ~65–75% of the workspace; the results panel is hidden
  until a run completes, then chat shrinks to ~38% and results take
  ~62%, collapsible. On small screens the results panel is a full-screen
  overlay. The conversation stays visible beside the scorecard.

## Architecture — and where an LLM fits later

```
chat.py        deterministic state machine: welcome -> industry ->
               scenario -> config -> run -> follow-ups. Never computes
               a number. Optional LLMAdapter.explain() may ADD an
               explanation AFTER a deterministic answer. It may not
               choose scenarios, set configs, compute metrics, or
               replace a deterministic block.
engine.py      flagship generators + Welch/CUPED/SRM/decision rules.
               Seeded, deterministic. Geo experiments are analysed at
               geo level (ZIP/DMA), never by counting impressions.
engine_extra.py  Phase 2 generators: the 9 remaining industry scenarios,
               same primitives and result shape as engine.py.
scenarios.py   the registry: industry cards, flagship configs, follow-ups.
followups.py   recomputations on stored data (exclude-top, segments,
               driver availability, publisher descriptive shares).
scorecard.py   executive + KPIs + charts + metrics table + PM/DS/watch-outs.
server.py      stdlib ThreadingHTTPServer. Sessions in memory, lost on
               restart (stated in the UI — a Phase 1 tradeoff).
static/        chat-first frontend, SVG charts, no CDN.
```

Adding an LLM later = implementing `LLMAdapter.explain()` and wiring it
in `server.py`. The engine, registry, decision rules, and tests do not
change. The LLM must never be the calculator.

## The three flagships (seed 42 defaults)

- **Reddit feed-ranking** (user-level, CUPED): +5.8% replies/user, LAUNCH.
  Heavy-tailed activity, new vs established users, novelty decay.
- **DoorDash ZIP promotion** (ZIP-level, matched pairs by baseline size,
  pre-period adjusted): +6.1% orders, but contribution is negative as
  designed ($5 paid on every treated order) → **Do Not Launch** on
  economics, with the reason stated on the scorecard. Driver strain is
  a separate guardrail check, not the verdict driver here.
- **Pharma DMA publisher incrementality** (DMA-level, matched pairs by
  population, claims-adjusted): +7.0% starts, LAUNCH. Publisher shares
  are descriptive only — a geo allocation experiment cannot identify
  individual publisher causal effects, and the follow-up says so.

## Phase 2: the other 9 scenarios, now runnable (seed 42 defaults)

- **Social:** Meta creator recommendations +5.5% sessions/user, LAUNCH;
  TikTok personalized notifications +6.3% sessions/user, LAUNCH (strong
  novelty decay — early trend well above late trend); LinkedIn
  connection recommendations +7.7% accepted connections/user, LAUNCH.
- **Marketplace:** DMA digital advertising +4.8% orders/DMA, LAUNCH at
  $4.20 per incremental order; Uber driver incentives +9.4% trips/city,
  LAUNCH at $15.90 per incremental trip (large cities +16.7%, small
  cities −4.1% — heterogeneous, not a uniform win); delivery-fee
  optimization +10.5% orders/ZIP, but contribution −$72,185 as designed
  ($2 given up on every treated order) → **Do Not Launch** on economics.
- **Pharma:** HCP omnichannel +5.2% interactions/HCP, LAUNCH (low prior
  engagement +9.5% vs high +3.9%); field visits + sampling +11.1%
  prescriptions/territory, LAUNCH at $14.30 per incremental prescription;
  cross-publisher budget reallocation +3.9% starts/DMA, **Inconclusive**
  at seed 42 (p=0.115) — a reallocation that does not prove itself at
  this scale, not a hidden failure.

Other seeds can give different verdicts. That is honest sampling
variation, not a bug: the scorecard reports the computed result, and
inconclusive runs are explained, not hidden.

## What is precomputed, honestly

The 14 example scenarios are the recorded 1M-user engine output from
`docs/scorecard_data.json`, labelled "precomputed demonstration"
everywhere. They cannot be recomputed live in Phase 1, and the chat
says so when asked. All 12 industry scenarios are generated and
analysed live on every run, with real stage-by-stage progress (each
stage is reported only when its function returns — no timers, no fake
percentages).

## 13th scenario: LinkedIn creator crowding-out (seed 42)

Viewer-randomized, not creator-randomized — that distinction is the
point. Primary: small-creator (1,000–10,000 followers) share of creator
sessions, CUPED on its pre-period value. Top tier is 100,000+ followers,
mid tier 10,000–100,000. At seed 42: small share 16.1% → 14.8% (−1.3pp),
top share +1.6pp, total sessions +5.7% → **Do Not Launch** as a
small-creator intervention: the gain went to top creators. The trend
chart plots the small-tier share itself. Follow-up "Which tier gained
the most?" recomputes all three tier shares by arm. Watch-out states
the limit plainly: this measures viewer exposure, not creator earnings
— a creator-level claim would need creator randomization.

## Known limits

- Sessions and saved experiments live in server memory only.
- Custom experiments are mapped to the closest runnable scenario or
  honestly declined — never guessed into a design.
- Downloads: scorecard JSON, config JSON, metrics CSV, and a simple
  text executive PDF (hand-rolled, stdlib). A full designed PDF is a
  later step.
- No login on the app. Synthetic demo data only; never put real data,
  credentials, or internal data in this repo or on the service.
