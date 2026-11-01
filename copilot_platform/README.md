# Experiment Copilot — Chat-first platform (Phase 1)

A live Python web service (stdlib HTTP only, no new dependencies) where
the chat is the product and the scorecard is an artifact the
conversation generates. The existing static showroom is untouched and
remains the "Example Scenarios" destination.

## Run locally

```bash
pip install -r requirements.txt
python -m copilot_platform.server --port 8000
# open http://localhost:8000
python copilot_platform/tests/test_platform.py   # 21 checks, no pytest needed
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

Other seeds can give different verdicts. That is honest sampling
variation, not a bug: the scorecard reports the computed result, and
inconclusive runs are explained, not hidden.

## What is precomputed, honestly

The 14 example scenarios are the recorded 1M-user engine output from
`docs/scorecard_data.json`, labelled "precomputed demonstration"
everywhere. They cannot be recomputed live in Phase 1, and the chat
says so when asked. The three flagships are generated and analysed
live on every run, with real stage-by-stage progress (each stage is
reported only when its function returns — no timers, no fake percentages).

## Known Phase 1 limits

- Sessions and saved experiments live in server memory only.
- The remaining 9 industry scenarios are listed as "Coming next" and
  are not faked as runnable.
- Custom experiments are mapped to the closest runnable scenario or
  honestly declined — never guessed into a design.
- Downloads: scorecard JSON, config JSON, metrics CSV, and a simple
  text executive PDF (hand-rolled, stdlib). A full designed PDF is a
  later step.
- No login on the app. Synthetic demo data only; never put real data,
  credentials, or internal data in this repo or on the service.
