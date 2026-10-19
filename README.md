# Experiment Copilot

An AI that reviews A/B tests like a product data scientist — then gets graded
on whether it catches the traps.

Existing analytics MCP servers (Amplitude, PostHog, GrowthBook) are thin data
pipes: they fetch numbers but do none of the *judgment*. This project is the
statistician's brain: SRM checks, peeking/novelty detection, CUPED, segment
interaction tests, guardrail thinking — exposed as MCP tools, grounded in a
methodology knowledge base, with an eval harness that plants known traps and
scores whether the agent catches them.

## Architecture

```
┌─────────────────────┐     ┌──────────────────────┐
│  synthetic data      │     │  MCP server (FastMCP) │
│  generator (data.py) │────▶│  analyze_experiment   │
│  SRM / peeking /     │     │  design_review        │
│  novelty / variance  │     │  power_analysis       │
│  traps, known truth  │     └──────────┬───────────┘
└─────────────────────┘                │
                          ┌────────────▼────────────┐
                          │  stats engine (stats.py) │
                          │  SRM χ² · Welch's t ·    │
                          │  CUPED · interaction     │
                          │  tests · decay/peeking   │
                          │  trajectory analysis     │
                          └─────────────────────────┘
```

Coming next (roadmap below): RAG methodology layer, LLM launch-memo agent,
LLM-judge for memo quality.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate the 6 trap scenarios and run the full review on each —
# the seed of the eval harness. Expect 6/6.
python scripts/quickstart.py
```

Example output for the SRM trap:

```
Scenario: srm_trap
Verdict: INVALID

SRM: DETECTED (p=1.2e-109, split=0.550)
Lift: +5.41% (95% CI [...], p=...)
...
Flags:
  ! SRM DETECTED ... Randomization is broken — do not trust any lift estimate.
```

## Use it as an MCP server

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "experiment-copilot": {
      "command": "/path/to/experiment-copilot/.venv/bin/python",
      "args": ["-m", "experiment_copilot.server"],
      "cwd": "/path/to/experiment-copilot"
    }
  }
}
```

Then ask Claude things like:
- "Review the experiment in `data/srm_trap.csv` — should we launch?"
- "Critique this experiment plan: {…}"
- "How many users per group do I need to detect a +5% lift on a 10% base rate?"

## The six seed scenarios

| Scenario | Truth | Correct verdict |
|---|---|---|
| `clean_win` | +5% real lift | launch |
| `clean_null` | no effect | don't launch |
| `srm_trap` | 55/45 allocation skew | invalid |
| `peeking_trap` | false significance on day 4, null at horizon | inconclusive |
| `novelty_trap` | +33% early, decays to ~null | inconclusive |
| `variance_trap` | +6% lift but outlier-inflated metric | inconclusive |

## The platform eval harness

`scripts/run_eval.py` grades the reviewer against the *platform* generator's
trap scenarios (`synthetic_data/generator.py`) — five platform profiles
(Facebook, Instagram, TikTok, Reddit, LinkedIn) × three metric families
(flagship, ads, trust & safety), each with a hidden `experiment.json` answer
key the reviewer never sees:

```bash
.venv/bin/python scripts/run_eval.py --platform tiktok --n-users 20000 --days 14
```

For each scenario the harness aggregates daily rows to per-user outcomes,
runs the checklist (SRM → guardrail veto → peeking/novelty/variance flags →
verdict), and grades the verdict plus per-trap detection against the answer
key. Datasets cache under `eval_data/` (gitignored); results land in
`eval_data/<platform>_results_s<seed>.json`.

| Scenario | Injected trap | Correct verdict |
|---|---|---|
| `clean_win` | none (+4–6% real lift) | launch |
| `srm` | 55/45 allocation skew | invalid |
| `novelty_decay` | lift fades exp(−day/5) | inconclusive |
| `guardrail_violation` | engagement up, reports up 35% | don't launch |
| `whale_variance` | 8 users at 40× activity | inconclusive |
| `null_result` | none (noise) | don't launch |

Two calibration notes baked into `src/experiment_copilot/eval.py`: the
variance check leans on the treatment/control std ratio (50%+ inflation
flags) rather than the 5-SD share rule, because healthy lognormal engagement
tails trip the share rule on their own; and the guardrail veto requires both
statistical (p < 0.01, paired daily t-test on reports) and practical (>10%
lift) significance, so rare-event noise can't veto a clean launch.

## v2 data design: baseline metrics at realistic MDEs

Step 1 of v2 replaces the 5%-lift demo data with Meta-scale baseline metrics
at realistic 1–2% MDEs (`src/experiment_copilot/data_v2.py`,
`src/experiment_copilot/power.py`). Key design choices:

**Power table** — for a two-arm test at α=0.05, 80% power, per-arm
n ≈ 15.68 × CV² / r². At the uniform n = 400,000 (200k/arm):

| Metric | MDE | n/arm needed (raw) | n/arm (CUPED R²=0.55) | Power @400k |
|---|---|---|---|---|
| engagement (CV≈1) | +2% | 39k | 18k | 1.00 |
| revenue/user (lognormal σ=1.6, CV≈3.5) | +2% | 467k | 210k | 0.45 → 0.78 w/ CUPED |
| converted (base 30%) | +2% | 92k | — | 0.99 |
| reports (base 0.1%) | +35% | 128k | — | 0.94 |

Two takeaways: the volume drivers are the heavy-tailed revenue primary and
the *rare guardrail metric* (detecting +35% on a 0.1% report rate needs
~256k users) — hence 400k, not millions. And the cheapest sample size is
variance reduction, not rows: a 28-day pre-period covariate (target CUPED
R² ≈ 0.5–0.6) is what makes the revenue read viable at all.

**Per-user grain, parquet.** One row per user (400k rows, not ~17M daily
rows). Daily structure exists only where a trap needs it: `enrollment_day`
(peeking/novelty) and `report_day` (guardrail). Full 400k outputs live in
`eval_data_v2/` (gitignored); committed 20k-user samples live in
`data_v2_sample/`. Generation is vectorized numpy — ~0.4s per scenario.

**The honest whale trap.** The v1 trap was dishonest: its outlier bomb
landed on a secondary metric while the headline ran on `converted`. In v2
the whales go into the *primary* revenue metric itself — 250 treatment users
(0.125% of the arm) set to a $400–800 spend tier against a ~$2.23 mean.
Headline revenue reads ~+36% with p ≈ 0; the std-ratio variance flag
(~3.0) catches it 100% of seeds. That is the actual Meta-scale whale case.

**Detector finding (measured, not opinion).** Running the real
`check_variance_health` across seeds: the outlier-share rule has 0%
false-positives on 28-day per-user totals but 100% on raw lognormal
observations — so it is grain-dependent and must stay gated to aggregated
metrics. On the honest whale trap the share rule is blind (0% at the
overnight operating point) while the std-ratio rule (bar 2.0) catches
90–100%. The std-ratio is the primary signal; the share rule is secondary.

**Calibration.** `scripts/calibrate_v2.py` verifies all 7 scenarios × 10
seeds against the real stats functions — 17/17 checks green, including
t ≈ 6 on the +2% clean win, the guardrail veto firing on reports +38%,
and the novelty trap's early/late heterogeneity. No detector changes were
needed for v2; the generator was tuned instead (fixed whale tier for seed
stability, higher log-space pre-period correlation so revenue CUPED lands).

## Roadmap

- [x] **Week 1** — synthetic data generator + stats engine + MCP tools *(this repo)*
- [x] **Week 3 (eval harness)** — trap scenarios × answer keys, deterministic
      grader (`src/experiment_copilot/eval.py` + `scripts/run_eval.py`); 6/6 on
      TikTok and Reddit profiles. Still open: LLM-judge for memo quality.
- [ ] **Week 2** — RAG corpus: 15–25 short methodology docs (SRM, peeking,
      CUPED, guardrails, launch frameworks), written in your own words with
      citations; local embeddings via sentence-transformers + FAISS
- [ ] **Week 3 (memo agent)** — LLM turns a review into a launch memo
- [ ] **Week 4** — README polish, architecture diagram, 2-min demo video,
      LinkedIn post: *"I built an AI that reviews A/B tests — then I built the
      test suite that grades the AI."*

## Notes on method

- Welch's t-test throughout (unequal variances are the norm, not the exception).
- SRM uses α=0.001: with big samples the χ² test is extremely powerful, and a
  marginal imbalance rarely invalidates anything — a hard mismatch means
  something is broken upstream.
- Heterogeneity is tested with an interaction test on segment lifts, not by
  counting significant segments (small segments miss significance on power
  alone).
- Peeking is treated as a *process* fact: pass `n_interim_looks` when the team
  monitored the experiment with interim decisions. The trajectory check catches
  the statistical signature (early significance that doesn't survive the
  horizon); the explicit parameter catches the reality.
