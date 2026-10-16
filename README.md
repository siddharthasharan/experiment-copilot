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
