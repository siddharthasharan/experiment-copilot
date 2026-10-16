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
full eval harness.

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

## Roadmap

- [x] **Week 1** — synthetic data generator + stats engine + MCP tools *(this repo)*
- [ ] **Week 2** — RAG corpus: 15–25 short methodology docs (SRM, peeking,
      CUPED, guardrails, launch frameworks), written in your own words with
      citations; local embeddings via sentence-transformers + FAISS
- [ ] **Week 3** — memo agent (LLM turns a review into a launch memo) +
      eval harness: trap scenarios × rubric scoring, LLM-judge for memo quality
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
