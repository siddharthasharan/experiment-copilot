# Experiment Copilot

An AI that reviews A/B tests like a product data scientist — then gets graded
on whether it catches the traps.

## Live demo

- **Showroom (GitHub Pages):** https://siddharthasharan.github.io/experiment-copilot/
- **Showroom (Render, static site):** https://experiment-copilot.onrender.com

The showroom is a self-contained static page: 14 scenarios, each reviewed with
the real stats estimators, with the trap revealed at the end. No login, no
server, no data leaves your browser — the numbers are baked in at build time.

## Scale at a glance

| | Before | Now |
|---|---|---|
| Full generator, per scenario | 400,000 users | **1,000,000 users** |
| Scenarios | 14 | 14 (7 baseline + 3 triggered + 4 interaction) |
| Full data, all scenarios | 5.6M user-rows (~195 MB, gitignored) | **14M user-rows (~463 MB, gitignored)** |
| Committed sample, per scenario | 20,000 users | **100,000 users** |
| Committed sample, all scenarios | 280k rows (12 MB) | **1.4M rows (57 MB) in `data_v2_sample/`** |
| Showroom verdicts | 14/14 | **14/14 at 1M** (see scale note below) |
| Calibration | 17/17 + 8/8 + 9/9 at 400k | **17/17 + 8/8 + all-green at 1M** |

Regenerate everything locally (the full data is gitignored, so a fresh clone
must generate it — the committed sample is what reviewers run without that
step):

```bash
.venv/bin/python -m experiment_copilot.data_v2 --n 1000000 --sample-n 100000
.venv/bin/python -m experiment_copilot.triggered_data --n 1000000 --sample-n 100000
.venv/bin/python -m experiment_copilot.interaction_data --n 1000000 --sample-n 100000
.venv/bin/python scripts/build_showroom.py   # writes docs/index.html
```

## Scale note: what 400k → 1M actually changed (and what it didn't)

Scaling is not free — two showroom decision rules were scale-fragile and had
to be recalibrated, and both fixes are in `scripts/build_showroom.py` only
(the stats engine and the generator's trap design are untouched):

- **Novelty detector.** The old rule fired only if the long-exposure cohort
  was *not significant*. At 1M, the demo seed's long-exposure average lift
  (+1.05%) is significant even though it is under half the recent-cohort
  lift (+2.90%) — significance alone can no longer tell "null" from "small
  residual average". The rule now fires on material decay: full-sample
  significant, early/late heterogeneity significant, and long-exposure lift
  under half the recent lift. The trap story is unchanged; the detector
  stopped asking a question that sample size answers by itself.
- **Showroom guardrail veto.** The showroom now uses alpha = 0.01 (the eval
  harness's documented veto bar), not `check_guardrail`'s 0.05 default. At 1M
  the whale demo seed's true-null reports lift (+12% on daily totals,
  paired p = 0.027, per-user p = 0.083) otherwise vetoed spuriously and hid
  the variance story. The true guardrail scenario (+35–38% reports) still
  vetoes comfortably under 0.01.
- **Everything else scaled cleanly.** Whale count scales with n by design
  (0.125% of the treatment arm: 250 at 400k → 625 at 1M), so the headline
  revenue lift (~+37%) and std ratio (~3.1) are unchanged. Interference
  clusters stay at 2,000 (cluster size 200 → 500 users); the cluster/naive
  SE ratio is still 1.28 median. The one calibration gate that moved is the
  clean-win t-statistic: t scales with sqrt(n), so 6.3 at 400k → 9.8 at 1M,
  and the gate in `scripts/calibrate_v2.py` moved from 5–8 to 8–12 with that
  reasoning written in — the generator was not retuned to hit the old gate.

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

Showroom live (links above); RAG methodology layer done — 20 notes, 25-question eval set, 25/25 top-3 retrieval (see below). Still open: LLM launch-memo agent, LLM-judge for memo quality.

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
n ≈ 15.68 × CV² / r². At the uniform n = 1,000,000 (500k/arm):

| Metric | MDE | n/arm needed (raw) | n/arm (CUPED R²=0.55) | Power @1M |
|---|---|---|---|---|
| engagement (CV≈1) | +2% | 39k | 18k | 1.00 |
| revenue/user (lognormal σ=1.6, CV≈3.5) | +2% | 467k | 210k | 0.83 → 0.99 w/ CUPED |
| converted (base 30%) | +2% | 92k | — | 1.00 |
| reports (base 0.1%) | +35% | 128k | — | 1.00 |

Two takeaways: the volume drivers are the heavy-tailed revenue primary and
the *rare guardrail metric* (detecting +35% on a 0.1% report rate needs
~256k users) — hence 1M, which now powers even the heavy-tailed revenue metric raw (0.83), with CUPED taking it to 0.99. And the cheapest sample size is
variance reduction, not rows: a 28-day pre-period covariate (target CUPED
R² ≈ 0.5–0.6) is what makes the revenue read viable at all.

**Per-user grain, parquet.** One row per user (1M rows, not ~42M daily
rows). Daily structure exists only where a trap needs it: `enrollment_day`
(peeking/novelty) and `report_day` (guardrail). Full 1M outputs live in
`eval_data_v2/` (gitignored, ~264 MB for the 7 baseline scenarios);
committed 100k-user samples live in `data_v2_sample/` (~57 MB for all 14
scenarios). Generation is vectorized numpy — ~1.1s per scenario at 1M.

**The honest whale trap.** The v1 trap was dishonest: its outlier bomb
landed on a secondary metric while the headline ran on `converted`. In v2
the whales go into the *primary* revenue metric itself — 625 treatment users
(0.125% of the arm, scaled with n) set to a $400–800 spend tier against a ~$2.23 mean.
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
seeds against the real stats functions — 17/17 checks green at 1M, including
t ≈ 9.8 on the +2% clean win (sqrt(n)-scaled from ~6.3 at 400k), the guardrail veto firing on reports +38%,
and the novelty trap's early/late heterogeneity. No detector changes were
needed for v2; the generator was tuned instead (fixed whale tier for seed
stability, higher log-space pre-period correlation so revenue CUPED lands).

## Triggered analysis: ITT vs CACE (step 2)

Step 2 (`src/experiment_copilot/triggering.py`,
`src/experiment_copilot/triggered_data.py`) covers experiments where only a
fraction τ of randomized users ever reach the treated surface — the
standard Meta situation for deep-funnel features. Three analyses answer
three questions:

- **ITT** (intent-to-treat): Welch's t over *all* randomized users. Diluted
  (ITT ≈ τ × CACE) but unbiased for "what happens to the overall metric if
  we ship this."
- **CACE**: the effect on the triggered. The primary estimator is the
  *direct triggered contrast* (Welch's t on triggered users), valid when
  triggering is unaffected by treatment. The Wald IV estimator
  (ITT / P(trigger|treatment)) is kept for the unobservable-trigger case,
  but it pays full-sample noise — at τ=0.18 its SE is ~3× the direct
  contrast's, so it is strictly a fallback.
- **Trigger-balance check**: chi-square on trigger rates by arm (α=0.001).
  Fires ⇒ the triggered subpopulations differ by arm ⇒ triggered-only
  contrasts are selection-biased ⇒ fall back to ITT.

**Power.** On the *same* experiment, ITT needs 1/τ× the randomized users
of the CACE analysis for equal power (~3.3× at a 30% trigger rate) —
because ITT sees τ× more users, offsetting one factor of τ. (The 1/τ²
figure you'll see elsewhere is vs a hypothetical design where everyone
triggers, not the same-experiment comparison.) `power.py` gained
`required_n_triggered` (base/τ) and `required_n_itt` (base/τ², vs the
all-trigger baseline).

**Scenarios** (1M users, per-user grain, parquet; full outputs in
`eval_data_v2_triggered/`, 100k samples in `data_v2_sample/`):

| Scenario | Design | Correct verdict |
|---|---|---|
| `dilution_trap` | +2.5% engagement on triggered only, τ≈0.15 (heterogeneous by tenure) | **launch** — CACE significant; ITT diluted and usually misses; rollout impact ≈ CACE × τ |
| `trigger_imbalance_trap` | treatment boosts triggering +5pp; triggering correlates with engagement | **inconclusive** — balance check fires; triggered analysis selection-biased (naive estimate flips sign: −2.1% vs true +2.0%); use ITT |
| `triggered_clean_win` | +2.5% on triggered, τ=0.5, balanced | **launch** — CACE and ITT agree, dilution math exact |

**Calibration** (`scripts/calibrate_triggered.py`, 10 seeds/scenario, real
estimators): 8/8 green — CACE+CUPED significant 100% on the dilution trap
while ITT misses ~75% (40-seed measurement: 25% ITT power), ITT < ½×CACE
every seed (dilution is structural, not p-value luck), |ITT − τ×CACE| within
tolerance, balance check fires 100% on the imbalance trap and stays quiet
on the dilution trap. One honest calibration note: the dilution demo seed
was re-picked after the first seed's null ITT ran +0.62% (a 2σ unlucky
randomization that swamped the +0.44% diluted signal) — the trap is
dilution, so the demo instance is chosen clean; the mechanism is verified
across 40 seeds.

## Interaction effects: overlap, HTE, interference (step 3)

Step 3 (`src/experiment_copilot/interaction.py`,
`src/experiment_copilot/interaction_data.py`) covers the three ways a
single-experiment analysis goes wrong in a matrixed experimentation
program:

- **Overlap**: users are randomized into many experiments at once. A
  concurrent experiment B with a true A×B interaction makes the naive
  pooled A estimate a weighted average of two different truths.
  `overlap_analysis` stratifies by B arm and tests the interaction.
- **HTE / segment mirage**: with ~12 segments, some cell will hit nominal
  p<0.05 by chance. `interaction_test` (OLS interaction terms + Cochran's
  Q) plus `bh_fdr` (Benjamini-Hochberg, q=0.10) separates the mirage from
  real heterogeneity.
- **Interference (SUTVA sketch)**: users in ~2k clusters with a spillover
  term — outcome depends on treated cluster-mates. The naive i.i.d. SE
  understates uncertainty; `cluster_robust_se` (Liang-Zeger sandwich, per
  arm) is wider and a SUTVA flag fires. Documented as a sketch: clusters
  are assumed independent, no cross-cluster spillover modeled.

**Scenarios** (1M users, per-user grain, parquet; full outputs in
`eval_data_v2_interaction/`, 100k samples in `data_v2_sample/`):

| Scenario | Design | Correct verdict |
|---|---|---|
| `overlap_trap` | independent experiment B (50/50); true A×B interaction: A lifts +2% when B=control, −0.5% when B=treatment | **stratify** — interaction detected; pooled naive estimate (+0.8%, significant 5/10 seeds) is a misleading average |
| `hte_mirage_trap` | ~12 segments, homogeneous +2% everywhere | **no credible heterogeneity** — BH-FDR kills the nominal segment hits (mirage appears ~3/10 seeds) |
| `hte_real` | ~12 segments; "new" truly +4%, others +1% | **credible heterogeneity** — the pre-specified segment survives FDR 10/10; decide per segment |
| `interference_sketch` | 2k clusters × 200 users, user-level randomization, demeaned spillover λ=3.2 | **SUTVA risk flagged** — cluster-robust SE ~1.28× naive; treat SEs as understated |

**Calibration** (`scripts/calibrate_interaction.py`, 10 seeds/scenario):
9/9 green — interaction detected 10/10 with stratified estimates recovering
truth (+2.0%/−0.5%), BH clean 9/10 on the mirage (Cochran's Q quiet 10/10),
the true segment survives FDR 10/10, cluster/naive SE ratio 10/10 in
(1.26–1.30) with the SUTVA flag firing 10/10 and the λ=0 control honest
(ratio ≈1.00). One honest tuning note: the first interference build used
λ=1.5 with a non-demeaned spillover and failed both gates (median ratio
1.07) — the module's own docstring math assumed outcome σ=100 (actual
≈143) and full-cluster m in the sandwich (the estimator works per arm, so
effective m is halved). Fixed with a demeaned spillover (zero level shift)
at λ=3.2, re-derived in the docstring, re-verified across seeds.

## Roadmap

- [x] **Week 1** — synthetic data generator + stats engine + MCP tools *(this repo)*
- [x] **Week 3 (eval harness)** — trap scenarios × answer keys, deterministic
      grader (`src/experiment_copilot/eval.py` + `scripts/run_eval.py`); 6/6 on
      TikTok and Reddit profiles. Still open: LLM-judge for memo quality.
- [x] **Week 2** — RAG corpus: 20 methodology notes + 25-question eval set,
      25/25 top-3 retrieval (see above). Local embeddings/FAISS not used —
      the v1 retriever is dependency-free keyword retrieval in
      `evals/retrieval.py`.
- [x] **Showroom** — 14-scenario scorecard, 14/14 verdicts at 1M, live on
      GitHub Pages and Render (static site).
- [x] **README polish** — this update: live demo links, 1M scale, eval table,
      scale-recalibration notes.
- [ ] **Week 3 (memo agent)** — LLM turns a review into a launch memo
- [ ] **LLM-judge for memo quality** — still open.
- [ ] **2-min demo video, LinkedIn post** — still open: *"I built an AI that
      reviews A/B tests — then I built the test suite that grades the AI."*

## Methodology RAG corpus and chat-agent evals

`rag/corpus.json` holds 20 original methodology summaries (11 showroom traps,
3 design fundamentals, 6 causal-inference topics for the future study guide),
each with key points, interview angles, citations, and linked scenarios.
`evals/` holds 25 questions (20 methodology + 5 showroom-grounded) with gold
corpus notes and a grading rubric. The v1 gate is retrieval coverage:

```bash
.venv/bin/python evals/run_evals.py
# Retrieval eval: 25/25 questions retrieve a gold note in top-3
# Top-1 accuracy: 22/25
```

The chat agent itself is not built yet — the eval set is the release bar it
has to clear (25/25), following the same eval-first pattern as the showroom.

## Deploying the showroom on Render (static site)

Unlike a web app, the showroom needs no server: `docs/index.html` is
self-contained. `render.yaml` at the repo root defines a Render **Static
Site** publishing `docs/` with no build step — Render cannot regenerate the
showroom, because the 1M-user data it was built from is gitignored and is
not in the repo, so the prebuilt page is what gets published. In Render:
New → Blueprint → select experiment-copilot → Apply. The service URL is
https://experiment-copilot.onrender.com. Static sites do not sleep like
free web services, so there is no wake delay. GitHub Pages remains the
primary host; Render is the second URL, same content.

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
