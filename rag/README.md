# Methodology RAG corpus

Original methodology summaries that ground the Experiment Copilot chat agent.
Each note is written from scratch (not copied from any source) and carries
citations to the canonical references, so the agent can cite real sources
instead of hallucinating them.

## Schema

Each note in `corpus.json`:

| field | purpose |
|---|---|
| `id` | stable key, referenced by eval questions |
| `title` | human-readable topic name |
| `summary` | the authoritative explanation the agent draws on (~100 words) |
| `key_points` | the atomic facts an answer is graded against |
| `interview_angle` | how this topic shows up in data-science interviews — seed of the future study guide |
| `citations` | canonical real-world references (books, papers) |
| `related_scenarios` | showroom scenario ids this note explains |
| `keywords` | retrieval triggers, weighted highest by the retriever |

## Coverage

20 notes across three groups:

- **Showroom traps (11):** SRM, peeking, novelty, whale variance, CUPED, guardrails,
  triggering (ITT vs CACE), trigger imbalance, overlapping experiments, HTE/FDR,
  interference/SUTVA — each linked to the scenario(s) demonstrating it.
- **Design fundamentals (3):** power/MDE, ratio metrics, cluster randomization,
  contamination.
- **Causal inference for the study guide (6):** difference-in-differences,
  matching/propensity scores, regression discontinuity, interrupted time series,
  Simpson's paradox. These have no showroom scenario yet — they are the bridge
  to the interview study guide.

## For the showroom chat UI

Copy `rag/corpus.json` to `docs/corpus.json` so the static page can fetch it.
The retriever logic in `evals/retrieval.py` is intentionally dependency-free so
it can be mirrored in vanilla JS with identical scoring.

## Extending toward the study guide

Add notes the same way: original summary, key points, interview angle,
citations. SQL, Python, and behavioral sections from the interview bank slot
in as new groups without changing the schema.
