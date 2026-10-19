# Chat-agent evals

Eval-driven development for the Experiment Copilot chat agent: define what
"correct" looks like *before* building the agent, then build the agent to
clear the bar.

## Files

| file | purpose |
|---|---|
| `eval_questions.json` | 25 questions, each with category, source, gold corpus notes, and `expected_key_points` (the grading rubric) |
| `reference_answers.json` | human-written reference answer per question — the current definition of correct |
| `retrieval.py` | dependency-free keyword retriever over `rag/corpus.json` |
| `run_evals.py` | v1 harness: retrieval coverage gate |

## Question set

- **20 methodology questions** from the interview question bank, section 3
  (10 experiment design + 10 causal inference). These double as study-guide
  practice questions later.
- **5 showroom-grounded questions** (`category: scenario`): the agent's core job —
  reasoning about a *specific* experiment's readout (peeking trap, whale trap,
  dilution trap, trigger imbalance, guardrail veto).

## What v1 measures

`python evals/run_evals.py` checks that every question retrieves at least one
of its gold-standard corpus notes in the top-3. This gates the RAG layer:
the corpus must cover the eval set and the retriever must find it.

Rubric validation is done by construction: every `expected_key_points` entry
was checked to be substantively addressed by its reference answer
(see the validation step in the build log).

## Scoring rubric (for the agent, when built)

Each expected key point scores 0 (absent), 1 (partial/implied), or 2 (stated
correctly). A passing answer needs full marks on every key point — a correct
verdict with the wrong reasoning still fails.

## Roadmap

1. **v1 (now):** retrieval coverage gate + rubric-validated reference answers.
2. **v2:** run the deterministic chat agent over the questions; score key-point
   coverage with keyword matching against reference answers.
3. **v3:** LLM-judge layer for answer quality (fluency, calibration of
   uncertainty) on top of the key-point rubric.
