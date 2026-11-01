"""Deterministic chat orchestrator — the conversation state machine.

This is the ONLY place that decides what the assistant says next.
It never computes a number itself: every number comes from engine.py
or followups.py running on generated data.

LLM hook (requirement 5): an LLM added later implements ONE method,
``LLMAdapter.explain(prompt) -> str | None``, and is called ONLY to
add an optional explanation AFTER a deterministic answer. It never
chooses a scenario, never sets a config, never computes a metric, and
never replaces a deterministic block. Swapping in an LLM therefore
means wiring one adapter here — the statistical engine, the registry,
and the decision rules do not change. By default ``llm`` is None and
the product is fully deterministic.
"""
from __future__ import annotations

import re
import uuid

from . import followups
from .scenarios import FLAGSHIP_CONFIGS, INDUSTRIES, INDUSTRY_SCENARIOS, industry_by_id


class LLMAdapter:
    """Optional explainer layer. Return None to skip LLM text entirely."""

    def explain(self, prompt: str):  # pragma: no cover - hook only
        return None


def new_session():
    return {"id": str(uuid.uuid4())[:8], "messages": [], "state": "welcome",
            "industry": None, "scenario_id": None, "config": None,
            "modify_target": None, "run": None, "df": None, "result": None,
            "trend": None, "example_id": None}


def _text(t):
    return {"type": "text", "text": t}


def _chips(items):
    return {"type": "chips", "items": items}


def welcome_blocks():
    return [
        _text("Hi! I'm Experiment Copilot. I can help you design, simulate, analyze, and interpret "
              "experiments across different industries.\n\nWhich industry would you like to explore?"),
        {"type": "industry_cards", "industries": INDUSTRIES},
        _chips(["Explore the 14 example scenarios"]),
    ]


def scenario_cards_blocks(industry):
    titles = INDUSTRY_SCENARIOS[industry["id"]]
    cards = []
    for t in titles:
        cfg = FLAGSHIP_CONFIGS.get(t)
        if cfg:
            cards.append({"title": cfg["title"], "scenario_id": cfg["id"],
                          "subtitle": cfg["hypothesis"], "runnable": True})
        else:
            cards.append({"title": t, "scenario_id": None,
                          "subtitle": "Coming next — not runnable yet in Phase 1.",
                          "runnable": False})
    return cards


class Orchestrator:
    def __init__(self, examples):
        """examples: list of precomputed scenario dicts from scorecard_data.json."""
        self.examples = examples
        self.llm = None  # set to an LLMAdapter instance to enable the optional layer

    # ------------------------------------------------------------ helpers
    def _llm_note(self, prompt):
        if self.llm is None:
            return []
        try:
            out = self.llm.explain(prompt)
        except Exception:
            return []
        if out:
            return [_text("Optional LLM explanation (does not change any number above):\n" + out)]
        return []

    def _config_block(self, cfg):
        return {"type": "config", "config": cfg,
                "actions": ["Run Experiment", "Modify Design", "Explain Methodology"]}

    def _example_by_id(self, ex_id):
        for e in self.examples:
            if e["id"] == ex_id:
                return e
        return None

    # ------------------------------------------------------------ main entry
    def handle(self, session, message="", action=None, payload=None):
        """Returns a list of blocks to append to the conversation."""
        payload = payload or {}
        text = (message or "").strip()
        low = text.lower()

        # Global commands, available in any state.
        if low in ("new experiment", "start over", "new chat"):
            session.update({"state": "welcome", "industry": None, "scenario_id": None,
                            "config": None, "run": None, "example_id": None})
            return welcome_blocks()
        if low == "explore the 14 example scenarios" or action == "example_list":
            session["state"] = "examples"
            cards = [{"title": e["title"], "scenario_id": e["id"],
                      "subtitle": f"{e['verdict']} · n={e['n_users']:,}",
                      "runnable": True} for e in self.examples]
            return [_text("Here are the 14 existing statistical scenarios, preserved from the showroom. "
                          "These are precomputed demonstrations (1M users, seed 42) — the numbers are the "
                          "recorded engine output, not a fresh run. Pick one to open its scorecard."),
                    {"type": "scenario_cards", "cards": cards}]

        if action == "industry":
            ind = industry_by_id(payload.get("industry_id", ""))
            if not ind:
                return [_text("I don't recognise that industry. Pick one of the three cards, or explore the example scenarios.")]
            session["industry"] = ind["id"]
            session["state"] = "scenario"
            return [_text(f"Great! Let's explore {ind['title'].lower()} experimentation.\n\nHere are the realistic experiments you can run."),
                    {"type": "scenario_cards", "cards": scenario_cards_blocks(ind)},
                    _text("Or describe a custom experiment in your own words — I'll map it to the closest runnable scenario, or tell you honestly if it isn't supported yet.")]

        if action == "scenario":
            sid = payload.get("scenario_id")
            cfg = FLAGSHIP_CONFIGS.get(sid)
            if cfg:
                session["scenario_id"] = sid
                session["config"] = dict(cfg)
                session["config"]["seed"] = 42
                session["state"] = "config"
                return [_text(f"Let's test whether {cfg['hypothesis'][0].lower() + cfg['hypothesis'][1:]}"
                              f"\n\nHere's the experimental design I recommend."),
                        self._config_block(session["config"])]
            # Non-runnable scenario card clicked (no longer reachable from
            # the industry cards in Phase 2, kept as a guard).
            return [_text("That scenario isn't runnable — I won't fake a run for it. "
                          "The runnable options are the 12 industry scenarios, plus the 14 example scenarios. "
                          "Pick one of those, or tell me what you wanted to test and I'll point you to the closest match.")]

        if action == "example":
            ex = self._example_by_id(payload.get("scenario_id", ""))
            if not ex:
                return [_text("I couldn't find that example scenario.")]
            session["example_id"] = ex["id"]
            session["state"] = "example_done"
            return [{"type": "example_scorecard", "example": ex}]

        if session["state"] == "running":
            return [_text("Your experiment is still running — the stages in the progress card update as each real stage finishes. Ask follow-ups once it completes.")]
        if session["state"] == "config":
            return self._handle_config(session, text, action)
        if session["state"] == "modify_ask":
            return self._handle_modify_ask(session, text)
        if session["state"] == "modify_value":
            return self._handle_modify_value(session, text)
        if session["state"] in ("done", "example_done"):
            return self._handle_followup(session, text, low)
        if session["state"] == "scenario":
            return self._handle_custom(session, text, low)
        if session["state"] == "welcome":
            return self._handle_custom(session, text, low)
        return [_text("Pick an industry card to start, or explore the 14 example scenarios.")]

    # ------------------------------------------------------------ config state
    def _handle_config(self, session, text, action):
        cfg = session["config"]
        low = text.lower()
        if action == "run" or low == "run experiment":
            session["state"] = "running"
            return [{"type": "run_started", "scenario_id": session["scenario_id"]}]
        if action == "modify" or low == "modify design":
            session["state"] = "modify_ask"
            return [_text("What would you like to change? One thing at a time:"),
                    _chips(["Duration", "Seed"])]
        if action == "explain" or low == "explain methodology":
            # The unit sentence is scenario-specific: a generic geo sentence
            # in a user-level explanation is confusing (and wrong in context),
            # even though the underlying principle is the same.
            if "ZIP" in cfg["unit"] or "DMA" in cfg["unit"]:
                unit_note = ("The analysis unit is the randomization unit — here that is the "
                             "ZIP/DMA, never the impressions or users inside a geo counted as "
                             "independent units.")
            else:
                unit_note = ("The analysis unit is the randomization unit — here that is the "
                             "user, so each user counts once, and impressions or sessions from "
                             "the same user are never treated as independent experiments.")
            blocks = [_text(f"Methodology for {cfg['name']}:\n\n"
                            f"Design: {cfg['unit']}. Primary metric: {cfg['primary']}. "
                            f"Method: {cfg['method']}.\n\n"
                            f"Guardrails: {cfg['guardrails']}\n\n"
                            f"{unit_note}"),
                      self._config_block(cfg)]
            return blocks + self._llm_note(f"Explain the methodology of {cfg['name']}")
        return [_text("You can run this design, modify it, or ask me to explain the methodology."),
                self._config_block(cfg)]

    def _handle_modify_ask(self, session, text):
        low = text.lower()
        if "duration" in low:
            session["modify_target"] = "duration"
        elif "seed" in low:
            session["modify_target"] = "seed"
        elif "primary" in low or "metric" in low:
            return [_text("The primary metric is fixed by the scenario design — changing it would mean a different analysis, not just a relabel. The secondary and guardrail metrics are already in the scorecard's metrics table after a run, and you can ask segment and exclusion follow-ups about them. Pick duration or seed to change, or run this design."),
                    _chips(["Duration", "Seed"])]
        else:
            return [_text("Pick one to change:"), _chips(["Duration", "Seed"])]
        session["state"] = "modify_value"
        prompts = {"duration": "What duration should I use? Reply with a number of days, e.g. 14.",
                   "seed": "What seed should I use? Reply with a whole number, e.g. 7. Same seed, same data.",
                   "primary": f"Current primary metric is {session['config']['primary']}. "
                              f"Reply with a secondary metric name to promote it, or keep the current one."}
        return [_text(prompts[session["modify_target"]])]

    def _handle_modify_value(self, session, text):
        target = session.get("modify_target")
        cfg = session["config"]
        if target == "duration":
            m = re.search(r"\d+", text)
            if m:
                cfg["duration"] = f"{m.group(0)} days"
        elif target == "seed":
            m = re.search(r"\d+", text)
            if m:
                cfg["seed"] = int(m.group(0))
        elif target == "primary":
            if text.strip() and text.strip().lower() not in ("keep", "same", "current"):
                cfg["primary"] = text.strip()
        session["state"] = "config"
        session["modify_target"] = None
        return [_text("Updated. Here's the revised design:"), self._config_block(cfg)]

    # ------------------------------------------------------------ custom text
    def _handle_custom(self, session, text, low):
        if not text:
            return [_text("Tell me what you'd like to test, in your own words, or pick a card above.")]
        # One exception to platform-first: a named scenario topic beats
        # the platform. "LinkedIn ... top creators crowd out small
        # creators" is the crowding-out scenario, not the LinkedIn
        # connections scenario — the topic is more specific than the
        # platform here. Check it before the platform names.
        if "crowd" in low or "follower" in low or "top creator" in low:
            return self.handle(session, action="scenario", payload={"scenario_id": "linkedin_creator_crowding_out"})
        # Platform names win over topic words. The old order checked
        # "creator" before "linkedin", so "In LinkedIn, do top creators
        # crowd out small creators?" routed to Meta — the topic word
        # beat the platform the user actually named. Check every
        # platform first, then fall back to topic words.
        if "reddit" in low:
            return self.handle(session, action="scenario", payload={"scenario_id": "reddit_feed_ranking"})
        if "linkedin" in low:
            return self.handle(session, action="scenario", payload={"scenario_id": "linkedin_connections"})
        if "meta" in low:
            return self.handle(session, action="scenario", payload={"scenario_id": "meta_creator_recommendations"})
        if "tiktok" in low:
            return self.handle(session, action="scenario", payload={"scenario_id": "tiktok_notifications"})
        if "doordash" in low:
            return self.handle(session, action="scenario", payload={"scenario_id": "doordash_zip_promotion"})
        if "uber" in low:
            return self.handle(session, action="scenario", payload={"scenario_id": "uber_driver_incentives"})
        if "cross-publisher" in low or "cross publisher" in low or "budget allocation" in low:
            return self.handle(session, action="scenario", payload={"scenario_id": "cross_publisher_budget"})
        if "field" in low or "sampling" in low:
            return self.handle(session, action="scenario", payload={"scenario_id": "field_visits_sampling"})
        if "hcp" in low or "omnichannel" in low:
            return self.handle(session, action="scenario", payload={"scenario_id": "hcp_omnichannel"})
        if "dma" in low and ("advertis" in low or "marketplace" in low):
            return self.handle(session, action="scenario", payload={"scenario_id": "marketplace_dma_ads"})
        if "pharma" in low or "publisher" in low or "dma" in low:
            return self.handle(session, action="scenario", payload={"scenario_id": "pharma_dma_publisher"})
        # Topic-word fallbacks, only when no platform was named.
        if "feed" in low:
            return self.handle(session, action="scenario", payload={"scenario_id": "reddit_feed_ranking"})
        if "creator" in low:
            return self.handle(session, action="scenario", payload={"scenario_id": "meta_creator_recommendations"})
        if "notification" in low:
            return self.handle(session, action="scenario", payload={"scenario_id": "tiktok_notifications"})
        if "connection" in low:
            return self.handle(session, action="scenario", payload={"scenario_id": "linkedin_connections"})
        if "promotion" in low or "zip" in low:
            return self.handle(session, action="scenario", payload={"scenario_id": "doordash_zip_promotion"})
        if "driver incentive" in low:
            return self.handle(session, action="scenario", payload={"scenario_id": "uber_driver_incentives"})
        if "fee" in low:
            return self.handle(session, action="scenario", payload={"scenario_id": "delivery_fee_optimization"})
        if "visit" in low:
            return self.handle(session, action="scenario", payload={"scenario_id": "field_visits_sampling"})
        # Existing scenario by name (old chat behaviour, preserved).
        for e in self.examples:
            if e["title"].lower() in low or e["id"].replace("_", " ") in low:
                return self.handle(session, action="example", payload={"scenario_id": e["id"]})
        return [_text("I can't map that to a runnable experiment yet, and I won't guess a design for it. "
                      "The runnable options right now are the 12 industry scenarios — Reddit, Meta, TikTok, "
                      "LinkedIn, DoorDash, DMA advertising, Uber, delivery-fee, Pharma DMA, HCP omnichannel, "
                      "field visits, cross-publisher — plus the 14 example "
                      "scenarios. Which of those is closest to what you had in mind?")]

    # ------------------------------------------------------------ follow-ups
    def _handle_followup(self, session, text, low):
        if session["state"] == "example_done":
            ex = self._example_by_id(session.get("example_id", ""))
            if ex:
                return [_text(f"That example is a precomputed demonstration, so I can't recompute it live — "
                              f"its numbers are the recorded 1M-user engine output. What it shows: "
                              f"{ex['description']} Verdict: {ex['verdict']}. "
                              f"To run a live experiment, start a new experiment and pick a flagship scenario."),
                        _chips(["New experiment", "Explore the 14 example scenarios"])]
        if not session.get("result") or session.get("df") is None:
            return [_text("There's no completed live experiment in this conversation yet. "
                          "Run a flagship experiment first, then ask follow-ups about it."),
                    _chips(["New experiment"])]
        sid = session["scenario_id"]
        df, result, cfg = session["df"], session["result"], session["config"]
        m = re.search(r"exclude (?:the )?(\d+|five|ten) largest", low)
        if m or "exclude" in low and "largest" in low:
            n = 5
            if m:
                word = m.group(1)
                n = {"five": 5, "ten": 10}.get(word, int(word) if word.isdigit() else 5)
            out = followups.exclude_top(df, result, sid, n=n)
            return [_text(out["text"]), _chips(cfg["followups"])]
        # Descriptive largest/smallest question ("what are the five
        # largest DMAs?") — not an exclusion request, so it must not
        # fall through to the generic fallback. "Biggest" is a synonym
        # for "largest" here.
        if "largest" in low or "smallest" in low or "biggest" in low:
            n = 5
            _m = re.search(r"(\d+|five|ten)", low)
            if _m:
                _w = _m.group(1)
                n = {"five": 5, "ten": 10}.get(_w, int(_w) if _w.isdigit() else 5)
            out = followups.largest_units(df, result, sid, n=n, smallest=("smallest" in low))
            return [_text(out["text"]), _chips(cfg["followups"])]
        if "tier" in low or "gained" in low or "crowd" in low:
            out = followups.tier_shares(df, result, sid)
            return [_text(out["text"]), _chips(cfg["followups"])]
        if "driver" in low or "delivery" in low and "hurt" in low:
            out = followups.driver_availability(df, result, sid)
            return [_text(out["text"]), _chips(cfg["followups"])]
        if "publisher" in low and ("best" in low or "perform" in low):
            out = followups.publisher_best(df, result, sid)
            return [_text(out["text"]), _chips(cfg["followups"])]
        if "new customer" in low or "new user" in low:
            out = followups.segment(df, result, sid, "new")
            return [_text(out["text"]), _chips(cfg["followups"])]
        if "established" in low or "returning" in low:
            # Latent Phase 2 bug, fixed here: the old code returned
            # "returning" for every non-Reddit scenario, so "Show results
            # for established users" on Meta/TikTok/LinkedIn/crowding-out
            # (whose segment value IS "established") fell through to an
            # unsupported answer. Only the two marketplace scenarios
            # with new/returning customers use "returning".
            seg = "returning" if sid in ("doordash_zip_promotion", "delivery_fee_optimization") else "established"
            out = followups.segment(df, result, sid, seg)
            return [_text(out["text"]), _chips(cfg["followups"])]
        if "payer" in low:
            seg = "high" if "high" in low else "low"
            out = followups.segment(df, result, sid, seg)
            return [_text(out["text"]), _chips(cfg["followups"])]
        # Generic Phase 2 segment phrases (large/small cities, DMAs,
        # territories; low/high prior-engagement HCPs). Existing branches
        # above already handle new/established/returning/payer.
        for _tok in ("large", "small"):
            if _tok in low:
                out = followups.segment(df, result, sid, _tok)
                if out["supported"]:
                    return [_text(out["text"]), _chips(cfg["followups"])]
        if "prior engagement" in low:
            seg = "low" if "low" in low else "high"
            out = followups.segment(df, result, sid, seg)
            return [_text(out["text"]), _chips(cfg["followups"])]
        if "why" in low and "significant" in low:
            out = followups.why_significant(result, cfg)
            return [_text(out["text"]), _chips(cfg["followups"])]
        if "cuped" in low or "method" in low or "explain" in low:
            out = followups.explain_method(result, cfg)
            return [_text(out["text"]) ] + self._llm_note(f"Explain {cfg['name']} methodology") + [_chips(cfg["followups"])]
        if "caused" in low or "conclude" in low or "cause" in low:
            unit = cfg["unit"]
            return [_text(f"On causality: this experiment randomizes at {unit}, so a significant lift "
                          f"supports a causal claim about the treatment as delivered in this design — "
                          f"not about a brand, a platform, or a publisher in general. It does not prove "
                          f"why users responded, and it does not generalize beyond the population and "
                          f"period tested without further evidence."),
                    _chips(cfg["followups"])]
        if "new experiment" in low:
            return self.handle(session, message="New experiment")
        return [_text("I can't answer that from this experiment's data, so I won't guess. "
                      "Supported follow-ups for this experiment are: " + ", ".join(cfg["followups"]) + "."),
                _chips(cfg["followups"])]
