"""Scenario registry — the single source of truth for chat cards.

A scenario is a dict the chat can render without knowing the engine:
industry, title, hypothesis, design fields, guardrails, method, and
the follow-ups it supports. Adding an LLM later means adding a new
planner that produces these same dicts — the engine and the registry
do not change.

Existing 14 scenarios are NOT redefined here: they are loaded from
docs/scorecard_data.json (the recorded 1M-user engine output) and
presented as Example Scenarios, clearly labelled precomputed.
"""
from __future__ import annotations

INDUSTRIES = [
    {"id": "social", "title": "Social Media & Consumer Platforms",
     "examples": "Reddit, Meta, Instagram, Facebook, TikTok, LinkedIn",
     "focus": "Feed ranking, engagement, retention, creator ecosystems, network effects, and monetization."},
    {"id": "marketplace", "title": "Two-Sided Marketplaces",
     "examples": "DoorDash, Uber, Uber Eats, Zomato",
     "focus": "Customer demand, merchant supply, driver availability, pricing, promotions, and geographic experimentation."},
    {"id": "pharma", "title": "Regulated Pharma & Omnichannel",
     "examples": "Pharmaceutical commercial analytics",
     "focus": "DTC advertising, digital publishers, HCP outreach, field visits, sampling, and patient outcomes."},
]

FLAGSHIP_CONFIGS = {
    "reddit_feed_ranking": {
        "id": "reddit_feed_ranking", "industry": "social",
        "title": "Reddit feed-ranking and meaningful engagement",
        "name": "Reddit feed-ranking experiment",
        "hypothesis": "A new feed ranking that surfaces higher-quality posts will increase meaningful replies per active user, without increasing reports.",
        "treatment": "New ranking (quality-weighted, diversity-aware)",
        "control": "Current ranking",
        "unit": "Users (randomized at user level)",
        "population": "20,000 active Reddit users (default interactive size; seed 42)",
        "baseline": "Pre-period replies per user, same users",
        "duration": "28 days",
        "primary": "Meaningful replies per active user",
        "secondary": ["Reports per user (guardrail)", "Creator diversity (guardrail)"],
        "guardrails": "Reports per user must not rise significantly by more than 10%; SRM must pass.",
        "method": "CUPED with pre-period replies; Welch 95% CI; SRM chi-square check",
        "followups": ["Why is this statistically significant?", "Explain CUPED",
                      "Show results for new customers", "Show results for established users"],
    },
    "doordash_zip_promotion": {
        "id": "doordash_zip_promotion", "industry": "marketplace",
        "title": "DoorDash ZIP-level delivery promotion",
        "name": "DoorDash $5 delivery promotion",
        "hypothesis": "A $5 delivery promotion in treated ZIPs will generate incremental orders profitably — that is, contribution after promotion costs stays positive.",
        "treatment": "$5 delivery promotion in treated ZIPs",
        "control": "No promotion",
        "unit": "ZIP codes (randomized at ZIP level; ZIPs are the analysis units)",
        "population": "200 ZIPs across 10 markets (default interactive size; seed 42)",
        "baseline": "Pre-period orders per ZIP (28 days before)",
        "duration": "28 days",
        "primary": "Orders per ZIP",
        "secondary": ["Delivery time (guardrail)", "Cancellation rate (guardrail)", "Contribution margin"],
        "guardrails": "Delivery time must not rise significantly by more than 10%; cancellations must not rise significantly by more than 20%; SRM must pass.",
        "method": "Pre-period adjusted ZIP-level analysis; Welch 95% CI across ZIPs; SRM chi-square check",
        "followups": ["Why is this statistically significant?", "Exclude the five largest markets",
                      "Show results for new customers", "Did the promotion hurt driver availability?"],
    },
    "pharma_dma_publisher": {
        "id": "pharma_dma_publisher", "industry": "pharma",
        "title": "DTC publisher advertising incrementality",
        "name": "Pharma DMA publisher incrementality",
        "hypothesis": "Increasing publisher allocation in treated DMAs will increase treatment initiations, after adjusting for claims reporting lag.",
        "treatment": "Increased publisher allocation in treated DMAs",
        "control": "Baseline publisher allocation",
        "unit": "DMAs (randomized at DMA level; DMAs are the analysis units, NOT impressions)",
        "population": "200 DMAs (default interactive size; seed 42)",
        "baseline": "Pre-period treatment starts per DMA",
        "duration": "56 days (8 weeks, for delayed treatment outcomes)",
        "primary": "Treatment starts per DMA (claims-adjusted)",
        "secondary": ["Claims completeness (guardrail)", "Cost per incremental treatment start"],
        "guardrails": "Claims completeness must not differ significantly by arm by more than 3%; SRM must pass.",
        "method": "Pre-period adjusted DMA-level analysis; claims-completeness adjustment; Welch 95% CI across DMAs",
        "followups": ["Why is this statistically significant?", "Exclude the five largest markets",
                      "Which publisher performed best?", "Show results for high payer access DMAs"],
    },
}

# Remaining scenarios per industry, as listed in the spec. These are
# selectable in chat; in Phase 1 they are offered as "coming next"
# with an honest explanation, rather than faked as runnable — the
# three flagships above are the runnable ones, plus the 14 examples.
INDUSTRY_SCENARIOS = {
    "social": ["reddit_feed_ranking",
               "Meta creator recommendations and retention",
               "TikTok personalized notifications and novelty decay",
               "LinkedIn connection recommendations and network activation"],
    "marketplace": ["doordash_zip_promotion",
                    "DMA-level digital advertising incrementality",
                    "Uber driver incentives and marketplace supply",
                    "Delivery-fee optimization and contribution margin"],
    "pharma": ["pharma_dma_publisher",
               "HCP omnichannel engagement",
               "Field visits and sampling effectiveness",
               "Cross-publisher marketing budget allocation"],
}


def industry_by_id(industry_id):
    for ind in INDUSTRIES:
        if ind["id"] == industry_id:
            return ind
    return None


def flagship_config(scenario_id):
    return FLAGSHIP_CONFIGS.get(scenario_id)
