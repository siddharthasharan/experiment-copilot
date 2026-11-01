"""Platform tests — run with: python copilot_platform/tests/test_platform.py
(no pytest needed; plain asserts with PASS/FAIL lines)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from copilot_platform.chat import Orchestrator, new_session, welcome_blocks
from copilot_platform.engine import run_flagship, decide
from copilot_platform import followups
from copilot_platform.scorecard import build_scorecard


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise AssertionError(name)


def main():
    df1, r1 = run_flagship("reddit_feed_ranking", seed=42)
    df2, r2 = run_flagship("reddit_feed_ranking", seed=42)
    check("reddit seed reproducible", df1.equals(df2) and r1["primary"]["lift_pct"] == r2["primary"]["lift_pct"])
    df3, _ = run_flagship("reddit_feed_ranking", seed=7)
    check("reddit seed changes data", not df1.equals(df3))
    check("srm fail -> invalid", decide({"significant": True, "lift_pct": 5.0}, False, True) == "INVALID")
    check("guardrail fail -> dont launch", decide({"significant": True, "lift_pct": 5.0}, True, False) == "DONT_LAUNCH")
    check("significant positive -> launch", decide({"significant": True, "lift_pct": 5.0}, False, False) == "LAUNCH")
    check("non-significant -> inconclusive", decide({"significant": False, "lift_pct": 5.0}, False, False) == "INCONCLUSIVE")
    check("srm balanced at default", r1["srm"]["failed"] is False)
    _, dd = run_flagship("doordash_zip_promotion", seed=42)
    check("doordash seed42 significant orders", dd["primary"]["significant"] is True)
    check("doordash seed42 contribution negative -> dont launch",
          dd["verdict"] == "DONT_LAUNCH" and dd["business"]["contribution_usd"] < 0)
    _, ph = run_flagship("pharma_dma_publisher", seed=42)
    check("pharma seed42 launch", ph["verdict"] == "LAUNCH")
    df_dd, res_dd = run_flagship("doordash_zip_promotion", seed=42)
    ex = followups.exclude_top(df_dd, res_dd, "doordash_zip_promotion", 5)
    check("exclude-top recomputes", ex["supported"] and ex["new_lift_pct"] != ex["old_lift_pct"] and ex["n_remaining"] == 195)
    seg = followups.segment(df_dd, res_dd, "doordash_zip_promotion", "new")
    check("segment recomputes", seg["supported"] and abs(seg["lift_pct"] - res_dd["primary"]["lift_pct"]) < 5)
    pub = followups.publisher_best(df_dd, res_dd, "doordash_zip_promotion")
    check("publisher unsupported outside pharma, honestly", pub["supported"] is False)
    df_ph, res_ph = run_flagship("pharma_dma_publisher", seed=42)
    pub2 = followups.publisher_best(df_ph, res_ph, "pharma_dma_publisher")
    check("publisher shares sum to 100", pub2["supported"] and abs(sum(p["share_pct"] for p in pub2["publishers"]) - 100) < 0.1)
    check("publisher caveat present", "cannot identify" in pub2["text"])
    orch = Orchestrator([])
    s = new_session()
    assert welcome_blocks()
    blocks = orch.handle(s, action="industry", payload={"industry_id": "marketplace"})
    check("chat industry -> scenario cards", any(b["type"] == "scenario_cards" for b in blocks))
    blocks = orch.handle(s, action="scenario", payload={"scenario_id": "doordash_zip_promotion"})
    check("chat scenario -> config card", any(b["type"] == "config" for b in blocks) and s["state"] == "config")
    orch.handle(s, action="modify", payload={})
    check("chat modify asks one question", s["state"] == "modify_ask")
    orch.handle(s, message="Seed")
    orch.handle(s, message="7")
    check("chat modify seed updates config", s["config"]["seed"] == 7 and s["state"] == "config")
    from copilot_platform.scenarios import FLAGSHIP_CONFIGS
    cfg = dict(FLAGSHIP_CONFIGS["doordash_zip_promotion"])
    cfg["seed"] = 42
    df_dd, res_dd = run_flagship("doordash_zip_promotion", seed=42)
    trend = res_dd.pop("trend")
    sc = build_scorecard(cfg, res_dd, trend, 42)
    check("scorecard lift matches engine", abs(sc["primary_lift_pct"] - res_dd["primary"]["lift_pct"]) < 1e-9)
    check("scorecard verdict matches engine", sc["verdict"] == res_dd["verdict"])
    print("All platform tests passed.")


if __name__ == "__main__":
    main()
