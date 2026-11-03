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
    # Phase 2: all 9 new scenarios run, reproduce, and scorecard-match.
    from copilot_platform.scenarios import FLAGSHIP_CONFIGS as _CFG, INDUSTRY_SCENARIOS as _IND
    _new = ["meta_creator_recommendations", "tiktok_notifications", "linkedin_connections",
            "marketplace_dma_ads", "uber_driver_incentives", "delivery_fee_optimization",
            "hcp_omnichannel", "field_visits_sampling", "cross_publisher_budget"]
    check("all 16 industry scenarios registered", len(_CFG) == 16 and
          all(s in _CFG for ind in _IND.values() for s in ind))
    for _sid in _new:
        _a, _ra = run_flagship(_sid, seed=42)
        _b, _rb = run_flagship(_sid, seed=42)
        check(f"{_sid} seed reproducible", _a.equals(_b) and _ra["primary"]["lift_pct"] == _rb["primary"]["lift_pct"])
        _c, _ = run_flagship(_sid, seed=7)
        check(f"{_sid} seed changes data", not _a.equals(_c))
        check(f"{_sid} srm passes at seed 42", _ra["srm"]["failed"] is False)
        _cfg = dict(_CFG[_sid]); _cfg["seed"] = 42
        _trend = _ra.pop("trend")
        _sc = build_scorecard(_cfg, _ra, _trend, 42)
        check(f"{_sid} scorecard lift matches engine", abs(_sc["primary_lift_pct"] - _ra["primary"]["lift_pct"]) < 1e-9)
        check(f"{_sid} scorecard verdict matches engine", _sc["verdict"] == _ra["verdict"])
        check(f"{_sid} scorecard segment matches engine",
              all(abs(s["lift_pct"] - next(x for x in _ra["segments"] if x["segment"] == s["segment"])["lift_pct"]) < 1e-9
                  for s in _sc["segments"]))
    _df_u, _r_u = run_flagship("uber_driver_incentives", seed=42)
    _drv = followups.driver_availability(_df_u, _r_u, "uber_driver_incentives")
    check("uber driver utilization follow-up", _drv["supported"] and "utilization" in _drv["text"])
    _df_f, _r_f = run_flagship("delivery_fee_optimization", seed=42)
    check("fee contribution negative -> dont launch",
          _r_f["verdict"] == "DONT_LAUNCH" and _r_f["business"]["contribution_usd"] < 0)
    _ex = followups.exclude_top(_df_f, _r_f, "delivery_fee_optimization", 5)
    check("fee exclude-top recomputes", _ex["supported"] and _ex["n_remaining"] == 195)
    _df_x, _r_x = run_flagship("cross_publisher_budget", seed=42)
    _pub = followups.publisher_best(_df_x, _r_x, "cross_publisher_budget")
    check("cross-publisher shares sum to 100", _pub["supported"] and abs(sum(p["share_pct"] for p in _pub["publishers"]) - 100) < 0.1)
    check("cross-publisher caveat present", "cannot identify" in _pub["text"])
    _df_h, _r_h = run_flagship("hcp_omnichannel", seed=42)
    _seg = followups.segment(_df_h, _r_h, "hcp_omnichannel", "low")
    check("hcp low segment recomputes", _seg["supported"] and abs(_seg["lift_pct"] - next(x for x in _r_h["segments"] if x["segment"] == "low")["lift_pct"]) < 1e-9)
    # Routing regression (2026-10-31): a message naming a platform AND a
    # topic word must route to the platform, not the topic — "LinkedIn
    # ... creators ..." used to route to Meta via "creator". (The full
    # crowding-out message now routes to the crowding-out scenario by
    # the more-specific-topic rule below; this case has no crowding-out
    # keywords, so platform-first applies.)
    _s3 = new_session()
    orch.handle(_s3, action="industry", payload={"industry_id": "social"})
    orch.handle(_s3, message="In linkedin i want to see connection recommendations for new members")
    check("platform name beats topic word in routing", _s3.get("scenario_id") == "linkedin_connections")
    # Crowding-out scenario (13th runnable): viewer-randomized tier shares.
    _df_co, _r_co = run_flagship("linkedin_creator_crowding_out", seed=42)
    check("crowding-out seed42 dont launch on falling small share",
          _r_co["verdict"] == "DONT_LAUNCH" and _r_co["primary"]["lift_pct"] < 0)
    check("crowding-out small share falls, top share rises",
          _r_co["primary"]["treatment_mean"] < _r_co["primary"]["control_mean"] and
          _r_co["top_share"]["treatment_mean"] > _r_co["top_share"]["control_mean"])
    _tiers = followups.tier_shares(_df_co, _r_co, "linkedin_creator_crowding_out")
    check("crowding-out tier shares sum to 100 per arm",
          _tiers["supported"] and abs(sum(_df_co[_df_co.group == "control"][["post_small_share_pct", "post_mid_share_pct", "post_top_share_pct"]].mean()) - 100) < 0.5)
    check("crowding-out tier verdict present", "crowding-out" in _tiers["text"])
    _seg_co = followups.segment(_df_co, _r_co, "linkedin_creator_crowding_out", "established")
    check("crowding-out established segment recomputes",
          _seg_co["supported"] and abs(_seg_co["lift_pct"] - next(x for x in _r_co["segments"] if x["segment"] == "established")["lift_pct"]) < 1e-9)
    _s4 = new_session()
    orch.handle(_s4, action="industry", payload={"industry_id": "social"})
    orch.handle(_s4, message="In linkedin i want to see if top creators crowd out small creators")
    check("crowding-out topic beats platform in routing", _s4.get("scenario_id") == "linkedin_creator_crowding_out")
    # Chat established-segment regression: the old handler returned
    # "returning" for non-Reddit scenarios, breaking Meta established.
    _s5 = new_session()
    orch.handle(_s5, action="industry", payload={"industry_id": "social"})
    orch.handle(_s5, action="scenario", payload={"scenario_id": "meta_creator_recommendations"})
    _df_m, _r_m = run_flagship("meta_creator_recommendations", seed=42)
    _trend_m = _r_m.pop("trend")
    _s5.update({"state": "done", "df": _df_m, "result": _r_m,
                "config": dict(_CFG["meta_creator_recommendations"])})
    _fb = orch.handle(_s5, message="Show results for established users")
    check("chat established segment works for Meta", "established users" in _fb[0]["text"])
    # Largest-units follow-up (2026-10-31): "what are the five largest
    # DMAs?" must be answered from the stored frame, not fall through.
    _df_x2, _r_x2 = run_flagship("cross_publisher_budget", seed=42)
    _lg = followups.largest_units(_df_x2, _r_x2, "cross_publisher_budget", 5)
    check("largest units lists 5 sorted by baseline",
          _lg["supported"] and len(_lg["units"]) == 5 and
          _lg["units"] == _df_x2.nlargest(5, "pre_starts")["dma"].tolist())
    check("largest units text names a DMA and a share", "dma" in _lg["text"] and "% of baseline" in _lg["text"])
    _sm = followups.largest_units(_df_x2, _r_x2, "cross_publisher_budget", 3, smallest=True)
    check("smallest units lists 3", _sm["supported"] and len(_sm["units"]) == 3)
    _df_dd2, _r_dd2 = run_flagship("doordash_zip_promotion", seed=42)
    _lg_dd = followups.largest_units(_df_dd2, _r_dd2, "doordash_zip_promotion", 5)
    check("largest ZIPs work for flagship geo", _lg_dd["supported"] and len(_lg_dd["units"]) == 5)
    _df_r2, _r_r2 = run_flagship("reddit_feed_ranking", seed=42)
    check("largest units honestly unsupported for user-level",
          followups.largest_units(_df_r2, _r_r2, "reddit_feed_ranking", 5)["supported"] is False)
    _s6 = new_session()
    orch.handle(_s6, action="industry", payload={"industry_id": "pharma"})
    orch.handle(_s6, action="scenario", payload={"scenario_id": "cross_publisher_budget"})
    _s6.update({"state": "done", "df": _df_x2, "result": _r_x2,
                "config": dict(_CFG["cross_publisher_budget"])})
    _fb2 = orch.handle(_s6, message="what are the five largest dmas")
    check("chat largest question answered", "largest DMAs" in _fb2[0]["text"] and "I can't answer" not in _fb2[0]["text"])
    _fb3 = orch.handle(_s6, message="Exclude the five largest DMAs")
    check("chat exclude-largest still works after largest branch", "Excluding the 5 largest units" in _fb3[0]["text"])
    # Switchback scenarios (2026-11-03): region x window randomization,
    # unit-mean analysis with cluster-robust SEs by region, washout
    # exclusion, and scorecard-followup agreement.
    _sb_ids = ["switchback_sos_pricing", "switchback_surge", "switchback_batching"]
    _sb_verdicts = {"switchback_sos_pricing": "LAUNCH", "switchback_surge": "DONT_LAUNCH",
                    "switchback_batching": "INCONCLUSIVE"}
    for _sid in _sb_ids:
        _a, _ra = run_flagship(_sid, seed=42)
        _b, _rb = run_flagship(_sid, seed=42)
        check(f"{_sid} seed reproducible", _a.equals(_b) and _ra["primary"]["lift_pct"] == _rb["primary"]["lift_pct"])
        _c, _ = run_flagship(_sid, seed=7)
        check(f"{_sid} seed changes data", not _a.equals(_c))
        check(f"{_sid} verdict at seed 42", _ra["verdict"] == _sb_verdicts[_sid])
        check(f"{_sid} srm passes at seed 42", _ra["srm"]["failed"] is False)
        _sb = _ra["switchback"]
        check(f"{_sid} washout excludes first window after a switch",
              _sb["washout_windows"] > 0 and _sb["washout_windows"] < _sb["total_windows"] and
              _ra["n_units"] == _sb["total_windows"] - _sb["washout_windows"])
        _prim = _a[~_a["washout"]]
        _tr = _prim["group"] == "treatment"
        _unit_diff = _prim.loc[_tr, _sb["value_col"]].mean() - _prim.loc[~_tr, _sb["value_col"]].mean()
        _ctrl_mean = _prim.loc[~_tr, _sb["value_col"]].mean()
        check(f"{_sid} primary is unweighted unit-mean difference",
              abs(_ra["primary"]["lift_pct"] - _unit_diff / _ctrl_mean * 100) < 1e-9)
        _weighted = ((_prim.loc[_tr, _sb["value_col"]] * _prim.loc[_tr, "volume"]).sum() / _prim.loc[_tr, "volume"].sum()
                     - (_prim.loc[~_tr, _sb["value_col"]] * _prim.loc[~_tr, "volume"]).sum() / _prim.loc[~_tr, "volume"].sum())
        check(f"{_sid} primary is not volume-weighted",
              abs(_weighted / _ctrl_mean * 100 - _ra["primary"]["lift_pct"]) > 1e-6)
        check(f"{_sid} cluster-robust SE present",
              _ra["primary"].get("n_clusters", 0) == _ra["n_regions"] and _ra["primary"].get("se", 0) > 0)
        check(f"{_sid} washout sensitivity computed",
              "lift_pct" in _sb["sensitivity"] and _sb["sensitivity"]["lift_pct"] != _ra["primary"]["lift_pct"])
        _cfg = dict(_CFG[_sid]); _cfg["seed"] = 42
        _trend = _ra.pop("trend")
        _sc = build_scorecard(_cfg, _ra, _trend, 42)
        check(f"{_sid} scorecard lift matches engine", abs(_sc["primary_lift_pct"] - _ra["primary"]["lift_pct"]) < 1e-9)
        check(f"{_sid} scorecard verdict matches engine", _sc["verdict"] == _ra["verdict"])
        check(f"{_sid} scorecard segment matches engine",
              all(abs(s["lift_pct"] - next(x for x in _ra["segments"] if x["segment"] == s["segment"])["lift_pct"]) < 1e-9
                  for s in _sc["segments"]))
        check(f"{_sid} scorecard states washout",
              any("washout" in w.lower() for w in _sc["watchouts"]) and "washout" in _sc["ds"]["sensitivity"].lower())
        _seg = followups.segment(_a, _ra, _sid, "large")
        check(f"{_sid} segment recomputes and matches scorecard",
              _seg["supported"] and abs(_seg["lift_pct"] - next(x for x in _ra["segments"] if "large" in x["segment"])["lift_pct"]) < 1e-9)
        _wo = followups.switchback_washout(_a, _ra, _sid)
        check(f"{_sid} washout follow-up explains carryover",
              _wo["supported"] and "Washout" in _wo["text"] and "carryover" in _wo["text"].lower())
        _lg = followups.largest_units(_a, _ra, _sid, 5)
        check(f"{_sid} largest regions listed", _lg["supported"] and len(_lg["units"]) == 5)
        _ex = followups.exclude_top(_a, _ra, _sid, 5)
        check(f"{_sid} exclude-top recomputes with cluster-robust estimator",
              _ex["supported"] and _ex["n_remaining"] < _ra["n_units"])
        _wo2 = followups.switchback_washout(_df_r2, _r_r2, "reddit_feed_ranking")
        check("washout honestly unsupported for user-level", _wo2["supported"] is False)
    _s7 = new_session()
    orch.handle(_s7, action="industry", payload={"industry_id": "marketplace"})
    _blocks_m = orch.handle(_s7, action="industry", payload={"industry_id": "marketplace"})
    _cards_m = [b for b in _blocks_m if b["type"] == "scenario_cards"][0]["cards"]
    check("marketplace cards include 3 switchback scenarios", len(_cards_m) == 7 and all(c["runnable"] for c in _cards_m))
    orch.handle(_s7, message="DoorDash switchback pricing test")
    check("switchback keyword routes to SOS switchback", _s7.get("scenario_id") == "switchback_sos_pricing")
    orch.handle(_s7, message="New experiment")
    orch.handle(_s7, action="industry", payload={"industry_id": "marketplace"})
    orch.handle(_s7, action="scenario", payload={"scenario_id": "switchback_surge"})
    _df_sb, _r_sb = run_flagship("switchback_surge", seed=42)
    _trend_sb = _r_sb.pop("trend")
    _s7.update({"state": "done", "df": _df_sb, "result": _r_sb,
                "config": dict(_CFG["switchback_surge"])})
    _fb_sb = orch.handle(_s7, message="Explain the washout and carryover")
    check("chat washout follow-up works", "Washout" in _fb_sb[0]["text"])
    _fb_sb2 = orch.handle(_s7, message="What are the five largest cities?")
    check("chat largest cities works for switchback", "largest cities" in _fb_sb2[0]["text"])
    _s2 = new_session()
    _blocks = orch.handle(_s2, action="industry", payload={"industry_id": "social"})
    _cards = [b for b in _blocks if b["type"] == "scenario_cards"][0]["cards"]
    check("social cards all runnable", len(_cards) == 5 and all(c["runnable"] for c in _cards))
    _blocks = orch.handle(_s2, action="scenario", payload={"scenario_id": "tiktok_notifications"})
    check("chat new scenario -> config card", any(b["type"] == "config" for b in _blocks) and _s2["state"] == "config")
    print("All platform tests passed.")


if __name__ == "__main__":
    main()
