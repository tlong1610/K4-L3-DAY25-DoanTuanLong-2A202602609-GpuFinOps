"""Tests for the "Your Turn" extensions: #3 cache_is_worth_it, #4 reasoning budget."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import generate
from finops import pricing
from missions import m2_inference_levers


# --- Your Turn #3: cache economics -----------------------------------------

def test_cache_write_cost():
    # 25% premium on a $3 input price + $0.05 storage
    assert abs(pricing.cache_write_cost_per_m(3.0) - 0.80) < 1e-9
    assert abs(pricing.cache_write_cost_per_m(0.20) - 0.10) < 1e-9


def test_cache_break_even_reads_by_tier():
    large = pricing.cache_break_even_reads(3.0)
    small = pricing.cache_break_even_reads(0.20)
    assert abs(large - 0.80 / 2.70) < 1e-9          # hand-computed
    assert small > large                             # fixed storage fee hurts cheap models
    assert pricing.cache_break_even_reads(3.0, read_discount=1.0) == float("inf")


def test_cache_is_worth_it_threshold():
    wc = pricing.cache_write_cost_per_m(3.0)
    n_star = pricing.cache_break_even_reads(3.0)
    assert pricing.cache_is_worth_it(n_star * 1.1, wc, price_in_per_m=3.0)
    assert not pricing.cache_is_worth_it(n_star * 0.9, wc, price_in_per_m=3.0)
    assert not pricing.cache_is_worth_it(n_star, wc, price_in_per_m=3.0)   # break-even is not a win
    assert not pricing.cache_is_worth_it(0, wc, price_in_per_m=3.0)


def test_estimate_cache_reads():
    assert pricing.estimate_cache_reads(0, 1000) == 0.0
    assert abs(pricing.estimate_cache_reads(500, 1000) - 1.0) < 1e-9   # h=0.5 -> N=1
    assert pricing.estimate_cache_reads(1000, 1000) == float("inf")
    assert pricing.estimate_cache_reads(10, 0) == 0.0


def test_m2_cache_policy_applied():
    generate.main()
    r2 = m2_inference_levers.run(verbose=False)
    policy = {(p["project"], p["tier"]): p["worth_it"] for p in r2["cache_policy"]}
    assert policy[("chat", "small")] and policy[("doc-qa", "large")]
    assert not policy[("web-search", "small")] and not policy[("nightly-eval", "large")]
    # paying for writes costs more than the free-writes model; selective caching beats caching everything
    assert r2["naive_cache_daily"] <= r2["optimized_daily"] < r2["cache_all_daily"]
    assert 60 <= r2["savings_pct"] <= 95
    per_m = [s["per_m"] for s in r2["lever_steps"]]
    assert per_m == sorted(per_m, reverse=True)      # every lever lowers $/1M-token


# --- Your Turn #4: reasoning budget ----------------------------------------

def test_reasoning_budget():
    generate.main()
    rb = m2_inference_levers.run(verbose=False)["reasoning"]
    assert rb["wh_pct"] > rb["cost_pct"] > rb["traffic_pct"] > 0   # energy bomb
    assert rb["wh_per_query_ratio"] >= 80
    by_cap = {s["cap_pct"]: s for s in rb["scenarios"]}
    assert by_cap[10]["saved_daily"] == 0 and by_cap[10]["demoted"] == 0   # already under 10%
    assert 0 < by_cap[5]["saved_wh_daily"] < by_cap[2]["saved_wh_daily"]  # tighter cap saves more
    assert by_cap[2]["reasoning_kept"] == int(0.02 * 2400)


# --- Your Turn #5: carbon-aware scheduling ---------------------------------

def test_carbon_schedule_unit():
    from missions.m3_purchasing import carbon_schedule
    cs = carbon_schedule({"job-a": 1000.0})               # 1 MWh
    by_region = {t["region"]: t for t in cs["regions"]}
    assert by_region["us-east-1"]["carbon_kg"] == 380.0   # 1000 kWh * 380 g/kWh
    assert by_region["europe-north1"]["energy_usd"] == 90.0
    assert cs["cleanest"] == "europe-north1" and cs["cheapest"] == "us-east-wa"
    assert by_region["europe-central2"]["balanced_score"] == 1.0   # worst on both axes
    # cleanest wins on total cost exactly at the break-even carbon price
    p = cs["carbon_price_breakeven"] / 1e6
    assert abs((0.09 + 30 * p) - (0.055 + 90 * p)) < 1e-3


def test_m3_carbon_only_interruptible_jobs():
    generate.main()
    from missions import m3_purchasing
    cs = m3_purchasing.run(verbose=False)["carbon"]
    jobs = {pj["job_id"] for pj in cs["per_job"]}
    assert "job-train-llm" in jobs and "job-infer-chat" not in jobs   # realtime inference stays put
    assert cs["carbon_saved_kg"] > 0 and cs["carbon_saved_pct"] > 90
    assert cs["energy_usd_saved"] > 0                                  # cleaner AND cheaper than us-east-1
