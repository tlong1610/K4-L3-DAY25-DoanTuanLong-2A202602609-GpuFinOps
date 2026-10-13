"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def cache_policy(rows) -> dict:
    """Your Turn #3: decide per (project, route_tier) whether prompt caching pays off.

    Re-reads per prefix are estimated from the group's cache-hit fraction and
    compared with the tier's break-even (write premium + storage vs read savings).
    """
    groups = defaultdict(lambda: {"requests": 0, "input": 0, "cached": 0})
    for r in rows:
        g = groups[(r["project"] or "(untagged)", r["route_tier"])]
        g["requests"] += 1
        g["input"] += int(num(r["input_tokens"]))
        g["cached"] += int(num(r["cached_input_tokens"]))

    policy = {}
    for (project, tier), g in groups.items():
        pin = MODEL_PRICES[tier][0]
        reads = pricing.estimate_cache_reads(g["cached"], g["input"])
        write_cost = pricing.cache_write_cost_per_m(pin)
        policy[(project, tier)] = {
            "project": project, "tier": tier, "requests": g["requests"],
            "hit_frac": round(g["cached"] / g["input"], 3) if g["input"] else 0.0,
            "avg_reads": round(reads, 2),
            "break_even_reads": round(pricing.cache_break_even_reads(pin), 2),
            "write_cost_per_m": write_cost,
            "worth_it": pricing.cache_is_worth_it(reads, write_cost, price_in_per_m=pin),
        }
    return policy


def _cached_cost(inp, out, cached, is_batch, tier, p) -> float:
    """Request cost with prompt caching, including the write (prefix written once per N reads)."""
    pin, pout = MODEL_PRICES[tier]
    written = cached / p["avg_reads"] if p["avg_reads"] > 0 else 0.0
    write_overhead = written / 1e6 * p["write_cost_per_m"] * (0.5 if is_batch else 1.0)
    return pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch) + write_overhead


def _optimized_cost(inp, out, cached, is_batch, tier, p) -> float:
    """Optimized request cost: cascade tier + batch + caching only where it pays off."""
    if p["worth_it"]:
        return _cached_cost(inp, out, cached, is_batch, tier, p)
    pin, pout = MODEL_PRICES[tier]
    return pricing.request_cost(inp, out, pin, pout, cached_in=0, batch=is_batch)


def reasoning_budget(rows, policy, caps=(0.10, 0.05, 0.02)) -> dict:
    """Your Turn #4: split $ and Wh by is_reasoning, then simulate a reasoning cap.

    Routing rule: only the most complex requests keep reasoning, using prompt length
    (input_tokens) as the complexity proxy. Demoted requests lose their thinking
    tokens — output is cut to the median non-reasoning output — and use the normal
    (x1) energy profile. Cost uses the optimized (M2) pricing.
    """
    reqs = []
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        reqs.append({
            "inp": inp, "out": out, "cached": int(num(r["cached_input_tokens"])),
            "batch": bool(int(num(r["is_batch"]))), "tier": r["route_tier"],
            "p": policy[(r["project"] or "(untagged)", r["route_tier"])],
            "reasoning": bool(int(num(r["is_reasoning"]))),
        })

    def cost(q, out=None):
        return _optimized_cost(q["inp"], q["out"] if out is None else out, q["cached"], q["batch"], q["tier"], q["p"])

    def wh(q, out=None, reasoning=None):
        return sustainability.wh_per_query(q["inp"] + (q["out"] if out is None else out),
                                           is_reasoning=q["reasoning"] if reasoning is None else reasoning)

    split = {}
    for flag in (True, False):
        g = [q for q in reqs if q["reasoning"] == flag]
        split[flag] = {"requests": len(g), "tokens": sum(q["inp"] + q["out"] for q in g),
                       "cost": sum(cost(q) for q in g), "wh": sum(wh(q) for q in g)}
    tot = {k: split[True][k] + split[False][k] for k in ("requests", "tokens", "cost", "wh")}
    share = {k: split[True][k] / tot[k] * 100 if tot[k] else 0.0 for k in tot}
    wh_ratio = ((split[True]["wh"] / split[True]["requests"]) / (split[False]["wh"] / split[False]["requests"])
                if split[True]["requests"] and split[False]["requests"] else 0.0)

    plain_outs = sorted(q["out"] for q in reqs if not q["reasoning"])
    demoted_out = plain_outs[len(plain_outs) // 2] if plain_outs else 0
    # most complex first: these keep reasoning under any cap
    ranked = sorted((q for q in reqs if q["reasoning"]), key=lambda q: q["inp"], reverse=True)

    scenarios = []
    for cap in caps:
        allowed = int(cap * len(reqs))
        kept, demoted = ranked[:allowed], ranked[allowed:]
        saved_cost = sum(cost(q) - cost(q, min(q["out"], demoted_out)) for q in demoted)
        saved_wh = sum(wh(q) - wh(q, min(q["out"], demoted_out), reasoning=False) for q in demoted)
        scenarios.append({
            "cap_pct": cap * 100, "reasoning_kept": len(kept), "demoted": len(demoted),
            "min_input_tokens": kept[-1]["inp"] if demoted and kept else None,
            "saved_daily": round(saved_cost, 4), "saved_monthly": round(saved_cost * 30, 2),
            "saved_wh_daily": round(saved_wh, 1),
            "wh_reduction_pct": round(saved_wh / tot["wh"] * 100, 1) if tot["wh"] else 0.0,
        })

    return {
        "traffic_pct": round(share["requests"], 1), "token_pct": round(share["tokens"], 1),
        "cost_pct": round(share["cost"], 1), "wh_pct": round(share["wh"], 1),
        "reasoning_cost_daily": round(split[True]["cost"], 4), "plain_cost_daily": round(split[False]["cost"], 4),
        "reasoning_wh_daily": round(split[True]["wh"], 1), "plain_wh_daily": round(split[False]["wh"], 1),
        "wh_per_query_ratio": round(wh_ratio, 1), "demoted_output_tokens": demoted_out,
        "scenarios": scenarios,
    }


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    policy = cache_policy(rows)
    reasoning = reasoning_budget(rows, policy)
    base_cost = opt_cost = naive_cache_cost = cache_all_cost = 0.0
    cascade_cost = cascade_cache_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        # naive caching: every cached token gets the read discount, writes are free
        naive_cache_cost += pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)
        p = policy[(r["project"] or "(untagged)", r["route_tier"])]
        # cache everything and pay for the writes
        cache_all_cost += _cached_cost(inp, out, cached, is_batch, r["route_tier"], p)
        # cache-aware: cache only where cache_is_worth_it() says it pays off
        opt_cost += _optimized_cost(inp, out, cached, is_batch, r["route_tier"], p)
        # stepwise levers for the $/1M-token waterfall: cascade -> + cache -> + batch
        cascade_cost += pricing.request_cost(inp, out, pin, pout)
        cascade_cache_cost += _optimized_cost(inp, out, cached, False, r["route_tier"], p)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    naive_pm = pricing.dollars_per_million(naive_cache_cost, total_tokens)
    cache_all_pm = pricing.dollars_per_million(cache_all_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0
    naive_savings_pct = (1 - naive_cache_cost / base_cost) * 100 if base_cost else 0.0
    lever_steps = [
        {"step": "Baseline (all large model)", "daily": base_cost},
        {"step": "+ Cascade routing", "daily": cascade_cost},
        {"step": "+ Cache-aware prompt caching", "daily": cascade_cache_cost},
        {"step": "+ Batch API", "daily": opt_cost},
    ]
    for st in lever_steps:
        st["per_m"] = round(pricing.dollars_per_million(st["daily"], total_tokens), 3)
        st["daily"] = round(st["daily"], 2)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + cache-aware caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print("$/1M-token by lever (cumulative):")
        for st in lever_steps:
            print(f"  {st['step']:30} ${st['daily']:>7,.2f}/day   ${st['per_m']:.3f}/1M-token")

        print("\n-- Your Turn #3: cache_is_worth_it() per (project, tier) --")
        print(f"{'project':14}{'tier':7}{'reqs':>6}{'hit':>7}{'reads N':>9}{'N*':>7}  cache?")
        for p in sorted(policy.values(), key=lambda x: (x["tier"], -x["avg_reads"])):
            print(f"{p['project']:14}{p['tier']:7}{p['requests']:>6}{p['hit_frac']:>7.3f}"
                  f"{p['avg_reads']:>9.2f}{p['break_even_reads']:>7.2f}  {'YES' if p['worth_it'] else 'no'}")
        print(f"naive caching (writes free) : ${naive_cache_cost:,.4f}/day   ${naive_pm:.4f}/1M-token  ({naive_savings_pct:.1f}% saved, unrealistic)")
        print(f"cache ALL, pay writes       : ${cache_all_cost:,.4f}/day   ${cache_all_pm:.4f}/1M-token")
        print(f"cache-aware (worth_it only) : ${opt_cost:,.4f}/day   ${opt_pm:.4f}/1M-token  "
              f"-> policy saves ${cache_all_cost - opt_cost:,.4f}/day vs caching everything")

        rb = reasoning
        print("\n-- Your Turn #4: reasoning budget --")
        print(f"reasoning = {rb['traffic_pct']}% of requests, {rb['token_pct']}% of tokens, "
              f"{rb['cost_pct']}% of $ (${rb['reasoning_cost_daily']:.2f}/day), "
              f"{rb['wh_pct']}% of energy ({rb['reasoning_wh_daily']:,.0f} Wh/day)")
        print(f"energy per reasoning query = {rb['wh_per_query_ratio']}x a normal query "
              f"(x{sustainability.REASONING_ENERGY_MULTIPLIER:.0f} multiplier x longer outputs)")
        print(f"rule: keep reasoning only for the longest prompts; demoted -> output capped at "
              f"{rb['demoted_output_tokens']} tok (median non-reasoning), x1 energy")
        print(f"{'cap':>6}{'kept':>6}{'demoted':>9}{'min input':>11}{'$ saved/day':>13}{'$/month':>9}{'Wh saved/day':>14}{'Wh -%':>7}")
        for sc in rb["scenarios"]:
            thr = sc["min_input_tokens"] if sc["min_input_tokens"] is not None else "-"
            print(f"{sc['cap_pct']:>5.0f}%{sc['reasoning_kept']:>6}{sc['demoted']:>9}{thr:>11}"
                  f"{sc['saved_daily']:>13.4f}{sc['saved_monthly']:>9.2f}{sc['saved_wh_daily']:>14,.1f}{sc['wh_reduction_pct']:>6.1f}%")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "naive_cache_daily": round(naive_cache_cost, 2), "naive_cache_per_m": round(naive_pm, 3),
        "cache_all_daily": round(cache_all_cost, 4), "cache_all_per_m": round(cache_all_pm, 4),
        "lever_steps": lever_steps,
        "reasoning": reasoning,
        "cache_policy": sorted(policy.values(), key=lambda x: (x["tier"], -x["avg_reads"])),
    }


if __name__ == "__main__":
    run()
