"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing, sustainability

DAYS = 30
CURRENT_REGION = "us-east-1"


def carbon_schedule(jobs_energy, regions=None) -> dict:
    """Your Turn #5: carbon-aware placement of interruptible jobs.

    `jobs_energy` maps job_id -> monthly kWh. Compares every region on electricity
    $ and gCO2e, then picks the cheapest, the cleanest, and the most balanced one
    (equal-weight min-max score of $/kWh and gCO2/kWh).
    """
    regions = regions or list(sustainability.REGION_CARBON)
    kwh = sum(jobs_energy.values())
    price = {r: sustainability.REGION_PRICE_KWH[r] for r in regions}
    co2 = {r: sustainability.REGION_CARBON[r] for r in regions}

    def norm(d, r):
        lo, hi = min(d.values()), max(d.values())
        return (d[r] - lo) / (hi - lo) if hi > lo else 0.0

    table = []
    for r in regions:
        table.append({
            "region": r, "usd_per_kwh": price[r], "gco2_per_kwh": co2[r],
            "energy_usd": round(sustainability.energy_cost_usd(kwh * 1000, r), 2),
            "carbon_kg": round(sustainability.carbon_g(kwh * 1000, r) / 1000, 1),
            "balanced_score": round(0.5 * norm(price, r) + 0.5 * norm(co2, r), 3),
        })
    cheapest = min(regions, key=price.get)
    cleanest = min(regions, key=co2.get)
    balanced = min(table, key=lambda t: t["balanced_score"])["region"]
    # internal carbon price ($/tCO2) at which the cleanest region also wins on total cost
    dp = price[cleanest] - price[cheapest]
    dc = (co2[cheapest] - co2[cleanest]) / 1e6  # tCO2 per kWh
    carbon_price_be = dp / dc if dc > 0 and dp > 0 else 0.0

    cur = next(t for t in table if t["region"] == CURRENT_REGION)
    best = next(t for t in table if t["region"] == cleanest)
    per_job = [{"job_id": j, "kwh": round(e, 1),
                "carbon_kg_current": round(sustainability.carbon_g(e * 1000, CURRENT_REGION) / 1000, 1),
                "carbon_kg_cleanest": round(sustainability.carbon_g(e * 1000, cleanest) / 1000, 1)}
               for j, e in jobs_energy.items()]
    return {
        "kwh_month": round(kwh, 1), "regions": table, "per_job": per_job,
        "cheapest": cheapest, "cleanest": cleanest, "balanced": balanced,
        "carbon_saved_kg": round(cur["carbon_kg"] - best["carbon_kg"], 1),
        "carbon_saved_pct": round((1 - best["carbon_kg"] / cur["carbon_kg"]) * 100, 1) if cur["carbon_kg"] else 0.0,
        "energy_usd_saved": round(cur["energy_usd"] - best["energy_usd"], 2),
        "carbon_price_breakeven": round(carbon_price_be),
    }


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    interruptible_kwh = {}
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        tier = pricing.recommend_tier(hpd, interruptible)
        run_hours = gpu_hours
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
            run_hours = sim["spot_effective_hours"]  # rework + checkpoint overhead burn energy too
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        if interruptible:
            interruptible_kwh[j["job_id"]] = run_hours * num(c["watts"]) / 1000.0
        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0
    carbon = carbon_schedule(interruptible_kwh)

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        cs = carbon
        print(f"\n-- Your Turn #5: carbon-aware scheduling ({len(cs['per_job'])} interruptible jobs, "
              f"{cs['kwh_month']:,.0f} kWh/month GPU energy) --")
        print(f"{'job':18}{'kWh':>9}{'kgCO2 ' + CURRENT_REGION:>20}{'kgCO2 ' + cs['cleanest']:>22}")
        for pj in cs["per_job"]:
            print(f"{pj['job_id']:18}{pj['kwh']:>9,.0f}{pj['carbon_kg_current']:>20,.1f}{pj['carbon_kg_cleanest']:>22,.1f}")
        print(f"{'region':17}{'$/kWh':>7}{'gCO2/kWh':>10}{'energy $':>10}{'kgCO2e':>9}{'score':>7}")
        for t in sorted(cs["regions"], key=lambda t: t["balanced_score"]):
            print(f"{t['region']:17}{t['usd_per_kwh']:>7.3f}{t['gco2_per_kwh']:>10}{t['energy_usd']:>10,.0f}"
                  f"{t['carbon_kg']:>9,.1f}{t['balanced_score']:>7.3f}")
        print(f"cheapest $: {cs['cheapest']}   cleanest CO2: {cs['cleanest']}   balanced: {cs['balanced']}")
        print(f"move all interruptible jobs {CURRENT_REGION} -> {cs['cleanest']}: "
              f"-{cs['carbon_saved_kg']:,.1f} kgCO2e/month ({cs['carbon_saved_pct']}%), "
              f"electricity ${cs['energy_usd_saved']:,.0f}/month saved")
        print(f"{cs['cleanest']} beats {cs['cheapest']} on total cost only if internal carbon price "
              f"> ${cs['carbon_price_breakeven']:,}/tCO2")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "carbon": carbon}


if __name__ == "__main__":
    run()
