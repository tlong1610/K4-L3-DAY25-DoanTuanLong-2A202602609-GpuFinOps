"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def cache_write_cost_per_m(
    price_in_per_m: float,
    write_premium: float = 0.25,    # Anthropic cache write ~1.25x base input (+25%)
    storage_per_m: float = 0.05,    # Gemini-style storage fee per 1M cached tokens per TTL window
) -> float:
    """Extra USD per 1M tokens to write (and keep) a prefix in cache, vs not caching it.

    The premium scales with the model's input price; the storage fee does not — so
    cheap models pay relatively more to cache.
    """
    return max(0.0, write_premium) * price_in_per_m + max(0.0, storage_per_m)


def cache_break_even_reads(
    price_in_per_m: float,
    write_premium: float = 0.25,
    storage_per_m: float = 0.05,
    read_discount: float = 0.10,
) -> float:
    """Re-reads per cached prefix needed before caching pays off.

    Each re-read saves price_in * (1 - read_discount); the write costs
    cache_write_cost_per_m(). Break-even N* = write cost / saving per read.
    Large model ($3/1M in): (0.75 + 0.05) / 2.70 ~= 0.30 reads.
    Small model ($0.20/1M in): (0.05 + 0.05) / 0.18 ~= 0.56 reads.
    """
    saved_per_read = price_in_per_m * (1.0 - read_discount)
    if saved_per_read <= 0:
        return float("inf")
    return cache_write_cost_per_m(price_in_per_m, write_premium, storage_per_m) / saved_per_read


def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost_per_m: float,
    read_discount: float = 0.10,
    price_in_per_m: float = 3.00,
) -> bool:
    """Caching only saves money when read savings exceed the cost of writing the cache.

    avg_cache_reads  : average re-reads per cached prefix
    write_cost_per_m : extra USD per 1M tokens to write the cache (see cache_write_cost_per_m)
    read_discount    : cached reads billed at this fraction of input price (0.10 = -90%)
    """
    saved = max(0.0, avg_cache_reads) * price_in_per_m * (1.0 - read_discount)
    return saved > write_cost_per_m


def estimate_cache_reads(cached_tokens: float, input_tokens: float) -> float:
    """Estimate avg re-reads per prefix from an observed cache-hit fraction.

    A prefix written once and read N times has hit fraction h = N / (N + 1),
    so N = h / (1 - h). Returns inf when every input token was a cache hit.
    """
    if input_tokens <= 0:
        return 0.0
    h = max(0.0, min(1.0, cached_tokens / input_tokens))
    return float("inf") if h >= 1.0 else h / (1.0 - h)


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def recommend_tier(hours_per_day: float, interruptible: bool, reserved_discount: float = 0.45) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    DOCUMENTED simple policy (instructor extension point — swap in your own):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    if interruptible and hours_per_day < 24:
        return "spot"
    if duty >= be:
        return "reserved"
    return "on_demand"


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
