"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly",
                 extra_sections: list[str] | None = None) -> str:
    """Return a markdown cost-optimization report.

    `extra_sections` are markdown blocks (analysis, extensions) appended before the footnote.
    """
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    lines = [
        "# NimbusAI — GPU Cost Optimization Report",
        "",
        f"**Period:** {period}  ",
        f"**Baseline spend:** ${baseline_usd:,.0f}  ",
        f"**Optimized spend:** ${optimized_usd:,.0f}  ",
        f"**Projected savings:** ${savings:,.0f}  (**{pct:.0f}%**)",
        "",
        "## Savings by lever",
        "",
        "| Lever | Savings (USD) |",
        "|---|---|",
    ]
    for name, amount in levers.items():
        lines.append(f"| {name} | ${amount:,.0f} |")
    if sustainability:
        lines += [
            "",
            "## Sustainability",
            "",
            f"- Energy per query: {sustainability.get('wh_per_query', 0):.2f} Wh",
            f"- Carbon per query: {sustainability.get('carbon_g', 0):.3f} gCO2e",
            f"- Cheapest+cleanest region: {sustainability.get('best_region', 'n/a')}",
        ]
    for section in extra_sections or []:
        lines += ["", section.rstrip()]
    lines += ["", "_Figures are June-2026 as-of snapshots; re-baseline before acting._"]
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str, baseline: float | None = None) -> str:
    """Write a savings chart PNG. Returns the path. No-op if matplotlib absent.

    With `baseline`, draws a true waterfall: baseline -> minus each lever -> optimized.
    Without it, a plain bar chart of savings per lever.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    names = list(levers.keys())
    vals = [levers[n] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    if baseline is None:
        ax.bar(names, vals, color="#2e548a")
        ax.set_ylabel("Savings (USD / month)")
    else:
        optimized = baseline - sum(vals)
        labels = ["Baseline"] + names + ["Optimized"]
        bottoms, heights, colors = [0.0], [baseline], ["#7a7a7a"]
        running = baseline
        for v in vals:
            running -= v
            bottoms.append(running)
            heights.append(v)
            colors.append("#2e8a57")
        bottoms.append(0.0)
        heights.append(optimized)
        colors.append("#2e548a")
        bars = ax.bar(labels, heights, bottom=bottoms, color=colors)
        for i, (bar, h, b) in enumerate(zip(bars, heights, bottoms)):
            txt = f"${h:,.0f}" if i in (0, len(labels) - 1) else f"-${h:,.0f}"
            ax.annotate(txt, (bar.get_x() + bar.get_width() / 2, b + h),
                        ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("GPU spend (USD / month)")
    ax.set_title("GPU cost savings by FinOps lever")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
