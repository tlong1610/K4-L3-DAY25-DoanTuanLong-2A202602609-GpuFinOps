"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import load_csv, num, catalog_by_type, ROOT
from finops import report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing, m4_allocation

DAYS = 30
HOURS_MONTH = 24 * DAYS
# one tier down for over-provisioned ("util-lie") GPUs
RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}
TARGET_MFU = 0.40  # healthy training MFU (~0.35-0.45); the honest H100s here sit at ~0.40-0.43

# Implementation effort per lever (1 = config change, 3 = benchmark + migration) -> ROI ranking
LEVER_EFFORT = {
    "Purchasing (spot/reserved)": (2, "Trung bình",
        "Đổi tier mua: job interruptible sang spot (cần checkpoint), job chạy ≥55% thời gian sang reserved. "
        "Rủi ro là cam kết 3 năm, nên chỉ reserve phần tải nền ổn định."),
    "Kill idle GPUs": (1, "Thấp",
        "Chỉ cần auto-shutdown / scale-to-zero khi util < 10%. Không đụng tới model, gần như không rủi ro."),
    "Inference (cascade/cache/batch)": (2, "Trung bình",
        "Sửa router (cascade), bật cache theo chính sách cache_is_worth_it, đẩy traffic không cần realtime sang Batch API."),
    "Right-size util-lies": (3, "Cao",
        "Phải profile (Nsight) và sửa nút thắt trước, hoặc benchmark GPU nhỏ hơn rồi migrate. Tốn công nhất."),
}


def _fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def _section_unit_economics(r2, levers, baseline, optimized, tokens_month) -> str:
    rows = [
        "## Đơn vị kinh tế: $/1M-token",
        "",
        "### Inference: từng đòn bẩy cộng dồn (M2)",
        "",
        "| Bước | $/ngày | $/1M-token | Giảm so với baseline |",
        "|---|---|---|---|",
    ]
    base_pm = r2["lever_steps"][0]["per_m"]
    for st in r2["lever_steps"]:
        drop = (1 - st["per_m"] / base_pm) * 100 if base_pm else 0.0
        rows.append(f"| {st['step']} | ${st['daily']:,.2f} | ${st['per_m']:.3f} | {drop:.1f}% |")
    (s_in, s_out), (l_in, l_out) = m2_inference_levers.MODEL_PRICES["small"], m2_inference_levers.MODEL_PRICES["large"]
    small_reqs = sum(p["requests"] for p in r2["cache_policy"] if p["tier"] == "small")
    all_reqs = sum(p["requests"] for p in r2["cache_policy"])
    rows += [
        "",
        f"Cascade là đòn bẩy inference lớn nhất: {r2['lever_steps'][0]['per_m']:.3f} → "
        f"{r2['lever_steps'][1]['per_m']:.3f} $/1M-token. Model nhỏ rẻ hơn {l_in / s_in:.0f}× (input) / "
        f"{l_out / s_out:.1f}× (output), và {small_reqs / all_reqs:.0%} request đủ dễ để model nhỏ xử lý. "
        "Cache và batch chồng lên sau đó, vì các chiết khấu nhân với nhau.",
        "",
        "### Toàn bộ fleet GPU quy về $/1M-token",
        "",
        f"Tổng chi GPU (inference + training/serving fleet) chia cho {tokens_month / 1e6:,.0f}M token phục vụ mỗi tháng.",
        "",
        "| Đòn bẩy | Tiết kiệm $/tháng | % baseline | Tương đương $/1M-token |",
        "|---|---|---|---|",
    ]
    m = tokens_month / 1e6
    for name, amount in levers.items():
        rows.append(f"| {name} | {_fmt_money(amount)} | {amount / baseline * 100:.1f}% | ${amount / m:.2f} |")
    rows += [
        f"| **Baseline → Optimized** | **{_fmt_money(baseline)} → {_fmt_money(optimized)}** | "
        f"**{(baseline - optimized) / baseline * 100:.1f}%** | **${baseline / m:.2f} → ${optimized / m:.2f}** |",
    ]
    return "\n".join(rows)


def _section_util_lie(r1, cat) -> str:
    rows = [
        "## Vì sao GPU-Util \"nói dối\" (M1)",
        "",
        "`nvidia-smi` báo GPU-Util = **% thời gian có ít nhất một kernel đang chạy** trong cửa sổ lấy mẫu. "
        "Nó không đo bao nhiêu SM/Tensor Core thực sự làm việc, cũng không đo bao nhiêu FLOPs sinh ra. "
        "Một kernel nhỏ chiếm 1 SM trên tổng 132 SM của H100 vẫn được tính là \"bận 100%\". "
        "MFU (FLOPs đạt được / FLOPs đỉnh) mới là thước đo thứ ta trả tiền để thuê.",
        "",
        f"| GPU | Loại | GPU-Util | MFU | MBU | Lãng phí so với MFU mục tiêu {TARGET_MFU:.2f} ($/tháng) |",
        "|---|---|---|---|---|---|",
    ]
    total = 0.0
    for lie in r1["lies"]:
        price = num(cat[lie["gpu_type"]]["on_demand_hr"])
        waste = max(0.0, 1 - lie["mfu"] / TARGET_MFU) * price * HOURS_MONTH
        total += waste
        rows.append(f"| {lie['gpu_id']} | {lie['gpu_type']} | {lie['gpu_util_pct']}% | {lie['mfu']:.3f} | "
                    f"{lie['mbu']:.3f} | {_fmt_money(waste)} |")
    healthy = [s for s in r1["summary"] if s["gpu_type"] == "H100" and s["gpu_util_pct"] >= 90
               and s not in r1["lies"]]
    lie_h100 = next((l for l in r1["lies"] if l["gpu_type"] == "H100"), None)
    rows.append("")
    if healthy and lie_h100:
        avg_mfu = sum(s["mfu"] for s in healthy) / len(healthy)
        rows.append(
            f"**So sánh trực tiếp:** {len(healthy)} con H100 khỏe có util "
            f"{min(s['gpu_util_pct'] for s in healthy)}–{max(s['gpu_util_pct'] for s in healthy)}%, "
            f"tức **gần bằng** {lie_h100['gpu_id']} ({lie_h100['gpu_util_pct']}%). Nhưng MFU trung bình của chúng là "
            f"{avg_mfu:.2f}, gấp **{avg_mfu / lie_h100['mfu']:.1f}×**. Nhìn GPU-Util thì không thể phân biệt hai nhóm này.")
        rows.append("")
    rows += [
        f"**Cơ chế gốc rễ:** ở các GPU bị flag, **cả MFU lẫn MBU đều thấp** "
        f"(MFU {min(l['mfu'] for l in r1['lies']):.2f}–{max(l['mfu'] for l in r1['lies']):.2f}, "
        f"MBU {min(l['mbu'] for l in r1['lies']):.2f}–{max(l['mbu'] for l in r1['lies']):.2f}). Vậy GPU không chạm trần compute, "
        "cũng không chạm trần băng thông HBM (roofline), tức nó đang *chờ*, không phải đang tính:",
        "- **Kernel launch overhead / batch quá nhỏ:** nhiều kernel ngắn, mỗi kernel chỉ phủ một phần nhỏ SM, nên clock \"bận\" mà Tensor Core rảnh.",
        "- **Đồng bộ host↔device, data loader CPU chậm, chờ NCCL all-reduce:** kernel vẫn \"đang chạy\" trong khi chờ, nên util vẫn cao.",
        "- **Memory stall:** với decode LLM (arithmetic intensity ~1–2 FLOP/byte, thấp hơn nhiều so với ridge ~295 của H100), "
        "workload bị giới hạn bởi băng thông. Khi đó nên tối ưu và đo bằng MBU, không phải MFU.",
        "",
        f"**Tác động tài chính:** ta trả nguyên giá giờ GPU nhưng chỉ nhận một phần FLOPs. Riêng các GPU bị flag lãng phí "
        f"khoảng **{_fmt_money(total)}/tháng** so với mức MFU {TARGET_MFU:.2f}. "
        "Cách xử lý: profile bằng Nsight, tăng batch size, dùng CUDA Graphs / torch.compile để giảm launch overhead. "
        "Nếu không sửa được thì right-size xuống GPU rẻ hơn (lever \"Right-size util-lies\").",
    ]
    return "\n".join(rows)


def _section_actions(levers, r2, r3, r4) -> str:
    ranked = sorted(levers.items(), key=lambda kv: kv[1] / LEVER_EFFORT[kv[0]][0], reverse=True)
    rows = [
        "## Đề xuất hành động (xếp theo ROI = $ tiết kiệm / công sức)",
        "",
        "| Ưu tiên | Hành động | Tiết kiệm $/tháng | Công sức | Lý do |",
        "|---|---|---|---|---|",
    ]
    for i, (name, amount) in enumerate(ranked, 1):
        _, effort, why = LEVER_EFFORT[name]
        rows.append(f"| {i} | {name} | {_fmt_money(amount)} | {effort} | {why} |")
    spot = [r["job_id"] for r in r3["recommendations"] if r["tier"] == "spot"]
    reserved = [r["job_id"] for r in r3["recommendations"] if r["tier"] == "reserved"]
    cap2 = next((s for s in r2["reasoning"]["scenarios"] if s["cap_pct"] == 2), None)
    rows += [
        "",
        f"1. **Purchasing trước tiên** vì đây là lever lớn nhất và không cần sửa code model. "
        f"Đưa {len(spot)} job interruptible ({', '.join(spot)}) sang spot kèm checkpoint. "
        f"Đưa {len(reserved)} job inference chạy ổn định ({', '.join(reserved)}) sang reserved, "
        f"vì duty cycle của chúng ≥ ngưỡng hòa vốn 55%. Riêng M3 đã tiết kiệm {r3['savings_pct']}% chi phí workload.",
        "2. **Kill idle là quick win:** công sức thấp nhất, làm ngay trong tuần đầu.",
        f"3. **Inference:** cascade + cache-aware + batch đưa $/1M-token từ ${r2['baseline_per_m']:.3f} xuống "
        f"${r2['optimized_per_m']:.3f}. Về tỷ lệ thì cao ({r2['savings_pct']}%), nhưng về tiền tuyệt đối thì nhỏ hơn "
        "purchasing vì hóa đơn token hiện nhỏ hơn hóa đơn fleet GPU. Lever này sẽ quan trọng dần khi traffic tăng.",
        "4. **Right-size cuối cùng:** nên sửa nút thắt (batch, CUDA Graphs) trước khi đổi phần cứng.",
    ]
    if cap2:
        rows.append(f"5. **Giới hạn reasoning ở ~2% traffic:** tiết kiệm {cap2['wh_reduction_pct']}% năng lượng inference "
                    f"(${cap2['saved_monthly']:.2f}/tháng). Đây chủ yếu là lever về carbon (xem Your Turn #4).")
    rows.append(f"6. **Chargeback:** tag coverage {r4['tag_coverage']:.0%} ≥ 80% nên có thể chuyển từ showback sang "
                "chargeback theo team. Nên gắn tag nốt phần request untagged để hóa đơn mỗi team đầy đủ.")
    return "\n".join(rows)


def _section_sustainability(cat, r2) -> str:
    jobs = load_csv("workloads.csv")
    kwh_month = sum(num(j["hours_per_day"]) * DAYS * num(j["num_gpus"]) * num(cat[j["gpu_type"]]["watts"])
                    for j in jobs) / 1000.0
    gpu_rent = sum(num(j["hours_per_day"]) * DAYS * num(j["num_gpus"]) * num(cat[j["gpu_type"]]["on_demand_hr"])
                   for j in jobs)
    rows = [
        "## Bền vững: carbon gắn với chi phí điện",
        "",
        f"Fleet GPU (workloads.csv) tiêu thụ khoảng **{kwh_month:,.0f} kWh/tháng** (chỉ tính TDP của GPU). "
        "Chi phí điện và carbon nếu chạy ở từng vùng:",
        "",
        "| Vùng | $/kWh | gCO2/kWh | Tiền điện $/tháng | tCO2e/tháng |",
        "|---|---|---|---|---|",
    ]
    regions = sorted(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get)
    for reg in regions:
        wh = kwh_month * 1000
        rows.append(f"| {reg} | {sustainability.REGION_PRICE_KWH[reg]:.3f} | {sustainability.REGION_CARBON[reg]} | "
                    f"{_fmt_money(sustainability.energy_cost_usd(wh, reg))} | {sustainability.carbon_g(wh, reg) / 1e6:.2f} |")
    wh = kwh_month * 1000
    cur_c, best_c = sustainability.carbon_g(wh, "us-east-1"), sustainability.carbon_g(wh, "europe-north1")
    cur_e, best_e = sustainability.energy_cost_usd(wh, "us-east-1"), sustainability.energy_cost_usd(wh, "europe-north1")
    cheapest = min(sustainability.REGION_PRICE_KWH, key=sustainability.REGION_PRICE_KWH.get)
    rb = r2["reasoning"]
    rows += [
        "",
        f"- **Vùng khuyến nghị: europe-north1** (thủy điện Na Uy, sạch nhất). So với us-east-1 hiện tại, carbon giảm "
        f"**{(1 - best_c / cur_c) * 100:.0f}%** ({cur_c / 1e6:.2f} → {best_c / 1e6:.2f} tCO2e/tháng) và tiền điện cũng giảm "
        f"({_fmt_money(cur_e)} → {_fmt_money(best_e)}/tháng). Ở đây cắt carbon và cắt tiền đi cùng chiều.",
        f"- **Rẻ nhất ≠ sạch nhất:** {cheapest} có giá điện thấp nhất "
        f"(${sustainability.REGION_PRICE_KWH[cheapest]}/kWh) nhưng carbon gấp "
        f"{sustainability.REGION_CARBON[cheapest] / sustainability.REGION_CARBON['europe-north1']:.0f}× europe-north1. "
        "Nếu chỉ tối ưu $ thì chọn vùng này (phân tích trade-off ở Your Turn #5). "
        "europe-central2 (Ba Lan) vừa đắt nhất vừa bẩn nhất, nên tránh.",
        f"- **Tiền điện chỉ ≈ {cur_e / gpu_rent * 100:.1f}% tiền thuê GPU** ({_fmt_money(cur_e)} so với {_fmt_money(gpu_rent)}/tháng on-demand). "
        "Vì vậy chuyển job interruptible (training, eval) sang vùng sạch gần như không tốn thêm tiền mà cắt mạnh carbon. "
        "Job inference realtime thì phải cân nhắc độ trễ tới người dùng: Bắc Âu xa user Mỹ/châu Á.",
        f"- **Reasoning là \"quả bom năng lượng\":** chỉ {rb['traffic_pct']}% request nhưng chiếm {rb['wh_pct']}% năng lượng inference "
        f"({rb['wh_per_query_ratio']}× Wh/query so với query thường).",
    ]
    return "\n".join(rows)


def _section_ext_cache(r2) -> str:
    rows = [
        "## Your Turn #3: `cache_is_worth_it()`",
        "",
        "Cache chỉ có lời khi tiền tiết kiệm từ các lần đọc lại lớn hơn chi phí ghi (phụ phí ghi +25% giá input "
        "và phí lưu trữ $0.05/1M token). Hòa vốn khi số lần đọc N\\* = phí ghi / (giá input × 0.9). "
        "Số lần đọc thực tế được ước lượng từ tỷ lệ cache hit h theo công thức N = h/(1−h).",
        "",
        "| Project | Tier | Requests | Cache hit | N thực tế | N\\* hòa vốn | Bật cache? |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in r2["cache_policy"]:
        rows.append(f"| {p['project']} | {p['tier']} | {p['requests']} | {p['hit_frac']:.1%} | {p['avg_reads']:.2f} | "
                    f"{p['break_even_reads']:.2f} | {'✅' if p['worth_it'] else '❌'} |")
    saved = r2["cache_all_daily"] - r2["optimized_daily"]
    rows += [
        "",
        f"- Cache mọi thứ (có trả phí ghi): ${r2['cache_all_per_m']:.4f}/1M-token. Chỉ cache nhóm đáng cache: "
        f"${r2['optimized_per_m']:.3f}/1M-token, tiết kiệm thêm ${saved:.2f}/ngày (≈ ${saved * DAYS:.2f}/tháng).",
        f"- Mô hình cũ coi phí ghi bằng 0 nên báo ${r2['naive_cache_per_m']:.3f}/1M-token. Con số đó lạc quan quá mức.",
        "- **Insight:** model nhỏ cần số lần đọc gần gấp đôi model lớn mới hòa vốn, vì phí lưu trữ cố định không giảm theo giá model. "
        "web-search và nightly-eval gần như không đọc lại prefix (N ≈ 0.08), nên bật cache ở đó là mất tiền.",
    ]
    return "\n".join(rows)


def _section_ext_reasoning(r2) -> str:
    rb = r2["reasoning"]
    rows = [
        "## Your Turn #4: Ngân sách Reasoning",
        "",
        "| | Reasoning | Thường |",
        "|---|---|---|",
        f"| Chi phí $/ngày | ${rb['reasoning_cost_daily']:.2f} ({rb['cost_pct']}%) | ${rb['plain_cost_daily']:.2f} |",
        f"| Năng lượng Wh/ngày | {rb['reasoning_wh_daily']:,.0f} ({rb['wh_pct']}%) | {rb['plain_wh_daily']:,.0f} |",
        f"| Tỷ lệ traffic / token | {rb['traffic_pct']}% / {rb['token_pct']}% | {100 - rb['traffic_pct']:.1f}% / {100 - rb['token_pct']:.1f}% |",
        "",
        f"Một query reasoning tốn **{rb['wh_per_query_ratio']}×** năng lượng so với query thường. Lý do là hai yếu tố nhân với nhau: "
        f"hệ số ×{sustainability.REASONING_ENERGY_MULTIPLIER:.0f} mỗi token (chuỗi suy luận dài, decode memory-bound) "
        "và output dài hơn nhiều lần do thinking tokens.",
        "",
        "**Quy tắc routing:** chỉ cho reasoning khi prompt đủ phức tạp. Dùng `input_tokens` làm proxy độ phức tạp, "
        f"request bị hạ cấp bỏ thinking tokens (output còn {rb['demoted_output_tokens']} token, bằng median của query thường).",
        "",
        "| Cap reasoning | Ngưỡng input_tokens | Giữ / hạ cấp | Tiết kiệm $/tháng | Tiết kiệm Wh/ngày | Giảm năng lượng |",
        "|---|---|---|---|---|---|",
    ]
    for sc in rb["scenarios"]:
        thr = f"≥ {sc['min_input_tokens']:,}" if sc["min_input_tokens"] is not None else "không cần"
        rows.append(f"| {sc['cap_pct']:.0f}% | {thr} | {sc['reasoning_kept']} / {sc['demoted']} | "
                    f"${sc['saved_monthly']:.2f} | {sc['saved_wh_daily']:,.0f} | {sc['wh_reduction_pct']}% |")
    rows += [
        "",
        f"- Cap 10% không tiết kiệm được gì vì hiện chỉ {rb['traffic_pct']}% traffic dùng reasoning. "
        "Tỷ lệ traffic thấp nhưng năng lượng vẫn bị reasoning chi phối, nên phải siết xuống khoảng 2%.",
        "- Với reasoning, lý do chính để giới hạn là **năng lượng và carbon**, không phải tiền: tiết kiệm $ nhỏ nhưng năng lượng giảm mạnh.",
    ]
    return "\n".join(rows)


def _section_ext_carbon(r3) -> str:
    cs = r3["carbon"]
    rows = [
        "## Your Turn #5: Carbon-aware Scheduling",
        "",
        f"Chỉ di chuyển **{len(cs['per_job'])} job interruptible** (training, finetune, eval, sandbox). Chúng chạy spot, "
        "có checkpoint và không phục vụ user realtime, nên không nhạy độ trễ. Năng lượng tính theo giờ chạy thực tế trên spot "
        f"(đã gồm rework và overhead checkpoint): **{cs['kwh_month']:,.0f} kWh/tháng**.",
        "",
        "| Job | kWh/tháng | kgCO2e ở us-east-1 | kgCO2e ở " + cs["cleanest"] + " |",
        "|---|---|---|---|",
    ]
    for pj in cs["per_job"]:
        rows.append(f"| {pj['job_id']} | {pj['kwh']:,.0f} | {pj['carbon_kg_current']:,.1f} | {pj['carbon_kg_cleanest']:,.1f} |")
    rows += [
        "",
        "| Vùng | $/kWh | gCO2/kWh | Tiền điện $/tháng | kgCO2e/tháng | Điểm cân bằng (0 = tốt nhất) |",
        "|---|---|---|---|---|---|",
    ]
    for t in sorted(cs["regions"], key=lambda t: t["balanced_score"]):
        rows.append(f"| {t['region']} | {t['usd_per_kwh']:.3f} | {t['gco2_per_kwh']} | ${t['energy_usd']:,.0f} | "
                    f"{t['carbon_kg']:,.1f} | {t['balanced_score']:.3f} |")
    rows += [
        "",
        "**Vùng tối ưu tùy theo ưu tiên của công ty:**",
        f"- **Rẻ nhất ($):** {cs['cheapest']}",
        f"- **Sạch nhất (CO2):** {cs['cleanest']}",
        f"- **Cân bằng (trọng số 50/50 sau khi chuẩn hóa min-max):** {cs['balanced']}",
        "",
        f"- Chuyển cả {len(cs['per_job'])} job từ us-east-1 sang {cs['cleanest']}: **giảm {cs['carbon_saved_kg']:,.0f} kgCO2e/tháng "
        f"({cs['carbon_saved_pct']}%)**, đồng thời tiết kiệm ${cs['energy_usd_saved']:,.0f}/tháng tiền điện. So với vùng hiện tại thì không có trade-off về tiền.",
        f"- Giữa {cs['cheapest']} và {cs['cleanest']} thì có trade-off thật. {cs['cleanest']} chỉ thắng về tổng chi phí khi công ty áp "
        f"**giá carbon nội bộ > ${cs['carbon_price_breakeven']:,}/tCO2**, cao hơn nhiều mức giá carbon thị trường "
        "(khoảng $50–150/t). Vì vậy câu trả lời phụ thuộc mục tiêu: nếu tối ưu tiền thì chọn "
        f"{cs['cheapest']}; nếu có cam kết net-zero hoặc báo cáo ESG thì chọn {cs['cleanest']}.",
        "- **Trade-off độ trễ:** vùng sạch nhất (Bắc Âu) xa user Mỹ/châu Á nhất. Điều này không ảnh hưởng job batch hay training, "
        "nhưng inference realtime nên giữ gần user. Ngoài ra cần tính phí egress khi chuyển dataset hoặc checkpoint lớn "
        "giữa các vùng, và các ràng buộc data residency (GDPR).",
    ]
    return "\n".join(rows)


def _section_showback(r4) -> str:
    total = sum(r4["by_team"].values())
    rows = [
        "## Showback theo team (M4)",
        "",
        "| Team | $/ngày (inference, đã tối ưu) | Tỷ trọng |",
        "|---|---|---|",
    ]
    for team, cost in sorted(r4["by_team"].items(), key=lambda kv: -kv[1]):
        rows.append(f"| {team} | ${cost:.2f} | {cost / total * 100:.1f}% |")
    rows += [
        "",
        f"Tag coverage **{r4['tag_coverage']:.0%}** ≥ ngưỡng 80%, nên có thể chuyển sang chargeback. "
        "Dữ liệu được xuất theo chuẩn FOCUS (`outputs/focus_export.csv`) để gộp được với hóa đơn của các cloud/neocloud khác.",
    ]
    return "\n".join(rows)


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    r4 = m4_allocation.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]

    idle_savings = r1["idle_waste_daily"] * DAYS
    rightsize_savings = 0.0
    for lie in r1["lies"]:
        cur = lie["gpu_type"]
        tgt = RIGHTSIZE_MAP.get(cur, cur)
        delta = num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"])
        rightsize_savings += max(0.0, delta) * 24 * DAYS

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size util-lies": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get),
    }

    sections = [
        _section_unit_economics(r2, levers, baseline, optimized, r2["total_tokens"] * DAYS),
        _section_util_lie(r1, cat),
        _section_actions(levers, r2, r3, r4),
        _section_showback(r4),
        _section_sustainability(cat, r2),
        _section_ext_cache(r2),
        _section_ext_reasoning(r2),
        _section_ext_carbon(r3),
    ]
    md = report.build_report(baseline, optimized, levers, sustainability=sust, extra_sections=sections)
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"), baseline=baseline)

    if verbose:
        if hasattr(_sys.stdout, "reconfigure"):  # Vietnamese text on Windows cp1252 consoles/pipes
            _sys.stdout.reconfigure(encoding="utf-8")
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


if __name__ == "__main__":
    run()
