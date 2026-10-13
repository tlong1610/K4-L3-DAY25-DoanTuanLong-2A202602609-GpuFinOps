# Lab 25 — GPU FinOps: Bài viết ngắn

**Học viên:** Đoàn Tuấn Long — 2A202602609
**Kết quả tự động:** `verify.py` 11/11 · `pytest` 23/23 (15 test gốc + 8 test cho extension)
**Extension đã làm:** #3 `cache_is_worth_it()` · #4 Ngân sách Reasoning · #5 Carbon-aware Scheduling
**Chi tiết số liệu:** [`report.md`](report.md) · biểu đồ waterfall [`savings.png`](savings.png)

---

## 1. Baseline vs. Optimized

| | Baseline | Optimized | Tiết kiệm |
|---|---|---|---|
| Tổng chi GPU / tháng | $27,133 | $14,641 | **$12,492 (46.0%)** |
| Toàn fleet quy về $/1M-token (226M token/tháng) | $120.06 | $64.79 | 46.0% |
| Riêng inference $/1M-token | $6.488 | $1.192 | 81.6% |

Đo bằng `$/1M-token` cho thấy điều mà `$/GPU-giờ` che mất. Phần inference rẻ đi 81.6% dù giá thuê GPU không đổi, vì cùng số tiền ta phục vụ được nhiều token hơn. Ngược lại, purchasing giảm giá giờ GPU nhưng không làm GPU hiệu quả hơn.

## 2. Phân tích từng đòn bẩy

| Đòn bẩy | $/tháng | Tỷ trọng trong tổng tiết kiệm |
|---|---|---|
| Purchasing (spot/reserved) | $10,040 | 80.4% |
| Inference (cascade/cache/batch) | $1,197 | 9.6% |
| Right-size util-lies | $655 | 5.2% |
| Kill idle GPUs | $600 | 4.8% |

- **Purchasing đóng góp lớn nhất** vì hóa đơn fleet GPU ($25,667/tháng on-demand) lớn hơn hóa đơn token rất nhiều. 5 job interruptible chuyển sang spot kèm checkpoint. 3 job inference có duty cycle 75–100% (trên ngưỡng hòa vốn 55% = 1 − 45% discount) chuyển sang reserved 3 năm. Riêng M3 giảm 39.1%.
- **Inference có tỷ lệ cao nhất (81.6%) nhưng tiền tuyệt đối nhỏ.** Theo $/1M-token cộng dồn: $6.488 → cascade **$1.523** → cache **$1.433** → batch **$1.192**. Cascade chiếm phần lớn: model nhỏ rẻ hơn 15× (input) / 37.5× (output), và 79% request đủ dễ cho model nhỏ. Các chiết khấu nhân với nhau (batch + 100% cache = 0.5 × 0.1 = 0.05).
- **Kill idle:** `gpu-h100-5` để trống 8 giờ/ngày, tốn $20/ngày ($600/tháng). Tắt tự động khi util < 10% là việc dễ nhất.

## 3. GPU-Util "nói dối"

| GPU | GPU-Util | MFU | MBU |
|---|---|---|---|
| gpu-h100-4 | 98.2% | 0.194 | 0.207 |
| gpu-a10g-1 | 96.9% | 0.268 | 0.302 |
| 4 con H100 khỏe (so sánh) | 93–95% | ~0.41 | ~0.44 |

`nvidia-smi` đo **% thời gian có kernel đang chạy**, không đo số SM/Tensor Core thực sự làm việc. `gpu-h100-4` có util cao hơn cả các H100 khỏe nhưng chỉ đạt khoảng một nửa MFU của chúng. Cả MFU lẫn MBU đều thấp, nên GPU không chạm trần compute hay trần băng thông của roofline. Nó đang chờ: kernel nhỏ hoặc launch overhead, batch quá nhỏ, đồng bộ host↔device, data loader, hoặc NCCL.

**Tác động tài chính:** so với MFU mục tiêu 0.40, hai GPU này phí khoảng **$1,165/tháng**. Ta trả nguyên giá giờ H100 ($2.5) mà chỉ nhận khoảng 1/5 FLOPs đỉnh. Nên sửa trước (profile bằng Nsight, tăng batch size, CUDA Graphs). Nếu không sửa được thì right-size xuống tier thấp hơn: H100→A100 và A10G→L4 tiết kiệm $655/tháng.

## 4. Các phần mở rộng

### #3 — `cache_is_worth_it()` (`finops/pricing.py`, `missions/m2_inference_levers.py`)

- **Mô hình:** ghi cache tốn thêm `0.25 × giá input + $0.05/1M` (phụ phí ghi kiểu Anthropic + phí lưu trữ kiểu Gemini). Mỗi lần đọc lại tiết kiệm `0.9 × giá input`. Điểm hòa vốn là **N\* = phí ghi / tiết kiệm mỗi lần đọc**: large **0.30**, small **0.56** lần đọc.
- **Đo trên dữ liệu:** số lần đọc thực tế được ước lượng từ tỷ lệ cache hit, `N = h/(1−h)`. chat và doc-qa có N ≈ 1.0, vượt ngưỡng nên **bật cache**. web-search và nightly-eval có N ≈ 0.08, dưới ngưỡng nên **tắt cache**. Nhóm untagged: large bật (0.39 > 0.30), small tắt (0.45 < 0.56).
- **Trước/sau:**

  | Kịch bản | $/1M-token |
  |---|---|
  | Code gốc (cache mọi thứ, coi phí ghi bằng 0) | $1.126 (lạc quan quá mức) |
  | Cache mọi thứ, trả đủ phí ghi | $1.230 |
  | Chỉ cache nhóm đáng cache | **$1.192** |

  Chính sách mới rẻ hơn 3.1% so với cache mù quáng.
- **Insight:** model nhỏ cần số lần đọc gần gấp đôi model lớn mới hòa vốn, vì phí lưu trữ cố định không giảm theo giá model. Nghĩa là cache càng khó có lời khi model càng rẻ.

### #4 — Ngân sách Reasoning (`missions/m2_inference_levers.py`)

| | Reasoning | Toàn bộ |
|---|---|---|
| Traffic | 8.4% | 2,400 request/ngày |
| Chi phí $ | 15.7% ($1.41/ngày) | $8.98/ngày |
| Năng lượng | **94.0%** (29,788 Wh/ngày) | 31,676 Wh/ngày |

- **Mỗi query reasoning tốn 172.6× năng lượng so với query thường.** Có hai yếu tố nhân với nhau. Thứ nhất là hệ số ×80 mỗi token, vì chuỗi suy luận dài và decode bị giới hạn bởi băng thông bộ nhớ. Thứ hai là output dài hơn khoảng 6× (trung vị 4,062 so với 641 token) do thinking tokens.
- **Quy tắc routing:** chỉ cho reasoning khi `input_tokens ≥ ngưỡng`, dùng độ dài prompt làm proxy cho độ phức tạp. Request bị hạ cấp bỏ thinking tokens.

  | Cap | Ngưỡng | $ tiết kiệm/tháng | Năng lượng giảm |
  |---|---|---|---|
  | 10% | không cần | $0 | 0% |
  | 5% | ≥ 2,034 token | $13.57 | 31.5% |
  | **2%** | **≥ 3,213 token** | **$23.44** | **68.2%** |

- **Insight:** cap 10% vô nghĩa với dữ liệu này vì hiện chỉ 8.4% traffic dùng reasoning. Tỷ lệ traffic thấp nhưng năng lượng vẫn bị reasoning chi phối. Với reasoning, lý do chính để giới hạn là **năng lượng và carbon**, không phải tiền.
- **Giả định:** dữ liệu không có cột độ phức tạp nên dùng độ dài prompt thay thế. Output của request bị hạ cấp lấy bằng trung vị của query thường (641 token).

### #5 — Carbon-aware Scheduling (`missions/m3_purchasing.py`)

- **Phạm vi:** chỉ di chuyển 5 job interruptible (train-llm, train-embed, finetune, dev-sandbox, batch-eval). Các job này chạy spot, có checkpoint và không phục vụ user realtime. Năng lượng tính theo giờ chạy thực tế trên spot, đã gồm rework và overhead checkpoint: **4,460 kWh/tháng**.
- **Bảng 5 vùng** (tiền điện và carbon/tháng cho 5 job này):

  | Vùng | $/kWh | gCO2/kWh | Tiền điện | kgCO2e |
  |---|---|---|---|---|
  | us-east-wa | 0.055 | 90 | $245 | 401 |
  | us-west-2 | 0.070 | 120 | $312 | 535 |
  | europe-north1 | 0.090 | 30 | $401 | **134** |
  | us-east-1 (hiện tại) | 0.120 | 380 | $535 | 1,695 |
  | europe-central2 | 0.180 | 660 | $803 | 2,943 |

- **Kết quả:** chuyển từ us-east-1 sang europe-north1 **giảm 1,561 kgCO2e/tháng (92.1%)** và tiết kiệm thêm $134/tháng tiền điện. Riêng job-train-llm chiếm 79% lượng carbon giảm được.
- **Vùng nào "tối ưu thực sự"?** Phụ thuộc ưu tiên của công ty:
  - Rẻ nhất về $: us-east-wa.
  - Sạch nhất về CO2: europe-north1.
  - Cân bằng (trọng số 50/50 sau khi chuẩn hóa): us-east-wa.

  europe-north1 chỉ thắng us-east-wa về tổng chi phí khi giá carbon nội bộ **> $583/tCO2**, cao hơn nhiều giá carbon thị trường. Vì vậy chọn europe-north1 là một quyết định ESG, không phải quyết định về tiền.
- **Trade-off:**
  - Bắc Âu xa user Mỹ/châu Á, nên không phù hợp cho inference realtime.
  - Chuyển dataset hoặc checkpoint lớn giữa các vùng tốn phí egress.
  - Cần tính tới ràng buộc data residency.

## 5. Khuyến nghị cho NimbusAI: 3 hành động đầu tiên

1. **Tuần 1–2, đổi tier mua GPU** ($10,040/tháng). Chuyển job interruptible sang spot kèm checkpoint. Chỉ reserve phần tải nền chạy ≥ 55% thời gian. Cùng lúc bật auto-shutdown cho GPU idle ($600/tháng, gần như không rủi ro).
2. **Tháng 1, đưa router inference lên production.** Gồm cascade, cache theo `cache_is_worth_it`, batch API cho eval/offline, và cap reasoning ở khoảng 2%. $/1M-token giảm từ $6.49 xuống $1.19, năng lượng inference giảm khoảng 68%. Lever này tăng giá trị khi traffic tăng.
3. **Đổi thước đo: theo dõi MFU/MBU thay cho GPU-Util và báo cáo chi phí theo $/1M-token.** Dùng chúng để tìm và sửa các GPU như `gpu-h100-4` trước khi mua thêm phần cứng. Tag coverage đã đạt 92% (≥ 80%), nên chuyển từ showback sang chargeback để mỗi team chịu trách nhiệm chi phí của mình. Đặt các job interruptible ở europe-north1: carbon giảm 92% so với us-east-1 và tiền điện của chúng cũng giảm ($535 → $401/tháng). Tiền điện chỉ chiếm 2.8% tiền thuê GPU, nên đi theo hướng xanh gần như không tốn thêm tiền.

---

## Phụ lục: trả lời câu hỏi oral check

1. **GPU-Util 98% có nghĩa là hiệu quả?** Không. Đó chỉ là % thời gian có kernel chạy. `gpu-h100-4` có util 98% nhưng MFU 0.19.
2. **Vì sao cần tag coverage ≥ 80% mới chargeback?** Phần chưa gắn tag sẽ không ai chịu hoặc bị chia sai cho team khác. Thu tiền dựa trên dữ liệu thiếu làm các team mất niềm tin vào hóa đơn. Dưới 80% chỉ nên showback.
3. **70% workload interruptible thì tối ưu thế nào?** Chuyển phần đó sang spot kèm checkpoint (tiết kiệm khoảng 35–60% sau khi trừ rework và overhead checkpoint). Phần 30% chạy ổn định thì reserved nếu duty cycle ≥ 55%. Chọn vùng sạch và rẻ cho job interruptible vì chúng không nhạy độ trễ.
4. **Khi nào $/GPU-giờ và $/1M-token cho kết quả ngược nhau?** Khi GPU rẻ hơn nhưng throughput thấp hơn, hoặc GPU đắt hơn nhưng batch tốt hơn. Ví dụ: hạ từ H100 xuống A10G làm $/GPU-giờ giảm, nhưng nếu tok/s giảm mạnh hơn thì $/1M-token lại tăng.
5. **Vì sao decode là memory-bound còn prefill là compute-bound?** Prefill xử lý cả prompt trong một lượt, nên mỗi lần đọc trọng số được dùng cho nhiều token (khoảng 455 FLOP/byte, trên ridge ~295 của H100). Decode sinh từng token một, mỗi bước phải đọc lại toàn bộ trọng số và KV cache cho rất ít FLOPs (khoảng 1–2 FLOP/byte). Vì vậy decode bị giới hạn bởi băng thông HBM, nên đo bằng MBU.
