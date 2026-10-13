# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,641  
**Projected savings:** $12,492  (**46%**)

## Savings by lever

| Lever | Savings (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,197 |
| Purchasing (spot/reserved) | $10,040 |
| Right-size util-lies | $655 |
| Kill idle GPUs | $600 |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cheapest+cleanest region: europe-north1

## Đơn vị kinh tế: $/1M-token

### Inference: từng đòn bẩy cộng dồn (M2)

| Bước | $/ngày | $/1M-token | Giảm so với baseline |
|---|---|---|---|
| Baseline (all large model) | $48.87 | $6.488 | 0.0% |
| + Cascade routing | $11.48 | $1.523 | 76.5% |
| + Cache-aware prompt caching | $10.80 | $1.433 | 77.9% |
| + Batch API | $8.98 | $1.192 | 81.6% |

Cascade là đòn bẩy inference lớn nhất: 6.488 → 1.523 $/1M-token. Model nhỏ rẻ hơn 15× (input) / 37.5× (output), và 79% request đủ dễ để model nhỏ xử lý. Cache và batch chồng lên sau đó, vì các chiết khấu nhân với nhau.

### Toàn bộ fleet GPU quy về $/1M-token

Tổng chi GPU (inference + training/serving fleet) chia cho 226M token phục vụ mỗi tháng.

| Đòn bẩy | Tiết kiệm $/tháng | % baseline | Tương đương $/1M-token |
|---|---|---|---|
| Inference (cascade/cache/batch) | $1,197 | 4.4% | $5.30 |
| Purchasing (spot/reserved) | $10,040 | 37.0% | $44.43 |
| Right-size util-lies | $655 | 2.4% | $2.90 |
| Kill idle GPUs | $600 | 2.2% | $2.65 |
| **Baseline → Optimized** | **$27,133 → $14,641** | **46.0%** | **$120.06 → $64.79** |

## Vì sao GPU-Util "nói dối" (M1)

`nvidia-smi` báo GPU-Util = **% thời gian có ít nhất một kernel đang chạy** trong cửa sổ lấy mẫu. Nó không đo bao nhiêu SM/Tensor Core thực sự làm việc, cũng không đo bao nhiêu FLOPs sinh ra. Một kernel nhỏ chiếm 1 SM trên tổng 132 SM của H100 vẫn được tính là "bận 100%". MFU (FLOPs đạt được / FLOPs đỉnh) mới là thước đo thứ ta trả tiền để thuê.

| GPU | Loại | GPU-Util | MFU | MBU | Lãng phí so với MFU mục tiêu 0.40 ($/tháng) |
|---|---|---|---|---|---|
| gpu-h100-4 | H100 | 98.2% | 0.194 | 0.207 | $927 |
| gpu-a10g-1 | A10G | 96.9% | 0.268 | 0.302 | $238 |

**So sánh trực tiếp:** 4 con H100 khỏe có util 93.1–95.2%, tức **gần bằng** gpu-h100-4 (98.2%). Nhưng MFU trung bình của chúng là 0.41, gấp **2.1×**. Nhìn GPU-Util thì không thể phân biệt hai nhóm này.

**Cơ chế gốc rễ:** ở các GPU bị flag, **cả MFU lẫn MBU đều thấp** (MFU 0.19–0.27, MBU 0.21–0.30). Vậy GPU không chạm trần compute, cũng không chạm trần băng thông HBM (roofline), tức nó đang *chờ*, không phải đang tính:
- **Kernel launch overhead / batch quá nhỏ:** nhiều kernel ngắn, mỗi kernel chỉ phủ một phần nhỏ SM, nên clock "bận" mà Tensor Core rảnh.
- **Đồng bộ host↔device, data loader CPU chậm, chờ NCCL all-reduce:** kernel vẫn "đang chạy" trong khi chờ, nên util vẫn cao.
- **Memory stall:** với decode LLM (arithmetic intensity ~1–2 FLOP/byte, thấp hơn nhiều so với ridge ~295 của H100), workload bị giới hạn bởi băng thông. Khi đó nên tối ưu và đo bằng MBU, không phải MFU.

**Tác động tài chính:** ta trả nguyên giá giờ GPU nhưng chỉ nhận một phần FLOPs. Riêng các GPU bị flag lãng phí khoảng **$1,165/tháng** so với mức MFU 0.40. Cách xử lý: profile bằng Nsight, tăng batch size, dùng CUDA Graphs / torch.compile để giảm launch overhead. Nếu không sửa được thì right-size xuống GPU rẻ hơn (lever "Right-size util-lies").

## Đề xuất hành động (xếp theo ROI = $ tiết kiệm / công sức)

| Ưu tiên | Hành động | Tiết kiệm $/tháng | Công sức | Lý do |
|---|---|---|---|---|
| 1 | Purchasing (spot/reserved) | $10,040 | Trung bình | Đổi tier mua: job interruptible sang spot (cần checkpoint), job chạy ≥55% thời gian sang reserved. Rủi ro là cam kết 3 năm, nên chỉ reserve phần tải nền ổn định. |
| 2 | Kill idle GPUs | $600 | Thấp | Chỉ cần auto-shutdown / scale-to-zero khi util < 10%. Không đụng tới model, gần như không rủi ro. |
| 3 | Inference (cascade/cache/batch) | $1,197 | Trung bình | Sửa router (cascade), bật cache theo chính sách cache_is_worth_it, đẩy traffic không cần realtime sang Batch API. |
| 4 | Right-size util-lies | $655 | Cao | Phải profile (Nsight) và sửa nút thắt trước, hoặc benchmark GPU nhỏ hơn rồi migrate. Tốn công nhất. |

1. **Purchasing trước tiên** vì đây là lever lớn nhất và không cần sửa code model. Đưa 5 job interruptible (job-train-llm, job-train-embed, job-finetune, job-dev-sandbox, job-batch-eval) sang spot kèm checkpoint. Đưa 3 job inference chạy ổn định (job-infer-chat, job-infer-rag, job-infer-search) sang reserved, vì duty cycle của chúng ≥ ngưỡng hòa vốn 55%. Riêng M3 đã tiết kiệm 39.1% chi phí workload.
2. **Kill idle là quick win:** công sức thấp nhất, làm ngay trong tuần đầu.
3. **Inference:** cascade + cache-aware + batch đưa $/1M-token từ $6.488 xuống $1.192. Về tỷ lệ thì cao (81.6%), nhưng về tiền tuyệt đối thì nhỏ hơn purchasing vì hóa đơn token hiện nhỏ hơn hóa đơn fleet GPU. Lever này sẽ quan trọng dần khi traffic tăng.
4. **Right-size cuối cùng:** nên sửa nút thắt (batch, CUDA Graphs) trước khi đổi phần cứng.
5. **Giới hạn reasoning ở ~2% traffic:** tiết kiệm 68.2% năng lượng inference ($23.44/tháng). Đây chủ yếu là lever về carbon (xem Your Turn #4).
6. **Chargeback:** tag coverage 92% ≥ 80% nên có thể chuyển từ showback sang chargeback theo team. Nên gắn tag nốt phần request untagged để hóa đơn mỗi team đầy đủ.

## Showback theo team (M4)

| Team | $/ngày (inference, đã tối ưu) | Tỷ trọng |
|---|---|---|
| assistant | $2.84 | 31.6% |
| search | $2.56 | 28.5% |
| eval | $1.82 | 20.2% |
| rag | $1.77 | 19.7% |

Tag coverage **92%** ≥ ngưỡng 80%, nên có thể chuyển sang chargeback. Dữ liệu được xuất theo chuẩn FOCUS (`outputs/focus_export.csv`) để gộp được với hóa đơn của các cloud/neocloud khác.

## Bền vững: carbon gắn với chi phí điện

Fleet GPU (workloads.csv) tiêu thụ khoảng **5,895 kWh/tháng** (chỉ tính TDP của GPU). Chi phí điện và carbon nếu chạy ở từng vùng:

| Vùng | $/kWh | gCO2/kWh | Tiền điện $/tháng | tCO2e/tháng |
|---|---|---|---|---|
| europe-north1 | 0.090 | 30 | $531 | 0.18 |
| us-east-wa | 0.055 | 90 | $324 | 0.53 |
| us-west-2 | 0.070 | 120 | $413 | 0.71 |
| us-east-1 | 0.120 | 380 | $707 | 2.24 |
| europe-central2 | 0.180 | 660 | $1,061 | 3.89 |

- **Vùng khuyến nghị: europe-north1** (thủy điện Na Uy, sạch nhất). So với us-east-1 hiện tại, carbon giảm **92%** (2.24 → 0.18 tCO2e/tháng) và tiền điện cũng giảm ($707 → $531/tháng). Ở đây cắt carbon và cắt tiền đi cùng chiều.
- **Rẻ nhất ≠ sạch nhất:** us-east-wa có giá điện thấp nhất ($0.055/kWh) nhưng carbon gấp 3× europe-north1. Nếu chỉ tối ưu $ thì chọn vùng này (phân tích trade-off ở Your Turn #5). europe-central2 (Ba Lan) vừa đắt nhất vừa bẩn nhất, nên tránh.
- **Tiền điện chỉ ≈ 2.8% tiền thuê GPU** ($707 so với $25,667/tháng on-demand). Vì vậy chuyển job interruptible (training, eval) sang vùng sạch gần như không tốn thêm tiền mà cắt mạnh carbon. Job inference realtime thì phải cân nhắc độ trễ tới người dùng: Bắc Âu xa user Mỹ/châu Á.
- **Reasoning là "quả bom năng lượng":** chỉ 8.4% request nhưng chiếm 94.0% năng lượng inference (172.6× Wh/query so với query thường).

## Your Turn #3: `cache_is_worth_it()`

Cache chỉ có lời khi tiền tiết kiệm từ các lần đọc lại lớn hơn chi phí ghi (phụ phí ghi +25% giá input và phí lưu trữ $0.05/1M token). Hòa vốn khi số lần đọc N\* = phí ghi / (giá input × 0.9). Số lần đọc thực tế được ước lượng từ tỷ lệ cache hit h theo công thức N = h/(1−h).

| Project | Tier | Requests | Cache hit | N thực tế | N\* hòa vốn | Bật cache? |
|---|---|---|---|---|---|---|
| doc-qa | large | 95 | 51.6% | 1.07 | 0.30 | ✅ |
| chat | large | 161 | 50.1% | 1.01 | 0.30 | ✅ |
| (untagged) | large | 40 | 28.2% | 0.39 | 0.30 | ✅ |
| nightly-eval | large | 77 | 7.2% | 0.08 | 0.30 | ❌ |
| web-search | large | 125 | 7.1% | 0.08 | 0.30 | ❌ |
| doc-qa | small | 422 | 51.0% | 1.04 | 0.56 | ✅ |
| chat | small | 571 | 50.6% | 1.03 | 0.56 | ✅ |
| (untagged) | small | 157 | 30.9% | 0.45 | 0.56 | ❌ |
| web-search | small | 448 | 7.2% | 0.08 | 0.56 | ❌ |
| nightly-eval | small | 304 | 7.2% | 0.08 | 0.56 | ❌ |

- Cache mọi thứ (có trả phí ghi): $1.2304/1M-token. Chỉ cache nhóm đáng cache: $1.192/1M-token, tiết kiệm thêm $0.29/ngày (≈ $8.67/tháng).
- Mô hình cũ coi phí ghi bằng 0 nên báo $1.126/1M-token. Con số đó lạc quan quá mức.
- **Insight:** model nhỏ cần số lần đọc gần gấp đôi model lớn mới hòa vốn, vì phí lưu trữ cố định không giảm theo giá model. web-search và nightly-eval gần như không đọc lại prefix (N ≈ 0.08), nên bật cache ở đó là mất tiền.

## Your Turn #4: Ngân sách Reasoning

| | Reasoning | Thường |
|---|---|---|
| Chi phí $/ngày | $1.41 (15.7%) | $7.57 |
| Năng lượng Wh/ngày | 29,788 (94.0%) | 1,888 |
| Tỷ lệ traffic / token | 8.4% / 16.5% | 91.6% / 83.5% |

Một query reasoning tốn **172.6×** năng lượng so với query thường. Lý do là hai yếu tố nhân với nhau: hệ số ×80 mỗi token (chuỗi suy luận dài, decode memory-bound) và output dài hơn nhiều lần do thinking tokens.

**Quy tắc routing:** chỉ cho reasoning khi prompt đủ phức tạp. Dùng `input_tokens` làm proxy độ phức tạp, request bị hạ cấp bỏ thinking tokens (output còn 641 token, bằng median của query thường).

| Cap reasoning | Ngưỡng input_tokens | Giữ / hạ cấp | Tiết kiệm $/tháng | Tiết kiệm Wh/ngày | Giảm năng lượng |
|---|---|---|---|---|---|
| 10% | không cần | 201 / 0 | $0.00 | 0 | 0.0% |
| 5% | ≥ 2,034 | 120 / 81 | $13.57 | 9,977 | 31.5% |
| 2% | ≥ 3,213 | 48 / 153 | $23.44 | 21,612 | 68.2% |

- Cap 10% không tiết kiệm được gì vì hiện chỉ 8.4% traffic dùng reasoning. Tỷ lệ traffic thấp nhưng năng lượng vẫn bị reasoning chi phối, nên phải siết xuống khoảng 2%.
- Với reasoning, lý do chính để giới hạn là **năng lượng và carbon**, không phải tiền: tiết kiệm $ nhỏ nhưng năng lượng giảm mạnh.

## Your Turn #5: Carbon-aware Scheduling

Chỉ di chuyển **5 job interruptible** (training, finetune, eval, sandbox). Chúng chạy spot, có checkpoint và không phục vụ user realtime, nên không nhạy độ trễ. Năng lượng tính theo giờ chạy thực tế trên spot (đã gồm rework và overhead checkpoint): **4,460 kWh/tháng**.

| Job | kWh/tháng | kgCO2e ở us-east-1 | kgCO2e ở europe-north1 |
|---|---|---|---|
| job-train-llm | 3,545 | 1,347.0 | 106.3 |
| job-train-embed | 506 | 192.4 | 15.2 |
| job-finetune | 266 | 101.0 | 8.0 |
| job-dev-sandbox | 76 | 28.9 | 2.3 |
| job-batch-eval | 66 | 25.3 | 2.0 |

| Vùng | $/kWh | gCO2/kWh | Tiền điện $/tháng | kgCO2e/tháng | Điểm cân bằng (0 = tốt nhất) |
|---|---|---|---|---|---|
| us-east-wa | 0.055 | 90 | $245 | 401.4 | 0.048 |
| us-west-2 | 0.070 | 120 | $312 | 535.1 | 0.131 |
| europe-north1 | 0.090 | 30 | $401 | 133.8 | 0.140 |
| us-east-1 | 0.120 | 380 | $535 | 1,694.6 | 0.538 |
| europe-central2 | 0.180 | 660 | $803 | 2,943.3 | 1.000 |

**Vùng tối ưu tùy theo ưu tiên của công ty:**
- **Rẻ nhất ($):** us-east-wa
- **Sạch nhất (CO2):** europe-north1
- **Cân bằng (trọng số 50/50 sau khi chuẩn hóa min-max):** us-east-wa

- Chuyển cả 5 job từ us-east-1 sang europe-north1: **giảm 1,561 kgCO2e/tháng (92.1%)**, đồng thời tiết kiệm $134/tháng tiền điện. So với vùng hiện tại thì không có trade-off về tiền.
- Giữa us-east-wa và europe-north1 thì có trade-off thật. europe-north1 chỉ thắng về tổng chi phí khi công ty áp **giá carbon nội bộ > $583/tCO2**, cao hơn nhiều mức giá carbon thị trường (khoảng $50–150/t). Vì vậy câu trả lời phụ thuộc mục tiêu: nếu tối ưu tiền thì chọn us-east-wa; nếu có cam kết net-zero hoặc báo cáo ESG thì chọn europe-north1.
- **Trade-off độ trễ:** vùng sạch nhất (Bắc Âu) xa user Mỹ/châu Á nhất. Điều này không ảnh hưởng job batch hay training, nhưng inference realtime nên giữ gần user. Ngoài ra cần tính phí egress khi chuyển dataset hoặc checkpoint lớn giữa các vùng, và các ràng buộc data residency (GDPR).

_Figures are June-2026 as-of snapshots; re-baseline before acting._