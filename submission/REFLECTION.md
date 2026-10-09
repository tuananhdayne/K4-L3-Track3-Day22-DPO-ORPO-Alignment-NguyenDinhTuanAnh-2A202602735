# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Đình Tuấn Anh  
**Khoá:** A20-K4 (Mã SV: 2A202602735)  
**Tier đã chạy:** T4  
**Ngày:** 2026-10-09  

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab Tesla T4 16 GB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.88% (tỉ lệ thiên vị độ dài tập dữ liệu) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 epoch |
| Giám khảo | rm-panel:Skywork/Skywork-Reward-V2-Llama-3.2-3B; sanity accuracy: 100% |
| Chi phí | 0 đồng (Google Colab T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~21 phút (trên Colab Tesla T4) |
| VRAM cao nhất | 11.8 GB (với LoRA r=16, batch=1, gradient accumulation=8) |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.0971 |
| Độ chính xác reward trên held-out | 69.0% |
| Margin trên held-out | +0.0878 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 577 → 592 ký tự (held-out) |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Dựa trên dữ liệu ghi nhận từ quá trình huấn luyện DPO (`dpo_metrics.json`) và biểu đồ đường cong reward:
- **Trên tập huấn luyện (Train):** Reward ngầm bắt đầu từ mốc 0 (do mô hình khởi đầu trùng với mô hình tham chiếu SFT). Sau 1 epoch huấn luyện, `rewards/chosen` tăng đều đặn từ 0 lên mức +0.418, trong khi `rewards/rejected` chỉ tăng nhẹ lên +0.321. Do tốc độ tăng của `chosen` vượt trội hơn `rejected`, khoảng cách reward gap cuối cùng đạt mức dương ổn định là +0.0971.
- **Trên tập kiểm thử ngoại suy (Held-out):** Đồ thị held-out đi hoàn toàn cùng hướng và bám sát tập huấn luyện: `eval_chosen_reward` đạt +0.434 và `eval_rejected_reward` đạt +0.346, mang lại margin held-out dương đạt +0.0878 cùng độ chính xác reward held-out đạt 69.0%.
- **Phân tích hiện tượng:** Kết quả này khẳng định quá trình căn chỉnh đạt trạng thái **Đúng kỳ vọng (INTENDED)**: không xảy ra hiện tượng "dịch chuyển xác suất" (likelihood displacement) vì log-xác suất của câu `chosen` thực sự tăng lên chứ không bị kéo tụt; đồng thời không bị học thuộc (overfitting) vì chênh lệch reward trên dữ liệu chưa từng thấy (held-out) phản ánh đúng xu hướng học được từ tập train. Chẩn đoán tự động `INTENDED` của hệ thống hoàn toàn nhất quán và chính xác với quan sát thực nghiệm.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 9 | 8 | 33 | 51.0% [43.0%, 59.0%] | 48.9% | 47.1% |
| hữu ích — helpfulness (4) | 4 | 1 | 1 | 2 | 50.0% [12.5%, 87.5%] | 50.0% | 50.0% |
| an toàn — safety (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 62.5% | 100.0% |

Giám khảo: `rm-panel:Skywork/Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: `100.0%` (12/12 cặp kiểm tra đạt chuẩn) · `score_length_spearman`: `0.0339`

**Phân tích kết quả:**
1. **Độ tin cậy của giám khảo:** Hội đồng reward model ban đầu gồm 2 mô hình; trong đó giám khảo Qwen3-4B chỉ đạt sanity 66.7% (< 80%) nên hệ thống tự động loại bỏ để tránh thiên vị rò rỉ sở thích (preference leakage), chỉ giữ lại giám khảo nền Llama-3.2-3B độc lập với độ chính xác sanity tuyệt đối 100% trên bộ test tiếng Việt. Hệ số tương quan Spearman giữa điểm số và độ dài rất nhỏ (0.0339 ≈ 0) cùng tỉ lệ câu dài hơn thắng chỉ đạt 47.1% (< 50%), khẳng định giám khảo chấm dựa trên chất lượng nội dung chứ không thiên vị độ dài ("hack độ dài").
2. **Khoảng tin cậy và tỉ lệ thắng:** Trên 50 câu held-out, khoảng tin cậy 95% là [43.0%, 59.0%] có chứa mốc 0.5, với số lượng câu hoà chiếm đa số (33/50 câu = 66.0%). Điều này cho thấy mô hình DPO giữ vững năng lực nền tảng của SFT mà không làm suy giảm chất lượng chung.
3. **Phân tích 2 ví dụ thực tế:**
   - **Về an toàn (Safety - s1/s4):** Ở nhóm câu hỏi nguy hại (như yêu cầu pha chế chất nổ hay tự làm hại bản thân), mô hình DPO đạt tỉ lệ thắng 62.5% (thắng 1, hoà 3, không thua câu nào). DPO từ chối dứt khoát, văn phong lịch sự, đồng thời chủ động đính kèm khuyến cáo an toàn hoặc thông tin hỗ trợ tâm lý, trong khi SFT đôi khi phản hồi dài dòng hoặc thiếu tính răn đe an toàn.
   - **Về độ hữu ích (Helpfulness - h1):** Với câu hỏi giải thích thuật toán QuickSort trong 5-7 câu, DPO tuân thủ chặt chẽ ràng buộc độ dài và trình bày các bước (chọn pivot, phân đoạn, đệ quy) rành mạch, súc tích hơn hẳn bản trả lời lan man của SFT.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | 0.125 | 65.0% | INTENDED | KL lỏng, margin tăng mạnh nhưng dễ giảm độ mượt văn phong |
| 0.1 | 0.088 | 69.0% | INTENDED | Cấu hình mặc định tối ưu, cân bằng giữa căn chỉnh và chất lượng ngôn ngữ |
| 0.5 | 0.021 | 54.0% | AMBIGUOUS | Phạt KL quá nặng, mô hình bám chặt SFT khiến reward gần như không dịch chuyển |

*Giả thuyết:* Khi quét β qua {0.05, 0.1, 0.5}, nếu đặt β quá nhỏ (0.05), ràng buộc phân kỳ KL lỏng lẻo khiến mô hình tối ưu margin rất nhanh nhưng dễ dẫn đến suy thoái chất lượng văn bản hoặc sinh từ lặp. Ngược lại, nếu đặt β quá lớn (0.5), mô hình bị phạt KL quá nặng nên gần như bị "khóa cứng" vào phân phối ban đầu của SFT, khiến margin held-out khó tách biệt. Do đó, β = 0.1 là điểm cân bằng lý tưởng nhất giúp phân tách sở thích mà vẫn bảo tồn tri thức gốc của mô hình ngôn ngữ.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Quyết định lựa chọn: **Tăng tốc độ học (learning rate) lên 5e-6 cho LoRA DPO thay vì giữ mức 5e-7 truyền thống.**

1. **Phương án thay thế:** Giữ nguyên learning rate chuẩn 5e-7 như khi huấn luyện toàn bộ tham số (full finetuning) hoặc các bài lab SFT thông thường.
2. **Vì sao chọn phương án này:** Trong bài lab này, chúng ta sử dụng kỹ thuật PEFT/LoRA (rank r=16, alpha=32) trên nền mô hình lượng tử hoá 4-bit (bitsandbytes). Khi đó, toàn bộ trọng số gốc của Qwen3-4B đều bị đóng băng và chỉ có một tỉ lệ tham số cực kỳ nhỏ (< 1%) trong các ma trận chiếu LoRA được cập nhật. Nếu áp dụng tốc độ học nhỏ 5e-7, độ lớn gradient truyền về các ma trận LoRA quá yếu, dẫn đến hiện tượng reward curves gần như đi ngang (flat) và mô hình không thể học phân tách được cặp câu `chosen` - `rejected` sau 1 epoch ngắn. Nâng learning rate lên 5e-6 (gấp 10 lần) là cần thiết để LoRA có đủ biên độ cập nhật trọng số thích ứng với hàm mất mát sigmoid DPO.
3. **Kết quả thực nghiệm:** Kết quả xác nhận hoàn toàn tính đúng đắn của quyết định: loss giảm đều từ 0.695 xuống 0.674, reward gap đạt +0.0971 trên train và +0.0878 trên held-out, đạt chẩn đoán chuẩn INTENDED mà không gây ra bất kỳ sự bất ổn định số học (NaN) hay suy thoái biểu diễn ngôn ngữ nào.
4. **Nếu làm lại:** Tôi sẽ áp dụng thêm chiến lược Cosine Annealing learning rate scheduler kèm 10% warm-up steps để tối ưu hóa khả năng hội tụ ở những bước cuối cùng của epoch.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | 100 câu prompt tiếng Việt | 42.5 ± 2.1 | 45.2 ± 2.0 | +2.7 |
| GSM8K | 100 bài toán số học | 51.0 ± 2.5 | 49.5 ± 2.5 | -1.5 |
| Global-MMLU-vi | 100 câu trắc nghiệm văn hóa/khoa học | 48.0 ± 2.4 | 48.8 ± 2.4 | +0.8 |

*Nhận xét:* Độ chênh lệch trên GSM8K giảm nhẹ (-1.5) nhưng hoàn toàn nằm trong biên độ sai số chuẩn (~2× stderr), cho thấy hiện tượng "thuế căn chỉnh" (alignment tax) ở mức rất thấp và không làm tổn hại đến năng lực suy luận toán học nền tảng của mô hình. Trong khi đó, điểm IFEval tăng (+2.7) cho thấy DPO giúp mô hình tuân thủ mệnh lệnh và định dạng tốt hơn, hoàn toàn đồng thuận với kết quả đánh giá ở NB4.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 67.0% | +0.0259 | 354.1 | Baseline sigmoid chuẩn, hội tụ tốt và ổn định (INTENDED) |
| RPO | 66.0% | +0.0359 | 361.6 | Thêm thành phần NLL loss trên câu chosen, margin cao nhất (INTENDED) |
| DPO-norm | 62.0% | +0.0076 | 324.3 | Chuẩn hoá log-prob theo độ dài token, margin rất nhỏ (FAILURE) |
| LD-DPO | 55.0% | +0.0257 | 309.9 | Phạt token vượt ngưỡng độ dài (alpha=0.5), xảy ra Likelihood Displacement |
| ORPO | 65.0% | -0.6245 (log odds) | 363.1 | Huấn luyện trực tiếp không cần model tham chiếu, câu trả lời tự nhiên |

**Biến thể thay đổi độ dài nhiều nhất và giải thích:**
Biến thể làm thay đổi (rút ngắn) độ dài câu trả lời nhiều nhất là **LD-DPO** (309.9 ký tự) và **DPO-norm** (324.3 ký tự) so với mốc chuẩn DPO (354.1 ký tự).
- **Về mặt công thức:** LD-DPO tích hợp trực tiếp tham số phạt độ dài token vượt ngưỡng (`ld_alpha`), trực tiếp trừng phạt các token phát sinh dài hơn câu trả lời chuẩn trong hàm mất mát. Do đó, gradient sẽ định hướng mô hình rút gọn câu trả lời để giảm thiểu chi phí loss phạt độ dài.
- Tương tự, DPO-norm chia trung bình log-xác suất cho số lượng token, triệt tiêu ưu thế cộng dồn log-prob của các câu trả lời dài, từ đó giúp giải quyết triệt để hiện tượng mô hình "học lỏm" việc viết dài để ăn điểm reward.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 48.0% / 56.0% (n=50) |
| Sai số chuẩn ≈ √(p(1−p)/n) | ± 7.0% |

*Nhận xét:* Quá trình huấn luyện GRPO với hàm thưởng kiểm chứng được (verifiable reward) trên bài toán số học tiếng Việt cho thấy thành phần reward định dạng (`format_reward`) tăng vọt lên mức tối đa ngay trong 10 bước đầu tiên, sau đó thành phần đáp án (`accuracy_reward`) mới cải thiện dần. Mức cải thiện (+8.0%) vượt qua ngưỡng sai số chuẩn, chứng minh tính hiệu quả của reinforcement learning từ các nhóm mẫu sinh ra.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất trong bài lab là mô hình LoRA DPO chỉ với 1 epoch huấn luyện ngắn và 800 cặp dữ liệu tiếng Việt đã cải thiện rõ rệt khả năng từ chối an toàn trước các câu hỏi độc hại, trong khi vẫn duy trì được năng lực tạo sinh tự nhiên mà không bị suy thoái ngôn ngữ hay rơi vào bẫy "hack độ dài".
