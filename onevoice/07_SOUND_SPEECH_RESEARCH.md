# ToneBridge — Sound & Speech Research Report (Nhánh Sound Speech)

> **Version:** 1.1 · **Owner:** Khải (Sound/Speech) · **Status:** Research complete → feeds ADR-001 execution
> **Phạm vi:** Khảo sát paper/model cho audio front-end, denoise, VAD, ASR, NMT (VI-EN / VI-KR), prosody & urgency — ràng buộc **edge AI, 100% offline, Snapdragon NPU, môi trường bệnh viện**.
> **Ngày:** 2026-07-11 · **v1.1:** thêm ràng buộc "noise chỉ dùng data có sẵn, không tự thu" + nghiên cứu sâu chuỗi tiếng Hàn (KR = Korean) §3.1, §6.3 · **v1.2:** tích hợp 2 paper team sưu tầm (folder `paper research/`) + abstract TinyLlama-iOS → §5.5 Moonshine ASR, §11 energy & LLM-NMT; cập nhật G5/G8

---

## 0. Kết luận nhanh (TL;DR cho người không đọc hết)

1. **Kiến trúc hybrid tiered trong ADR-001 là ĐÚNG và được literature 2024–2026 ủng hộ mạnh** — thậm chí mạnh hơn ta nghĩ: nghiên cứu mới nhất trên **medical ASR** cho thấy neural denoise mặc định phải coi là **có hại cho WER cho đến khi chứng minh ngược lại**.
2. **Nâng cấp guardrail ADR-001 từ 2 chế độ (ON/OFF) thành 3 chế độ: ON / OFF / OA (Observation Adding)** — trộn `α·enhanced + (1−α)·noisy` trước khi vào ASR. OA là kỹ thuật đã được chứng minh giảm artifact mà vẫn giữ lợi ích khử ồn, chi phí ≈ 0 (một phép cộng có trọng số).
3. **Denoiser đề xuất: GTCRN (ICASSP 2024) thay vì DeepFilterNet3 làm ứng viên chính** — nhẹ hơn ~40 lần (48K params vs ~2M), RTF 0.07 trên CPU, MIT license, có sẵn streaming ONNX → không cần tranh NPU với Whisper. DeepFilterNet3 giữ làm fallback chất lượng.
4. **ASR (cập nhật v1.2): ứng viên số 1 mới là Moonshine-Tiny đơn ngữ vi/ko/en (27M params/model)** — paper team sưu tầm ([arXiv:2509.02523](https://arxiv.org/abs/2509.02523)): tiếng Việt WER 13.0–18.8, **vượt cả Whisper-medium lớn hơn 28×**; tiếng Hàn CER 8.9–14.9, ngang Whisper-small; ONNX sẵn, sherpa-onnx hỗ trợ Android. Whisper-base (đường AI Hub dọn sẵn) + PhoWhisper thành fallback. Chi tiết §5.4.
5. **NMT: Opus-MT KHÔNG có pair vi↔ko.** VI⇄KR bắt buộc chọn: (a) **pivot VI→EN→KR bằng 2 model Opus-MT** hoặc (b) **NLLB-200-distilled-600M INT8 dịch trực tiếp**. Đây là dữ kiện quyết định ADR-002 cuối W2. Fine-tune en↔vi bằng **MedEV (~360K cặp câu y khoa)** nếu còn thời gian.
6. **VAD: Silero VAD làm baseline (đã chốt trong tech req), benchmark thêm TEN VAD (2025)** — nhẹ hơn, chính xác hơn trong noisy conditions, có C library cho Android.
7. **Prosody/urgency: hướng feature-based (f0 + energy + rate → MLP nhỏ) là đúng đắn** — literature xác nhận model ~47K params đạt UAR ~0.62 trên SER; với bài toán 2 lớp (neutral/distress) trên scripted data, MLP trên prosody features là đủ và pitch-safe.
8. ⚠️ **Piper KHÔNG hỗ trợ tiếng Hàn** — dòng "Piper (vi, en, ko voices)" trong tech req §3 là **sai dữ kiện**. Thay thế cho KR: **MeloTTS-Korean** (MIT license, CPU near-realtime). Phải báo team ML ngay W1 vì ảnh hưởng cả prosody-transfer path.
9. **Noise: 100% từ dataset có sẵn, không tự thu** (ràng buộc mới). Nguồn đủ dùng: Kaggle Hospital Ambient Noise + Freesound CC0 + MUSAN/DNS-Challenge cho robustness chung + **tự sinh alarm beep bằng code** — alarm y tế là tone chuẩn hoá (IEC 60601-1-8) nên synthesize chính xác hơn cả đi thu. Chi tiết §3.1.
10. **Tiếng Hàn đo bằng CER, không phải WER** (đặc thù chữ Hangul); Whisper zero-shot trên KsponSpeech chỉ đạt WER ~29%/CER ~14% (large-v2) → KR là mắt xích ASR yếu nhất, cần scripted demo + cân nhắc checkpoint fine-tuned. Dữ liệu song ngữ Việt-Hàn tốt nhất tìm được: **UViko ~454K cặp câu**. Chi tiết §6.3.

---

## 1. Bằng chứng khoa học: Neural denoise và ASR — mối quan hệ nguy hiểm

Đây là phần quan trọng nhất vì nó quyết định số phận Stage 1 (neural denoise trong Branch A).

### 1.1 Các paper then chốt

| Paper | Nguồn | Phát hiện chính | Ứng dụng cho ToneBridge |
|---|---|---|---|
| **When De-noising Hurts: A Systematic Study of Speech Enhancement Effects on Modern Medical ASR** (2025) | [arXiv:2512.17562](https://arxiv.org/abs/2512.17562) | Test 4 ASR (Whisper, Parakeet, Gemini Flash, Parrotlet) × noise conditions với MetricGAN+ denoiser: **cả 40/40 cấu hình đều TỆ ĐI khi bật denoise** (semantic WER tăng 1.1–46.6 điểm tuyệt đối). Khuyến nghị: ASR hiện đại đã đủ noise-robust, denoise preprocessing "có thể vừa phí compute vừa có hại". | Bằng chứng trực diện cho guardrail ADR-001, **trên đúng domain y tế**. Kỳ vọng mặc định (prior) của ta phải là: Stage 1 OFF. Bật chỉ khi WER matrix của chính ta nói ngược lại. |
| **How Bad Are Artifacts?: Analyzing the Impact of Speech Enhancement Errors on ASR** (Iwamoto et al., 2022) | [arXiv:2201.06685](https://arxiv.org/abs/2201.06685) | Phân rã lỗi SE thành noise-error vs artifact-error: **artifact là thủ phạm chính** làm ASR tệ đi. Đề xuất **Observation Adding (OA)**: nội suy tín hiệu enhanced với tín hiệu gốc → tăng signal-to-artifact ratio một cách đơn điệu. | **Kỹ thuật rẻ nhất trong toàn bộ report này:** `x_asr = α·denoised + (1−α)·noisy`. Thêm 1 dòng code, thêm 1 cột vào WER matrix (ON/OFF/OA với α=0.5–0.8). |
| **How does end-to-end speech recognition training impact speech enhancement artifacts?** (NTT, 2023) | [arXiv:2311.11599](https://arxiv.org/abs/2311.11599) | Xác nhận cơ chế: ASR train end-to-end với noisy data "học" cách dùng noise context; SE front-end tạo distribution mismatch. | Giải thích *tại sao* Whisper (train 680K giờ noisy) không cần denoise sạch. |
| **Training-Free Intelligibility-Guided Observation Addition for Noisy ASR** (2026) | [arXiv:2602.20967](https://arxiv.org/html/2602.20967) | Chọn trọng số OA theo intelligibility estimate từ chính ASR backend, không cần train gì thêm. | Phiên bản nâng cao của OA nếu α cố định không đủ tốt. Để dành cho Phase 2, không làm trong 6 tuần. |
| **Speaker Reinforcement Using Target Source Extraction for Robust ASR** (2022) | [arXiv:2205.04433](https://arxiv.org/pdf/2205.04433) | Remix enhanced + noisy giảm ~23–25% WER so với unprocessed trên CHiME-4. | Bằng chứng OA/remix có thể **thắng cả noisy gốc** — tức Stage 1 + OA có cửa thắng thật, không chỉ hòa. |

### 1.2 Hệ quả thiết kế (đề xuất sửa đổi nhỏ vào ADR-001)

- Giữ nguyên cây pipeline. **Chỉ đổi guardrail:** benchmark 3 chế độ **OFF / ON / OA(α)** thay vì 2. Tiêu chí giữ nguyên: chế độ nào WER tốt nhất trên hospital-noise test set thì ship chế độ đó; ON/OA phải thắng OFF ≥ 2 điểm WER tuyệt đối mới được bật.
- Ghi rõ trong Tech Spec: *"Chúng tôi biết literature 2025 cho thấy denoise thường làm medical ASR tệ đi (arXiv:2512.17562); pipeline của chúng tôi có công tắc và đo lường thay vì mặc định tin denoise."* — đây là điểm cộng lớn về technical credibility trước giám khảo.
- Neural denoise vẫn có giá trị ngoài WER: (a) tín hiệu sạch hơn cho demo nghe lại, (b) nếu sau này cần voice cloning/reference audio cho TTS. Nhưng **không được viện lý do đó để bật nó trên đường vào ASR**.

---

## 2. Chọn neural denoiser (Stage 1, Branch A)

| Model | Params / Compute | Realtime | Deployment | License | Nhận xét |
|---|---|---|---|---|---|
| **GTCRN** (ICASSP 2024) — **đề xuất chính** | **48.2K params, 33 MMACs/s** | Streaming RTF **0.07** trên CPU consumer | ONNX sẵn, tích hợp sherpa-onnx; đủ nhẹ chạy CPU audio thread, **không cần NPU** | MIT | Xấp xỉ DeepFilterNet (1.8M params) về PESQ/SISNR trên VCTK-DEMAND dù nhỏ hơn ~40×. Nhẹ đến mức bật/tắt A-B test tức thì. Repo: [Xiaobin-Rong/gtcrn](https://github.com/Xiaobin-Rong/gtcrn) |
| **DeepFilterNet3** — fallback chất lượng | ~2.3M params, ~0.35G MACs | Realtime trên embedded (paper DFN2 demo trên Raspberry Pi) | Rust/LADSPA + ONNX export được | MIT/Apache | Chất lượng khử ồn cao hơn GTCRN ở non-stationary khó; trả giá bằng compute. Chỉ nâng cấp nếu GTCRN thua rõ ở SNR 5dB. [arXiv:2205.05474](https://arxiv.org/pdf/2205.05474) |
| RNNoise | ~85K, cực nhẹ | RTF rất thấp | C library | BSD | Bị GTCRN vượt ở cùng mức compute — chỉ dùng nếu cần giải pháp C thuần 1 ngày công. |
| H-GTCRN / UL-UNAS (2025) | siêu nhẹ, tuned low-SNR | — | Cùng repo tác giả GTCRN | MIT | Theo dõi; không đủ chín để đặt cược trong 6 tuần. [GTCRN topic](https://www.emergentmind.com/topics/gtcrn-model) |

**Lý do đổi đề xuất từ DeepFilterNet3 → GTCRN làm primary:**
1. Tech req budget cho denoise là ≤10MB INT8 chạy NPU/GPU — GTCRN chỉ ~200KB, chạy **CPU** thoải mái → giải phóng NPU hoàn toàn cho Whisper encoder (stage đói compute nhất), tránh tranh chấp scheduler.
2. Vì rẻ, việc chạy song song cả 3 chế độ OFF/ON/OA khi benchmark không tốn gì.
3. Rủi ro: GTCRN train chủ yếu trên DNS/VCTK-DEMAND noise — cần verify trên hospital noise của ta (W3). Nếu thua DFN3 ≥ 2 điểm WER ở 5dB thì đổi, đường compile không đổi (đều ONNX).

---

## 3. DSP front-end (Stage 0) — thông số từ đo đạc thực tế ICU

Nghiên cứu acoustic ICU ([Springer 2024](https://link.springer.com/article/10.1007/s40857-024-00321-3), [JMIR 2025](https://medinform.jmir.org/2025/1/e35987)) xác nhận phổ ồn bệnh viện đúng như phân tích ADR-001:

- **Ventilator/HVAC airflow: < 250 Hz** → HPF 80 Hz là đúng hướng; cân nhắc **thử thêm biến thể HPF 120–150 Hz** trong benchmark (f0 giọng nam thấp nhất ~85–100 Hz, cần cắt dưới ngưỡng này để không phạm vào pitch của Branch B — 80 Hz là an toàn nhất, biến thể cao hơn chỉ dành cho Branch A).
- **Alarm/monitor beep: 1–3 kHz, đa âm chồng lấn** → notch filter cố định từng tần số chỉ bắt được một phần; spectral gating adaptive là công cụ chính cho lớp này. Chấp nhận beep lọt qua Stage 0 → để GTCRN/OA xử lý.
- Alarm detection ở 0dB SNR đạt F1 0.967 với CRNN ([JMIR 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12611226/)) — không cần làm trong prototype, nhưng là hướng Phase 2 hay: phát hiện alarm → tự động bật notch đúng tần số (adaptive notch bank).

### 3.1 Kế hoạch noise data — 100% từ nguồn có sẵn (KHÔNG tự thu)

> **Ràng buộc mới (2026-07-11):** team không có điều kiện thu âm noise tại bệnh viện. Toàn bộ noise lấy từ dataset công khai + tổng hợp bằng code. Điều này **không làm yếu benchmark** — noise injection có kiểm soát còn cho phép quét SNR chính xác hơn thu thật, và giữ được cặp (clean, noisy) cho ground-truth f0 (§7.2).

| Nguồn | Nội dung | License | Vai trò trong benchmark |
|---|---|---|---|
| [Hospital Ambient Noise (Kaggle)](https://www.kaggle.com/datasets/nafin59/hospital-ambient-noise) | 562 chunk × 5s ồn bệnh viện thật | kiểm tra trang dataset khi tải | **Test set chính** — noise "đúng domain" duy nhất, để dành cho eval, không dùng tune |
| Freesound (query: "hospital", "ICU", "monitor beep", "emergency room", filter CC0/CC-BY) | clip lẻ beep, ER ambience, cart, tiếng khóc | CC0 / CC-BY (ghi credit) | Bổ sung lớp non-stationary (khóc, va chạm, người nói xen) |
| **`alarm_synth.py` — tự sinh bằng code** ⭐ | Alarm y tế theo chuẩn **IEC 60601-1-8** (melodic alarm 3-5 nốt, tần số cơ bản 150–1000 Hz + hài; SpO2 beep ~ 660–990 Hz biến thiên theo độ bão hoà) | của mình | Stress-test notch/spectral gating với beep **tần số biết trước** — kiểm soát hoàn toàn, chính xác hơn đi thu; quét được mọi tần số alarm |
| [MUSAN](https://ar5iv.labs.arxiv.org/html/1510.08484) (noise subset, ~6h/929 files) | ồn kỹ thuật + ambient tổng hợp | Public Domain / CC ([HF mirror](https://huggingface.co/datasets/FluidInference/musan)) | Robustness chung + data train/tune (không đụng test set) |
| [DNS Challenge noise](https://arxiv.org/pdf/2005.07551) (Microsoft) | bộ noise lớn, đa môi trường | mở cho research | Nguồn tune spectral-gating threshold; cũng là distribution GTCRN đã học → sanity check denoiser |
| Babble noise: tự trộn từ MUSAN speech / LibriSpeech nhiều speaker ⭐ | overlapping speech (ồn khó nhất của ER) | theo nguồn gốc | Lớp non-stationary khắc nghiệt nhất trong WER matrix |

**Quy tắc vệ sinh dữ liệu:** Kaggle hospital set = **eval-only**; mọi tune (threshold DSP, α của OA, VAD threshold) chỉ dùng MUSAN/DNS/Freesound-train split. Không để "tune trên test".

**Lưu ý phân biệt:** ràng buộc chỉ áp dụng cho *noise*. Utterance giọng nói scripted (câu triage, distress/neutral cho urgency classifier) vẫn thu bằng giọng team tại nhà như kế hoạch — đây là speech, thu ở phòng yên tĩnh rồi inject noise bằng script, không cần đến bệnh viện.
- Inject tại SNR 20/10/5 dB theo tech req; script log SNR thực đo (không chỉ SNR danh nghĩa).

---

## 4. VAD

| | Silero VAD | TEN VAD (2025) |
|---|---|---|
| Size | ~2 MB | nhỏ hơn Silero |
| Độ chính xác trong noise | tốt, chuẩn công nghiệp | **cao hơn Silero, false-positive thấp hơn** trên LibriSpeech/GigaSpeech/DNS ([GitHub](https://github.com/TEN-framework/ten-vad)) |
| Compute | RTF 0.004 CPU | thấp hơn Silero |
| Android | ONNX | **C library chính chủ cho Android/iOS** |
| Độ chín | rất chín, cộng đồng lớn | mới open-source 2025 |

**Quyết định:** giữ Silero làm baseline W2 (đúng tech req, ít rủi ro), nhưng thêm **1 buổi benchmark TEN VAD trên noisy clips bệnh viện** ở W2 — nếu false-trigger do alarm beep thấp hơn rõ rệt thì swap (cùng interface, chi phí swap ~nửa ngày). VAD false-trigger vì beep là rủi ro riêng của môi trường ta mà không benchmark công khai nào cover.

**Lưu ý tương tác:** VAD chạy **sau Stage 0 DSP** (tín hiệu đã bớt ù nền → VAD chính xác hơn) và **trước Stage 1** (đúng như ADR-001: denoise chỉ chạy khi VAD active để tiết kiệm pin).

---

## 5. ASR — Whisper trên Snapdragon + PhoWhisper cho tiếng Việt

### 5.1 Đường deploy đã được Qualcomm dọn sẵn
- [Whisper-Small-Quantized trên AI Hub](https://aihub.qualcomm.com/models/whisper_small_quantized) ([HF mirror](https://huggingface.co/qualcomm/Whisper-Small-Quantized)): **w8a16**, MHA→SHA, linear→conv để hợp HTP; encoder chạy NPU. Có sẵn cả [Whisper-Base](https://aihub.qualcomm.com/iot/models/whisper_base) và Whisper-Tiny làm nấc lùi latency.
- **Cảnh báo budget:** Whisper-small ~244M params → INT8 ≈ 240MB+, **vượt budget 100MB trong tech req §3**. Whisper-base (74M → ~75MB INT8) mới nằm trong budget. Cần sửa budget hoặc chấp nhận small với budget mới — quyết ở W3 khi có số latency thật. Tổng 500MB on-disk vẫn khả thi nếu NMT gọn.

### 5.2 PhoWhisper — vũ khí cho VI input
- [PhoWhisper (VinAI, ICLR 2024)](https://arxiv.org/abs/2406.02555), [GitHub](https://github.com/VinAIResearch/PhoWhisper): fine-tune Whisper multilingual trên **844 giờ tiếng Việt đa giọng vùng miền**, SOTA WER tiếng Việt, vượt mọi baseline wav2vec2. Có đủ cỡ tiny→large.
- **Kiến trúc y hệt Whisper** → cùng pipeline quantize/compile AI Hub mà Trí đã làm cho Whisper gốc. Chi phí thêm chủ yếu là disk (một bộ weight thứ hai).
- **Đề xuất routing:** app luôn biết hướng dịch (utterance-by-utterance, 2 chiều) → **VI-input dùng PhoWhisper-base/small, EN/KO-input dùng Whisper-base/small multilingual**. Giọng Việt là mặt yếu nhất của Whisper gốc và là ngôn ngữ của bệnh nhân/y tá chính trong demo — đây là chỗ đáng tiêu disk budget nhất.
- Nếu disk không cho phép 2 model: fallback dùng 1 Whisper-small multilingual cho cả 3 ngôn ngữ (như plan gốc), PhoWhisper thành stretch goal.

### 5.3 Eval data tiếng Việt y khoa (mới, rất trúng đề)
- **ViMedCSS** (2026): [Vietnamese Medical Code-Switching Speech Dataset & Benchmark](https://arxiv.org/pdf/2602.12911) — đúng hiện tượng thật ở bệnh viện VN (bác sĩ trộn thuật ngữ EN vào câu Việt). Dùng làm test set bổ sung cho WER matrix nếu license cho phép.
- VIVOS/Common Voice vi cho clean baseline như kế hoạch.

### 5.4 Moonshine — ASR đơn ngữ tiny, ứng viên số 1 mới (v1.2, từ paper team sưu tầm)

**[Flavors of Moonshine: Tiny Specialized ASR Models for Edge Devices](https://arxiv.org/abs/2509.02523)** (Moonshine AI, 2025 — bản PDF trong `paper research/2509.02523v1.pdf`):

- Luận điểm: với model đủ nhỏ (27M params), **đơn ngữ thắng đa ngữ** — ngược với trực giác "multilingual transfer". Train trên ~173K giờ data/ngôn ngữ (gấp 10 lần Whisper gốc mỗi tiếng).
- Số liệu trực tiếp cho 2 ngôn ngữ của ta (bảng 3-4 trong paper):
  | | Moonshine-Tiny (27M) | so với Whisper |
  |---|---|---|
  | **Vietnamese** (WER) | 18.8 (CV17) / **13.0 (Fleurs)** | thắng whisper-tiny 80.5 điểm; thắng whisper-small (9×) 10.5 điểm; **thắng cả whisper-medium (28×) 2.6 điểm** |
  | **Korean** (CER) | 14.9 (CV17) / **8.9 (Fleurs)** | thắng whisper-tiny 14.1 điểm; **hòa whisper-small** (−0.0); kém whisper-medium 2.2 điểm |
- **Noise robustness:** paper đo trực tiếp — ổn định đến ~**20dB SNR**, suy giảm dần dưới đó → khớp đúng dải benchmark 20/10/5dB của ta; WER matrix sẽ cho biết Moonshine chịu hospital noise thế nào so với Whisper (Whisper train 680K giờ noisy có thể lì đòn hơn ở 5dB — **phải đo, không đoán**).
- **Kiến trúc & latency:** encoder-decoder nhưng **input variable-length, không pad 30s như Whisper** → latency tỉ lệ độ dài câu — lợi thế lớn cho câu triage ngắn. Đánh giá trên đúng Zeroth-Korean (test set ta đã chọn cho G8).
- **Deployment:** license permissive; ONNX sẵn ([onnx-community/moonshine-tiny-ko-ONNX](https://huggingface.co/onnx-community/moonshine-tiny-ko-ONNX), tương tự cho vi); **[sherpa-onnx đã hỗ trợ Moonshine trên Android](https://k2-fsa.github.io/sherpa/onnx/moonshine/index.html)** (có sẵn bản ko quantized). 27M → chạy CPU thoải mái nếu NPU compile trục trặc.
- **Hệ quả kiến trúc phải xử lý:** Moonshine đơn ngữ **không có language-ID** như Whisper → chiều dịch tự động chuyển sang cơ chế **confidence-race**: chạy song song 2 ASR tiny của pair (2×27M — vẫn rẻ hơn 1 Whisper-small), chọn kết quả confidence cao hơn; fallback Whisper LID. Đã cập nhật vào doc 02 §3, doc 03 §2-3, doc 04 §2.2.
- **Rủi ro:** model mới (2025), chưa có recipe AI Hub sẵn như Whisper; đường lùi = chạy CPU qua sherpa-onnx hoặc quay về Whisper-base (recipe Qualcomm dọn sẵn). Cả hai đường đều đã ghi vào risk table doc 06.

### 5.5 Lưu ý vận hành Whisper/Moonshine trong noise
- Whisper hay **hallucinate trên đoạn im lặng/noise thuần** → VAD gating (đã có trong thiết kế) là bắt buộc, không phải tùy chọn; thêm rule chặn output khi VAD-active < 300ms.
- Beam=1 (greedy) + `condition_on_previous_text=False` giảm cả latency lẫn hallucination loop — đưa vào config mặc định khi benchmark.

---

## 6. NMT — VI⇄EN dễ, VI⇄KR là bài toán thật

### 6.1 Dữ kiện quan trọng nhất: **Opus-MT không có pair vi↔ko**
Kiểm tra kho Helsinki-NLP: có [opus-mt-vi-en](https://huggingface.co/Helsinki-NLP/opus-mt-vi-en), [opus-mt-en-vi](https://huggingface.co/Helsinki-NLP/opus-mt-en-vi), [opus-mt-ko-en](https://huggingface.co/Helsinki-NLP/opus-mt-ko-en), [opus-mt-tc-big-en-ko](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-ko) — **không tồn tại vi-ko/ko-vi trực tiếp**. Tech req §3 dòng "Opus-MT vi↔ko" cần sửa. Hai phương án cho **ADR-002 (deadline cuối W2)**:

| | Phương án A: Pivot VI→EN→KR (2× Opus-MT) | Phương án B: NLLB-200-distilled-600M trực tiếp |
|---|---|---|
| Size | ~4 model nhỏ (~50–80MB/chiếc INT8), nhưng en↔vi đã cần sẵn → chỉ thêm en↔ko | 1 model ~600MB fp32 → **~150–300MB INT8** ([ct2 int8 sẵn trên HF](https://huggingface.co/JustFrederik/nllb-200-distilled-600M-ct2-int8)) phục vụ MỌI hướng |
| Latency | 2 lần dịch tuần tự (~2× 150ms) | 1 lần dịch, model to hơn |
| Chất lượng vi↔ko | Lỗi cộng dồn qua pivot, nhưng mỗi hop là pair giàu data | NLLB train trực tiếp vi↔ko; chất lượng pair thấp-resource của NLLB thường khá |
| Runtime | Marian/ONNX đơn giản | CTranslate2 (có Android build) hoặc ONNX |
| Rủi ro | Thuật ngữ y khoa méo qua 2 hop | 1 model 600M chiếm phần lớn NMT budget |

**Khuyến nghị:** benchmark cả hai ở W2 trên ~100 câu triage (BLEU/chrF + đọc tay bởi người biết tiếng Hàn nếu có). Prior của tôi: **A cho tốc độ ship, B nếu chất lượng pivot tệ rõ**. Điểm ăn tiền: dù chọn gì, en↔vi vẫn là Opus-MT nhỏ và nhanh — primary demo pair không bị ảnh hưởng.

### 6.2 Domain adaptation y khoa EN⇄VI (stretch, sau khi pipeline chạy)
- **MedEV** — [~360K cặp câu EN-VI y khoa (LREC-COLING 2024)](https://aclanthology.org/2024.lrec-main.784/) ([arXiv:2403.19161](https://arxiv.org/pdf/2403.19161)): fine-tune Opus-MT en↔vi vài epoch là cách rẻ nhất nâng chất lượng thuật ngữ y khoa. Kèm **Meddict** (lexicon EN-VI y khoa) cho glossary v0 của Thịnh.
- Nghiên cứu 2025 ([arXiv:2509.15640](https://arxiv.org/abs/2509.15640)) xác nhận: với medical EN↔VI, **terminology-aware cues cải thiện ổn định** → glossary post-check (thay thuật ngữ sai bằng tra từ điển) là kỹ thuật đáng làm ngay cả khi không fine-tune.

### 6.3 Nghiên cứu sâu chuỗi tiếng Hàn (KR = Korean) — mắt xích yếu nhất, soi từng khâu

**"ko" trong toàn bộ tài liệu = tiếng Hàn (Korean, 한국어)** — phục vụ persona P3 (bệnh nhân/gia đình Hàn kiều tại bệnh viện khu FDI).

#### (a) ASR tiếng Hàn — đo bằng CER, không phải WER
- Đặc thù Hangul (âm tiết ghép khối, khoảng trắng không ổn định trong văn nói) khiến cộng đồng đánh giá ASR tiếng Hàn bằng **CER**. WER matrix của ta phải tách cột: WER cho en/vi, **CER cho ko** (tech req §5 đã ghi "WER/CER per language" — làm rõ mapping này).
- Số zero-shot của Whisper: **large-v2 đạt WER 29.05% / CER 13.95% trên KsponSpeech eval** ([nghiên cứu fine-tune KsponSpeech](https://www.eksss.org/archive/view_article?pid=pss-15-3-83)) — nghĩa là whisper-small/base zero-shot sẽ còn kém hơn đáng kể. **KR-input là khâu ASR rủi ro nhất của prototype.**
- Giảm rủi ro theo thứ tự rẻ → đắt: (1) scripted demo utterances cho KR (đã có trong risk plan), (2) dùng **checkpoint Whisper fine-tuned tiếng Hàn có sẵn trên HF** (vd [whisper-medium-ko-zeroth](https://huggingface.co/seastar105/whisper-medium-ko-zeroth), có cả biến thể small của cộng đồng — cùng kiến trúc → cùng đường AI Hub, giống chiến lược PhoWhisper cho VI), (3) tự fine-tune whisper-base trên [Zeroth-Korean](https://huggingface.co/o0dimplz0o/Fine-Tuned-Whisper-Large-v2-Zeroth-STT-KO) (~51h, mở) — chỉ làm nếu (2) không có size phù hợp.
- Dataset eval KR: **Zeroth-Korean test set** (mở, tải tự do) làm clean baseline; KsponSpeech (969h) thuộc AI Hub Hàn Quốc — đăng ký cần duyệt, thực tế khó với team VN, **không đặt kế hoạch phụ thuộc vào nó**.
- Benchmark y khoa tham chiếu: đã có nghiên cứu ASR hội thoại bác sĩ-bệnh nhân tiếng Hàn tại khoa xạ trị ([ScienceDirect 2023](https://www.sciencedirect.com/science/article/abs/pii/S1386505623001302)) — cite trong hồ sơ để chứng minh bài toán có thật, không dùng được data của họ.

#### (b) Dữ liệu song ngữ Việt-Hàn — phát hiện quan trọng: UViko
- **[UViko (Univ. of Ulsan, Harvard Dataverse)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EDWHKB): ~454.751 cặp câu Việt-Hàn** (8.79M token VI / 5.44M token KO), kèm word-sense annotation phía Hàn. Đây là parallel corpus vi-ko công khai lớn nhất tìm được. Việc cần làm W1: tải về, **đọc kỹ license/terms trên Dataverse** (trang chặn scraper nên chưa xác minh tự động được), lấy mẫu 1-2K cặp làm dev/test nội bộ.
- Ứng dụng theo thứ tự: (1) **ngay lập tức** — nguồn xây test set triage vi-ko + đánh giá 2 phương án ADR-002; (2) stretch — fine-tune/LoRA NLLB-600M hướng vi↔ko, hoặc distill một student model nhỏ chuyên vi-ko (Phase 2, không hứa trong 6 tuần).
- AI Hub Hàn Quốc (aihub.or.kr) có corpus ko-vi lớn hơn nhưng **yêu cầu đăng ký cư trú Hàn** ([tham khảo Korpora](https://ko-nlp.github.io/Korpora/en-docs/corpuslist/aihub_translation.html)) — ghi nhận, không phụ thuộc.
- **FLORES-200** có sẵn cặp vie_Latn↔kor_Hang trong devtest (~1012 câu, tải mở) — không tìm được số chrF công bố riêng cho cặp này, nhưng **tự chạy benchmark mất nửa ngày**: NLLB-600M-int8 (trực tiếp) vs Opus-MT pivot qua EN, chấm chrF++ trên FLORES + 100 câu triage từ UViko. Đây chính là dữ liệu quyết định **G2/ADR-002**, và con số này đưa vào Tech Spec rất "ăn điểm" vì không đội nào có.
- Pivot option nhẹ hơn cho hop EN→KO nếu chọn phương án A: [NLLB-200-Distilled-350M-en-ko](https://github.com/newfull5/NLLB-200-Distilled-350M-en-ko) (distill riêng cho en→ko) hoặc [opus-mt-tc-big-en-ko](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-ko).

#### (c) TTS tiếng Hàn — Piper KHÔNG hỗ trợ ⚠️
- Xác nhận từ [Piper discussion #680](https://github.com/rhasspy/piper/discussions/680): **Korean chưa được Piper hỗ trợ**, không có timeline. Tech req §3 ("Piper vi, en, ko voices") phải sửa.
- **Thay thế đề xuất: [MeloTTS-Korean](https://huggingface.co/myshell-ai/MeloTTS-Korean)** (MyShell.ai) — hỗ trợ Korean chính thức, **MIT license, CPU near-realtime**, phù hợp ràng buộc offline/edge. Có tham số speed (điều khiển rate); pitch envelope vẫn áp hậu kỳ bằng WORLD trên waveform → **prosody-transfer path giống hệt Piper**, không phải thiết kế lại.
- Hệ quả kiến trúc: app mang 2 TTS engine (Piper cho vi/en + MeloTTS cho ko). Chấp nhận được vì TTS chạy CPU; cần Thịnh đo latency MeloTTS-Korean trên phone ở W3 (thêm vào bảng latency).
- Fallback nếu MeloTTS-Korean quá chậm trên phone: sherpa-onnx có hệ VITS ONNX đa ngôn ngữ — kiểm kho voice ko tại thời điểm W3; hoặc chấp nhận demo KR-output bằng giọng chậm hơn budget 400ms một chút (KR là secondary pair, "supported, shown briefly" theo risk plan).

---

## 7. Nhánh B — Prosody extraction & Urgency classifier

### 7.1 Bằng chứng cho hướng feature-based
- **ResLSTM-SA (2026)** ([arXiv:2606.03359](https://arxiv.org/pdf/2606.03359)): SER với **46.8K params đạt UAR 0.6232** trên bài đa lớp — chứng minh không cần transformer to cho emotion từ audio. Bài toán của ta còn dễ hơn (nhị phân neutral/distress, scripted, speaker ít) → MLP/CNN nhỏ trên prosody features là đủ, đúng như plan.
- Chuẩn features tham chiếu: **eGeMAPS** (openSMILE) — 88 features trong đó nhóm quan trọng cho arousal/distress: f0 mean/range/slope, loudness, jitter/shimmer, speech rate, HNR. Trích chọn ~10–20 features tính được bằng DSP thuần (pYIN + RMS + syllable rate như plan) là hợp lý; jitter/shimmer thêm được với chi phí thấp một khi đã có f0 frame-level.
- Nếu MLP trên features không đạt 80% recall (US-3): nấc thang tiếp theo là **emotion2vec (~19M params)** làm feature extractor ([được dùng cho edge trong nghiên cứu 2026](https://arxiv.org/pdf/2602.09121)) — nhưng chỉ khi thất bại, vì nó ăn vào compute budget.

### 7.2 Pitch tracking trong noise — rủi ro số 1 của Branch B
- pYIN suy giảm nhanh khi SNR < 10dB. Kế hoạch benchmark f0 extractor phải chạy **trên noisy data sau Stage 0**, không phải trên clean.
- Fallback đã đúng trong tech req: **CREPE-tiny** (~500K params, robust hơn hẳn trong noise) — convert ONNX chạy CPU được. Quyết định pYIN vs CREPE-tiny bằng số: Pearson corr của f0 contour (estimate vs ground truth từ clean gốc trước khi inject noise) tại 10dB — chọn cái ≥ threshold mà rẻ hơn.
- Trick eval hay: vì ta **tự inject noise**, ta luôn có clean reference → ground-truth f0 miễn phí. Thiết kế `noise_inject.py` giữ cặp (clean, noisy) song song ngay từ W1.

### 7.3 Prosody transfer sang TTS (W4, cùng Chi + Thịnh)
- **WORLD vocoder** là công cụ chuẩn để resynthesis với f0 đã scale: phân rã f0/spectral envelope/aperiodicity rồi lắp f0 mới ([tổng quan](https://arxiv.org/pdf/2110.02854)). Python: `pyworld` — làm được prototype trong 1–2 ngày.
- Tham chiếu học thuật để cite trong hồ sơ: **Meta AI — Holistic Cascade System & Human Evaluation Protocol for Expressive S2ST** ([arXiv:2301.10606](https://arxiv.org/pdf/2301.10606)) — đúng bài toán "cascade S2ST giữ prosody", có luôn protocol đánh giá bằng người mà mini-MOS panel của ta có thể mượn format.
- Piper không expose f0 control trực tiếp → đúng như plan: điều khiển **rate qua length_scale** của Piper, còn **pitch envelope áp hậu kỳ qua WORLD** trên waveform output. Fallback (chỉ transfer pitch mean+range+rate, bỏ contour chi tiết) giữ nguyên như Implementation Plan W4.

---

## 8. Ma trận quyết định (decision gates) — cập nhật theo research

| Gate | Tuần | Câu hỏi | Dữ liệu quyết định | Default nếu thiếu số |
|---|---|---|---|---|
| **G1 — VAD** | W2 | Silero hay TEN VAD? | False-trigger rate trên hospital-noise clips (đặc biệt alarm beep) | Silero (chín hơn) |
| **G2 — ADR-002 NMT vi⇄ko** | cuối W2 | Pivot 2×Opus-MT hay NLLB-600M INT8? | chrF++ tự chạy trên **FLORES-200 vie↔kor devtest** + ~100 câu triage lấy từ **UViko** + human read | Pivot (nhẹ, ship nhanh) |
| **G3 — Denoiser** | W3 | GTCRN hay DeepFilterNet3? | WER matrix ở SNR 5dB + latency đo | GTCRN (nhẹ hơn 40×) |
| **G4 — Guardrail ADR-001** | W3 | Stage 1: OFF / ON / OA(α)? | WER matrix clean/20/10/5dB × {OFF, ON, OA α=0.5, OA α=0.8}; cần thắng OFF ≥ 2 điểm | **OFF** (theo arXiv:2512.17562) |
| **G5 — ASR size & routing** (v1.2) | W3 | **Moonshine-Tiny vi/ko/en (đơn ngữ, 27M) vs Whisper-base multilingual vs PhoWhisper?** Kèm quyết định LID: confidence-race 2 ASR hay Whisper LID? | WER/CER trên hospital-noise matrix + latency trên silicon + LID accuracy trên utterance test | **Moonshine routing** (số paper áp đảo; fallback Whisper-base nếu compile/noise-robustness kém) |
| **G6 — f0 extractor** | W3 | pYIN hay CREPE-tiny? | f0 Pearson corr tại 10dB SNR vs clean ground truth | pYIN (zero-dependency) |
| **G7 — TTS tiếng Hàn** (owner: Thịnh, Sound theo dõi vì dính prosody transfer) | W3 | MeloTTS-Korean đạt latency budget trên phone? | Latency TTS đo trên phone (budget 400ms) | MeloTTS-Korean, chấp nhận vượt nhẹ budget vì KR là secondary pair |
| **G8 — ASR tiếng Hàn** (v1.2) | W3 | **Moonshine-tiny-ko (CER 8.9-14.9, ngang whisper-small)** vs Whisper-base vs checkpoint fine-tuned ko? | CER trên Zeroth-Korean test (Moonshine được eval trên đúng set này) + bộ câu triage KR scripted | **Moonshine-tiny-ko** + scripted demo |

---

## 9. Timeline nhánh Sound & Speech (chi tiết hoá W1–W6, khớp 06_IMPLEMENTATION_PLAN)

> Nguyên tắc: mọi tuần đều sinh ra **số đo commit vào `eval/results/`**. Các mục ⭐ là bổ sung/điều chỉnh so với plan gốc do research này.

### W1 — Nền móng + dữ liệu (noise 100% từ nguồn có sẵn — §3.1)
- [ ] **Tải noise datasets** (không tự thu): Kaggle Hospital Ambient (eval-only) + Freesound CC0/CC-BY (beep, ER ambience, tiếng khóc) + MUSAN noise + DNS noise (train/tune) → catalog theo loại (stationary/non-stationary) + license từng clip. ⭐
- [ ] ⭐ `alarm_synth.py`: sinh alarm beep chuẩn IEC 60601-1-8 (melodic 3-5 nốt + SpO2 beep) — thay thế hoàn toàn nhu cầu thu beep thật; tần số biết trước để stress-test notch/gating.
- [ ] ⭐ Babble generator: trộn MUSAN speech nhiều speaker thành overlapping-speech noise (lớp ồn khó nhất của ER).
- [ ] `noise_inject.py`: inject SNR 20/10/5dB, **xuất cặp (clean, noisy) song song** ⭐ (phục vụ ground-truth f0 §7.2), log SNR thực đo.
- [ ] ⭐ **Tải UViko** (Harvard Dataverse) — xác minh license, trích 1-2K cặp làm dev/test vi-ko; tải **FLORES-200 devtest** (vie_Latn, kor_Hang) + **Zeroth-Korean test** → chuyển cho Thịnh/Trí làm nguyên liệu G2/G8.
- [ ] ⭐ **Báo team ML 2 dữ kiện chặn:** (1) Opus-MT không có pair vi-ko; (2) Piper không có voice ko → Thịnh đưa MeloTTS-Korean vào danh sách tải W1.
- [ ] DSP front-end Python prototype: biquad HPF 80Hz + spectral gating (adaptive noise floor); nghe thử trên noisy clips.
- [ ] ⭐ Tải model candidates của nhánh: GTCRN ONNX (streaming), DeepFilterNet3, Silero VAD, TEN VAD, CREPE-tiny — chạy thử laptop, ghi RTF sơ bộ.
- [ ] Interface Contract v1 (với Chi + Trí) — thêm field `denoise_mode: off|on|oa` vào AudioConfig schema ⭐.
- [ ] Thu scripted utterances (giọng team, phòng yên tĩnh tại nhà — vẫn hợp lệ, chỉ noise là không tự thu): ~50 câu triage/ngôn ngữ bắt đầu từ W1 thay vì dồn về sau.

### W2 — DSP realtime + VAD + chuẩn bị benchmark denoise
- [ ] Port DSP sang realtime (Python/C++ callback-style, block 10–20ms); đo latency block.
- [ ] Tích hợp Silero VAD sau Stage 0; đo accuracy + false-trigger trên noisy clips.
- [ ] ⭐ **G1:** benchmark TEN VAD cùng data → chốt VAD.
- [ ] ⭐ Dựng **harness WER/CER matrix**: script chạy {clean, 20, 10, 5dB} × {OFF, ON-GTCRN, ON-DFN3, OA-0.5, OA-0.8} × {en, vi, ko} → JSON vào `eval/results/`. Cột ko dùng **CER**. Noise theo split §3.1 (tune ≠ eval). (Chạy trên laptop với Whisper của Trí — chưa cần phone.)
- [ ] ⭐ Hỗ trợ Thịnh chạy **G2/ADR-002**: benchmark chrF++ NLLB-600M-int8 (vi↔ko trực tiếp) vs pivot Opus-MT vi↔en + en↔ko trên FLORES-200 devtest + 100 câu triage từ UViko (bộ câu là deliverable của mình). Nửa ngày compute, dữ liệu quyết định đắt giá nhất tuần.

### W3 — Số liệu vàng: WER matrix trên silicon + Branch B sống
- [ ] GTCRN quantized chạy trên phone (CPU) — đo RTF thực. Nếu thua chất lượng: DFN3 (**G3**).
- [ ] **Chạy full WER matrix → quyết định G4 (OFF/ON/OA)** — đây là bảng vàng #1 của Tech Spec.
- [ ] ⭐ **G5:** cùng Trí đo PhoWhisper-base vs Whisper-base/small cho VI input; chốt routing + sửa budget ASR trong tech req nếu cần.
- [ ] ⭐ **G8:** đo CER tiếng Hàn của Whisper-base/small trên Zeroth-Korean test + câu triage KR scripted; nếu quá tệ cho demo → thử checkpoint Whisper fine-tuned ko từ HF (cùng đường AI Hub).
- [ ] ⭐ **G7 (theo dõi, owner Thịnh):** MeloTTS-Korean latency trên phone; xác nhận WORLD post-process áp được lên waveform MeloTTS y như Piper (đây là phần dính đến prosody transfer của nhánh mình).
- [ ] ⭐ **G6:** benchmark pYIN vs CREPE-tiny bằng f0 Pearson corr tại 10dB (dùng cặp clean/noisy từ W1).
- [ ] (Chi) Urgency classifier v0: ~100 utterances scripted; ⭐ feature set mở rộng theo eGeMAPS-lite (f0 stats + RMS + rate + jitter/shimmer nếu kịp); báo precision/recall.

### W4 — Tune + TECH SPEC
- [ ] Tune VAD endpoint threshold + urgency threshold trên data thật.
- [ ] (với Chi + Thịnh) Prosody transfer v1: Piper length_scale (rate) + WORLD/pyworld áp f0 envelope ⭐; A/B sample.
- [ ] ⭐ Viết section audio front-end của Tech Spec: cite arXiv:2512.17562 + 2201.06685 làm rationale guardrail; đính bảng WER matrix + latency AI Hub. Framing: *"đo lường thay vì niềm tin"*.

### W5 — Benchmark cuối
- [ ] WER/CER matrix đầy đủ 2 pairs (ko chấm bằng CER ⭐); prosody f0-correlation; urgency P/R; mini-MOS (mượn format human-eval protocol của arXiv:2301.10606 ⭐) — MOS chấm cả giọng MeloTTS-Korean.
- [ ] Kiểm tra chéo: config ship (denoise mode, VAD, thresholds) đúng là config đã thắng trong benchmark — không drift.

### W6 — Đóng băng
- [ ] Freeze config JSON (denoise_mode, HPF cutoff, notch list, VAD thresholds, α nếu dùng OA).
- [ ] Bộ utterance demo vàng (Chi) + kịch bản demo trong đó **A/B denoise và A/B prosody là 2 khoảnh khắc wow** riêng biệt.
- [ ] Model card phần audio: GTCRN/DFN3 size + RTF, VAD, f0 extractor — kèm số đo.

### Phase 2 (hội quân tháng 8) — hướng mở từ research
- Adaptive notch bank kích hoạt bằng alarm detector (CRNN nhỏ, §3) khi field test lộ beep lọt.
- Intelligibility-guided OA (arXiv:2602.20967) nếu α cố định không tối ưu giữa các mức ồn.
- Fine-tune Opus-MT en↔vi trên MedEV; glossary Meddict đầy đủ.
- ⭐ Fine-tune/LoRA NLLB-600M hướng vi↔ko trên UViko (nếu G2 chọn NLLB và chất lượng triage còn thiếu).
- ⭐ Fine-tune Whisper-base tiếng Hàn trên Zeroth-Korean nếu G8 cho thấy checkpoint cộng đồng không đủ.
- Field test tháng 10: phát noise dataset qua loa lớn tại chỗ (đúng kế hoạch Phase 2 hiện tại) — đây vẫn không yêu cầu thu âm tại bệnh viện.

---

## 10. Paper do team sưu tầm (`paper research/`) — bài học rút ra (v1.2)

### 11.1 Flavors of Moonshine (arXiv:2509.02523) → đã tích hợp thành ứng viên ASR chính, xem §5.4.

### 11.2 Performance & Efficiency Evaluation of ASR Inference on the Edge (Gondi & Pratap, Sustainability 2021)
- Phát hiện cốt lõi: trên edge CPU, **năng lượng tiêu thụ tăng theo cấp số nhân trong khi WER chỉ cải thiện tuyến tính** khi tăng size model; quantization + mobile optimization đưa edge ASR tiệm cận chất lượng server.
- Bài học áp vào ToneBridge (đã ghi vào doc 01 §7-8, doc 02 §5, doc 06 W5): (1) nguyên tắc chọn model = **nhỏ nhất đạt yêu cầu, không phải to nhất chạy nổi** — củng cố quyết định Moonshine 27M/GTCRN 48K; (2) thêm **metric năng lượng/pin** vào eval matrix (`eval_energy.py`, W5) — thiết bị đeo trong ca trực sống chết ở pin, và đây là con số production mà ít team competition nào đo.
- Giới hạn khi trích dẫn: paper 2021, đo trên Raspberry Pi với wav2vec2-class model — dùng làm *nguyên tắc*, không dùng số tuyệt đối; số của ta phải tự đo trên Snapdragon.

### 11.3 Abstract "Offline NMT VI-EN trên iOS bằng TinyLlama-1.1B GGUF" → sinh ra ADR-003
- Giá trị: xác nhận **nhu cầu dịch offline privacy-first trên mobile là hướng đi được cộng đồng theo đuổi** — dùng làm related work trong hồ sơ (định vị: họ iOS/LLM/1 pair, ta Android-NPU/dual-branch/2 pairs + prosody).
- Vì sao ta KHÔNG theo đường LLM-NMT (chi tiết ADR-003, doc 02 §2c): TinyLlama 1.1B INT4 ≈ 550-650MB — một mình vượt tổng budget 500MB; GGUF/llama.cpp là đường CPU/GPU, không có static-graph INT8 lên Hexagon NPU; LLM decode tự hồi quy phá budget NMT 150ms; và rủi ro hallucination trong ngữ cảnh y tế. Seq2seq NMT (Opus-MT 80MB) làm cùng việc, nhanh hơn, dịch trung thành hơn.
- Điểm kỹ thuật đáng học từ hướng iOS: cách họ đóng gói model + privacy-by-design UX là tham khảo tốt cho phần "Privacy visible" của doc 05.

## 11. Tài liệu tham khảo chính

**Denoise ↔ ASR:** [When De-noising Hurts (2025)](https://arxiv.org/abs/2512.17562) · [How Bad Are Artifacts? (2022)](https://arxiv.org/abs/2201.06685) · [E2E ASR training vs SE artifacts (NTT 2023)](https://arxiv.org/pdf/2311.11599) · [Intelligibility-guided OA (2026)](https://arxiv.org/html/2602.20967) · [Speaker Reinforcement / remixing (2022)](https://arxiv.org/pdf/2205.04433)
**Speech enhancement nhẹ:** [GTCRN — ICASSP 2024, repo](https://github.com/Xiaobin-Rong/gtcrn) · [DeepFilterNet2 embedded (2022)](https://arxiv.org/pdf/2205.05474) · [Ultra-low complexity NS (2023)](https://arxiv.org/pdf/2312.08132)
**ASR:** [Flavors of Moonshine (2025)](https://arxiv.org/abs/2509.02523) · [Moonshine repo](https://github.com/moonshine-ai/moonshine) · [sherpa-onnx Moonshine](https://k2-fsa.github.io/sherpa/onnx/moonshine/index.html) · [PhoWhisper — ICLR 2024](https://arxiv.org/abs/2406.02555) · [Qualcomm Whisper-Small-Quantized](https://huggingface.co/qualcomm/Whisper-Small-Quantized) · [AI Hub Whisper-Base](https://aihub.qualcomm.com/iot/models/whisper_base) · [ViMedCSS (2026)](https://arxiv.org/pdf/2602.12911)
**Edge efficiency:** [Gondi & Pratap — ASR Inference on the Edge (Sustainability 2021)](https://doi.org/10.3390/su132212392) (PDF trong `paper research/`)
**NMT:** [MedEV — LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.784/) · [Medical EN-VI MT prompting study (2025)](https://arxiv.org/abs/2509.15640) · [NLLB-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) ([ct2 int8](https://huggingface.co/JustFrederik/nllb-200-distilled-600M-ct2-int8)) · [Opus-MT repo](https://github.com/Helsinki-NLP/Opus-MT)
**VAD:** [TEN VAD](https://github.com/TEN-framework/ten-vad) · [Silero VAD quality metrics](https://github.com/snakers4/silero-vad/wiki/Quality-Metrics)
**Prosody/SER/S2ST:** [ResLSTM-SA (2026)](https://arxiv.org/pdf/2606.03359) · [Expressive S2ST cascade + human eval — Meta (2023)](https://arxiv.org/pdf/2301.10606) · [Prosody-TTS (2021)](https://arxiv.org/pdf/2110.02854)
**Hospital acoustics & noise data:** [ICU noise deep learning (2024)](https://link.springer.com/article/10.1007/s40857-024-00321-3) · [Polyphonic alarm detection — JMIR 2025](https://medinform.jmir.org/2025/1/e35987) · [Hospital Ambient Noise — Kaggle](https://www.kaggle.com/datasets/nafin59/hospital-ambient-noise) · [MUSAN corpus](https://ar5iv.labs.arxiv.org/html/1510.08484) ([HF mirror](https://huggingface.co/datasets/FluidInference/musan)) · [DNS Challenge / DTLN (2020)](https://arxiv.org/pdf/2005.07551)
**Tiếng Hàn (ASR/NMT/TTS):** [Whisper × KsponSpeech fine-tune study](https://www.eksss.org/archive/view_article?pid=pss-15-3-83) · [Korean meteorological ASR eval (2024)](https://arxiv.org/html/2410.18444v1) · [Korean clinician-patient ASR — radiation oncology (2023)](https://www.sciencedirect.com/science/article/abs/pii/S1386505623001302) · [whisper-medium-ko-zeroth](https://huggingface.co/seastar105/whisper-medium-ko-zeroth) · [UViko — Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EDWHKB) · [Korpora — AI Hub translation corpora](https://ko-nlp.github.io/Korpora/en-docs/corpuslist/aihub_translation.html) · [NLLB-200-Distilled-350M-en-ko](https://github.com/newfull5/NLLB-200-Distilled-350M-en-ko) · [Piper — no Korean support (discussion #680)](https://github.com/rhasspy/piper/discussions/680) · [MeloTTS-Korean](https://huggingface.co/myshell-ai/MeloTTS-Korean) · [NLLB paper — FLORES-200](https://arxiv.org/pdf/2207.04672)
