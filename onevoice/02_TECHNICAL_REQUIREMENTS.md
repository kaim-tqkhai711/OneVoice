# ToneBridge — Technical Requirements & Architecture Decisions

> **Version:** 1.1 · **Owner:** Tech Lead · **Status:** Active
> Includes ADR-001 (audio front-end), ADR-002 (NMT vi⇄ko — pending W2), ADR-003 (NMT engine class: seq2seq vs on-device LLM).
> **v1.1 (2026-07-11):** cập nhật theo research nhánh Sound & Speech (doc 07) + 2 paper mới (Moonshine ASR, ASR edge energy): sửa 2 dữ kiện sai (Opus-MT không có pair vi-ko; Piper không có voice ko), thêm Moonshine vào ASR stack, guardrail denoise 3 chế độ, metric năng lượng cho production.

---

## 1. Platform Constraints (LOCKED)

| Constraint | Value |
|-----------|-------|
| Target device | Android phone, Snapdragon SoC (8 Gen 2/3 class preferred; min: any Hexagon-NPU device team owns) |
| Runtime | 100% on-device. Zero network calls at inference time. |
| Deployment toolchain | Qualcomm AI Hub (compile, quantize, profile) → TFLite / QNN / ONNX Runtime delegate |
| Quantization | Fully **static** INT8 (INT4 where supported) — NPU yêu cầu static quantization, dynamic-activation PTQ không tương thích |
| Language pairs | EN⇄VI, VI⇄KR |
| Audio input | Phone mic, 16 kHz mono PCM (resample từ 48 kHz nếu cần) |

## 2. ADR-001 — Audio Front-End: Idea 1 vs Idea 2 vs Hybrid

### Context
Hai phương án được đề xuất:
- **Idea 1:** Neural denoise model (DeepFilterNet/NAFNet-class) → ASR (Whisper) → NMT.
- **Idea 2:** Waveform-level/DSP denoise tuned cho hospital noise → sau đó pipeline model như Idea 1.

### Analysis

**Hospital noise profile** gồm 2 lớp khác nhau về bản chất:

| Noise type | Ví dụ | Đặc tính | Xử lý tốt nhất |
|-----------|-------|----------|----------------|
| Stationary / tuần hoàn | Monitor beep (tần số cố định ~1kHz), quạt thông gió, ù nền HVAC, ù điện 50Hz | Phổ ổn định theo thời gian | **DSP** (HPF, notch, spectral gating) — rẻ, latency ~0, deterministic |
| Non-stationary | Người nói chồng lấn, tiếng khóc, va chạm kim loại, cart lăn | Phổ thay đổi nhanh | **Neural denoiser** — DSP bó tay |

**Điểm chết của Idea 1 thuần:**
1. Neural denoise chạy **trên toàn bộ luồng** = tốn compute liên tục trên phone.
2. Denoiser tạo artifact; Whisper vốn được train với noisy data — denoise quá tay có thể làm **WER tệ đi** (đã ghi nhận trong literature và thực nghiệm cộng đồng). Không được mặc định denoise = tốt.
3. Neural denoise **bóp méo pitch contour** → phá Branch B (prosody). Đây là lý do loại trừ tuyệt đối việc đặt neural denoise trước điểm rẽ nhánh.

**Điểm chết của Idea 2 thuần:** DSP không xử lý được overlapping speech / non-stationary noise — chính là loại ồn khó nhất của ER.

### Decision — Hybrid tiered front-end (Idea 2 đúng vị trí + Idea 1 đúng chỗ)

```
Mic (16kHz PCM)
   │
   ▼
[Stage 0 — DSP front-end]  ← SHARED, luôn bật, latency ≈ 0
   • High-pass filter 80 Hz (loại rumble, ù điện)
   • Adaptive spectral gating (noise floor ước lượng từ non-speech frames)
   • Optional notch filters cho beep tần số cố định (config theo môi trường)
   │
   ├──────────────► [Branch B — EMOTION]
   │                 Pitch-safe: KHÔNG neural denoise
   │                 → f0/energy/rate extraction → urgency classifier
   │
   ▼
[Stage 1 — Neural denoise]  ← CHỈ trong Branch A
   DeepFilterNet-class, quantized, chạy khi VAD active
   │
   ▼
[Branch A — MEANING]
   VAD → ASR → NMT
```

**Rationale:**
- DSP làm sạch lớp stationary cho **cả hai nhánh** với chi phí gần bằng 0 và **không phá pitch**.
- Neural denoise chỉ gánh phần non-stationary còn lại, chỉ chạy trong Branch A, chỉ khi VAD active → tiết kiệm pin/compute.
- Branch B nhận tín hiệu sau DSP nhẹ → prosody contour nguyên vẹn.

**Guardrail bắt buộc (điều kiện để giữ neural denoise) — v1.1 nâng từ 2 lên 3 chế độ:**
> Benchmark WER/CER với `denoise_mode ∈ {OFF, ON, OA}` trên hospital-noise test set, trong đó **OA (Observation Adding)** = trộn `α·denoised + (1−α)·noisy` trước ASR (α quét 0.5/0.8) — kỹ thuật đã chứng minh giảm artifact của denoiser mà giữ lợi ích khử ồn ([Iwamoto 2022, arXiv:2201.06685](https://arxiv.org/abs/2201.06685)). Chế độ nào WER tốt nhất thì ship; ON/OA phải thắng OFF ≥ 2 điểm tuyệt đối mới được bật. Quyết định bằng số liệu, không bằng cảm giác "nghe sạch hơn".
>
> **Prior mặc định = OFF:** nghiên cứu 2025 trên chính medical ASR ([When De-noising Hurts, arXiv:2512.17562](https://arxiv.org/abs/2512.17562)) cho thấy 40/40 cấu hình denoise-trước-ASR đều làm WER tệ đi. Ta không mặc định tin denoise — ta đo. (Chi tiết evidence: doc 07 §1.)

### Consequences
- (+) Compute thấp nhất có thể; prosody sống; có nút xoay (Stage 1 on/off) để tune bằng data.
- (−) Thêm 1 stage DSP phải code tay (scipy/custom C++/Oboe) — giao cho Sound team, ~2-3 ngày công.
- (−) Notch filter config là per-environment — chấp nhận hard-code cho demo, ghi rõ trong roadmap.

## 2b. ADR-002 — NMT cho vi⇄ko (Status: PENDING, chốt cuối W2)

**Context:** Kho Helsinki-NLP **không có model Opus-MT vi↔ko** (đã kiểm chứng — chỉ có vi↔en, ko↔en). Hai phương án:
- **A — Pivot:** vi→en→ko bằng 2 model Opus-MT (en↔vi đã cần sẵn; thêm en↔ko ~80MB). Latency 2 hop, lỗi cộng dồn nhưng mỗi hop là pair giàu data.
- **B — Trực tiếp:** NLLB-200-distilled-600M INT8 (~300MB, CTranslate2/ONNX) dịch thẳng vi↔ko, dùng chung được mọi hướng.

**Data quyết định:** chrF++ tự chạy trên FLORES-200 devtest (vie_Latn↔kor_Hang) + ~100 câu triage trích từ corpus UViko (~454K cặp vi-ko, Harvard Dataverse) + human read. Default nếu thiếu số: A (nhẹ, ship nhanh). Chi tiết doc 07 §6.3(b).

## 2c. ADR-003 — NMT engine class: dedicated seq2seq, KHÔNG dùng on-device LLM (Status: DECIDED)

**Context:** Có hướng đi khác đang phổ biến: dùng LLM nhỏ chạy on-device làm translator — ví dụ nghiên cứu triển khai **TinyLlama-1.1B-Chat GGUF dịch VI↔EN offline trên iOS** (paper team sưu tầm, folder `paper research/`). Hướng này chứng minh nhu cầu dịch offline privacy-first là có thật và làm được trên mobile — nhưng ta **không** đi theo, vì với target production trên **Qualcomm Snapdragon/Hexagon NPU**:

1. **Size:** TinyLlama 1.1B ≈ 550-650 MB (INT4 GGUF) — một mình nó vượt cả tổng budget 500MB của ta; Opus-MT ~80MB/pair làm cùng việc, tốt hơn ở pair đã có data.
2. **Sai đường silicon:** GGUF/llama.cpp chạy CPU/GPU, **không có đường static-graph INT8 lên Hexagon NPU** — đi ngược ràng buộc §1 (static quantization, QNN/HTP). Đường LLM-on-NPU của Qualcomm (Genie/QAIRT) tồn tại nhưng là stack khác, nặng effort, không đáng cho một tác vụ NMT thuần.
3. **Latency & năng lượng:** LLM decode tự hồi quy từng token trên CPU phá vỡ budget NMT 150ms/utterance; paper edge-ASR energy trong `paper research/` (Gondi & Pratap 2021) cho quy luật chung: **năng lượng tăng theo cấp số khi chất lượng tăng tuyến tính** — chọn model nhỏ nhất đạt yêu cầu, không chọn model to nhất chạy nổi.
4. **An toàn y tế:** LLM chat dễ diễn giải/thêm thắt (hallucinate) thay vì dịch trung thành — rủi ro không chấp nhận được cho triage. Seq2seq NMT dịch sát và chặn được bằng glossary constraint.

**Decision:** NMT = dedicated seq2seq (Opus-MT / NLLB-distilled) quantize static INT8. LLM on-device chỉ được xem xét lại ở Phase 2+ cho vai trò phụ (rescoring/glossary-aware rewrite) nếu stack Genie trên NPU chín muồi.

## 3. Model Stack (per stage) — v1.1

| Stage | Model (primary → fallback) | Size target (quantized) | Runtime | Owner |
|-------|-------|------------------------|---------|-------|
| DSP front-end | Custom (biquad HPF 80Hz + spectral gate + notch bank) | n/a (code) | CPU (audio thread) | Sound |
| VAD | Silero VAD (baseline) → benchmark TEN VAD trên hospital noise (gate G1, W2) | ~2 MB | CPU | Sound |
| Neural denoise (Branch A) | **GTCRN** (48K params, ~0.2 MB, ICASSP'24) → DeepFilterNet3 nếu thua ở SNR 5dB (gate G3, W3) | ≤ 10 MB | **CPU** (GTCRN đủ nhẹ → giải phóng NPU cho ASR) | Sound + ML |
| ASR | **Per-language routing:** Moonshine-Tiny-vi + Moonshine-Tiny-ko (27M/model, [arXiv:2509.02523](https://arxiv.org/abs/2509.02523) — vi vượt Whisper-medium, ko ngang Whisper-small) + Moonshine-Tiny/Base-en → fallback: Whisper-base multilingual duy nhất; PhoWhisper-base cho vi nếu Moonshine-vi kém trên test nội bộ (gate G5/G8, W3) | ~30 MB/lang (Moonshine INT8) — tổng 3 lang ≈ 90-120 MB, rẻ hơn 1 Whisper-small (~250 MB) | NPU (encoder) hoặc CPU (Moonshine đủ nhẹ; sherpa-onnx đã hỗ trợ Android) | ML |
| LID (chiều dịch) | Hệ quả của ASR đơn ngữ: chạy song song 2 ASR tiny của pair, chọn theo confidence (2×27M — rẻ) → fallback: Whisper LID | (dùng chung ASR) | như ASR | ML |
| NMT | en↔vi: Opus-MT (+ fine-tune MedEV nếu kịp). **vi↔ko: KHÔNG tồn tại Opus-MT** → ADR-002 (W2): pivot vi↔en↔ko (2×Opus-MT) vs NLLB-200-distilled-600M INT8 trực tiếp | ≤ 80 MB/pair (Opus-MT) hoặc ~300 MB (NLLB dùng chung mọi hướng) | NPU/CPU | ML |
| Prosody extraction | Custom DSP: f0 (pYIN → CREPE-tiny nếu thua ở 10dB, gate G6), RMS energy, syllable-rate | ≤ 5 MB | CPU | Sound |
| Urgency classifier | Lightweight MLP/CNN trên prosody features (tự train; eGeMAPS-lite feature set) | ≤ 1 MB | CPU | Sound |
| TTS | Piper (vi, en). **Piper KHÔNG hỗ trợ ko** → **MeloTTS-Korean** (MIT, CPU near-realtime) cho tiếng Hàn (gate G7, W3) | ~60 MB/voice | CPU | ML |
| Prosody transfer | Post-process trên waveform TTS: rate qua length_scale/speed param; pitch envelope qua WORLD (pyworld) resynthesis — **cùng một path cho cả Piper lẫn MeloTTS** | code | CPU | Sound + ML |

**Tổng model budget:** ≤ 500 MB on-disk. Ước tính stack đề xuất: ASR ~120 + NMT ~160 (Opus×2 pair pivot) hoặc ~300 (NLLB) + TTS ~180 (3 voice) + còn lại < 10 → **~470 MB worst-case, đạt budget**. Production note: model đóng gói kèm manifest (`models.json`: tên, version, SHA256, quant scheme) để verify integrity lúc load — bắt buộc cho production, rẻ cho prototype.

## 4. Latency Budget (p50, end-of-speech → first TTS sample)

| Stage | Budget |
|-------|--------|
| VAD endpoint detection | 200 ms (inherent trailing silence) |
| Neural denoise (utterance ≤ 5s) | 150 ms (GTCRN streaming RTF 0.07 → thực tế kỳ vọng < 100 ms) |
| ASR (Moonshine-Tiny INT8 / Whisper-base) | 500 ms — lợi thế Moonshine: input variable-length (không pad 30s như Whisper) → latency tỉ lệ độ dài câu, câu triage ngắn kỳ vọng nhanh hơn đáng kể |
| NMT | 150 ms (lưu ý: nếu ADR-002 chọn pivot, vi⇄ko là 2 hop — budget riêng cho pair này 300 ms) |
| Prosody extract (song song với ASR) | 0 ms (overlap) |
| TTS synthesis + prosody transfer | 400 ms (đo riêng MeloTTS-Korean — engine khác Piper) |
| **Total** | **≤ 1.4 s** (vi⇄ko cho phép ≤ 1.55 s nếu pivot) |

Đo bằng: AI Hub profiling (per-model) + on-phone tracing (end-to-end). Mọi số trong Tech Spec tháng 7 phải là **số đo, không phải số ước**.

## 5. Evaluation Requirements

**Datasets:**
- Clean: LibriSpeech test-clean subset (EN), VIVOS/Common Voice vi (VI), **Zeroth-Korean test** (KR) — chỉ dùng subset nhỏ cho benchmark.
- Noise: **100% từ dataset có sẵn, KHÔNG tự thu** (ràng buộc team) — Kaggle Hospital Ambient (eval-only) + Freesound CC0 + MUSAN/DNS (tune) + alarm beep **tự sinh bằng code** theo chuẩn IEC 60601-1-8 + babble tự trộn. Inject tại SNR 20/10/5 dB. Quy tắc: noise để tune ≠ noise để eval. (Chi tiết doc 07 §3.1.)
- NMT: FLORES-200 devtest (mọi hướng) + ~100 câu triage (en/vi tự soạn; vi-ko trích từ UViko).
- Medical phrases (speech): tự thu ~50 câu kịch bản triage mỗi ngôn ngữ bằng giọng team ở phòng yên tĩnh (hợp lệ — chỉ noise là không tự thu), sau đó inject noise bằng script.

**Metrics matrix (bắt buộc có trong Tech Spec):**

| Metric | Điều kiện |
|--------|-----------|
| WER (en, vi) / **CER (ko — chuẩn cộng đồng cho Hangul)** | clean / 20dB / 10dB / 5dB × denoise **OFF/ON/OA** |
| NMT chrF++ | FLORES-200 + triage set, per direction (dữ liệu ADR-002) |
| E2E latency p50/p95 | on-phone, airplane mode |
| Per-stage latency | AI Hub profiling screenshot |
| **Năng lượng/pin** (production metric — theo Gondi & Pratap 2021, `paper research/`) | % pin + mAh cho session 15' liên tục (Android `batterystats`/`dumpsys`); ghi nhận thermal throttling nếu có |
| Prosody fidelity | Pearson corr của f0 contour (source vs output), + MOS mini-panel (team tự chấm blind, mượn format human-eval của Meta expressive-S2ST) |
| Urgency classifier | Precision/recall trên scripted distress set |

## 6. Engineering Standards

- **Python (training/eval/tooling):** type hints bắt buộc, Pydantic cho config, Google docstrings. Tensor shape annotation bắt buộc tại mọi transformation: `# x: [B, T] -> [B, n_mels, T']`.
- **Android app:** Kotlin, Jetpack Compose. Audio I/O qua Oboe/AAudio (low-latency path). Model inference qua ONNX Runtime Android / TFLite với QNN delegate.
- **Determinism:** seed cố định trong mọi script eval; requirements.txt pin version; kết quả benchmark phải reproduce được.
- **Logging:** mọi eval run ghi ra JSON (metric, model hash, dataset hash, timestamp) — commit vào `eval/results/`.
- **No cloud calls:** CI check grep network APIs trong inference path; demo build chạy airplane mode.

## 7. Privacy & Security Requirements

- Audio buffer chỉ tồn tại in-memory trong session; không ghi file trừ khi bật dev-flag (và dev-flag bị strip khỏi demo build).
- Audit log: append-only, mỗi entry `{session_id, ts, lang_pair, duration_s, urgency_flag}`, hash-chained (`entry_hash = SHA256(prev_hash || entry)`) → tamper-evident. **Không có trường audio/transcript.**
- Dashboard serve trên localhost / LAN hotspot của chính phone — không internet.
