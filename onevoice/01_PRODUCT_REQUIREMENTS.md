# ToneBridge — Product Requirements Document (PRD)

> **Version:** 1.1 (đồng bộ research doc 07 + model stack v1.1: metric CER cho KO, pin/năng lượng, risk mitigations mới) · **Owner:** Product Owner (Khải) · **Status:** Active
> **Scope:** Competition prototype (phone-based), OneVoice AI Challenge — Healthcare track

---

## 1. Product Vision

**One-liner:** The on-device medical interpreter that translates not only *what* a patient says — but *how* they say it.

ToneBridge is a hands-free, screen-free medical interpreter running 100% offline on a Snapdragon device. It translates patient–clinician conversations in real time while **preserving the patient's vocal prosody** (distress, pain, breathlessness) — treating emotion as clinical information that must survive translation.

**Positioning guardrail (non-negotiable):** ToneBridge is a *communication and urgency-awareness aid under clinical supervision*. It is **not** a diagnostic device and must never be described as one in any UI copy, docs, or pitch material.

## 2. Problem Statement

In emergency rooms and triage, a foreign or speech-impaired patient must be understood in seconds. Existing tools fail on three assumptions:

| # | Assumption | Reality in ER |
|---|-----------|---------------|
| 1 | "You have a free hand" | Clinician's hands are on the patient |
| 2 | "The room is quiet" | Alarms, monitors, overlapping speech (SNR thấp) |
| 3 | "Cloud is fine" | Patient audio legally cannot leave premises; connectivity unreliable |

Additionally, all existing translators **flatten prosody** into neutral text, discarding the urgency signal that drives correct triage prioritization.

## 3. Target Users & Personas

| Persona | Context | Key need |
|---------|---------|----------|
| **P1 — Triage nurse (VN)** | Receives foreign patients at intake, hands busy | Understand symptoms + urgency instantly, hands-free |
| **P2 — Foreign patient (EN speaker)** | Tourist/expat in distress, may have impaired speech | Be understood — words *and* urgency |
| **P3 — Korean patient/family (KR speaker)** | Korean expat community (largest foreign patient group at FDI-area hospitals) | Communicate symptoms to VN staff |
| **P4 — Head nurse (dashboard user)** | Oversees ward | See urgency flags, audit log; zero cloud dependency |

## 4. Language Scope (LOCKED for prototype)

- **Pair 1: EN ⇄ VI** — primary demo pair (best model quality, richest medical terminology).
- **Pair 2: VI ⇄ KR** — strategic pair (Korean expat/FDI patient population).
- Architecture must be language-agnostic (adding a pair = swapping NMT weights + TTS voice, no pipeline change).

## 5. Prototype Scope (what we ARE building)

Team = students, remote until mid-August. Prototype = **Android app on a Snapdragon phone**. The lanyard device is the *product vision*, the phone is the *vehicle to prove the AI*.

**In scope (MVP):**
- [MVP-1] Two-way speech translation EN⇄VI and VI⇄KR, utterance-by-utterance (not continuous streaming).
- [MVP-2] 100% offline inference — airplane-mode demo must work.
- [MVP-3] Zero-UI interaction: VAD auto-triggers capture; no button needed to translate (a manual push-to-talk fallback is allowed as a hidden dev option).
- [MVP-4] Dual-branch pipeline: translated speech replayed **with the speaker's prosody envelope**; prosody transfer can be toggled ON/OFF for demo contrast.
- [MVP-5] Urgency flag: high-distress utterances raise a visible/silent flag.
- [MVP-6] Offline clinician dashboard (local web app served on-device or LAN) showing session metadata + urgency events. **Metadata only — never audio, never transcript content.**
- [MVP-7] Benchmark evidence: WER/CER (clean vs hospital-noise-injected, denoise ON/OFF) + per-stage latency measured via Qualcomm AI Hub.

**Out of scope (explicitly NOT building now):**
- Custom hardware / lanyard enclosure (Phase 2, hội quân tháng 8).
- Continuous simultaneous interpretation (streaming).
- Speaker diarization for >2 speakers.
- Any diagnosis/triage-scoring feature.
- Cloud sync, accounts, OTA — all future.

## 6. User Stories & Acceptance Criteria

**US-1 (Core translation):** As a nurse, when a patient speaks English, I hear Vietnamese within ~1.5s of them finishing, without touching the device.
- AC: end-to-end latency (end-of-speech → start-of-TTS) ≤ 1.5s p50, ≤ 2.5s p95 on target phone; works in airplane mode.

**US-2 (Prosody preservation):** As a nurse, when a patient speaks with distress, the translated voice carries that distress.
- AC: A/B demo — same utterance with prosody transfer ON vs OFF is clearly distinguishable; pitch contour correlation between source and output ≥ agreed threshold (defined in eval plan).

**US-3 (Urgency flag):** As a nurse, when a patient's voice shows high distress, I see an urgency indicator.
- AC: classifier flags ≥80% of scripted "distress" test utterances; false-positive rate documented (no hard target for prototype, but measured).

**US-4 (Noise robustness):** As a nurse in a noisy ER, translation still works.
- AC: WER degradation from clean → 80dB injected hospital noise ≤ agreed budget (see eval plan); denoise ON/OFF comparison table exists.

**US-5 (Privacy):** As a hospital admin, I can verify no audio leaves the device.
- AC: app functions fully in airplane mode; dashboard log contains zero audio/transcript fields; audit entries are hash-chained (tamper-evident).

**US-6 (Dashboard):** As a head nurse, I can see session history and urgency events on a local dashboard.
- AC: dashboard reachable offline; shows timestamp, language pair, duration, urgency flags per session.

## 7. Success Metrics (competition-oriented)

| Metric | Target | Why it wins |
|--------|--------|-------------|
| E2E latency p50 | ≤ 1.5 s / utterance | Technical Excellence (50%) |
| WER (EN, clean) | ≤ 15% | Baseline credibility |
| WER (VI) / CER (KO) clean | đo & báo cáo (KO chấm bằng CER — chuẩn cộng đồng cho Hangul) | Chứng minh cả 3 ngôn ngữ được đo nghiêm túc |
| WER delta (clean → noisy, best denoise mode) | ≤ +10 pts | Noise-robustness proof — kèm bảng OFF/ON/OA cho thấy quyết định bằng số liệu |
| Prosody A/B demo | Audibly convincing | Innovation (25%) |
| Offline proof | Airplane-mode live demo | Edge AI requirement |
| Pin/năng lượng | đo & báo cáo (% pin cho session 15') | Production-readiness — thiết bị đeo phải sống qua ca trực; chọn model nhỏ nhất đạt yêu cầu thay vì model to nhất chạy nổi |
| AI Hub profiling | Real-silicon numbers in Tech Spec | Beats "promise-ware" teams |

## 8. Risks & Mitigations (product-level)

| Risk | Impact | Mitigation |
|------|--------|-----------|
| VI/KR ASR quality thấp hơn EN | Demo lắp bắp | **Moonshine-Tiny đơn ngữ vi/ko (27M, vi vượt cả Whisper-medium — paper trong `paper research/`)** + PhoWhisper fallback cho vi; scripted demo utterances + medical glossary; EN⇄VI là primary demo pair |
| vi⇄ko NMT không có model trực tiếp (Opus-MT không có pair này) | KR pair yếu | ADR-002 chốt W2 bằng benchmark FLORES-200 + UViko: pivot qua EN vs NLLB-600M trực tiếp; worst-case KR demo "supported, shown briefly" |
| TTS tiếng Hàn: Piper không hỗ trợ ko | Không có giọng KR | MeloTTS-Korean (MIT, CPU) — đã chốt trong model stack v1.1; prosody transfer dùng chung path WORLD nên không tăng effort |
| Prosody transfer sounds robotic | Kills the wow moment | Fallback: pitch+rate-only transfer (skip energy); A/B rehearsed with best samples |
| Latency vượt budget trên phone | Fails US-1 | Model đã nhỏ sẵn (Moonshine 27M, GTCRN 48K); còn vượt → quantize harder (INT4), utterance length cap |
| Pin tụt nhanh khi session dài | Thiết bị chết giữa ca — chết production story | Đo năng lượng từ W5 (`eval_energy.py`); nguyên tắc từ paper edge-ASR energy: model nhỏ nhất đạt yêu cầu, không phải to nhất chạy nổi |
| Ethics challenge from judges | Credibility hit | Guardrail statement everywhere; urgency = decision *support* only |
| Team remote, tích hợp trễ | Prototype không kịp | Interface Contract tuần 1 + weekly integration demo (see Implementation Plan) |

## 9. Document Map

| Doc | File |
|-----|------|
| Technical requirements + audio architecture decision | `02_TECHNICAL_REQUIREMENTS.md` |
| Product flow (user + data flow) | `03_PRODUCT_FLOW.md` |
| Backend/system structure | `04_BACKEND_ARCHITECTURE.md` |
| UI/UX design | `05_UIUX_DESIGN.md` |
| Implementation plan (6 weeks) | `06_IMPLEMENTATION_PLAN.md` |
