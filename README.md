<div align="center">

# 🎙️ ToneBridge

### The on-device medical interpreter that translates not only *what* a patient says — but *how* they say it.

**OneVoice AI Challenge — Healthcare Track**
`100% Offline Edge AI` · `Zero-UI` · `Snapdragon NPU via Qualcomm AI Hub`

> ⚕️ *A communication and urgency-awareness aid under clinical supervision — **not** a diagnostic device.*

</div>

---

## 🩺 The Problem

In high-pressure triage, a foreign or speech-impaired patient must be understood in **seconds**. Every existing translator makes three fatal assumptions for an emergency room:

1. **"You have a free hand."** — App translators need you to pick up a phone, unlock, and tap. In triage, both of the clinician's hands are on the patient.
2. **"The room is quiet."** — ERs are full of monitor beeps, alarms, and overlapping speech. Consumer ASR collapses as SNR drops.
3. **"Cloud is fine."** — Patient audio is legally sensitive and often cannot leave the premises; ER connectivity is unreliable anyway.

## 💡 The Insight

In triage, *how* something is said — distress, pain, breathlessness — is itself **clinical information**. Standard translators flatten prosody into neutral text. **ToneBridge treats prosody as a signal that must survive translation**, so a nurse hears the urgency, not just the sentence.

---

## 🏗️ Architecture — The Dual-Branch Pipeline

The core technical idea: **aggressive denoising rescues accuracy but flattens emotion.** So we split denoising into two paths — one optimized for *meaning*, one for *feeling* — and fuse them at synthesis.

```
                          ┌─────────────────────────────────────────────────────────┐
                          │  BRANCH A — MEANING  (accuracy-optimized)                │
                  ┌──────▶│  aggressive denoise → VAD → Whisper-small ASR → NMT      │──┐
                  │       └─────────────────────────────────────────────────────────┘  │
   🎤 Mic array ──┤                                                                     ├──▶ 🔊 Prosody-transfer TTS
                  │       ┌─────────────────────────────────────────────────────────┐  │     (translated text spoken with
                  └──────▶│  BRANCH B — EMOTION  (prosody-preserving)                │──┘      patient's original prosody)
                          │  pitch-safe denoise → prosody envelope → urgency flag    │            │
                          └─────────────────────────────────────────────────────────┘            ▼
                                                                                          🚨 silent urgency flag → staff
```

| Stage | Model / Method | Runtime |
|-------|----------------|---------|
| Voice activity gate (Zero-UI) | Silero VAD | CPU / DSP |
| Denoise (Branch A, aggressive) | NAFNet-style denoiser | Hexagon NPU |
| Denoise (Branch B, pitch-safe) | Light spectral denoise | DSP |
| ASR | Whisper-small (quantized, dysarthria-tuned) | Hexagon NPU |
| NMT | Opus-MT EN⇄VN + clinical glossary | Hexagon NPU |
| Prosody engine | Pitch / energy / rate envelope extraction + transfer | Sound team module |
| Urgency classifier | Lightweight prosody-feature classifier | DSP |
| TTS | Piper (prosody-transfer) | Hexagon NPU |

> **Why two branches?** Strong denoising improves WER/CER but destroys the pitch contour that carries emotion. Branch A maximizes intelligibility; Branch B preserves the prosodic envelope. This directly resolves the denoise-vs-prosody conflict that breaks single-pipeline translators in noisy wards.

---

## ⚡ Edge Optimization (Qualcomm AI Hub)

All models are compiled, quantized, and profiled on **Qualcomm AI Hub Workbench**, targeting the Snapdragon Hexagon NPU.

| Target | Value |
|--------|-------|
| Quantization | INT8 / INT4 (fully static, NPU-compatible) |
| End-to-end latency | < 1 s / utterance (target) |
| Quality retention | ~90% of full precision at INT4 |
| Audio sent off-device | **0 bytes** |

**Benchmarking focus:** WER & CER on **dysarthric and accented speech**, measured clean vs. noise-injected and denoise on/off, on AI Hub's hosted Snapdragon devices — so numbers come from real silicon, not estimates.

```bash
# Example AI Hub workflow (see /scripts)
pip install qai-hub qai-hub-models
qai-hub configure --api_token <YOUR_TOKEN>
python -m qai_hub_models.models.whisper_small_v2.export --device "Snapdragon 8 Elite QRD"
```

---

## 📁 Repository Structure

```
tonebridge/
├── README.md
├── docs/
│   ├── ToneBridge_PitchDeck.pdf        # competition pitch deck
│   └── architecture.md                 # detailed dual-branch design
├── src/
│   ├── audio/                          # mic capture, dual-mode denoise, VAD   [Sound/Speech]
│   ├── prosody/                        # envelope extraction + transfer + urgency  [Sound/Speech]
│   ├── asr/                            # Whisper quantization + WER/CER eval    [ML/NLP]
│   ├── nmt/                            # Opus-MT EN-VN + clinical glossary      [ML/NLP]
│   ├── tts/                            # Piper prosody-transfer synthesis       [ML/NLP]
│   └── pipeline/                       # branch orchestration + fusion
├── edge/
│   └── aihub/                          # AI Hub compile/profile scripts         [ML/NLP]
├── app/                                # offline dashboard + mesh audit log     [Web3/Fullstack]
├── eval/
│   ├── benchmarks/                     # WER/CER + latency results
│   └── datasets/                       # (gitignored — no patient data committed)
└── scripts/
```

---

## 👥 Team & Ownership

The product's hardest bottleneck — **recovering medical prosody under chaotic ER noise** — maps directly onto our team's rarest skill.

| Role | Count | Owns |
|------|:-----:|------|
| 🔊 **Sound & Speech** | 2 | Branch B prosody engine, medical-grade prosody templates, pitch-contour extraction/transfer, dual-mode denoising, urgency classifier, dysarthric-speech WER/CER |
| 🧠 **ML / NLP** | 3 | INT4/INT8 quantization, EN–VN medical NMT, ASR/TTS deployment, Qualcomm AI Hub compilation & latency profiling |
| 🌐 **Web3 / Fullstack** | 1 | Offline clinician dashboard, tamper-evident privacy-preserving audit log (metadata only, never audio), hospital mesh sync — no cloud |

---

## 🔐 Privacy by Architecture

- 🚫 **Raw audio never leaves the device** — all inference runs on the NPU.
- 🧾 **Audit trail logs metadata only** (urgency flag, timestamp) — never conversation content.
- 🕸️ **Devices sync over the hospital's internal mesh** — no cloud server, ever.
- 📡 **Works in connectivity dead zones** — shielded ERs, basement triage.

---

## 🚀 Getting Started

> ⚠️ Prototype stage (competition PoC). Runs the full pipeline on a Snapdragon Android device or on AI Hub's hosted devices.

```bash
git clone https://github.com/<team>/tonebridge.git
cd tonebridge
pip install -r requirements.txt

# Run the demo pipeline (mic → translate → prosody-transfer playback)
python -m src.pipeline.run --pair en-vn --demo
```

**Demo highlights to watch for:**
- 🎚️ Toggle **prosody transfer on/off** — hear the same sentence with and without preserved emotion.
- 🔇 Toggle **denoise on/off** under injected ER noise — watch WER recover.

---

## 🗺️ Roadmap

- [x] **Now (Idea + PoC):** Prosody-preserving EN⇄VN interpreter on Snapdragon, profiled on AI Hub.
- [ ] **Next (Product):** Sterilizable lanyard wearable; offline OTA glossary packs; mesh urgency broadcast; clinical validation & medical-device certification.
- [ ] **Future (Platform):** Same engine extends to MIT-based aphasia rehab (*CadenceCare*) and voice restoration for non-verbal patients (*VoxRestore*).

---

## ⚖️ Responsible Use

ToneBridge is a **communication and urgency-awareness aid operating under clinical supervision**. It does **not** diagnose, and it does not replace professional medical judgment or human interpreters where those are required. Prosody and urgency outputs are decision *support* signals only. Any real-world deployment requires clinical validation and compliance with applicable medical-device regulations.

---

<div align="center">

Built for the **OneVoice AI Challenge** · Saigon AI Hub × Qualcomm

</div>
