> Bản lưu README root trước cập nhật ngày 09/10/2026. Các số liệu, scope và link tương đối bên dưới thuộc vị trí/tài liệu cũ. Hướng dẫn hiện tại: [README chính](../../README.md).

<div align="center">

# 🎙️ ToneBridge

### On-Device Medical Speech-to-Speech Interpreter with Vocal Urgency & Prosody Preservation

**OneVoice AI Challenge — Healthcare Track**
*Organized by Saigon AI Hub × Qualcomm*

`100% Offline Edge AI` · `Zero-UI Triage Aid` · `Dual-Branch Architecture` · `Snapdragon / Hexagon NPU Ready`

> ⚕️ **Clinical Positioning & Safety Guardrail:**
> ToneBridge is a communication and urgency-awareness decision-support aid under clinical supervision — **not** a diagnostic device. It operates with a Clinician-in-the-Loop paradigm and semantic safety validation.

---

[Key Highlights](#-key-highlights) •
[The Problem & Insight](#-the-clinical-problem--core-insight) •
[Architecture](#-system-architecture) •
[Measured Benchmarks](#-measured-benchmarks--laptop-results) •
[Quick Start](#-quick-start-guide) •
[Dành cho Teammate](#-dành-cho-teammate-hướng-dẫn-phát-triển-tiếp) •
[Project Map](#-repository-structure--doc-map)

---

</div>

## 🌟 Key Highlights

- **Dual-Branch Pipeline**: Aggressive acoustic processing optimizes speech recognition (*Branch A - Meaning*), while a parallel pitch & energy engine captures vocal distress (*Branch B - Emotion*).
- **100% Offline & Zero Data Leakage**: Evaluated with strict socket sandboxing (`OfflineGuard`); zero bytes leave the device, zero cloud dependencies.
- **Sub-Second Latency on Edge**: End-to-end PTT-to-audio latency of **~371 ms** (2s speech) and **~671 ms** (4s speech) on laptop reference CPU; target < 1.0s on Snapdragon Hexagon NPU.
- **Ultra-Compact Footprint**: Full pipeline resident footprint is only **274.8 MiB** on disk (Zipformer ASR INT8 + Marian NMT INT8 + Piper TTS + SwiftF0).
- **Clinical Semantic Safety Check**: Real-time rule & lexicon validator catching medication name distortions, dose drops, and negated instructions before speech synthesis.
- **0-Stubs Reference Prototype Verified**: Gate G-L passed with 81 passing unit & integration tests, clean-clone verification, and comprehensive benchmark reporting.

---

## 🩺 The Clinical Problem & Core Insight

In emergency room (ER) intake and triage, clinicians must assess non-native or speech-impaired patients within seconds. Standard consumer translation applications collapse under three clinical realities:

| ER Reality | Standard Translation Apps | ToneBridge Solution |
|---|---|---|
| **Hands-Busy Triage** | Requires picking up, unlocking, and tapping a smartphone screen | **Zero-UI / Auto-VAD / PTT**: Wearable, screen-free or single-trigger interaction |
| **Acoustic Noise Chaos** | Fails under monitor beeps, alarms, and background chatter | **Calibrated Edge Models**: Evaluated under DEMAND, babble, and medical alarm noise |
| **Strict Privacy & Dead Zones** | Streams raw audio to cloud servers; fails in basement ERs | **100% On-Device**: Pure ONNX Runtime & Sherpa-ONNX execution; airplane-mode native |
| **Flattened Urgency** | Flattens all vocal emotion into monotone robotic speech | **Prosody & Urgency Extraction**: Distress, breathlessness, and acute pain are preserved as clinical signals |

> 💡 **Core Insight:** In clinical triage, ***how*** something is said is vital diagnostic information. If a patient says *"I feel chest pressure"* with intense shortness of breath and high pitch strain, that prosody carries urgency that the triage nurse must hear immediately.

---

## 🏗️ System Architecture

ToneBridge decouples linguistic meaning from emotional acoustics to resolve the fundamental conflict where aggressive speech denoising rescues word error rates but destroys the pitch contour carrying human distress.

```
                                  ╔══════════════════════════════════════════════════════════╗
                                  ║         BRANCH A: MEANING (Accuracy-Optimized)           ║
                             ┌───▶║  VAD (Silero) ──▶ ASR (Zipformer-vi INT8)                ║───┐
                             │    ║       ──▶ Constrained NMT (Opus-MT vi-en INT8)           ║   │
                             │    ╚══════════════════════════════════════════════════════════╝   │
                             │                                                                   ▼
     🎤 Mic / Audio Input ───┤                                                       ┌───────────────────────┐
      (16 kHz Mono Audio)    │                                                       │ Semantic Safety Check │
                             │                                                       │   & Clinical Gate     │
                             │                                                       └───────────┬───────────┘
                             │    ╔══════════════════════════════════════════════════════════╗   │
                             │    ║         BRANCH B: EMOTION & URGENCY (Prosody Engine)     ║   │
                             └───▶║  SwiftF0 Pitch Extractor ──▶ 21 Prosody Features         ║───┘
                                  ║       ──▶ Urgency Classifier (LOW / HIGH / UNKNOWN)      ║
                                  ╚══════════════════════════════════════════════════════════╝
                                                                   │
                                                                   ▼
                                                       ┌───────────────────────┐
                                                       │ TTS (Piper EN)        │ ──▶ 🔊 Synthesized Speech
                                                       │ + Urgency Advisory    │ ──▶ 🚨 Silent Clinical Flag
                                                       └───────────────────────┘
```

### 1. Branch A — Meaning
1. **Voice Activity Detection (VAD)**: Silero VAD (via `sherpa-onnx`) strips leading/trailing silences.
2. **Denoising Module**: Evaluated extensively in [ADR-001](Prototype/docs/ADR-001-denoise.md). For single-channel microphone inputs under overlapping speech, default is set to `OFF` to prevent phase distortion, with GTCRN available as a configurable arm.
3. **ASR (Automated Speech Recognition)**: Offline Transducer `Zipformer-vi` (Apache-2.0, quantized INT8, **69.9 MiB**). RTF ~0.040.
4. **Constrained NMT (Neural Machine Translation)**: `Helsinki-NLP/opus-mt-vi-en` quantized INT8 (**125.4 MiB**) with custom greedy KV-cache decoding loop and **Clinical Glossary Constrained Decoding** to prevent clinical hallucination.
5. **Semantic Safety Check & Decision Gate**: Validates clinical integrity (medication names, units, dosages, negation). The Gate emits one of 5 actions: `SPEAK`, `CONFIRM`, `REPEAT`, `ABSTAIN`, or `LOG`.
6. **TTS (Text-to-Speech)**: `Piper EN` (`en_US-ljspeech-medium`, **77.7 MiB**) with high naturalness and fast synthesis.

### 2. Branch B — Emotion & Urgency
- **SwiftF0 Pitch Extractor**: Compact ONNX model (**1.1 MiB**) computing frame-level fundamental frequency $F_0$.
- **21 Normalized Prosody Features**: Median-relative pitch deviations (semitones), energy dynamics, voice fraction, spectral slope.
- **Urgency Classification**: Maps acoustic prosody into actionable triage advisory levels (`LOW`, `HIGH`, `UNKNOWN`).

### 3. Model Inventory & Memory Footprint

| Subsystem | Model / Framework | Format & Quant | Disk Size | Resident Memory | License |
|---|---|---|---|---|---|
| **VAD** | Silero VAD | ONNX | 0.6 MiB | ~20.6 MiB | MIT |
| **ASR** | Zipformer Vietnamese | INT8 Transducer | 69.9 MiB | ~121.9 MiB | Apache-2.0 |
| **NMT** | Opus-MT vi-en | INT8 Marian | 125.4 MiB | ~176.5 MiB | Apache-2.0 |
| **Safety** | Clinical Rule Validator | Pure Python/JSON | < 0.1 MiB | ~0.0 MiB | MIT |
| **TTS** | Piper EN (ljspeech) | ONNX + espeak data | 77.7 MiB | ~72.1 MiB | Public Domain / GPL-3.0* |
| **Branch B** | SwiftF0 + NumPy stats | ONNX + pure NumPy | 1.1 MiB | ~5.2 MiB | MIT |
| **Total** | **Full Pipeline (`--full`)** | **All Edge ONNX** | **274.8 MiB** | **~457 MiB idle** | Open & Compliant |

*(GPL audit details documented in [docs/TTS_GPL_OPTIONS.md](Prototype/docs/TTS_GPL_OPTIONS.md))*

---

## 📊 Measured Benchmarks (Laptop Results)

All figures below are measured on the shipping reference pipeline (`0 stubs`, INT8 models, 2 threads, offline guard active). Full benchmark reports are available in [reports/LAPTOP_RESULTS.md](Prototype/reports/LAPTOP_RESULTS.md).

### 1. End-to-End Latency
Measured across audio duration crops (p50 and p95 confidence intervals):

| Input Audio Duration | PTT Release ➔ English Text | PTT Release ➔ First Audio Sample | Real-Time Factor (RTF) |
|---|---|---|---|
| **2.0 seconds** | **137 ms** | **371 ms** [292, 436] | 0.18 |
| **4.0 seconds** | **250 ms** | **671 ms** [593, 723] | 0.16 |
| **6.0 seconds** | **333 ms** | **929 ms** [810, 1011] | 0.15 |

### 2. Clinical Semantic Safety & Recall
- **Safety Recall on Real NMT Errors**: **90.7% [77.9%, 97.4%]** (caught dosage drops, drug confusions, unit discrepancies).
- **False-Block Rate on Valid Inputs**: Low false-block (**0.0% - 3.7%**), ensuring critical translations are spoken promptly.
- **Seeded Clinical Error Recall**: **97.7%** detection across 1,279 synthetic failure modes.

### 3. Noise Robustness Findings ([ADR-001](Prototype/docs/ADR-001-denoise.md))
- Under standard background noise (DEMAND office/hallway) and medical alarm beeps, Zipformer ASR achieves **9.3% - 13.0% WER** at 10 dB to 0 dB SNR.
- Neural single-channel denoisers (GTCRN) introduce phase artifacts in severe multi-speaker babble; empirical evidence supported locking default denoise to `OFF` for the single-mic configuration.

---

## 🚀 Quick Start Guide

### 1. Requirements & Prerequisites
- Python 3.10+ (tested on Windows 11 & Linux)
- Git & Git LFS
- Windows PowerShell or Bash terminal

### 2. Setup Environment
```bash
# Clone repository
git clone https://github.com/kaim-tqkhai711/OneVoice.git
cd OneVoice/Prototype

# Setup virtual environment
python -m venv .venv

# Activate environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements-bench.txt
```

### 3. Model Checkpoints
Models are stored locally under `Prototype/models/` to guarantee 100% offline reproducibility without remote downloads.
Check [Prototype/docs/OFFLINE_FILES.md](Prototype/docs/OFFLINE_FILES.md) for the exact SHA-256 manifests.

### 4. Run End-to-End Translation (`--full`)
Translate a Vietnamese audio clip to English with full safety checks, prosody extraction, and gate logic:

```powershell
$env:PYTHONPATH = "src"
python -m tonebridge.cli --wav models\asr\zipformer-vi-int8\test_wavs\0.wav --out out.wav --full --assert-offline --log results\turns.jsonl
```

### 5. Run Verification & Test Suite
```powershell
# Run all 81 unit & pipeline tests
python -m pytest tests/ -q

# Verify offline integrity (asserts zero socket connections and zero external network calls)
python -m pytest tests/test_offline.py -q

# Run runtime manifest audit
python tools/offline_audit.py manifest
```

---

## 👥 Dành cho Teammate: Hướng dẫn phát triển tiếp

> 🔔 **LƯU Ý QUAN TRỌNG:** Pipeline tham chiếu trên laptop (`Prototype/`) đã hoàn thiện **0 stubs, 100% offline, 81 tests xanh**. Toàn đội cần phối hợp để thực hiện các bước tiếp theo theo kế hoạch.

### 🎯 Phân công nhiệm vụ theo vai trò

| Thành viên | Mã Speaker | Vai trò chính | Nhiệm vụ ưu tiên cần làm ngay |
|---|:---:|---|---|
| **Khải** (Lead) | `spk01` | System Architecture & Sound/Speech | Giám sát tích hợp, review PR, điều phối thí nghiệm trên Snapdragon 712. |
| **Chi** | `spk02` | Sound / Speech & Branch B | **1.** Thu âm theo bộ kit.<br>**2.** Sau khi đủ audio cả nhóm: chạy `tools/train_urgency.py` để huấn luyện MLP urgency và đánh giá LOSO F1.<br>**3.** Hoàn thiện ngưỡng `UNKNOWN` / `HIGH` / `LOW`. |
| **Trí** | `spk03` | ML / NLP — ASR Engine | **1.** Thu âm theo bộ kit.<br>**2.** Đánh giá in-domain WER khi có audio team (`tools/run_asr_eval.py`).<br>**3.** Nghiên cứu bộ lọc "Noise quá lớn" để kích hoạt REPEAT gate trong môi trường babble. |
| **Thịnh** | `spk04` | ML / NLP — NMT & Safety & TTS | **1.** Thu âm theo bộ kit.<br>**2.** Điền bản dịch tham chiếu chuẩn vào [docs/nmt_ref_template.csv](Prototype/docs/nmt_ref_template.csv) để test BLEU/chrF.<br>**3.** Mở rộng glossary y tế và thẩm định licensing của Piper/espeak-ng theo [docs/TTS_GPL_OPTIONS.md](Prototype/docs/TTS_GPL_OPTIONS.md). |
| **Tân** | `spk05` | ML / NLP — Edge & Qualcomm AI Hub | **1.** Thu âm theo bộ kit.<br>**2. RỦI RO R1 (Deadline D5):** Lấy token Qualcomm AI Hub, chạy thử script `tools/aihub_submit.py` để profile model trên QCS6490 / Snapdragon 8 Gen.<br>**3.** Cắm máy Snapdragon SD712 qua adb, chạy `tools/device_probe.sh` đo hệ số $k$. |
| **Bảo** | `spk06` | Mobile / Web3 / Fullstack | **1.** Thu âm theo bộ kit.<br>**2.** Triển khai Android App (Scope P2): dựng UI Compose / Kotlin tích hợp Sherpa-ONNX & ONNX Runtime.<br>**3.** Hoàn thiện Clinician Local Dashboard & Hash-chain Audit Log (metadata only, offline). |

---

### 🚨 VIỆC GẤP SỐ 1: BỘ THU ÂM NHÓM (`recording_kit/`)

Tất cả 6 thành viên cần hoàn thành thu âm theo tài liệu hướng dẫn:
👉 **[Hướng dẫn thu âm chi tiết (HUONG_DAN_THU_AM.md)](Prototype/recording_kit/HUONG_DAN_THU_AM.md)**

- **Mục tiêu:** Cung cấp dữ liệu huấn luyện bộ phân loại vocal urgency và đo in-domain WER.
- **Thời gian thực hiện:** Khoảng 12 - 15 phút mỗi người.
- **Nội dung:** Đọc 30 câu trong `recording_kit/script_vi.csv`, mỗi câu 2 lần (1 lần `NEUTRAL`, 1 lần `URGENT`). Tổng cộng 60 file audio WAV/FLAC.
- **Quy tắc đặt tên file:** `<speaker_id>_<NEU|URG>_<sentence_id>.wav` (Ví dụ: `spk03_NEU_s01.wav`, `spk03_URG_s01.wav`).
- **Gửi file:** Nén thư mục thành `spkXX.zip` kèm file `speaker_metadata_template.csv` đã điền và gửi cho Khải.

---

### 📱 Hướng dẫn thử nghiệm trên thiết bị thật (Snapdragon SD712)
1. Cài đặt Android Platform Tools (`adb`) trên máy tính:
   ```powershell
   winget install Google.PlatformTools
   ```
2. Bật **USB Debugging** trên điện thoại Snapdragon và kết nối với máy tính.
3. Chạy script thăm dò phần cứng:
   ```bash
   bash tools/device_probe.sh
   ```
4. Đo đạc benchmark và hiệu chuẩn hệ số $k$ (`RTF_phone / RTF_laptop`) theo hướng dẫn trong [docs/DEVICE_SESSION.md](Prototype/docs/DEVICE_SESSION.md).

---

## 📁 Repository Structure & Doc Map

```
OneVoice/
├── README.md                                  # Tài liệu tổng quan dự án (File này)
├── .gitignore                                 # Git ignore chuẩn (loại trừ model nặng, .venv, .vscode)
├── ToneBridge_TechProposal_v3_1_Revised.docx   # Hồ sơ đề xuất kỹ thuật nộp BTC (v3.1)
├── ToneBridge_Solo_Prototype_Timeline.xlsx    # Kế hoạch tiến độ chi tiết
│
├── onevoice/                                  # Tài liệu đặc tả yêu cầu & thiết kế tổng thể
│   ├── 01_PRODUCT_REQUIREMENTS.md            # Product Requirements Document (PRD v1.1)
│   ├── 02_TECHNICAL_REQUIREMENTS.md          # Technical Specs & Hardware Targets
│   ├── 03_PRODUCT_FLOW.md                    # Zero-UI Interaction Flow & State Machine
│   ├── 04_BACKEND_ARCHITECTURE.md            # Interface Contracts & Schemas
│   ├── 05_UIUX_DESIGN.md                     # Clinician Dashboard & Mobile Specs
│   ├── 06_IMPLEMENTATION_PLAN.md             # Kế hoạch hành động 6 tuần của team
│   └── 07_SOUND_SPEECH_RESEARCH.md           # Nghiên cứu chuyên sâu về Prosody & Speech
│
├── paper research/                            # Các bài báo nghiên cứu nền tảng
│
└── Prototype/                                 # MÃ NGUỒN PIPELINE THAM CHIẾU (100% OFFLINE)
    ├── configs/                               # File cấu hình JSON (pipeline, glossary, safety, gate)
    ├── docs/                                  # Architecture Decision Records (ADRs) & Designs
    │   ├── ADR-001-denoise.md                 # Đánh giá thực nghiệm lọc ồn (OFF vs GTCRN vs OA)
    │   ├── ADR-002-nmt-direction.md           # Quyết định hướng dịch VI->EN vs VI->KO
    │   ├── BRANCH_B_DESIGN.md                 # Thiết kế chi tiết trích xuất 21 đặc trưng Prosody
    │   ├── OFFLINE_FILES.md                   # Danh mục toàn bộ file model offline
    │   └── DEVICE_SESSION.md                  # Hướng dẫn chạy benchmark trên Snapdragon
    ├── recording_kit/                         # Bộ công cụ thu âm kịch bản cho 6 thành viên
    │   ├── HUONG_DAN_THU_AM.md                # Hướng dẫn thu âm 12 phút
    │   ├── script_vi.csv                      # Kịch bản 30 câu thoại lâm sàng
    │   └── speaker_metadata_template.csv      # Mẫu metadata người đọc
    ├── reports/                               # Báo cáo thực nghiệm & kết quả đo thật
    │   ├── LAPTOP_RESULTS.md                  # Báo cáo số liệu đo đạc hoàn chỉnh trên laptop
    │   ├── OVERNIGHT_SUMMARY.md               # Tóm tắt tiến độ qua đêm & kết quả tối ưu
    │   └── NMT_QUALITY.md                     # Đánh giá chất lượng dịch thuật NMT
    ├── src/tonebridge/                        # Mã nguồn Python cốt lõi
    │   ├── pipeline.py                        # Điều phối luồng 2 nhánh (Branch A & B)
    │   ├── cli.py                             # Giao diện dòng lệnh CLI
    │   ├── branch_b.py                        # Trích xuất đặc trưng cảm xúc & ngữ điệu
    │   ├── safety.py                          # Semantic Safety Checker v1.2
    │   ├── gate.py                            # Decision Gate (5 actions)
    │   ├── nmt_constraints.py                 # Ràng buộc từ vựng y tế cho NMT
    │   ├── offline_guard.py                   # Sandbox chặn và giám sát kết nối mạng
    │   └── stages/                            # Các module ASR, NMT, TTS, VAD, Denoise
    ├── tests/                                 # Bộ kiểm thử tự động (81 tests passed)
    └── tools/                                 # Scripts đo đạc, benchmark, và kiểm toán
```

---

## 📜 Quy định lập trình (Portability Law)

Để đảm bảo mã nguồn chạy được mượt mà trên thiết bị di động Snapdragon (Android / NPU) mà không bị phụ thuộc vào môi trường máy tính cồng kềnh, nhóm tuân thủ nghiêm ngặt **Quy luật khả chuyển (Portability Law)**:

1. **Đường suy luận (Inference Path)** chỉ sử dụng đồ thị ONNX thông qua `ONNX Runtime` hoặc `sherpa-onnx`, kết hợp với logic thuần (Pure Python / Kotlin) đọc cấu hình từ file JSON.
2. **Không phụ thuộc thư viện nặng trong Runtime**: Tuyệt đối **không** import `torch`, `librosa`, `pyworld`, `transformers` trong luồng suy luận chính. Các thư viện này chỉ dùng trong khâu huấn luyện offline hoặc kiểm thử đối chuẩn.
3. **100% Offline**: Mọi model phải được đọc từ thư mục cục bộ `models/`. Mọi request mạng trong quá trình chạy inference đều bị chặn bởi `OfflineGuard`.
4. **Git Workflow**: Không commit file model nhị phân nặng (`.onnx`, `.bin`) hoặc file audio thô vào Git. Tuân thủ file `.gitignore`. Tạo branch cho từng tính năng và tạo Pull Request có review trước khi gộp vào `main`.

---

## ⚖️ Tuyên bố trách nhiệm (Responsible Use)

ToneBridge là một công cụ hỗ trợ giao tiếp và nhận biết mức độ khẩn cấp dưới sự giám sát của nhân viên y tế. Thiết bị **không** thực hiện chẩn đoán y khoa và **không** thay thế nhận định chuyên môn của bác sĩ hay thông dịch viên y tế chuyên nghiệp. Tín hiệu ngữ điệu và cờ báo độ khẩn cấp chỉ mang tính chất tham khảo cho quá trình phân loại bệnh nhân (triage support).

---

<div align="center">

**ToneBridge Team — OneVoice AI Challenge**
*Saigon AI Hub × Qualcomm*

</div>
