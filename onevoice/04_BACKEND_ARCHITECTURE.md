# ToneBridge — System / Backend Architecture

> **Version:** 1.1 (đồng bộ model stack v1.1 doc 02: GTCRN/CPU, ASR per-lang Moonshine, MeloTTS-ko, denoise_mode 3 chế độ, model manifest) · Lưu ý ngữ nghĩa: "backend" của ToneBridge KHÔNG phải server cloud.
> Toàn bộ "backend" sống **trên thiết bị**: inference engine + local services + on-device dashboard server.

---

## 1. High-Level Component Map

```
┌───────────────────────────── ANDROID APP (Snapdragon phone) ─────────────────────────────┐
│                                                                                           │
│  ┌─────────────┐    ┌──────────────────────── CORE ENGINE (native/Kotlin+JNI) ─────────┐  │
│  │  UI LAYER   │    │                                                                  │  │
│  │  (Compose)  │◄──►│  AudioService          PipelineOrchestrator                      │  │
│  │  - Session  │    │  - Oboe input stream   - state machine (IDLE→…→SPEAKING)         │  │
│  │  - Live view│    │  - DSP front-end       - fork Branch A/B                         │  │
│  │  - Urgency  │    │  - ring buffer         - fusion + playback                       │  │
│  └─────────────┘    │                                                                  │  │
│                     │  ┌─ Branch A ─────────────────┐   ┌─ Branch B ──────────────┐    │  │
│                     │  │ Denoise GTCRN (CPU)        │   │ ProsodyExtractor (CPU)  │    │  │
│                     │  │   mode: off|on|oa          │   │ UrgencyClassifier (CPU) │    │  │
│                     │  │ VAD Silero/TEN (CPU)       │   └─────────────────────────┘    │  │
│                     │  │ ASR per-lang (NPU/CPU)     │                                  │  │
│                     │  │   Moonshine-vi/ko/en       │                                  │  │
│                     │  │   (fb: Whisper-base)       │                                  │  │
│                     │  │ NMT (NPU/CPU)              │                                  │  │
│                     │  │   Opus-MT en⇄vi            │                                  │  │
│                     │  │   vi⇄ko: theo ADR-002      │                                  │  │
│                     │  └────────────────────────────┘                                  │  │
│                     │  TTS (Piper vi/en · MeloTTS ko) + ProsodyTransfer/WORLD (CPU)    │  │
│                     └──────────────────────────────────────────────────────────────────┘  │
│                                                                                           │
│  ┌──────────────────────┐        ┌──────────────────────────────┐                        │
│  │  LOCAL DATA LAYER    │        │  DASHBOARD SERVER (on-device)│                        │
│  │  - SQLite (metadata) │◄──────►│  - embedded HTTP (Ktor)      │◄─── Browser (LAN/     │
│  │  - Hash-chain audit  │        │  - read-only JSON API        │      hotspot, offline) │
│  │  - Model store       │        │  - static dashboard SPA      │                        │
│  └──────────────────────┘        └──────────────────────────────┘                        │
└───────────────────────────────────────────────────────────────────────────────────────────┘

          OFFLINE TOOLING (laptop, không dính runtime):
          eval harness (Python) · AI Hub compile/profile scripts · dataset prep
```

## 2. Module Contracts (Interface Contract — chốt Tuần 1)

Đây là "hợp đồng" giữa 3 nhóm. Mọi thay đổi schema phải qua PR + review chéo.

### 2.1 AudioService → Pipeline
```
UtteranceBuffer {
  session_id: str
  utterance_id: str
  pcm: float32[]          # 16 kHz mono, ĐÃ qua DSP front-end
  sample_rate: 16000
  t_start, t_end: float
}
```

### 2.2 Branch A output
```
MeaningResult {
  utterance_id: str
  src_lang: "en"|"vi"|"ko"   # từ LID: chạy song song 2 ASR đơn ngữ của pair,
                              # chọn theo confidence (fallback: Whisper LID) — chốt tại G5
  asr_text: str
  asr_confidence: float       # avg logprob normalized
  tgt_lang: "en"|"vi"|"ko"
  nmt_text: str
  denoise_mode: "off"|"on"|"oa"   # mode thực dùng cho utterance này (trace cho eval)
  timings_ms: {denoise, vad, asr, nmt}
}
```

### 2.3 Branch B output
```
ProsodyResult {
  utterance_id: str
  f0_contour: float[]         # 10ms hop, Hz, 0 = unvoiced
  energy_envelope: float[]    # RMS, 10ms hop
  speaking_rate: float        # syllables/s (proxy)
  urgency_score: float        # [0,1]
  timings_ms: {extract, classify}
}
```

### 2.4 Fusion → TTS
```
SynthesisRequest {
  utterance_id: str
  text: str                   # nmt_text
  engine: "piper"|"melo"      # piper cho vi/en · MeloTTS cho ko (Piper không có voice ko)
  voice: str                  # voice id theo tgt_lang
  prosody: ProsodyResult | null   # null = prosody transfer OFF
}
# Prosody transfer là post-process TRÊN WAVEFORM output (WORLD/pyworld)
# → interface không phụ thuộc engine; thêm engine mới không đổi schema.
```

### 2.5 Audit entry (Data layer)
```
AuditEntry {
  session_id, utterance_id: str
  ts: iso8601
  lang_pair: "en-vi"|"vi-ko"
  duration_s: float
  urgency_flag: bool
  prev_hash: hex64
  entry_hash: hex64           # SHA256(prev_hash || canonical_json(entry_without_hashes))
}
# TUYỆT ĐỐI KHÔNG có field audio, asr_text, nmt_text trong AuditEntry.
```

## 3. Threading & Runtime Model

| Thread/Executor | Việc | Ghi chú |
|-----------------|------|---------|
| Audio thread (Oboe callback) | DSP front-end + VAD scoring + ring buffer | Realtime-safe: không alloc, không lock dài, không JNI ngược |
| Pipeline executor | Orchestrator state machine, fork/join Branch A+B | Coroutine-based |
| NPU inference | ASR encoder, NMT | Qua ONNX Runtime/TFLite + QNN delegate; warm-up lúc mở session. Static shapes bắt buộc (HTP); w8a16 theo recipe AI Hub |
| CPU inference | Denoise GTCRN (48K params — cố ý đặt ở CPU để NPU trọn cho ASR/NMT), VAD, prosody, urgency, TTS | |
| Dashboard server | Ktor embedded, port 8080 localhost/LAN | Read-only |

**Model lifecycle:** load + warm-up toàn bộ model khi Start Session (chấp nhận 3-5s loading screen) → giữ trong memory suốt session → release khi End Session. Không lazy-load giữa chừng (tránh latency spike).

## 4. Data Layer

- **SQLite** (Room): bảng `sessions`, `audit_entries`. Không lưu audio/transcript.
- **Model store:** `/data/.../models/` — model đóng gói trong APK expansion hoặc tải 1 lần lúc setup (setup được phép online; **inference thì không**). Kèm **manifest `models.json`** {name, version, quant_scheme, sha256} — app verify hash lúc load, từ chối model lệch manifest (production requirement, rẻ cho prototype).
- **Config:** JSON per-environment (VAD threshold, urgency threshold, notch freqs, `denoise_mode: off|on|oa`, `oa_alpha`) — hot-reload được từ dev screen để tune nhanh khi field test. Config ship = config thắng benchmark, có test CI so khớp.

## 5. Dashboard (vai trò của Bảo)

- **Server:** Ktor embedded trong app, endpoints:
  - `GET /api/sessions` — list sessions + counts
  - `GET /api/sessions/{id}/events` — audit entries
  - `GET /api/integrity` — verify hash chain, trả `{valid: bool, checked: n}`
- **Client:** static SPA (Svelte/React build) đóng gói trong assets — mở từ browser laptop cùng hotspot. Không CDN, mọi asset local.
- **Bảo mật mức prototype:** read-only + LAN only. (Auth/TLS = roadmap, ghi rõ limitation.)

## 6. Offline Tooling (repo `tools/`, chạy trên laptop)

| Tool | Chức năng | Owner |
|------|-----------|-------|
| `aihub_compile.py` | Submit compile/quantize job lên AI Hub, tải model artifact | ML |
| `aihub_profile.py` | Profile model trên hosted Snapdragon, xuất bảng latency (JSON+MD) | ML |
| `eval_wer.py` | WER/CER matrix: lang × SNR × denoise {off,on,oa} (ko chấm CER) | Sound + ML |
| `eval_prosody.py` | f0 correlation source vs output | Sound |
| `noise_inject.py` | Trộn noise dataset vào clean set tại SNR chỉ định; **xuất cặp (clean, noisy) song song** cho ground-truth f0 | Sound |
| `alarm_synth.py` | Sinh alarm beep y tế chuẩn IEC 60601-1-8 (thay thu âm — noise 100% từ nguồn có sẵn) | Sound |
| `eval_nmt.py` | chrF++ trên FLORES-200 + triage set, per direction (data cho ADR-002) | ML |
| `eval_energy.py` | Pin/năng lượng per session qua `dumpsys batterystats` (metric production) | Sound + ML |
| `make_glossary.py` | Build medical glossary constraint cho NMT (Meddict seed) | ML |

Mọi tool: Python 3.11, type-hinted, Pydantic config, kết quả ghi `eval/results/*.json` (commit vào repo — số liệu là tài sản).

## 7. Repo Layout

```
tonebridge/
├── docs/                    # 6 file MD này + tech spec
├── app/                     # Android (Kotlin/Compose)
│   ├── audio/               # Oboe, DSP front-end, VAD        [Sound]
│   ├── pipeline/            # orchestrator, branch A/B, fusion
│   ├── inference/           # ORT/TFLite wrappers, QNN delegate [ML]
│   ├── prosody/             # extractor, urgency, transfer     [Sound]
│   ├── data/                # Room, audit hash-chain           [Bảo]
│   ├── dashboard/           # Ktor server + SPA assets         [Bảo]
│   └── ui/                  # Compose screens
├── models/                  # exported/quantized artifacts (git-lfs hoặc release assets)
├── tools/                   # offline tooling (trên)
├── eval/
│   ├── datasets/            # scripts tải/prep (data KHÔNG commit)
│   └── results/             # JSON kết quả (COMMIT)
└── .github/workflows/       # CI: lint, unit test, no-network-in-inference check
```

## 8. CI/CD (mức prototype)

- PR bắt buộc: ktlint/detekt (Kotlin), ruff+mypy (Python), unit tests pass.
- Custom check: grep network API (OkHttp/HttpURLConnection/sockets) trong `app/pipeline|inference|prosody` → fail nếu xuất hiện (bảo vệ lời hứa offline).
- Nightly (optional): chạy eval smoke test trên 10 file audio.
