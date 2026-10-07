# ToneBridge — Product Flow

> **Version:** 1.1 (đồng bộ v1.1: LID bằng confidence-race 2 ASR đơn ngữ, TTS 2 engine, denoise 3 chế độ) · User flows + data flow qua Dual-Branch Pipeline.

---

## 1. Actors & Modes

| Actor | Interaction |
|-------|-------------|
| Patient (EN/KR speaker) | Nói tự nhiên — zero UI |
| Nurse (VI speaker) | Nói tự nhiên; liếc urgency indicator khi cần |
| Head nurse | Xem dashboard offline |

**App modes:** `IDLE` → `LISTENING` (VAD armed) → `CAPTURING` → `PROCESSING` → `SPEAKING` → quay lại `LISTENING`.

## 2. Core Flow — One Translation Turn (Zero-UI)

```
┌────────────┐   speech detected   ┌────────────┐  end-of-speech  ┌─────────────┐
│ LISTENING   │ ──────────────────► │ CAPTURING  │ ───────────────► │ PROCESSING  │
│ (VAD armed) │                     │ (buffering)│   (VAD ~200ms    │ (dual branch│
└────────────┘                      └────────────┘    silence)      └──────┬──────┘
      ▲                                                                    │
      │                        TTS playback xong                           ▼
      │                             ┌────────────┐               ┌─────────────┐
      └─────────────────────────────│  SPEAKING  │◄──────────────│   FUSION    │
                                    └────────────┘               └─────────────┘
```

**Step-by-step:**
1. App ở `LISTENING`: audio thread chạy DSP front-end liên tục, Silero VAD chấm điểm từng frame 30ms. Compute gần như bằng 0 (chưa model nặng nào chạy).
2. VAD vượt ngưỡng speech → `CAPTURING`: buffer PCM (đã qua DSP) vào ring buffer. Utterance cap 15s (auto-cut).
3. VAD phát hiện ~200ms silence → end-of-utterance → `PROCESSING`, buffer đóng băng và fork thành 2 nhánh **song song**:
   - **Branch A (meaning):** denoise theo `denoise_mode` (off/on/oa — mode ship do benchmark quyết, xem ADR-001) → **LID + ASR**: chạy song song 2 ASR đơn ngữ của pair đang chọn (Moonshine-tiny 27M/model — đủ rẻ để chạy cả hai), lấy kết quả có confidence cao hơn làm chiều dịch (fallback: Whisper language-ID) → NMT sang ngôn ngữ đích.
   - **Branch B (emotion):** trích f0 contour, RMS energy envelope, speaking-rate → (i) urgency score, (ii) prosody envelope chuẩn hóa theo thời gian.
4. **Fusion:** TTS theo ngôn ngữ đích (Piper cho vi/en · MeloTTS cho ko) synthesize câu dịch → post-process áp prosody envelope trên waveform (scale pitch/duration/energy theo Branch B — cùng một path cho cả 2 engine). Urgency score vượt ngưỡng → bắn event lên UI + audit log.
5. `SPEAKING`: phát audio dịch qua loa. UI hiển thị minimal transcript (xem UIUX doc).
6. Playback xong → về `LISTENING`. Hướng dịch tiếp theo tự xác định bằng language-ID của utterance kế (EN vào → ra VI; VI vào → ra EN/KR theo pair đang chọn).

## 3. Direction & Language Handling

- User chọn **language pair** một lần khi mở session (EN⇄VI hoặc VI⇄KR) — đây là thao tác chạm duy nhất của cả session.
- Trong session, **chiều dịch tự động** trên mỗi utterance bằng confidence-race giữa 2 ASR đơn ngữ của pair (hệ quả của việc dùng ASR monolingual Moonshine — xem doc 02 §3; fallback Whisper LID): nghe EN → nói ra VI; nghe VI → nói ra EN (hoặc KR). Không nút bấm đổi chiều.
- Nếu chênh lệch confidence giữa 2 ASR thấp (< threshold): phát lại câu "Xin nhắc lại / Please repeat" bằng cả 2 ngôn ngữ (graceful degradation, không đoán bừa).

## 4. Urgency Flag Flow

```
Branch B features ──► Urgency classifier ──► score ∈ [0,1]
                                              │
                              score ≥ 0.7 ────┼──► UI: viền màu hổ phách + icon pulse
                                              │    Audit log: {ts, urgency: true}
                                              │    (KHÔNG âm thanh cảnh báo — tránh làm bệnh nhân hoảng thêm)
                              score < 0.7 ────┴──► không action
```

Nguyên tắc: cờ khẩn là **tín hiệu hỗ trợ**, nurse quyết định. Không auto-escalate, không notification đi xa hơn màn hình thiết bị + dashboard.

## 5. Session & Dashboard Flow

```
[Mở app] → [Chọn pair + Start session] → N translation turns → [End session]
                                              │
                                              ▼ (mỗi turn)
                                   append audit entry (hash-chained)
                                              │
                                              ▼
[Head nurse mở dashboard trên browser LAN] → xem: sessions, turns count,
                                              urgency events timeline, integrity check ✓
```

Dashboard **đọc-only**, không điều khiển app. Data = metadata only.

## 6. Error & Edge-Case Flows

| Tình huống | Hành vi |
|-----------|---------|
| ASR trả rỗng / confidence thấp | Phát "Please repeat" song ngữ; không dịch bừa |
| Utterance > 15s | Auto-cut tại 15s, xử lý phần đã thu, UI hint "nói ngắn hơn" |
| 2 người nói chồng lấn | VAD/denoise cố gắng; nếu ASR confidence thấp → repeat flow. (Diarization = out of scope, ghi rõ limitation) |
| Model load fail | App không vào được session; hiển thị lỗi rõ ràng + log |
| Pin < 15% | Banner cảnh báo (model nặng ngốn pin) |
| TTS voice thiếu cho pair | Chặn từ lúc chọn pair (không cho start) |

## 7. Demo Flow (competition — 3.5 phút)

1. **Setup (15s):** bật airplane mode TRƯỚC mặt giám khảo → mở app → chọn EN⇄VI.
2. **Happy path (45s):** câu triage EN → nghe VI; câu VI đáp → nghe EN. Zero touch.
3. **Wow #1 — Prosody A/B (45s):** phát cùng câu distress: prosody OFF (giọng phẳng) vs ON (giữ hoảng loạn). Đây là moment ăn điểm Innovation.
4. **Wow #2 — Noise A/B (60s):** loa ngoài phát ER ambience ~80dB (clip từ noise dataset — chính bộ đã benchmark) → dịch vẫn chạy; show màn hình so sánh WER theo denoise mode OFF/ON/OA. Điểm nói thêm với giám khảo: "literature 2025 nói denoise thường làm medical ASR tệ đi — chúng tôi không tin mặc định, chúng tôi đo cả 3 chế độ."
5. **Urgency + dashboard (30s):** câu distress → viền amber sáng → mở dashboard trên laptop (LAN) cho thấy event + hash-chain integrity ✓.
6. **Close (15s):** slide latency đo từ AI Hub. "Số thật, silicon thật, không cloud."

Backup: video quay sẵn toàn bộ flow, phòng sự cố live.
