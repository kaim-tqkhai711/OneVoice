# ToneBridge — UI/UX Design

> **Version:** 1.1 (đồng bộ v1.1: model name từ manifest, denoise 3 chế độ trên dev panel) · Triết lý: đây là sản phẩm **Zero-UI** — màn hình tồn tại để *chứng minh* và *trấn an*, không phải để *vận hành*.
> Nghịch lý cần quản lý: sản phẩm khoe "không cần nhìn màn hình", nhưng demo cần màn hình để giám khảo *thấy* điều đang xảy ra. UI được thiết kế cho khán giả demo + nurse liếc nhanh, không phải cho thao tác.

---

## 1. Design Principles

1. **Zero-touch trong session.** Sau khi bấm Start, không một thao tác chạm nào cần thiết cho việc dịch.
2. **Glanceable, không readable.** Nurse chỉ *liếc* 0.5s: trạng thái + urgency. Không bắt đọc đoạn văn.
3. **Trạng thái luôn rõ.** Người dùng phải biết ngay app đang nghe / xử lý / nói — bằng màu + motion, không bằng chữ.
4. **Calm by default, loud when it matters.** Giao diện tĩnh lặng; chỉ urgency mới được phép "ồn" thị giác. Không âm báo động (tránh làm bệnh nhân hoảng thêm).
5. **Privacy visible.** Trạng thái offline (airplane icon) hiển thị thường trực — biến lời hứa thành thứ nhìn thấy được.

## 2. Design Tokens

| Token | Giá trị | Dùng cho |
|-------|--------|----------|
| `ink` | #0A1F24 | nền tối chính |
| `teal` | #1FB6A6 | trạng thái LISTENING, brand |
| `teal-deep` | #0E7C72 | accent phụ |
| `coral` | #FF6B5E | SPEAKING, sóng âm phát |
| `amber` | #FFB443 | **duy nhất** cho urgency |
| `paper` | #F4F7F6 | nền sáng (dashboard) |
| Type | Sora (heading), Inter (body), JetBrains Mono (số liệu) | đồng bộ pitch deck |
| Motion | breathing 1.2s (listening), pulse 0.6s (urgency) | trạng thái sống |

Contrast tối thiểu WCAG AA; mọi trạng thái phân biệt được cả khi mù màu (shape + motion, không chỉ màu).

## 3. Screens (chỉ 4 màn hình — cưỡng lại mọi cám dỗ thêm)

### S1 — Home / Session Setup
```
┌─────────────────────────────┐
│  ToneBridge        ✈ OFFLINE│   ← badge offline thường trực
│                             │
│      Chọn cặp ngôn ngữ      │
│  ┌───────────┐ ┌──────────┐ │
│  │  EN ⇄ VI  │ │  VI ⇄ KR │ │   ← 2 card lớn, 1 chạm
│  └───────────┘ └──────────┘ │
│                             │
│  ┌─────────────────────────┐│
│  │      ▶ START SESSION    ││   ← nút duy nhất, full-width
│  └─────────────────────────┘│
│  models ready ✓  ·  v0.1    │
└─────────────────────────────┘
```
- Chạm duy nhất của cả phiên: chọn pair + Start. Loading model 3-5s có progress + dòng tên model thật đang load, ví dụ "loading Moonshine-vi INT8 → NPU" (khoe kỹ thuật một cách chính đáng; string lấy từ manifest `models.json`, không hard-code — model có thể đổi theo gate G5).

### S2 — Live Session (màn hình chính, Zero-UI)
```
┌─────────────────────────────┐
│  EN ⇄ VI          ✈  ● REC │
│                             │
│        ╭───────────╮        │
│        │  ~~~~~~~  │        │   ← orb sóng âm trung tâm:
│        │  ~~~~~~~  │        │      teal breathing = LISTENING
│        ╰───────────╯        │      teal ripple    = CAPTURING
│                             │      spinner nhỏ    = PROCESSING
│   "Chest pain since when?"  │      coral wave     = SPEAKING
│   ────────────────────────  │
│   「Đau ngực từ khi nào?」   │   ← transcript 2 dòng cuối cùng,
│                             │      mờ dần sau 8s (glanceable)
│                             │
│  [⏸ pause]        [■ end]  │   ← 2 nút nhỏ, góc dưới
└─────────────────────────────┘
```
- **Urgency state:** viền màn hình amber + orb chuyển amber pulse + haptic nhẹ. Không popup, không âm thanh.
- Transcript hiển thị để *giám khảo/nurse xác nhận*, không phải để đọc hội thoại — tự mờ, không scroll history (privacy + glanceable).
- Dev gesture ẩn (long-press orb): panel toggle prosody ON/OFF + denoise 3 trạng thái OFF/ON/OA (segmented control) — phục vụ demo A/B.

### S3 — Session End Summary
```
│  Session #12 kết thúc        │
│  ── 14 utterances · 6m32s    │
│  ── 1 urgency event  ⚑      │
│  ── audit chain ✓ intact     │
│  [Done]                      │
```
Tối giản, xác nhận dữ liệu đã ghi (metadata-only) + chain integrity.

### S4 — Dashboard (web, laptop/tablet của head nurse)
```
┌────────────────────────────────────────────┐
│ ToneBridge Ward Dashboard        ✈ LAN-only│
│                                            │
│ Hôm nay: 8 sessions · 2 urgency events     │
│ ┌────────────────────────────────────────┐ │
│ │ Timeline (giờ) ────⚑────────⚑───────── │ │
│ └────────────────────────────────────────┘ │
│ Sessions                                   │
│ #12  en-vi  6m32s  1⚑   chain ✓           │
│ #11  vi-ko  3m10s  0    chain ✓           │
│ ...                                        │
│ [Verify integrity] → "34/34 entries valid" │
└────────────────────────────────────────────┘
```
Read-only. Không nội dung hội thoại — chỉ metadata. Nút Verify integrity là "sân khấu" cho phần audit hash-chain của Bảo khi demo.

## 4. State Machine ↔ Visual Mapping

| State | Orb | Màu | Motion |
|-------|-----|-----|--------|
| LISTENING | vòng mảnh | teal | breathing chậm 1.2s |
| CAPTURING | waveform live | teal sáng | amplitude theo mic |
| PROCESSING | orb thu nhỏ | teal-deep | spinner quỹ đạo |
| SPEAKING | waveform phát | coral | amplitude theo TTS |
| URGENCY (overlay) | viền màn hình | amber | pulse 0.6s + haptic |
| ERROR (repeat) | orb rung nhẹ | muted | shake 1 lần + text song ngữ |

## 5. Accessibility & Môi trường thực

- ER ồn → không phụ thuộc audio cue cho nurse; mọi tín hiệu có kênh thị giác + haptic.
- Găng tay y tế → mục tiêu chạm tối thiểu đã đạt by design; 2 nút pause/end kích thước ≥ 48dp.
- Ánh sáng gắt → dark theme mặc định, contrast cao; text tối thiểu 16sp.
- Song ngữ UI: label chính bằng VI + EN song song (người dùng chủ là nurse VN).

## 6. UX của demo (thiết kế cho giám khảo)

- Badge `✈ OFFLINE` luôn trong khung hình — nhắc liên tục điều kiện thi.
- Dev panel A/B (prosody, denoise) thiết kế *đẹp có chủ đích* vì nó sẽ lên hình lúc demo — 1 toggle lớn "Prosody Transfer" + 1 segmented control "Denoise: OFF · ON · OA" (3 chế độ là chi tiết kỹ thuật ăn điểm, cho giám khảo thấy đây là nút được quyết bằng benchmark).
- Màn urgency phải "ăn camera": viền amber đủ dày để thấy từ xa 3m.
