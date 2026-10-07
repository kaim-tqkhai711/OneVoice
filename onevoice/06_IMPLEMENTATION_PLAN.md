# ToneBridge — Implementation Plan (6 tuần, remote → hội quân)

> **Version:** 1.1 (đồng bộ model stack v1.1 + gates G1-G8 của doc 07; noise 100% dataset có sẵn) · Team: Khải, Chi (Sound/Speech) · Trí, Thịnh, Tân (ML/NLP) · Bảo (Web3/Fullstack)
> Mô hình: remote/async đến giữa tháng 8, sau đó hội quân tích hợp phần cứng.
> Ràng buộc thực tế: **sinh viên, quỹ giờ lệch nhau** → kế hoạch tính theo *deliverable*, không theo giờ công.

---

## 0. Nguyên tắc vận hành (nhắc lại từ kick-off, không thương lượng)

1. **GitHub = nguồn sự thật.** Branch theo module, PR + review chéo bắt buộc, không push thẳng `main`.
2. **Async-first.** Update tiến độ bằng văn bản; 1 sync call giữa tuần (gỡ blocker); **demo nội bộ mỗi Chủ nhật** — show thứ *chạy được*.
3. **Interface Contract chốt Tuần 1** (schema trong `04_BACKEND_ARCHITECTURE.md` §2). Đổi schema = PR riêng + cả 3 nhóm approve.
4. **Số liệu là tài sản:** mọi kết quả eval commit vào `eval/results/`. Không có số → không được claim trong hồ sơ.
5. **Definition of Done (mọi task):** code trên `main` + test/eval đi kèm + chạy được bởi người *không phải* tác giả.

## 1. Milestones khớp lịch BTC

| Mốc BTC | Deliverable của team | Tuần |
|---------|---------------------|------|
| ~~Tháng 6 — Ghi danh~~ | ✅ Xong | — |
| **Tháng 7 — Tech Spec** | Hồ sơ + bảng số ĐO THẬT (AI Hub latency, WER matrix sơ bộ) | W4 |
| Tháng 8-9 — Prototype | App end-to-end trên phone + hội quân ráp form factor | W6 → offline sprint |
| Tháng 10 — Field test | Test môi trường ồn thật, tune config | sau W6 |

## 2. Lộ trình 6 tuần

### W1 — Nền móng & Hợp đồng giao diện
| Owner | Deliverable |
|-------|-------------|
| Cả team | Repo + CI dựng xong; mọi người chạy được 1 model mẫu qua AI Hub (chứng chỉ nhập môn — ai kẹt phải kêu trước Thứ 5) |
| Khải + Chi + Trí | **Interface Contract v1 merged** (schemas §2 doc 04) |
| Trí, Thịnh, Tân | Chọn & tải model candidates: ASR = **Moonshine-Tiny vi/ko/en (ONNX sẵn trên HF, sherpa-onnx hỗ trợ Android)** + Whisper-base + PhoWhisper-base; NMT = Opus-MT en↔vi + en↔ko + NLLB-600M-ct2-int8 (**lưu ý: Opus-MT vi↔ko KHÔNG tồn tại** — xem ADR-002); TTS = Piper vi/en + **MeloTTS-Korean (Piper không có ko)**. Chạy inference laptop baseline |
| Khải, Chi | **Noise 100% từ dataset có sẵn (không tự thu):** tải Kaggle Hospital Ambient (eval-only) + Freesound CC0 + MUSAN/DNS (tune) + viết `alarm_synth.py` (beep IEC 60601-1-8) + babble generator; script `noise_inject.py` xuất cặp (clean, noisy); tải UViko + FLORES-200 + Zeroth-Korean cho ADR-002/G8; draft DSP front-end (Python prototype) |
| Bảo | Skeleton Android app (Compose, 4 màn hình rỗng) + Room + Ktor hello-world |

**Gate W1:** Interface Contract merged + mọi người đã chạy AI Hub. Không đạt → họp khẩn, không trôi sang W2.

### W2 — Xương sống Branch A (laptop trước, phone sau)
| Owner | Deliverable |
|-------|-------------|
| Trí | ASR candidates INT8 static quant (Moonshine qua ONNX path; Whisper qua AI Hub recipe sẵn); **WER (en/vi) + CER (ko)** baseline clean vào `eval/results/` — bảng so Moonshine vs Whisper là input cho G5/G8 |
| Thịnh | Opus-MT en⇄vi chạy + glossary y khoa v0 (seed Meddict); **ADR-002 chốt cuối W2** bằng số: chrF++ trên FLORES-200 vie↔kor + ~100 câu triage từ UViko — pivot 2×Opus-MT vs NLLB-600M-int8; đồng thời xác nhận ADR-003 (seq2seq, không LLM) trong biên bản |
| Tân | ONNX Runtime + QNN delegate chạy được 1 model trên phone thật (proof lên silicon) |
| Khải | DSP front-end port sang code chạy realtime (HPF + spectral gate) + Silero VAD tích hợp; đo VAD accuracy trên noisy clips |
| Chi | Prosody extractor v0 (pYIN f0 + RMS + rate) chạy trên utterance file; xuất `ProsodyResult` đúng schema |
| Bảo | Audit hash-chain + unit test; API `/api/sessions` trả mock data |

**Gate W2:** Branch A chạy end-to-end **trên laptop** với audio file: wav → denoise → ASR → NMT text. Demo Chủ nhật: dịch 5 câu triage EN→VI từ file.

### W3 — Lên silicon + Nhánh B sống
| Owner | Deliverable |
|-------|-------------|
| Trí + Tân | ASR (model thắng G5) + NMT chạy trên phone qua NPU/CPU; **bảng latency per-stage từ AI Hub profiling** (input trực tiếp cho Tech Spec); **G5/G8:** chốt routing ASR (Moonshine vi/ko/en vs Whisper-base) bằng WER/CER + latency trên silicon |
| Thịnh | TTS trên phone: Piper (vi/en) + **MeloTTS-Korean (G7: đo latency, xác nhận WORLD post-process áp được như Piper)** |
| Khải | Denoise GTCRN (fallback DFN3 — G3) chạy trên phone (CPU); **WER/CER matrix: clean/20/10/5 dB × denoise OFF/ON/OA** — dữ liệu quyết định guardrail ADR-001 (G4); G6: pYIN vs CREPE-tiny bằng f0 corr tại 10dB |
| Chi | Urgency classifier v0: tự thu + gán nhãn ~100 utterances scripted (neutral/distress), train MLP nhỏ, báo precision/recall |
| Bảo | Live Session screen nối vào pipeline mock; dashboard SPA v0 hiển thị data thật từ SQLite |

**Gate W3:** có số latency thật trên silicon + WER matrix sơ bộ. Đây là hai bảng vàng của hồ sơ tháng 7.

### W4 — Hợp nhất + NỘP TECH SPEC
| Owner | Deliverable |
|-------|-------------|
| Chi + Thịnh | **Prosody transfer v1:** áp f0/duration envelope lên Piper output; A/B sample nghe được |
| Khải | Tune ngưỡng VAD endpoint + urgency threshold trên data thật |
| Trí + Tân | Pipeline orchestrator trên phone: mic → 2 nhánh → fusion → loa. **First full turn on-device** 🎉 |
| Bảo | Urgency UI (amber) + audit ghi thật + integrity endpoint |
| **Khải (PO) + cả team** | **Tổng hợp & nộp Tech Spec** — kiến trúc + ADR-001 + bảng latency AI Hub + WER matrix + roadmap. **Deadline cứng của BTC.** |

**Gate W4:** Tech Spec nộp. Một turn dịch hoàn chỉnh chạy trên phone (dù còn xù xì).

### W5 — Kín vòng & đo đạc toàn diện
| Owner | Deliverable |
|-------|-------------|
| Trí, Thịnh, Tân | Tối ưu latency về budget (§4 doc 02); fix memory (model warm-up, không reload giữa session); vi⇄ko pair hoàn thiện |
| Khải, Chi | Benchmark cuối: WER/CER matrix đầy đủ 2 pairs (ko chấm CER), prosody f0-correlation, urgency P/R; mini MOS panel nội bộ (blind, gồm cả giọng MeloTTS-ko); **`eval_energy.py`: % pin + mAh cho session 15' (metric production)** |
| Bảo | Dashboard hoàn thiện + polish 4 màn hình theo doc 05; airplane-mode test toàn trình |
| Cả team | Bug bash Chủ nhật: mỗi người 30' dùng app như nurse, log mọi khó chịu |

**Gate W5:** demo flow 6 bước (doc 03 §7) chạy trọn không lỗi ≥ 3 lần liên tiếp.

### W6 — Đóng băng & chuẩn bị hội quân
| Owner | Deliverable |
|-------|-------------|
| Cả team | **Feature freeze đầu tuần.** Chỉ fix bug. |
| Khải | Kịch bản demo + pitch nói; quay **video backup** toàn flow |
| Chi | Bộ utterance demo "vàng" (thu chuẩn, chọn best) cho A/B prosody |
| Trí, Thịnh, Tân | Tài liệu model card (kích thước, quant scheme, số đo) — appendix hồ sơ |
| Bảo | Build release APK ổn định + checklist thiết bị demo (phone chính + phone backup cài sẵn) |

**Exit criteria trước hội quân:** APK release chạy demo flow trên 2 phone; video backup tồn tại; mọi số liệu nằm trong `eval/results/`; docs cập nhật. → Những ngày gặp mặt tháng 8 chỉ dành cho form factor + tích hợp phần cứng + tập pitch.

## 3. Phase 2 — Hội quân (nửa cuối tháng 8, phác thảo)

- Ngày 1-2: ráp phần cứng demo (phone + mic gắn kẹp áo/lanyard mock, loa mini) — mô phỏng form factor.
- Ngày 3-4: field-style test (phát ER noise thật qua loa lớn, đo lại WER on-site, tune config JSON).
- Ngày 5+: tập pitch + demo ≥ 5 lần full-run; quay video final.

## 4. Rủi ro thực thi & phương án

| Rủi ro | Dấu hiệu sớm | Phương án |
|--------|--------------|-----------|
| vi⇄ko NMT kém (không có Opus-MT trực tiếp) | chrF++ W2 thấp cả 2 phương án | ADR-002 chọn phương án đỡ tệ hơn; nếu vẫn kém → demo primary EN⇄VI, VI⇄KR là "supported, shown briefly"; Phase 2: LoRA NLLB trên UViko |
| ASR chậm/kém trên phone | latency/WER W3 > budget | Đường lùi nhiều nấc: Moonshine-tiny (27M) ↔ Whisper-base ↔ Whisper-tiny; cap utterance 10s |
| Moonshine không compile được lên NPU qua AI Hub (model mới, chưa có recipe sẵn) | compile job fail W2 | Chạy Moonshine trên CPU qua sherpa-onnx (27M đủ nhanh — đường Android đã có sẵn); NPU dành cho NMT; Whisper-base giữ làm đường NPU đã được Qualcomm dọn sẵn |
| LID confidence-race đoán sai chiều dịch | demo W4 dịch ngược chiều | Threshold chênh lệch confidence + repeat flow (doc 03 §3); fallback cuối: nút đổi chiều thủ công trong dev panel |
| Prosody transfer robotic | A/B W4 không thuyết phục | Fallback: chỉ transfer pitch mean + range + rate (bỏ contour chi tiết) — vẫn đủ tạo khác biệt A/B |
| Thành viên bận thi cử | miss 2 demo Chủ nhật liên tiếp | PO redistribute task; scope cắt theo thứ tự: vi⇄ko → dashboard polish → urgency (KHÔNG BAO GIỜ cắt: offline, dual-branch, latency) |
| AI Hub quota/queue chậm | job pending lâu W2-3 | Đăng ký account sớm W1, batch job qua đêm, giữ fallback local ONNX benchmark |

## 5. Cadence tổng kết

- **Thứ 4:** sync call 30' (blocker only).
- **Chủ nhật:** integration demo 45' — mỗi nhóm show working software; PO ghi biên bản quyết định vào repo (`docs/decisions/`).
- **Mỗi PR:** ít nhất 1 reviewer khác nhóm khi chạm Interface Contract.
