# D1: NMT bake-off + walking skeleton (2026-10-06)

## 1. Tóm tắt
- Bake-off SMaLL-100 (vi→ko trực tiếp) **trượt 3/3 ngưỡng** đặt trước → chọn phương án B (P0 = VI→EN). ADR-002 draft: `docs/ADR-002-nmt-direction.md`.
- Hạng mục 2–5 của D1 xong: contracts/config/telemetry, skeleton wav→wav (stub), license table v0, bộ thu âm.
- `tools/device_probe.sh` đã viết nhưng **chưa chạy với thiết bị thật** và máy này **không có `adb` trong PATH**.
- 10 test pytest xanh.

## 2. Bake-off NMT: số đo
Điều kiện chung (`tools/bench_hop.py`): laptop Intel Core Ultra (Family 6 Model 170), Windows 11, ORT 1.23.2 CPU EP, `intra_op=2`, INT8 động (avx2), greedy, max_len 48, 30 câu lâm sàng × 3 lượt = 90 lần chạy sau 3 warm-up. **Số laptop x86, không phải SD712.**

| Model | Tham số | License | INT8 | p50 [CI95] | p95 [CI95] | Peak RSS (suy luận) |
|---|---|---|---|---|---|---|
| opus-mt-tc-big-en-ko | 209,158,401 | CC-BY-4.0 | 265 MB | 171 ms [152–196] | 608 ms [287–664] | 966 MB |
| small100 (vi→ko) | 332.7M | MIT | 581 MB | 218 ms [203–239] | 373 ms [336–421] | 1596 MB |

- Lần chạy đầu của small100 (p50 307 / p95 498 ms, RSS 6.05 GB) bị nhiễu: chạy khi máy đang làm việc khác và RSS gồm cả bước quantize FP32. Bảng trên là lần chạy lại sạch.
- tc-big en→ko: bản dịch **rác** (cả ví dụ trong model card cũng sai) vì `source.spm` có từ không nằm trong `vocab.json`. Số latency chỉ mang tính tham khảo.

### Ngưỡng nhận (đặt trước) vs kết quả, SMaLL-100
| Ngưỡng | Yêu cầu | Đo được | Kết quả |
|---|---|---|---|
| Tiếng Hàn đọc được, đúng chủ đề | ≥ 9/10 | 8/10 | **TRƯỢT** |
| p50 laptop | ≤ 150 ms | 218 ms | **TRƯỢT** |
| INT8 | ≤ 350 MB | 581 MB | **TRƯỢT** |

Giả định của ngưỡng 150 ms: SD712 chậm hơn laptop 3–4 lần (est., chưa đo). Theo giả định đó small100 trên SD712 vào khoảng 650–870 ms cho một hop (est.).

### 10 cặp nguồn–đích (nguyên văn)
| # | vi | ko | Ghi chú |
|---|---|---|---|
| 1 | Bạn có bị đau ngực không? | 당신은 가슴에 고통을 느낀가요? | đọc được, ngữ pháp hơi sai |
| 3 | Bạn bị đau này bao lâu rồi? | 여러분은 얼마나 오래 이 고통을 겪었습니까? | đọc được |
| 4 | Bạn có dị ứng với loại thuốc nào không? | 어떤 약에 알레르기가 있습니까? | đúng |
| 5 | Tôi bị dị ứng với penicillin. | 나는 페니실린에 알레르기가있다. | đúng |
| 6 | Uống hai viên paracetamol mỗi sáu giờ. | 6 시간마다 두 개의 파라세타мол을 마십시오. | **tên thuốc lẫn ký tự Cyrillic** |
| 10 | Tôi không thở được. | 난 숨을 수 없어. | **sai nghĩa** ("tôi không thể trốn") |
| 14 | Tôi thấy chóng mặt và tức ngực. | 나는 가두르고 가슴을 느꼈습니다. | **vô nghĩa** |
| 22 | Chúng tôi sẽ tiêm cho bạn năm miligam. | 우리는 당신에게 5 밀리그램을 주입 할 것입니다. | đúng |
| 27 | Cô ấy không có dị ứng thuốc nào được biết. | 그녀는 알레르기가있는 약은 없었습니다. | **phạm vi phủ định bị lệch** |
| 30 | Tôi cần giúp đỡ ngay bây giờ. | 나는 지금 도움을 필요로. | thiếu động từ nhưng hiểu được |

**Giới hạn:** chỉ một người chấm, không phải người bản ngữ. Điểm 8/10 là "đọc được + đúng chủ đề", cận trên. Ba lỗi liên quan an toàn (6, 10, 27) là đúng loại lỗi mà Semantic Safety Check phải bắt.

## 3. Việc đã làm (đường dẫn)
| Hạng mục | File | Trạng thái |
|---|---|---|
| 2. Contracts + config + telemetry | `src/tonebridge/{contracts,config,telemetry}.py` | Xong, `tests/test_core.py` |
| 3. Skeleton wav→wav | `src/tonebridge/{pipeline,cli}.py`, `stages/{base,stubs}.py` | Xong, `tests/test_pipeline.py` (gồm test Branch B không nhận audio đã denoise, và CLI end-to-end) |
| 4. License table v0 | `LICENSES.md` | Xong; mục **U** còn cần xác minh |
| 5. Bộ thu âm | `recording_kit/{HUONG_DAN_THU_AM.md,script_vi.csv,speaker_metadata_template.csv}` | Xong, chờ bạn gửi |
| Q5. Device probe | `tools/device_probe.sh` | Cú pháp OK, đường "không có adb" OK; chưa thử với máy |

Lệnh chạy: `cd Prototype && .venv/Scripts/python -m pytest -q` (pass = 10 passed).
Skeleton: `PYTHONPATH=src .venv/Scripts/python -m tonebridge.cli --wav in.wav --out out.wav` (pass = exit 0, 1 dòng JSONL, `out.wav` @22050).

## 4. Kết quả xác minh checkpoint/license (nguồn chính)
- Zipformer VI hợp lệ: `zzasdf/viet_iter3_pseudo_label` (Apache-2.0, offline). `hynt/Zipformer-30M-RNNT-6000h` là **CC-BY-NC-ND-4.0** → loại.
- Silero VAD, GTCRN, SwiftF0: MIT; sherpa-onnx: Apache-2.0; opus-mt-vi-en: Apache-2.0 (GitHub/HF API).
- Piper EN: chọn `en_US-ljspeech-medium` (public domain, MODEL_CARD). Lessac/Amy có điều khoản Blizzard 2013, chưa đọc.
- Chưa xác minh: điều khoản dữ liệu huấn luyện của Zipformer VI, WebRTC APM build, **espeak-ng (GPL-3.0) khi sherpa-onnx phonemize voice Piper**, noise datasets.

## 5. Deviation / thay đổi plan
1. **#1 (sửa):** P0 = VI→EN; VI→KO không demo. Lý do: bake-off + ADR-002.
2. **#2:** không có Zipformer VI streaming → "<450 ms tail" (§4.2) đo lại là thời gian decode toàn bộ utterance.
3. **Chưa làm (đã lên D2):** OA mixer có căn mẫu (cross-correlation, assert lệch 0 mẫu) + test; công thức convex `β·x_enh + (1−β)·x_raw`, chuẩn hóa RMS về `x_raw`.

## 6. Timeline D2 hiện hành (10 h, tính theo quyết định của bạn)
| # | Việc | Giờ |
|---|---|---|
| 1 | Timebox GPU/DSP trên SD712 | 2.0 |
| 2 | APK sherpa-onnx dựng sẵn chạy trên SD712 | 1.5 |
| 3 | AI Hub QCS6490 profile 1 model nhỏ (cần token) | 1.5 |
| 4 | **RTF Zipformer VI 2 thread + kích thước INT8 thật**, rồi harness WER | 2.0 |
| 5 | opus-mt-vi-en: kiểm tra tokenization bằng ví dụ model card + `bench_hop.py` + glossary v0 | 1.5 |
| 6 | ADR-001: noise mixer + GTCRN + OA (β dev) + bảng WER, chạy nền | 1.5 tương tác |
| | **Tổng** | **10.0** |

Chú ý: hạng mục 4 chưa làm D1. Nếu RTF × độ dài câu trung bình ăn hết budget ASR thì báo ngay.

## 7. Blocker / việc cần bạn
- Cắm máy SD712, bật USB debugging, **cài platform-tools (`adb`) và thêm vào PATH** trước đầu D2.
- Token AI Hub (D2 #3).
- Gửi `recording_kit/` cho 5 teammate (mã speaker: Khải `spk01`, Chi `spk02`, Trí `spk03`, Thịnh `spk04`, Tân `spk05`, Bảo `spk06`; đổi nếu bạn gán khác).
- Xác nhận cách ghi deviation #1/#2 trong ADR-002 (đang ở trạng thái draft).
