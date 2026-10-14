# Milestone B — Output Gate kiểm chứng, an toàn lâm sàng & privacy

Ngày: **14/10/2026**. Cơ sở: `TECH_LEAD_AUDIT_2026-10-14.md` và `MILESTONE_A_2026-10-14.md`.

Đã triển khai P0-3, P0-4 và P0-6 ở mức logic/runtime có regression. Không thêm production mock/stub. Các stage skeleton của Milestone A vẫn chỉ được dùng trong mode stub/benchmark và ở biên kiểm thử.

**Giới hạn nghiệm thu:** logic 9 phản ví dụ và privacy được kiểm thử; chưa kiểm chứng lâm sàng độc lập hoặc đo lại ASR babble 0 dB. Checkout thiếu model/corpus. Vì vậy quality policy của full runtime mặc định **không tự động SPEAK**: âm thanh yếu yêu cầu REPEAT; bằng chứng mạnh nhưng chưa calibration yêu cầu CONFIRM. Không coi ngưỡng proxy 6 dB là SNR vật lý hoặc là xác suất transcript đúng.

## 1. Diff so với Milestone A

| Task | File tạo/chỉnh sửa | Hành vi sau thay đổi |
|---|---|---|
| P0-3 | **Mới** `src/tonebridge/clinical_relations.py` | Grammar VI/EN theo mệnh đề; tuple subject–drug–amount–unit–frequency–action–negation–modality; bổ sung symptom, bound, context. So sánh multiset một-một, bảo toàn binding và số lần xuất hiện |
| P0-3 | `src/tonebridge/safety.py` | Relation comparison bắt buộc; lỗi parse critical, thêm/đổi relation đều `passed=False`. Chẩn đoán dose dùng cùng Decimal parser, không còn veto sai từ integer/set scan. Legacy slots/intensity là kiểm tra bổ sung |
| P0-3 | `configs/safety_lexicon_vi_en.json` | Version `v2.0-relations`; thêm aliases triệu chứng. Giữ thuốc/đơn vị/aliases có sẵn; OOV có anchor được so tên nguyên vẹn, không suy đoán alias thuốc |
| P0-3 | `src/tonebridge/nmt_constraints.py` | Constraint số dùng Decimal từ relation parser; `không phẩy năm`/`nửa` không bị ép thành `5`/`0`. Cú pháp critical mơ hồ không sinh numeric constraint đoán mò |
| P0-4 | **Mới** `src/tonebridge/quality_gate.py` | RMS dBFS, clipping fraction, frame-energy SNR proxy, VAD ratio, voicing quality, word rate và repeated bigrams; reject/confirm độc lập token confidence |
| P0-4 | `src/tonebridge/stages/asr_sherpa.py`, `stages/vad_silero.py`, `pipeline.py` | ASR trả acoustic evidence; VAD ratio là tổng speech span thực, không phải first-to-last span có padding. Pipeline đo raw PTT trước denoise/crop và gắn chất lượng SwiftF0 từ Branch B |
| P0-4 | `src/tonebridge/gate.py` | Full cần acoustic evidence; missing/weak/incomplete/uncalibrated evidence không đến TTS. Safety critical vẫn CONFIRM/ABSTAIN; chỉ SPEAK/SPEAK_CUE có approved text |
| P0-4/P0-7 | `src/tonebridge/config.py`, `configs/pipeline.json`, `stages/factory.py` | Thêm `quality` vào effective settings/fingerprint; calibration cần identifier rõ ràng. Hash gồm phiên bản parser/policy và actual lexicon bytes. Startup capabilities mô tả policy chưa calibration |
| P0-4 | `tools/asr_conf_gate_check.py`, `tools/snr_gate_probe.py` | Probe dùng cùng công thức runtime; bổ sung acoustic evidence/quality rejection, bad-transcript recall, false reject, automatic coverage khi có dữ liệu. Confidence probe chưa đo voicing nên phải ghi incomplete context; kết quả không phải calibration full |
| P0-6 | `src/tonebridge/contracts.py` | Nội dung giữ trong bộ nhớ để safety/TTS; `TurnRecord.model_dump()`/`model_dump_json()` mặc định metadata-only. Explicit API opt-in là `context={"eval_mode": True}` |
| P0-6 | `src/tonebridge/telemetry.py` | `JsonlLogger` default allowlist metadata; bỏ text, identifiers, arbitrary reasons/error strings/network details. Enum/code và số hữu hạn; mã lỗi không biết thành `StageError`. Refuse legacy/mixed/corrupt log, không xoá nội dung cũ |
| P0-6 | `src/tonebridge/cli.py`, `runtime.py` | `--eval-mode` rõ ràng; JSONL/stdout cùng projection. Tất cả CLI modes dùng process riêng; suppress cả Python và native stdout/stderr của worker. Argument/config/IO errors không echo user values/message/path. IPC nội bộ giữ kết quả cho TTS/evaluation, không ghi raw IPC vào log |
| Tests/docs | **Mới** `tests/test_safety_relations.py`, `test_quality_gate.py`, `test_privacy_logging.py`; `README.md` | Test chống unsafe TTS, positive controls, quality rejects, privacy/error/native-output boundaries. Test CLI cũ cần kiểm text được cập nhật thêm `--eval-mode`, vẫn giữ assertions và toàn bộ test Milestone A |

## 2. Kết quả 9 phản ví dụ audit, Mục 5

Nguồn/bản dịch sai được giữ đúng như audit. ASR confidence trong các ca này là **0.99**, urgency UNKNOWN. Không dùng confidence thấp để che lỗi safety.

| # | Phản ví dụ | Safety bản sai | Gate bản sai | Bản đúng tương ứng |
|---|---|---|---|---|
| 1 | Aspirin 2 mg / paracetamol 5 mg bị đảo binding | `passed=False` | CONFIRM | `Take two milligrams of aspirin and five milligrams of paracetamol.` — PASS |
| 2 | Không dị ứng penicillin nhưng dị ứng aspirin bị đảo scope | `passed=False` | CONFIRM | `I am not allergic to penicillin but allergic to aspirin.` — PASS |
| 3 | `0,5` mg → `5` mg | `passed=False` | CONFIRM | `Take 0.5 milligrams of paracetamol.` — PASS |
| 4 | Tự ý thêm aspirin | `passed=False` | CONFIRM | `I take paracetamol.` — PASS |
| 5 | Mỗi 6 giờ → mỗi 2 giờ | `passed=False` | CONFIRM | `Take two tablets of paracetamol every six hours.` — PASS |
| 6 | Có thể dị ứng → khẳng định dị ứng | `passed=False` | CONFIRM | `I may be allergic to penicillin.` — PASS |
| 7 | Ngừng → tiếp tục | `passed=False` | CONFIRM | `Stop taking aspirin.` — PASS |
| 8 | Đau ngực → đau đầu | `passed=False` | CONFIRM | `I have chest pain.` — PASS |
| 9 | Clopidogrel ngoài lexicon → aspirin | `passed=False` | CONFIRM | `I am allergic to clopidogrel.` — PASS khi giữ nguyên opaque identity |

**9/9 bản sai bị chặn; 9/9 bản đúng `passed=True`, `confirm=False`.** Gate isolation của positive controls cho SPEAK khi không bật yêu cầu acoustic của full runtime; full runtime vẫn áp dụng quality policy bảo thủ riêng. Pipeline regression cho từng bản sai xác nhận **0 lần gọi TTS**, output buffer rỗng, không có approved speak text.

Các regression bổ sung kiểm tra:

- Decimal comma/dot/trailing zero/number words/half; malformed decimals và cardinal mơ hồ fail closed. Giá trị và phép đổi interval dùng Decimal, không dùng float để đối chiếu liều.
- Subject swaps, reordered independent relations, inherited negation cho danh sách thuốc, `chưa`/`not yet`, lifetime `never`, uncertain/future/advised/question modality.
- OOV thêm vào mệnh đề phối hợp, duplicate drug/dose, unknown unit/cú pháp; symptom ngoài grammar chặn thay vì mặc nhiên PASS.
- Conditions/history/alcohol/today/meals/site/infusion duration được binding vào relation context; đổi/bỏ context critical bị chặn.
- Tập **1.224 reference tổng hợp có sẵn**: **0 critical false blocks, 0 confirms**. Đây là compatibility regression trên mẫu đã được dùng để hoàn thiện grammar, **không phải test holdout độc lập**, không dùng để ước lượng false-alarm ngoài thực tế.

## 3. Quality gate và mức độ kiểm chứng

Quality settings mặc định: VAD ratio tối thiểu 0.15, RMS tối thiểu −45 dBFS, SNR proxy tối thiểu 6 dB, clipping tối đa 0.01, voicing quality tối thiểu 0.35, word rate tối đa 8/s, repeated-bigram fraction tối đa 0.6. **Các ngưỡng này chưa calibration.**

- Test high token confidence 0.999 vẫn REPEAT khi bất kỳ evidence yếu: energy, SNR proxy, clipping, VAD ratio, voicing, implausible rate hoặc repetition.
- Full thiếu evidence → REPEAT; thiếu VAD/voicing context → CONFIRM; evidence mạnh nhưng `calibrated=false` → CONFIRM.
- Stationary-noise và tone+noise fixture ở SNR danh định 0 dB bị REPEAT. Đây là kiểm thử deterministic policy, không phải babble corpus hoặc bằng chứng ASR nhận sai.
- Fixture sạch có biến thiên năng lượng cho evidence hữu hạn; mặc định vẫn CONFIRM. Chỉ test policy với calibration identifier **unit-test-only** mới kiểm nhánh cho phép tiếp tục; không bật calibration trong runtime config.
- Test pipeline xác nhận quality dùng raw audio và speech coverage, không bị waveform sau denoise che mất nhiễu.

**Chưa có số đo mới về bad-transcript recall, false reject hoặc coverage trên audio thật.** Bộ probe đã hỗ trợ tái đo khi provision model và FLEURS/noise corpora. Full default automatic SPEAK coverage là **0 theo policy** trước calibration; đây là lựa chọn fail-closed, không phải đạt mục tiêu độ chính xác. Không tái sử dụng raw-confidence kết quả cũ làm chứng nhận quality mới.

## 4. Privacy policy

Default row: `logging_policy=metadata-v1`; audio length, stage latencies, endpoint timing, total/RTF/RSS, direction/mode, config hash, gate action, status và safe error codes. Không có `asr_text`, `nmt_text`, `speak_text`, session/utterance identifiers, arbitrary reasons, capabilities hoặc network detail chứa nội dung.

`--eval-mode`: `logging_policy=evaluation-v1`; cho phép text/reasons chi tiết. Default path tách riêng `results/eval-turns.jsonl` với runtime `results/turns.jsonl`. Explicit logging API cũng cần `eval_mode=True` và serialization context tương ứng. Không tự xóa hoặc migrate log cũ: chọn file mới nếu logger từ chối policy hiện có.

Tests dùng PHI sentinel kiểm default model dump, JSONL, stdout/stderr của CLI, filenames/IDs, malicious raw dictionary fields, exception messages tại ASR/NMT/safety/Branch B/TTS, malformed/mixed logs, invalid CLI arguments/config/audio/IO. Kiểm cả `os.write` và Windows native `WriteFile`, đồng thời xác nhận output handles được restore. Eval opt-in giữ text như dự kiến.

## 5. Nghiệm thu

Môi trường: Windows x64, **Python 3.13.9**, virtualenv sạch `.venv/clean` của Milestone A, không dùng system-site-packages.

```powershell
.venv/clean/Scripts/python.exe -m pytest tests/ -q -rs -p no:cacheprovider
.venv/clean/Scripts/python.exe -m pip check
.venv/clean/Scripts/python.exe tools/check_runtime.py --mode stub --profile dev
.venv/clean/Scripts/python.exe -m compileall -q src/tonebridge tools/asr_conf_gate_check.py tools/snr_gate_probe.py
```

- [x] **251 passed, 7 skipped**, không failure/collection error. Giữ 154 test Milestone A; thêm 97 case regression.
- [x] `pip check`: **No broken requirements found**.
- [x] Preflight dev/stub: **PASS**, 15 distribution/import checks; config hash tại checkout hiện tại `a0e78500cd527429`.
- [x] Compile check và `git diff --check`: PASS (Git chỉ cảnh báo line-ending conversion).
- [x] 9 phản ví dụ: chặn 9/9; positive controls pass 9/9; không gọi TTS.
- [x] PCM uint8/int32, silence/overlong/invalid buffers và timeout/process failures từ Milestone A tiếp tục pass.
- [x] Default privacy và explicit eval opt-in pass; lỗi không echo transcript.
- [x] Effective config chứa thresholds chất lượng, phiên bản parser/policy và lexicon mới.

7 integration skips giữ nguyên lý do Milestone A: SwiftF0 asset; Branch B golden/FLEURS; frontend models; noise data; hai offline real-model cases; real Branch A model packs. Không biến integration thiếu assets thành mock PASS.

## 6. Phần cần kiểm chứng trước khi dùng thực tế

1. Provision model/corpus được pin, chạy full preflight; đánh giá quality trên clean/demand/babble theo SNR, có VAD/voicing thực. Report bad-transcript recall, false reject, coverage và confidence intervals; freeze thresholds cùng calibration provenance trước khi bật automatic speech.
2. Validation độc lập cho relation grammar với chuyên gia và câu ngoài template: tên/aliases thuốc OOV, ngữ cảnh nhiều mệnh đề, thời gian, liều/phác đồ, điều kiện và triệu chứng mới. Mở grammar bằng regression giữ binding; không nới fail-closed bằng cách bỏ qua từ critical.
3. Checker đối chiếu **ASR transcript với NMT**; một transcript sai nhưng dịch trung thành vẫn có thể giữ relation đúng. Acoustic proxy không chứng minh lời nói gốc. Trained urgency và kiểm chứng lâm sàng chưa nằm trong kết quả này.

Milestone B hoàn thành phần implementation/regression nêu trên; hệ thống vẫn là PoC với output policy bảo thủ, chưa production-ready cho quyết định lâm sàng.
