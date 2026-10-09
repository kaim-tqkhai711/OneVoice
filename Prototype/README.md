# Prototype ToneBridge — hướng dẫn chạy trên laptop

Base **TechProposal v3.1**; scope hiện tại: laptop CPU, **vi-en / en-vi / en-ko / ko-en**. Tiếng Hàn dùng mã `ko`, không dùng `kr`.

Cập nhật **09/10/2026**, branch `codex/laptop-four-direction-integration`. VI→EN đã thử micro/loa thật; EN→VI đã sinh WAV từ smoke. Hai chiều tiếng Hàn còn dừng CONFIRM vì safety chưa có bilingual review.

[README repository](../README.md) · [Hướng dẫn tích hợp/chia việc](docs/LAPTOP_FOUR_DIRECTION_INTEGRATION.md) · [Báo cáo kiểm chứng](reports/LAPTOP_FOUR_DIRECTION_VERIFICATION_2026-10-09.md)

## 1. Tạo môi trường

Môi trường đã kiểm chứng: Windows 11, Python 3.12, CPU. Mở PowerShell tại thư mục `OneVoice/Prototype`. Các lệnh sau không cần activate virtualenv:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-laptop.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip check
```

Dùng [requirements-laptop.lock.txt](requirements-laptop.lock.txt) cho bản tích hợp. Các file `requirements-nmt*.txt` dành cho môi trường text độc lập của branch NMT/safety; tránh cài chồng các lock vì có pin numpy khác nhau. Trên Windows, dùng `-X utf8` cho tools; watchdog tự truyền cờ này cho CLI.

## 2. Tải và kiểm tra assets

Máy mới cần mạng trong bước setup; weights không nằm trong Git. Chạy lần lượt:

```powershell
.venv/Scripts/python.exe -X utf8 tools/provision_laptop.py
.venv/Scripts/python.exe -X utf8 tools/inventory_baseline_nmt.py
.venv/Scripts/python.exe -X utf8 tools/provision_nmt_laptop.py --directions en-vi ko-en
.venv/Scripts/python.exe -X utf8 tools/provision_argos_enko.py
.venv/Scripts/python.exe -X utf8 -c "import sys;sys.path.insert(0,'tools');from provision_laptop import provision;provision([])"
.venv/Scripts/python.exe tools/check_laptop_assets.py
```

Lệnh `provision([])` cuối cùng cập nhật manifest chung sau khi đã bổ sung các model. Adapter kiểm hash file inference trước khi load. Môi trường đã kiểm chứng có 452 assets khớp hash; số lượng này mô tả bộ assets tại lần kiểm chứng, không phải yêu cầu số file cho mọi máy.

[Registry laptop](configs/nmt/models_laptop.json) lưu đường dẫn và policy model. EN→VI/KO→EN dùng Marian CPU local, checkpoint được chuyển sang safetensors qua weights-only load và kiểm state_dict. VI→EN dùng ONNX INT8. EN→KO dùng ứng viên Argos/CTranslate2; checkpoint Helsinki EN→KO ban đầu vẫn cách ly. Gói Argos chưa có license cụ thể trong package và chưa được chốt cho phát hành.

Inference dùng assets local. `--assert-offline` kiểm tra hoạt động mạng trong lượt chạy; smoke/benchmark hiện ghi nhận 0 network attempts và 0 OS connections.

## 3. Chạy một WAV

Ví dụ VI→EN với WAV đi kèm ASR:

```powershell
.venv/Scripts/python.exe tools/run_laptop.py --timeout 60 -- --config configs/laptop_vi-en.json --wav models/asr/zipformer-vi-int8/test_wavs/0.wav --out results/laptop_current/vi-en-demo-new.wav --assert-offline
```

Ví dụ EN→VI:

```powershell
.venv/Scripts/python.exe tools/run_laptop.py --timeout 60 -- --config configs/laptop_en-vi.json --wav models/asr/zipformer-en-int8/test_wavs/0.wav --out results/laptop_current/en-vi-demo-new.wav --assert-offline
```

| Chiều | Config |
|---|---|
| VI→EN | [laptop_vi-en.json](configs/laptop_vi-en.json) |
| EN→VI | [laptop_en-vi.json](configs/laptop_en-vi.json) |
| EN→KO | [laptop_en-ko.json](configs/laptop_en-ko.json) |
| KO→EN | [laptop_ko-en.json](configs/laptop_ko-en.json) |

Đổi `--wav` thành WAV của bạn và dùng đường dẫn `--out` mới mỗi lượt: CLI không ghi đè output có sẵn. Input được downmix/resample về mono 16 kHz; preset hiện giới hạn 15 giây. WAV quá dài yêu cầu REPEAT.

Mẫu WAV của nhà cung cấp có thể nằm ngoài grammar clinical và bị CONFIRM; không phải mẫu nào cũng sinh audio. Chỉ SPEAK/SPEAK_CUE ghi WAV. Thêm `--play` để phát WAV đã qua gate trên Windows.

Runtime thật là mặc định; `--full`/`--real` là alias. `--stub` dùng pipeline giả/tone cho debug, không dùng để đánh giá model thật.

## 4. Đọc kết quả và log

| Gate | Hành vi |
|---|---|
| SPEAK | Tạo audio bản dịch đã qua kiểm tra. |
| SPEAK_CUE | Tạo audio bản dịch kèm cue theo policy. |
| REPEAT | Yêu cầu đọc lại; không tạo audio. |
| CONFIRM | Cần kiểm tra/xác nhận nội dung; không tạo audio. |
| ABSTAIN | Không tiếp tục lượt lỗi/không đủ điều kiện; không tạo audio. |

Hai chiều Hàn hiện không được phát tự động. CLI chưa có bước tương tác cho người dùng duyệt câu CONFIRM để phát. Lỗi model/input/stage/TTS trả ABSTAIN; audio TTS sinh một phần được loại bỏ.

Xem `gate.action`, `safety_status` và `output_written` trong kết quả/log để biết lượt có được phát hay không. CLI trả exit code 1 khi lỗi, nhưng một lượt bị safety chặn có thể vẫn trả 0. Watchdog trả 124 khi timeout và ghi sự kiện; exit code 0 không đồng nghĩa với SPEAK.

Log mặc định: `results/laptop_current/turns.jsonl`. Log loại transcript, translation và nội dung trong diagnostic. `--log-content` chỉ bật khi chủ động debug với dữ liệu được phép lưu; `--log` đổi đường dẫn. Không commit audio/log chứa dữ liệu người dùng.

NMT nhận transcript tiếng Anh sau định dạng tối thiểu: viết hoa ký tự đầu và thêm dấu kết thúc nếu thiếu. Transcript gốc được giữ; từ/số trong câu không được sửa. Đây chưa phải khôi phục dấu câu tổng quát. UNK, thiếu EOS, truncation hoặc evidence/constraints không hợp lệ chặn TTS; PASS của checker không chứng minh ASR đúng.

## 5. Thử microphone/loa VI→EN

Chuẩn bị micro và loa, rồi chạy:

```powershell
.venv/Scripts/python.exe -X utf8 tools/microphone_laptop_check.py --seconds 15 --out results/laptop_current/mic-demo-new.wav
```

Tool load model trước, chờ Enter ở READY. Sau khi nhấn Enter, đọc **“Xin chào”** ngay khi hiện RECORDING. Nếu nhận đúng và gate cho SPEAK, máy phát **“Hello”**. Audio nguồn giữ trong RAM; WAV đầu ra chỉ tạo khi gate cho phép.

Người dùng đã xác nhận nghe rõ “Hello” trong một lượt thử thật. Ba chiều còn lại chưa có xác nhận nghe trực tiếp; hai chiều Hàn cần hoàn tất review safety trước.

CLI chung cũng hỗ trợ timed capture:

```powershell
.venv/Scripts/python.exe tools/run_laptop.py --timeout 60 -- --config configs/laptop_vi-en.json --record-seconds 15 --out results/laptop_current/mic-cli-new.wav --play --assert-offline
```

CLI bắt đầu capture sau khi load model, không chờ Enter như tool ở trên. Đây là thu theo thời lượng, chưa có nút PTT. Metric first sample đo lúc có mẫu audio đầu tiên; chưa phải thời điểm người dùng nghe thấy âm thanh.

## 6. Regression và benchmark

Chạy regression:

```powershell
$env:PYTHONPATH="src"
.venv/Scripts/python.exe -X utf8 -m pytest -q -rs -p no:cacheprovider --basetemp results/laptop_current/pytest-readme-new
```

Dùng `--basetemp` riêng: pytest dọn nội dung thư mục đó. Lần kiểm chứng tích hợp ghi nhận **204 passed, 3 skipped**, một warning từ generation config của test fixture. Skip do thiếu Branch B golden/FLEURS, GTCRN và noise corpus; máy mới thiếu assets có thể skip thêm.

Ví dụ smoke/benchmark EN→VI:

```powershell
.venv/Scripts/python.exe -X utf8 tools/smoke_integrated_laptop.py --direction en-vi --out-dir results/laptop_current/integration-final/en-vi
.venv/Scripts/python.exe -X utf8 tools/bench_laptop_runtime.py --config configs/laptop_en-vi.json --manifest configs/laptop_integration_smoke_en-vi.jsonl --out-dir results/laptop_current/bench-readme/en-vi --repeats 34
```

Chạy smoke trước vì nó tạo greeting fixture dùng trong manifest benchmark. Smoke chạy audio thật và 20 câu text dev của chiều đã chọn. Đổi đồng bộ direction/config/manifest/thư mục khi đo chiều khác; đo mỗi chiều trong process riêng. Giữ holdout nguyên trạng.

Bản tích hợp đã đo 608 lượt + 12 warmup, mỗi chiều 3–4 WAV độc lập được lặp. Chưa đạt corpus 100 câu độc lập mỗi chiều. EN→VI peak RSS 1182 MiB; RAM đã plateau trên mẫu lặp sau khi giữ worker cố định, chưa có đo qua đêm.

Nếu gọi Pipeline từ code, dùng context manager hoặc `pipe.close()` sau lượt cuối để đóng worker. Một instance phục vụ các lượt tuần tự.

## 7. Việc còn lại

- NMT/safety: review [80 câu dev](reports/LAPTOP_DEV_REVIEW_PACKET_2026-10-09.jsonl) bằng người biết ngôn ngữ; sửa lỗi model/coverage checker, review reference và lexicon Hàn, kiểm license EN→KO. Không dùng checker tự gán nhãn gold.
- Runtime/audio: ít nhất 100 WAV độc lập mỗi chiều; ASR confidence/noise calibration; đo RAM dài hạn với câu đa dạng; thử hardware EN→VI, rồi hai chiều Hàn sau review.
- Urgency hiện UNKNOWN; denoise/urgency classifier tắt trong preset. Android/NPU, nút PTT và incremental TTS chưa được kiểm chứng. Marian CPU còn cần tối ưu nếu release gate yêu cầu dưới 1 GiB RAM.

Bạn phụ trách runtime/audio, bạn của bạn phụ trách NMT/safety. Tạo branch riêng từ branch tích hợp; thống nhất trước khi sửa contract/config/factory/registry dùng chung. Xem [phân chia chi tiết](docs/LAPTOP_FOUR_DIRECTION_INTEGRATION.md).

## Tài liệu liên quan

- [TechProposal v3.1](../ToneBridge_TechProposal_v3_1_Revised_Template.docx).
- [Báo cáo kiểm chứng](reports/LAPTOP_FOUR_DIRECTION_VERIFICATION_2026-10-09.md) và [summary JSON](reports/LAPTOP_FOUR_DIRECTION_SUMMARY_2026-10-09.json).
- [Audit ngày 08/10/2026](docs/ToneBridge_Deep_Codebase_Audit_2026-10-08.md), [handoff runtime](docs/LAPTOP_RUNTIME_HANDOFF.md), [handoff NMT/safety](docs/NMT_SAFETY_HANDOFF.md): tài liệu lịch sử, cần đọc cùng báo cáo tích hợp.
- [README Prototype trước cập nhật](docs/README_PROTOTYPE_LEGACY_2026-10-09.md): giữ để tham khảo lịch sử; lệnh và đường dẫn cũ có thể không còn phù hợp.
