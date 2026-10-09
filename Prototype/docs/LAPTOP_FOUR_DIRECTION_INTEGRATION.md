# Laptop bốn chiều — hướng dẫn sau tích hợp

Branch: `codex/laptop-four-direction-integration`. Ghép runtime `05b71b9` và NMT/safety `ef221c5`. Scope: laptop CPU, VI→EN / EN→VI / EN→KO / KO→EN; Hàn dùng mã `ko`.

## Trạng thái sử dụng

Factory mặc định đã chọn ASR, NMT, checker và TTS đúng chiều. VI→EN giữ ONNX INT8; EN→VI và KO→EN dùng Marian CPU local; EN→KO dùng ứng viên Argos/CTranslate2 INT8 riêng. Checkpoint Helsinki EN→KO ban đầu vẫn cách ly: 25.330 source pieces không có mapping, probe `Hello.` chứa UNK. Không tự dựng ID từ thứ tự SentencePiece.

Korean safety chưa có bilingual review: hai chiều Hàn trả CONFIRM và không tạo/phát audio. Đây là đường dịch và chặn safety hoạt động; chưa phải demo phát tiếng Hàn tự động. Gói Argos không kèm license cụ thể; chỉ là ứng viên thử nghiệm, cần review trước khi chọn cho bản phát hành.

Model thật có nhiều lỗi dịch clinical. Checker cũng có false block do lexicon hẹp, ví dụ `chest pains`. Xem [báo cáo](../reports/LAPTOP_FOUR_DIRECTION_VERIFICATION_2026-10-09.md) và [gói review](../reports/LAPTOP_DEV_REVIEW_PACKET_2026-10-09.jsonl). PASS không chứng minh ASR đúng hay câu phù hợp sử dụng clinical.

## Setup trên máy mới

PowerShell tại `OneVoice/Prototype`; Python 3.12. `.venv`, weights, WAV và log cục bộ không nằm trong Git.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-laptop.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -X utf8 tools/provision_laptop.py
.venv/Scripts/python.exe -X utf8 tools/inventory_baseline_nmt.py
.venv/Scripts/python.exe -X utf8 tools/provision_nmt_laptop.py --directions en-vi ko-en
.venv/Scripts/python.exe -X utf8 tools/provision_argos_enko.py
.venv/Scripts/python.exe -X utf8 -c "import sys;sys.path.insert(0,'tools');from provision_laptop import provision;provision([])"
.venv/Scripts/python.exe tools/check_laptop_assets.py
```

Setup có mạng, inference local-only. Model HF chỉ có `.bin` được nạp `torch.load(weights_only=True)`, kiểm `state_dict` và chuyển thành safetensors; shared `lm_head` chỉ được bổ sung khi model thực sự dùng chung storage. Manifest lưu hash của cả nguồn và file đã chuyển. Adapter kiểm hash các file inference trước khi load. Trên Windows chạy `-X utf8`; watchdog tự truyền cờ này.

Các vocab OPUS VI/EN/KO có một số SentencePiece bị lược khỏi vocab công bố. Giữ nguyên ID công bố; không thêm ID. Runtime EN→VI/KO→EN chấp nhận inventory này theo policy explicit ở registry; bất kỳ UNK trong câu nguồn/đích đều chặn TTS. Adapter Marian standalone mặc định vẫn strict nếu không bật policy đó.

`requirements-nmt*.txt` là môi trường text độc lập của branch bạn ấy; môi trường tích hợp dùng `requirements-laptop.lock.txt` hoặc `requirements-laptop-nmt.txt`, tránh cài chồng hai lock có numpy khác nhau.

## Chạy và kiểm tra

```powershell
.venv/Scripts/python.exe tools/run_laptop.py --timeout 60 -- --config configs/laptop_en-vi.json --wav path/to/english.wav --out results/laptop_current/translated.wav --assert-offline
```

Đổi config tương ứng cho chiều khác. Output phải là đường dẫn mới; chỉ SPEAK/SPEAK_CUE mới tạo WAV. `--play` phát WAV đã được duyệt. Log mặc định ở `results/laptop_current/turns.jsonl`, loại transcript/translation và nội dung nằm trong diagnostic. `--log-content` chỉ bật khi chủ động debug với dữ liệu được phép lưu.

NMT tiếng Anh nhận câu ASR đã định dạng tối thiểu: viết hoa ký tự đầu và thêm dấu chấm cuối nếu thiếu dấu kết thúc. Không thay từ/số/casing bên trong câu; transcript gốc vẫn giữ nguyên. Evidence ghi các thao tác định dạng. Đây chưa phải khôi phục dấu câu tổng quát; checker vẫn phải chặn sai phủ định/câu hỏi/dose.

Lỗi model/input/stage/TTS trả ABSTAIN; audio sinh một phần bị loại. Watchdog diệt process nếu quá timeout và ghi sự kiện vào log. Pipeline dùng hai worker bền qua các lượt liên tiếp để hạn chế native thread cache tăng RAM; gọi `pipe.close()` hoặc dùng `with Pipeline(...)` sau lượt cuối. Một instance phục vụ các lượt tuần tự.

```powershell
.venv/Scripts/python.exe -X utf8 tools/smoke_integrated_laptop.py --direction en-vi --out-dir results/laptop_current/integration-final/en-vi
.venv/Scripts/python.exe -X utf8 tools/bench_laptop_runtime.py --config configs/laptop_en-vi.json --manifest configs/laptop_integration_smoke_en-vi.jsonl --out-dir results/laptop_current/stability/en-vi --repeats 34
$env:PYTHONPATH="src"
.venv/Scripts/python.exe -X utf8 -m pytest -q -rs -p no:cacheprovider --basetemp results/laptop_current/pytest-local-new
```

Dùng thư mục `--basetemp` riêng cho mỗi lần chạy; pytest dọn nội dung thư mục đó. Smoke tạo greeting TTS làm fixture và đọc mẫu ASR có sẵn, rồi chạy 20 câu dev của đúng chiều. Không dùng holdout để chỉnh code. Benchmark lặp mẫu phải báo số WAV độc lập; không gọi 100 lượt lặp là 100 câu độc lập.

Thử hardware có người đọc sẵn sàng:

```powershell
.venv/Scripts/python.exe -X utf8 tools/microphone_laptop_check.py --seconds 15 --out results/laptop_current/mic-new.wav
```

Tool chờ Enter sau khi load model; đọc `Xin chào` ngay sau RECORDING, rồi nghe `Hello`. Audio nguồn giữ trong RAM. Đây là capture theo thời lượng, chưa có nút PTT; hoàn tất playback API không thay thế xác nhận nghe từ người dùng. Người dùng đã xác nhận một lượt VI→EN thật trong phiên tích hợp này.

## Việc tiếp theo có thể chia song song

Người phụ trách NMT/safety: review 80 câu trong `LAPTOP_DEV_REVIEW_PACKET_2026-10-09.jsonl`; phân biệt lỗi model với lexicon/grammar thiếu coverage. Điền `reviewer`, `clinical_correct` (boolean), `reason` từ người biết ngôn ngữ, không dùng chính checker để gán gold. Đây là dev/draft; giữ holdout nguyên trạng. Khi xuất labels cho `evaluate_text_nmt.py`, chỉ xuất ID đã review của đúng chiều. Thuốc/dose sai hiện bị chặn nhưng cần model/glossary đủ tốt để giảm confirmation. Review lexicon Hàn, reference Hàn và license ứng viên EN→KO; không tự đổi `korean_reviewed` để làm demo phát tiếng.

Người phụ trách runtime: thu hoặc nhận corpus ASR/WAV độc lập tối thiểu 100 câu mỗi chiều, benchmark sau khi model/checker được chốt; tiếp tục đo RAM dài hạn với câu đa dạng, tiếng ồn, silence, timeout, đổi direction. ASR confidence chưa được hiệu chuẩn, không suy ra confidence cao là câu đúng. Kiểm thử hardware EN→VI với người đọc phù hợp; tiếng Hàn cần reviewer trước.

Hai bên nên tạo branch mới từ branch tích hợp này. Text owner sửa adapter/lexicon/tools text và tests tương ứng; runtime owner sửa pipeline/audio/CLI/benchmark. Nếu cần đổi contract/config/factory, trao đổi trước để tránh sửa cùng file.

Nguồn model: [Helsinki EN→VI](https://huggingface.co/Helsinki-NLP/opus-mt-en-vi), [Helsinki KO→EN](https://huggingface.co/Helsinki-NLP/opus-mt-ko-en), [ứng viên Helsinki EN→KO cách ly](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-ko), [Argos index đã pin](https://github.com/argosopentech/argospm-index/blob/ff90de60728f7c1338ff6b75974e4c89b2442d22/index.json), [CTranslate2 EOS và input-length API](https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html).
