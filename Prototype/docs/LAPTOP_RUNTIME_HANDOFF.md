# Laptop runtime và ranh giới làm việc — 09/10/2026

Scope hiện tại: laptop CPU, VI→EN / EN→VI / EN→KO / KO→EN. Mã ngôn ngữ Hàn là `ko`.

Phần audio/runtime đã có adapter ASR/TTS cho cả VI, EN, KO và routing cho bốn chiều. E2E bằng NMT/safety thật hiện chạy VI→EN; ba chiều mới chờ module text của người phụ trách NMT/safety. Chọn chiều chưa đăng ký sẽ báo lỗi và không phát audio. Không tự dùng checker VI→EN cho chiều khác.

## Chạy ngay trong checkout này

Mở PowerShell tại `OneVoice/Prototype`. `.venv` và model assets đã được dựng trong lần triển khai này.

```powershell
.\.venv\Scripts\python.exe tools/run_laptop.py --timeout 60 -- --config configs/laptop_vi-en.json --wav models/asr/zipformer-vi-int8/test_wavs/0.wav --out results/demo_vi-en.wav --assert-offline
```

`--out` phải là đường dẫn mới. File có sẵn bị từ chối để tránh nhầm audio của lượt cũ với lượt bị chặn. WAV được ghi hoàn chỉnh vào file tạm, rồi công bố bằng thao tác exclusive; PCM16 dùng được với playback Windows. Thêm `--play` nếu muốn phát ra loa. Hiện tổng hợp toàn câu rồi phát, chưa có incremental synthesis. `first_audio` là thời điểm có sample TTS, không phải thời điểm người nghe nghe thấy.

`--full` và `--real` đều trỏ đến runtime có safety/gate. Test double phải dùng `--stub`; output của chế độ này là tone. `real_branch_a()` chỉ còn phục vụ các tool nghiên cứu cũ.

Log mặc định không ghi transcript, translation hay `speak_text`. Dùng `--log-content` khi chủ động cần nội dung cho debug/evaluation. Log có `status`, `error_stage`, `error_type`, gate action, timing và trạng thái ghi output.

Thu microphone chỉ xảy ra khi gọi rõ `--record-seconds`; audio nguồn giữ trong RAM. Ví dụ:

```powershell
.\.venv\Scripts\python.exe tools/run_laptop.py --timeout 60 -- --config configs/laptop_vi-en.json --record-seconds 5 --out results/demo_mic.wav --play --assert-offline
```

Capture dùng sample rate mặc định của thiết bị rồi chuyển một lần về mono 16 kHz. Có thông báo RECORDING. Đây là capture theo thời lượng, chưa có nút PTT nhấn/thả. Hardware microphone và loa chưa được thử trực tiếp trong lần triển khai; đã test boundary bằng thiết bị giả.

Watchdog chạy inference trong process riêng: timeout giết process và các inference thread, trả mã 124. Chạy CLI trực tiếp hoặc gọi `Pipeline.run()` không có hard timeout; muốn giới hạn native hang phải dùng launcher. Không có tự fallback sang model khác chưa được kiểm chứng.

## Hợp đồng để bạn phụ trách NMT/safety bàn giao

Người phụ trách text tạo module mới, ví dụ `src/tonebridge/partner_text.py`, với hàm:

```python
def build_text_stages(cfg, threads=2):
    # Chọn model và checker đúng cfg.direction.
    # Load từ assets cục bộ; không tự download trong inference.
    return nmt, safety
```

Đăng ký qua config `text_factory`, hoặc không sửa config chung mà dùng CLI:

```powershell
.\.venv\Scripts\python.exe tools/run_laptop.py --timeout 60 -- --config configs/laptop_en-ko.json --text-factory tonebridge.partner_text:build_text_stages --wav input_en.wav --out results/demo_en-ko.wav --assert-offline
```

Interface đã chốt:

- `nmt.translate(text, src, tgt) -> MtResult`; `src_text` phải giữ nguyên text ASR và lang phải khớp chiều.
- `MtResult` có `terminated_by_eos`, `truncated`, `constraints_satisfied`. Ba chiều mới phải báo EOS evidence; thiếu bằng chứng bị chặn. Truncated, không EOS hoặc constraints không đạt đều không được phát.
- `safety.check(mt) -> SafetyReport`; `passed=False` hoặc `confirm=True` chặn TTS, kể cả nếu custom gate cố duyệt.
- Adapter/checker phải có logic tương ứng từng ngôn ngữ. Cấu trúc quan trọng không phân tích được phải yêu cầu xác nhận, không ngầm pass.
- `AsrResult.confidence` hiện vẫn là token-probability proxy chưa calibrated theo ngôn ngữ; không xem confidence cao là bảo đảm transcript đúng.
- Checker/NMT VI→EN cũ được giữ nguyên thuật toán. Nó vẫn là lexical checker; NMT cũ chưa báo EOS/truncation. Chỗ này thuộc phần người phụ trách text cần sửa, không được coi là đã giải quyết bằng runtime.

Bạn sửa `cli.py`, `config.py`, `contracts.py`, `pipeline.py`, `gate.py`, `stages/base.py`, `stages/factory.py`, `stages/audio_factory.py`, các ASR/TTS adapter, config audio và tool runtime. Bạn phụ trách text sửa NMT/safety/glossary/lexicon, module factory text riêng và test của chúng. Không cùng sửa `stages/factory.py`; module text được nối bằng `text_factory`.

## Model audio và thay đổi so với tài liệu cũ

| Thành phần | Asset hiện dùng | Ghi chú |
|---|---|---|
| ASR VI | Zipformer VI INT8 2025-04-20 | Decoder được quantize cục bộ; hash thực nằm trong manifest mới |
| ASR EN | Zipformer EN INT8 2023-04-01 | CPU; source revision được pin |
| ASR KO | Zipformer Korean INT8 2024-06-24 | Có thể trả Hangul không có dấu cách; gate dùng character ratio cho tiếng Hàn |
| TTS EN | Piper ljspeech-medium qua sherpa-onnx | Giữ voice baseline; espeak-ng license caveat cũ vẫn còn |
| TTS VI/KO | Supertonic 3 INT8 qua sherpa-onnx | CPU/local, language truyền qua GenerationConfig; candidate laptop, chưa đánh giá phát âm y khoa |

Nguồn adapter: [sherpa transducer](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html) và [Supertonic Python API](https://k2-fsa.github.io/sherpa/onnx/tts/all/Korean/supertonic-3-ko.html). Supertonic thay MeloTTS ở laptop reference; đây là deviation cần leader chốt khi freeze model. Model card upstream [Supertonic 3](https://huggingface.co/Supertone/supertonic-3) khai báo OpenRAIL; LICENSE trong archive là MIT cho code, không dùng nó để kết luận license weights. Piper VI vivos đã được probe nhưng không chọn vì MODEL_CARD ghi dataset CC BY-NC-SA 4.0 và có unknown-phoneme warnings. Archive thử đó không nằm trong manifest active restoration.

NMT VI→EN phục hồi bằng public ONNX export của cùng checkpoint family từ [Xenova/opus-mt-vi-en](https://huggingface.co/Xenova/opus-mt-vi-en), revision được pin trong tool provision. Nó không phải bản export cục bộ đã dùng trong audit cũ. Không áp các con số WER/safety/latency cũ cho bộ assets mới. `configs/runtime_files.json` và báo cáo cũ được giữ nguyên; `configs/laptop_assets_manifest.json` ghi SHA256/size của bộ phục hồi hiện tại. License inventory và các caveat của model/data cũ vẫn cần được xử lý trước release.

Preset laptop tắt urgency classifier và dùng UNKNOWN; cấu hình cũ vẫn có thể chạy Branch B feature extraction. Denoise giữ OFF. Input quá 15 giây mặc định yêu cầu nói lại thay vì cắt âm thầm. Có thể nâng giới hạn đến 60 giây trong config cho benchmark được ghi rõ.

## Kiểm tra và benchmark

Lượt kiểm tra cuối: **114 passed, 3 skipped**. Ba test bỏ qua vì thiếu FLEURS/golden data hoặc GTCRN (denoise mặc định OFF). `pip check` không phát hiện dependency lỗi; 401 file trong manifest đã khớp size/SHA256. Test mới bao phủ bốn chiều routing, lỗi từng stage/partial TTS, source mismatch, translation completion evidence, privacy logs, WAV encoding/resampling, microphone boundary giả, factory bàn giao và process watchdog. Microphone/loa thực và quality holdout vẫn chưa được xác minh.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools/check_laptop_assets.py
.\.venv\Scripts\python.exe tools/audio_smoke.py
.\.venv\Scripts\python.exe tools/bench_laptop_runtime.py --config configs/laptop_vi-en.json --manifest configs/laptop_smoke_vi-en.jsonl --out-dir results/my_stability_run --repeats 34
```

Audio smoke dùng model-provided samples cho ASR, tạo WAV mẫu cho TTS; chưa có human listening assessment hoặc corpus clinical. Các WAV mẫu ở `results/laptop_current/audio/`. Không đánh giá chất lượng dịch bằng việc pipeline không crash.

Manifest benchmark là JSONL gồm `id`, `wav` (tương đối với thư mục chứa manifest), `direction`. Mỗi process chạy một chiều; warm-up không tính vào số lượt đo. Tool báo số WAV độc lập, số lần lặp, gate coverage, lỗi, RAM và latency. Không trộn lượt không có audio vào first-sample latency, và không giả định first sample là first audible.

TurnRecord và benchmark mới có `runtime_manifest_sha256` để gắn với manifest đã dùng. Đây là hash của manifest; chạy `check_laptop_assets.py` để kiểm chứng content assets, vì runtime không rehash hàng trăm MB mỗi lượt. Nên kiểm tra lại trước mỗi benchmark/freeze.

Kết quả triển khai lưu tại `results/laptop_current/`: 102 lượt VI→EN trên **3 WAV lặp lại 34 lần**, 3 warm-up, 102 OK, 0 error; p50 first-sample 514 ms, p95 791 ms; peak RSS 895 MiB, RSS lượt đầu/cuối khoảng 871/895 MiB. Offline guard ghi 0 Python network attempts và 0 OS connections. Đây là runtime stability smoke, không phải holdout 100 utterances độc lập, không chứng minh semantic safety. RAM có tăng khoảng 24 MiB trong lượt đo; chưa có cơ sở kết luận leak hoặc chứng minh ổn định dài hạn. Các timing được đo trước lượt test bổ sung cuối, có thể biến động theo machine load.

## Dựng lại môi trường

Python 3.12; tạo `.venv` riêng. Runtime dependencies ở `requirements-laptop.txt`. `requirements-laptop.lock.txt` pin cả bộ test/dev đã dùng; cài lock cần thêm `--extra-index-url https://download.pytorch.org/whl/cpu` vì Torch CPU chỉ phục vụ test training cũ. Inference không import Torch/Transformers/HF libraries.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-laptop.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe tools/provision_laptop.py
```

Provision là bước setup có mạng, tách khỏi inference. Model files nằm trong `models/` và được git-ignore; cần chạy provision ở checkout mới. Giữ manifest cùng báo cáo để truy nguồn assets. Không tải lại dataset test cũ hoặc sửa checker chỉ để làm đẹp số liệu.
