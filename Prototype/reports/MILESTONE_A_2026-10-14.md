# Milestone A — PoC tái lập & ổn định runtime

Đã triển khai P0-1, P0-2, P0-5 và P0-7 trong audit ngày 14/10/2026. Phạm vi là packaging/preflight, explicit modes, audio validation, bounded failures và effective configuration; không tuyên bố đã sửa safety theo quan hệ hoặc hoàn thiện urgency model.

## Thay đổi

| Task | Implementation |
|---|---|
| P0-1 | Tách runtime/dev/bench requirements, pin direct dependencies; `tools/check_runtime.py` kiểm version/import, config và size/SHA-256 tất cả assets của mode |
| P0-2 | `--mode stub|branch-a-bench|full` bắt buộc; bỏ `--real/--full` và config fallback; startup/turn log mode và capabilities |
| P0-5 | Normalize PCM uint8/int16/int32/float, reject stereo/NaN/Inf/rate sai/quá dài; silence/empty/too-short VAD short-circuit; empty ASR và truncated NMT không đến TTS; bounded branch/stage calls, discard partial audio; process watchdog cho CLI native hangs/crashes |
| P0-7 | `pipeline.json` là nguồn settings duy nhất; factory dùng snapshot cho model paths, gate/Branch B/VAD/decode/bonus/threads; fingerprint gồm resource pins, actual lexicon bytes/version và mode |

Các file mới chính: `src/tonebridge/audio.py`, `assets.py`, `preflight.py`, `runtime.py`, `tools/check_runtime.py`, các requirements và `Prototype/README.md`.

Không thêm production mock/stub. Mode stub và benchmark dùng các skeleton stage đã tồn tại; full dùng safety checker/gate thực và ghi rõ urgency UNKNOWN vì chưa train MLP.

## Nghiệm thu

Môi trường kiểm chứng: Windows x64, Python 3.13.9. Đã tạo virtualenv sạch tại `.venv/clean` **không dùng system-site-packages**, cài từ `requirements-dev.txt` và chạy:

```powershell
.venv/clean/Scripts/python.exe -m pip check
.venv/clean/Scripts/python.exe tools/check_runtime.py --mode stub --profile dev
.venv/clean/Scripts/python.exe -m pytest tests/ -q -rs -p no:cacheprovider
```

- `pip check`: **No broken requirements found**.
- Preflight dev/stub: **PASS**, imports và direct version pins khớp.
- Test suite: **154 passed, 7 skipped**; không có failure/collection error.
- Runtime/dev và benchmark requirements đã resolve thành công từ môi trường không có packages.
- Đã cài thêm profile benchmark vào virtualenv sạch: preflight `--mode stub --profile bench` PASS với 20 distribution/import checks, bao gồm `optimum.onnxruntime` và Transformers; `pip check` vẫn PASS.
- Full preflight chọn **370** assets, benchmark chọn **368**; GTCRN thêm một asset khi ON/OA. File sai size/hash, duplicate/escaping manifest, unpinned model path đều bị phát hiện.
- CLI thiếu mode/config sai/full thiếu assets: fail-fast, không fallback và không viết WAV.
- Test xác nhận PCM uint8 `[0,255]` và int32 extrema được normalize đúng; input quá ngắn, silence hoặc quá dài không gọi ASR/TTS sai.
- Test fault injection kiểm từng stage, partial TTS failure, branch/TTS timeout và native-style process hang/crash. Process được terminate/reap; turn lỗi không có `speak_text` hoặc output audio.
- Test config kiểm thresholds thay đổi hành vi, lexicon/asset/decode/path/mode thay đổi fingerprint; child config bị làm hỏng không ảnh hưởng runtime.

## 7 integration test skip

1. Branch B UNKNOWN/provenance cần `models/branch_b/swiftf0_model.onnx` đã pin.
2. Branch B golden vectors cần FLEURS data.
3. GTCRN alignment cần model/test WAV.
4. SNR trên real clips cần data.
5. Full pipeline offline cần full packs/test WAVs.
6. Native import/SwiftF0 copy identity cần full packs.
7. Real Branch A WAV→WAV cần real packs/test WAV.

Những prerequisite này đã thiếu trước triển khai. Skip là explicit cho integration có asset/data, không dùng fake result thay inference thật. Checkout vẫn chưa có full model pack, nên **chưa chạy lại full ASR/NMT/TTS inference**. Full mode hiện trả asset-preflight failure đúng thiết kế cho đến khi pack được provision và khớp manifest.

## Timeout và giới hạn

Direct `Pipeline.run` dùng worker thread daemon để trả lỗi trong budget và retire instance nếu timeout. Không tái dùng instance/native sessions đó. Python không thể kill thread native; CLI thật dùng spawned process có load/turn deadline và cleanup terminate/kill/join để khắc phục giới hạn này. Model startup nằm ngoài endpoint latency; đây chưa phải persistent serving benchmark.

Các vấn đề safety theo quan hệ, calibrated noisy-ASR rejection, privacy metadata-only, trained urgency và Android MVP vẫn thuộc các milestone sau. Hướng dẫn chạy hiện hành nằm tại `Prototype/README.md`; README ở repository root vẫn chứa định hướng sản phẩm cũ.
