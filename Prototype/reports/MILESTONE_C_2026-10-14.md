# Milestone C — Alpha trên laptop: urgency, first-clause TTS & E2E

Ngày nghiệm thu: **14/10/2026**. Cơ sở: audit Tech Lead, Milestone A và Milestone B cùng ngày.

**Kết luận cập nhật sau provisioning:** đã triển khai urgency artifact loader, cue disable minh bạch, first-clause synthesis/delivery, benchmark E2E cold/warm với RSS nhị phân, fail-fast batch/checkpoint và provisioner xác thực asset. Suite cuối đạt **336 passed, 6 skipped**, giữ toàn bộ 309 tests trước provisioning. Đã có **369/370 assets full**, **367/368 assets branch-a-bench** khớp size/SHA-256. **Chưa nghiệm thu E2E inference/TTS thật**: còn merged decoder NMT không tái tạo được đúng hash gốc; các candidate sai hash đã bị xóa. Preflight cả hai mode exit 1 với đúng một missing asset. Benchmark thật yêu cầu 100 turns nhưng thực đo 0, không có latency/RSS để công bố. Chi tiết cập nhật ở Mục 8; số stub và checkpoint ban đầu trong Mục 4/6 được giữ như lịch sử.

## 1. Diff logic so với Milestone B

| Phần | File tạo/chỉnh sửa | Hành vi |
|---|---|---|
| Urgency loader | **Mới** `src/tonebridge/urgency_model.py` | Load ONNX CPU + standardizer JSON; validate schema/order, class order, model hash, dtype/shape, số hữu hạn và scale dương; normalize 21 features, stable softmax → P(HIGH) |
| Branch B | `src/tonebridge/branch_b.py` | Load classifier khi đủ paths; không còn callable raw-output chưa ràng buộc; giữ acoustic evidence khi classifier chưa cấu hình; runtime không tự build band graph bị thiếu |
| Effective config | `config.py`, `configs/pipeline.json`, `configs/branch_b.json`, `assets.py`, `stages/factory.py` | Runtime chỉ đọc pipeline.json. Null artifacts → UNCONFIGURED; cấu hình artifact đầy đủ phải có manifest pins. Paths, tuning, schema/feature order và asset pins vào fingerprint |
| Contracts/gate | `contracts.py`, `gate.py`, `stages/stubs.py`, `telemetry.py` | Classifier status/cue flag rõ ràng; unavailable/untuned classifier không SPEAK_CUE. Stub/bench urgency DISABLED. Metadata-only log có enum status/cue state; log metadata an toàn của B vẫn append được |
| Training/export | `urgency_train.py` | ONNX export kèm feature/class/schema metadata; export standardizer kèm model SHA-256. Không train hoặc xuất weights giả thành model thực tế |
| First-clause TTS | `stages/tts_piper.py`, `pipeline.py` | Piper yield first-clause chunks trước khi synth phần còn lại; `decode.tts_first_clause` bật/tắt baseline. Pipeline callback nhận audio ngay sau gate approval; buffered default vẫn discard partial audio khi lỗi |
| Playback | **Mới** `playback.py`, `requirements-playback.txt`; `cli.py`, `runtime.py`, `tools/check_runtime.py` | `--playback` opt-in, lazy mono sounddevice stream trong worker có watchdog; abort queue khi lỗi. Profile playback kiểm imports/pins. Không mặc định bật loa khi benchmark |
| E2E benchmark | **Mới** `src/tonebridge/benchmark.py`, `tools/bench_e2e.py` | ≥50 turns, default 100; persistent sessions, cold/warm riêng, wall-clock distributions; actual process-tree RSS; metadata-only rows; thiếu assets → BLOCKED_PREFLIGHT, không fallback |
| Batch/checkpoint integrity | `tools/final_test.sh`, `run_adr001_v2.py`; **Mới** `benchmark_integrity.py`, `check_final_outputs.py` | Kiểm từng PID/exit code; fail-fast/kill/reap sibling; M3 phụ thuộc M2; marker sau cả 5 job thành công và outputs hợp lệ. Resume phải khớp fingerprint, coverage/counts/rates; atomic outputs |
| Metric correction | `src/tonebridge/evalkit/metrics.py` | Empty hypothesis là pure deletions với jiwer 4; bỏ token giả `<empty>` |
| Tests/docs | **Mới** `test_urgency_model.py`, `test_tts_streaming.py`, `test_bench_e2e.py`, `test_benchmark_integrity.py`; cập nhật test Branch B/privacy, README/design | Kiểm classifier schema/softmax, gate cue suppression, streaming order, missing assets, watchdog, PID failures và marker integrity |

## 2. Urgency: artifact contract và trạng thái

Schema: `tonebridge-urgency-v1`; input float `z[B,21]`, output float `logits[B,2]`, classes **LOW, HIGH**. Standardizer JSON chứa:

- Exact ordered `FEATURE_NAMES` (21), `class_names`, `schema_version`.
- `mean[21]` hữu hạn, `scale[21]` hữu hạn và >0; arrays được giữ read-only.
- `model_sha256` khớp ONNX; graph metadata phải đồng ý với scaler.

Loader lấy vector theo schema, **không theo insertion order của dict**, tính `z=(x-mean)/scale`, chạy graph rồi dùng `exp(logits-max)/sum(exp(...))`. Không lấy logits raw làm confidence. NaN/Inf, shape/class/schema/hash sai và scale không hợp lệ bị từ chối.

| Cấu hình / evidence | Label/status | Cue |
|---|---|---|
| `mlp_path=null` hoặc `standardizer=null` | UNCONFIGURED, reasons chỉ rõ trường thiếu | Tắt; acoustic evidence vẫn được tính |
| Paths đủ nhưng artifact thiếu/hỏng/unpinned | Fail-fast preflight/loader | Không giả định READY, không fallback |
| Verified classifier, evidence không đủ | UNKNOWN + READY classifier status, reasons acoustic cụ thể | Tắt |
| Verified classifier, evidence đủ, `tuned=false` | LOW/HIGH, P(HIGH), READY; reason `urgency_threshold_untuned` | Tắt |
| Verified classifier + `tuned=true` | LOW/HIGH, READY | Chỉ khi qua thêm score/voicing và mọi safety/quality check |
| Stub/branch-a-bench | DISABLED | Tắt |

`configs/branch_b.json` là tài liệu contract, không được đọc như cấu hình độc lập. Đặt paths trong `pipeline.json.branch_b` và pin thêm hai files trong `runtime_files.json`; full sẽ chọn 372 assets nếu có cả classifier/scaler ngoài 370 assets hiện tại.

Hiện cả paths là null, `tuned=false`. Chưa có recordings/weights thật và **không có F1/Recall urgency mới**. Các graph nhỏ trong pytest là fixture kiểm contract, không phải trained artifacts hoặc evidence về accuracy. Chính sách ASR quality uncalibrated từ B được giữ nguyên; cấu hình default full không tự động SPEAK.

## 3. First-clause readiness, delivery và playback

Piper helper first-clause đã được nối vào `stream()`. Lấy và yield audio mệnh đề đầu trước khi gọi synth phần còn lại; zero-length first clause bị lỗi, không được âm thầm bỏ câu đầu và chỉ đọc phần sau. Từng synthesis call vẫn buffered ở sherpa; đây là **clause streaming**, không phải token streaming.

Toàn bản dịch phải qua semantic/quality gate **trước** TTS. `Pipeline.run(..., on_audio=callback)` cho consumer nhận chunk sớm, thay vì đợi toàn turn hoàn tất. Test với remainder đang bị chặn xác nhận callback đã nhận chunk đầu khi turn chưa xong. Ca gate CONFIRM không gọi TTS/callback.

CLI mặc định ghi WAV sau khi hoàn tất và discard partial audio khi turn lỗi. `--playback` chủ động mở stream PortAudio trong process riêng; lỗi abort audio còn queued, timeout có thể terminate worker. Audio đã phát sau approval không thể thu hồi nếu TTS/device lỗi ở phần còn lại; chế độ live vì vậy được opt-in rõ ràng.

Các đại lượng tách riêng:

- `endpoint_to_first_audio_ms`: buffer TTS đầu tiên sẵn sàng, không gồm model startup.
- `endpoint_to_first_delivery_ms`: callback đã trả về; benchmark callback chỉ gửi thông báo IPC, không phát loa.
- `cold_first_buffer_received_ms`: parent nhận thông báo buffer đầu, tính từ trước spawn.
- Audible onset thực qua loa: **chưa đo**, để `null`. Cần device/loopback để nghiệm thu âm thanh nghe được; không dùng readiness thay phép đo này.

Đã cài/import-check sounddevice **0.5.3** và query PortAudio version thành công. Test sink kiểm open/write/drain/abort/close bằng fixture; không công bố physical playback performance từ fixture. API tham chiếu: [sounddevice streams](https://python-sounddevice.readthedocs.io/en/0.5.3/api/streams.html).

## 4. E2E measurement trên laptop

Artifact chính: [bench_e2e_stub_c_idle_2026-10-14.json](../results/bench_e2e_stub_c_idle_2026-10-14.json). Đo sau khi pytest kết thúc, không chạy benchmark đồng thời với suite; các artifact chạy sớm hơn được giữ nhưng không dùng cho bảng này.

Môi trường: **Windows AMD64, Python 3.13.9**, CPU string `Intel64 Family 6 Model 165 Stepping 3, GenuineIntel`, **12 logical CPUs**, RAM vật lý **23.899 GiB**. Config hash stub: `ef739998783c9ab4`.

Workload: 4 waveform tone tổng hợp mono 16 kHz, dài 0.5/1/2/3 s, lặp qua **100 turns** bằng các skeleton đã có. ASR fixed/sidecar, tagged NMT, always-pass safety, urgency DISABLED, v0 gate, sine-tone TTS. Tất cả stage objects cùng nằm trong pipeline worker. **Đây là E2E control-flow/overhead thực của stub, không phải latency/quality của ASR/NMT/Piper/Branch B models.** Không có audio device trong benchmark.

| Đại lượng | n | min | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|
| Warm total processing (ms) | 99 | 9.1595 | **9.7423** | **10.4581** | 10.9182 |
| Warm first-buffer readiness (ms) | 99 | 5.2366 | **5.6040** | **6.0226** | 6.6523 |
| Warm callback completion (ms) | 99 | 7.1796 | 7.6653 | 8.1935 | 8.7008 |
| Warm worker RSS (MiB) | 99 | 61.1367 | 61.8672 | 61.8711 | 61.8711 |

Cold được báo riêng, **n=1**, không suy ra p95 của cold cohort:

| Cold metric | Đo được |
|---|---:|
| Stage factory verification/load | 14.4731 ms (stub, không NN model load) |
| Spawn/import/factory ready | 574.4281 ms |
| Cold first-turn processing sau load | 10.5834 ms |
| Cold E2E từ trước spawn đến nhận kết quả turn | **587.4560 ms** |
| Cold từ trước spawn đến parent nhận thông báo buffer đầu | 582.9095 ms |

**100 processed / 100 successful / 100 audio turns; gate SPEAK=100.** Warm reuse cùng stage/session instances, không spawn/load lại mỗi turn. Report còn lưu distribution của toàn 100 processing samples; cold startup chỉ nằm trong cold metric riêng, không cộng vào warm.

### RSS và đơn vị

- Actual sampled parent + descendants resident sum peak: **130,748,416 bytes = 124.6914 MiB = 0.121769 GiB**.
- Worker OS lifetime peak từ turn records: **61.9180 MiB**; không cộng các lifetime peaks khác thời điểm thành tổng giả.
- Tree sampler: requested sleep interval 10 ms, **72 snapshots**; query/scheduler thêm overhead. Có thể bỏ lỡ peak ngắn; shared resident pages có thể tính ở nhiều process. Không gọi RSS sum là unique physical allocation.
- `MiB=bytes/2^20`, `GiB=bytes/2^30`. Field cũ `peak_rss_mb` giữ như API alias **MiB**, kèm `rss_unit=MiB` và `peak_rss_mib`; report mới dùng tên `_mib`/`_gib`.

Không kết luận RSS model thật đạt budget từ số stub này. Pipeline full gồm SwiftF0/band/safety/gate được tool hỗ trợ, nhưng chưa chạy được do assets vắng mặt. Không lấy các benchmark isolated-stage cũ cộng vào để giả lập full RSS/latency.

### Branch A thật và full

Checkpoint **trước provisioning**; trạng thái hiện tại được cập nhật ở Mục 8.

Artifact chặn đúng: [bench_e2e_branch_a_c_2026-10-14.json](../results/bench_e2e_branch_a_c_2026-10-14.json).

| Mode | Requested turns | Measured turns | Trạng thái | Latency/RSS |
|---|---:|---:|---|---|
| stub | 100 | 100 | COMPLETE | Số đo ở trên |
| branch-a-bench | 100 | **0** | BLOCKED_PREFLIGHT, **368/368** active assets missing, exit 1 | `null` |
| full preflight | — | **0** | **370/370** active assets missing, exit 1 | Chưa đo |

Người dùng xác nhận đã quét `D:\bachkhoa`: không có Zipformer/Opus-MT/Piper pack; chỉ có SwiftF0 bundled graph trong venv. Không dùng bundled graph đó để bypass inventory hoặc biến missing packs thành một benchmark giả.

## 5. Benchmark integrity

`final_test.sh` dùng Bash 5.1+ `wait -n -p`, lưu từng task/PID/exit code. Ba M1 và M2 chạy song song; M3 chỉ bắt đầu sau M2 exit 0. Job thất bại khiến batch dừng, kill/reap sibling; lock bảo vệ chạy trùng. Chỉ khi **5/5 exit 0** và validator xác nhận output coverage/provenance/counts mới atomically tạo `FINAL_TEST_DONE` kèm hashes. Marker cũ hoặc NMT/safety output cũ bị từ chối để review/archive, không bị ghi đè âm thầm.

Fault tests chạy **Bash thật 5.2.37 / OS subprocesses thật trong temp root**:

- Lỗi exit 7 ở từng `m1_demand`, `m1_babble`, `m1_alarm`, `m2`, `m3` → không marker.
- M2 fail → M3 không launch.
- Sibling 5 s bị thu hồi khi job khác fail; batch fail-fast, không còn PID fixture sống.
- Cả 5 job exit 0 nhưng output hỏng → validator fail, không marker.
- Cả 5 job và outputs hợp lệ → marker; chạy lại bị từ chối.

Không chạy lại frozen held-out experiment thật trong milestone này. Không chạm/xóa historical final marker/results để tạo một run giả.

ADR checkpoint `adr001-v3` fingerprint bao gồm effective config + actual config file hash, asset pins, source code/runtime versions, split/inputs/reference/noise hashes, seed, decode/chunk/grid settings. Legacy/mismatch, truncated samples, negative counts hoặc rates không khớp counts bị từ chối. Output atomically cập nhật sau cell; `complete=true` chỉ sau đủ requested grid. OFF không khởi tạo GTCRN/classical denoise không sử dụng. Full dataset run vẫn cần models/noise/corpus, chưa được tái đo ở máy này.

## 6. Nghiệm thu tests/dependencies

Virtualenv sạch của A/B: `.venv/clean`, không dùng system-site-packages.

```powershell
.venv/clean/Scripts/python.exe -m pytest tests/ -q -rs -p no:cacheprovider
.venv/clean/Scripts/python.exe -m pip check
.venv/clean/Scripts/python.exe tools/check_runtime.py --mode stub --profile dev
.venv/clean/Scripts/python.exe tools/check_runtime.py --mode stub --profile playback
.venv/clean/Scripts/python.exe tools/check_runtime.py --mode full
```

- **309 passed, 7 skipped**, 62.70 s; giữ toàn bộ 251 test B, thêm **58** cases. Assertions Branch B thiếu model được cập nhật theo contract UNCONFIGURED/DISABLED mới; không xóa test.
- `pip check`: **No broken requirements found**.
- Dev/stub preflight: **PASS**, 15 pinned distributions/imports.
- Playback/stub preflight: **PASS**, 9 pinned distributions/imports, gồm sounddevice 0.5.3.
- Full preflight: **FAIL đúng**, 370 missing assets, không dependency/version failure.
- `git diff --check`, Python compile checks và Bash syntax: PASS; chỉ có cảnh báo Git LF/CRLF.
- 7 integration skips giữ nguyên scope/lý do A/B: pinned SwiftF0; golden/FLEURS; frontend assets; noise data; hai offline model tests; real Branch A models. Không đổi missing integration thành mocked PASS.
- Privacy B được giữ: default metadata-only, eval-mode opt-in, native worker output suppression. Log metadata hợp lệ của B vẫn append; old text-bearing/unsafe log vẫn bị từ chối.

## 7. Checklist sign-off và runbook còn lại

- [x] Artifact loader/scaler/schema/softmax và explicit cue disable.
- [x] First-clause synth/delivery sau gate approval, buffered failure policy và playback opt-in.
- [x] ≥100-turn stub cold/warm E2E + actual process-tree RSS, đơn vị MiB/GiB.
- [x] PID/exit-code supervision, atomic completion và fingerprinted resume/fault tests.
- [x] Giữ 251 test B; full suite/dependency checks.
- [x] Provisioner nguồn chính thức, size/SHA-256 bắt buộc, cleanup/fail-fast; 369/370 full assets hợp lệ (Mục 8).
- [x] Chuẩn bị ba speech WAV thật mono 16 kHz, ≤15 s, kèm source/hash provenance.
- [ ] Cả hai preflight exit 0: còn một merged decoder NMT chưa khớp pin gốc.
- [ ] ≥100-turn **branch-a-bench model thật**, full resident-stage RSS và so sánh first-clause vs whole-utterance trên cùng inputs.
- [ ] Đo audible onset qua output device/loopback.
- [ ] Real recordings/urgency weights, speaker-disjoint evaluation và calibration quality/cue trước khi bật automatic clinical output.

Sau khi provision đúng packs và speech WAV 16-kHz mono (≤15 s), dùng fresh output paths:

```powershell
.venv/clean/Scripts/python.exe tools/check_runtime.py --mode branch-a-bench
.venv/clean/Scripts/python.exe tools/bench_e2e.py --mode branch-a-bench --wav speech1.wav speech2.wav --turns 100 --out results/bench_real_fresh.json
.venv/clean/Scripts/python.exe tools/check_runtime.py --mode full
.venv/clean/Scripts/python.exe tools/bench_e2e.py --mode full --wav speech1.wav speech2.wav --turns 100 --out results/bench_full_fresh.json
```

Full default có thể CONFIRM/REPEAT tất cả turns do quality chưa calibration; tool báo TTS coverage và first-audio sample count thực (0 nếu bị chặn). Không bật calibration/tuning để ép benchmark có SPEAK. Milestone C hoàn tất implementation/control-flow regression; **sign-off Alpha inference thực còn bị chặn bởi missing artifacts/data**.

## 8. Cập nhật provision và nghiệm thu Alpha — 14/10/2026

### Diff bổ sung

| File | Hành vi / bằng chứng |
|---|---|
| **Mới** `tools/provision_models.py` | Chọn inventory theo chính `pipeline.json`/`selected_assets`; mode bench 368, full 370. HTTPS stream có size/time limit; staging dưới project, lock độc quyền; tar whitelist, chặn traversal/link/duplicate/special files; SwiftF0 từ wheel đã cài hoặc PyPI wheel được kiểm digest. Kiểm cả size và SHA-256 trước install; không sửa manifest hoặc override hash. File hỏng bị xóa và exit 1, staging/source weights/archive được dọn. File tốt được reuse, installation lỗi rollback các file mới của nhóm |
| NMT provisioning | FP32 export từ Helsinki revision `c8d2853e77f5fae31124d993e0b35176b1c8914e`, không `trust_remote_code`; checkpoint PyTorch cũng được pin/hash-check. Optimum dynamic arm64, `is_static=False`, `per_channel=False`, hai graph encoder/merged decoder. Tokenizer saved qua Transformers. Verified tokenizer/config/encoder được giữ độc lập; decoder thiếu vẫn chặn cả factory và preflight |
| **Mới** `tests/test_provision_models.py` | 26 cases: mode counts, size/hash mismatch + deletion, atomic group validation/rollback/idempotence, archive traversal/links/duplicates/whitelist, HTTP errors/oversize/downgrade, lock cleanup, offline import, wheel fallback, giữ chỉ file NMT đã verified, route ASR decoder từ zzasdf và fail-fast khi explicit quantizer path sai |
| `tests/test_cli_modes.py` | Missing-model regression dùng config copy trong temp checkout, tiếp tục assert fail-closed ngay cả khi project thực có `models/`; không skip assertion chỉ vì folder tồn tại |
| **Mới** `docs/MODEL_PROVISIONING.md`; README/OFFLINE_FILES | Catalog source/revision/license notes, trust contract, original-pack import và runbook; báo đúng blocker hiện tại |
| **Mới** `configs/pipeline_bench_whole.json` | Snapshot config để so sánh, chỉ đổi `decode.tts_first_clause=false`; không thay `pipeline.json` runtime, fingerprint khác được ghi ở benchmark |
| `results/inputs/*` | Ba public speech WAV gốc từ ASR repo, kèm `provenance.json`: immutable URL, file SHA-256 và kiểm mono/16 kHz/duration |

Runtime vẫn không download/build model tự động; provisioning là bước online riêng. Không thêm runtime mock/stub hoặc trained weights giả; byte payloads trong fault tests chỉ kiểm installer contract.

### Inventory xác thực

| Nhóm | Số file khớp pin | Nguồn / kết quả |
|---|---:|---|
| Silero | 1/1 | sherpa `asr-models/silero_vad.onnx` |
| Zipformer VI | 4/4 | sherpa release encoder/joiner/tokens; decoder INT8 từ zzasdf revision `e827965a37aab92a4455566fac49c0e80a23afef` khớp `0cf67da0…`. Download trực tiếp nguồn này đã được kiểm lại thành công |
| Piper + espeak-ng-data | 357/357 | sherpa `tts-models/vits-piper-en_US-ljspeech-medium.tar.bz2`; voice + tokens + 355 phonemizer files |
| Marian VI→EN | **5/6** | Encoder, source.spm, target.spm, vocab.json và config.json đều khớp. Merged decoder bị từ chối |
| SwiftF0 | 1/1 | Byte-identical `swift-f0==0.3.0` wheel model |
| band_feats | 1/1 | Local ONNX graph build khớp hash đã pin |
| Tổng full | **369/370** | GTCRN OFF và classifier/scaler null, không nằm trong inventory default |
| Tổng branch-a-bench | **367/368** | Không chọn hai Branch B graphs |

Manifest giữ nguyên byte trong bước provisioning: SHA-256
`075b9e87d0a5c1e74294442ca6ba94583573c9ca851518ad9645775e126f91ee`.
Không cộng test WAVs vào số model assets. Theo [MODEL_PROVISIONING.md](../docs/MODEL_PROVISIONING.md), nguồn model card ASR/NMT là Apache-2.0, Silero/SwiftF0 MIT, dữ liệu LJ Speech public domain; bundled espeak-ng GPL-3.0 và review linking/distribution đã nêu ở tài liệu trước vẫn giữ nguyên.

### Blocker merged decoder: provenance đúng chưa đủ bảo đảm byte-identical

File cần phục hồi:
`models/nmt/vi-en-int8-arm64/decoder_model_merged_quantized.onnx`.

| Thuộc tính | Pin gốc | Candidate bị xóa |
|---|---|---|
| Bytes | 81,649,685 | 81,649,685 |
| SHA-256 | `ea35145e4b0b6757a9b1bde31c0363423115418fad23c0379cd1d3a9a2d5518d` | `14afbaeb4e4771d7866fd7286bdc759314fa045858027ee13a8b0416f65d3400` |

Đã chạy đúng provenance người dùng xác nhận và lệnh `tools/provision_models.py --mode branch-a-bench`; exit **1**, vì decoder không khớp SHA. Đã thử ORT **1.22.1** trong runtime và **1.23.2** trong thư mục công cụ riêng (không đổi dependency runtime). Cùng export/hash seed **0** cho cùng digest candidate trên; retry seed 0 tái lập được candidate nhưng **không phải artifact gốc**. Candidate từ lượt đầu chưa cố định seed có digest khác (`d2a056c7…`) và cũng bị xóa.

Mã Optimum đã cài tại `onnx/transformations_utils.py::_create_name_sharing_dict()` chọn tên shared initializer bằng `set.pop()`. Việc này có phụ thuộc Python hash randomization. Báo cáo cũ dùng Python 3.10, chưa ghi hash seed/export environment đầy đủ. Đây là rủi ro tái lập đã xác định từ source, **chưa đủ bằng chứng quy toàn bộ mismatch cho riêng hash seed**. Helsinki repo cung cấp weights gốc, không cung cấp chính binary merged decoder đã pin. Không kết luận candidate tương đương về chất lượng và không sửa manifest để biến lỗi thành PASS.

Artifacts:

- [provision_full_final_2026-10-14.json](../results/provision_full_final_2026-10-14.json): full 369 verified, fail đúng decoder.
- [provision_branch_a_final_2026-10-14.json](../results/provision_branch_a_final_2026-10-14.json): đúng lệnh mode branch-a-bench, 367 verified, exit 1.
- [provision_full_ort1232_seed0_2026-10-14.json](../results/provision_full_ort1232_seed0_2026-10-14.json): experiment isolated ORT 1.23.2 trước khi giữ subgroup NMT đã verified.

Các group totals trong report chỉ liệt kê nhóm hoàn tất; `assets_verified` phản ánh inventory thực sau khi nhóm NMT dừng giữa chừng. Mismatch file và mọi staging đã được dọn; directory `models/` chỉ chứa pinned assets đã pass.

### Preflight và benchmark đã thực thi

| Lệnh / artifact | Exit | Kết quả thực |
|---|---:|---|
| `check_runtime.py --mode branch-a-bench` ([JSON](../results/preflight_branch-a-bench_provision_2026-10-14.json)) | **1** | 368 checked, **1 missing decoder**, không dependency/version failure |
| `check_runtime.py --mode full` ([JSON](../results/preflight_full_provision_2026-10-14.json)) | **1** | 370 checked, **1 missing decoder**, không dependency/version failure |
| `bench_e2e.py --mode branch-a-bench --turns 100 --out results/bench_real_branch_a.json` ([JSON](../results/bench_real_branch_a.json)) | **1** | `BLOCKED_PREFLIGHT`, 100 requested / **0 processed**, first-clause=true, config `0d1d9378479f1de8` |
| Whole-utterance cùng WAVs / 100 requested ([JSON](../results/bench_real_branch_a_whole.json)) | **1** | `BLOCKED_PREFLIGHT`, **0 processed**, first-clause=false, config `7ef849bcf534ee06` |

Speech inputs: original `test_wavs/0.wav`, `1.wav`, `2.wav` từ HF ASR revision
`b2745a435379992ad3f299635468db0c34918e1e`; WAV files được hash-check trước `load_wav()`. Mono 16 kHz, **3.74 / 2.40 / 1.55 s**, không resample hoặc thay bằng tone. Public samples được lưu ở `results/inputs/zipformer_vi_{0,1,2}.wav` với [provenance](../results/inputs/provenance.json).

| Số đo real E2E yêu cầu | First-clause | Whole-utterance |
|---|---|---|
| Cold model load / cold first turn | **Chưa đo** | **Chưa đo** |
| Warm first-buffer p50/p95 | **Chưa đo**, n=0 | **Chưa đo**, n=0 |
| Warm completion p50/p95/min/max | **Chưa đo**, n=0 | **Chưa đo**, n=0 |
| Process-tree peak RSS (MiB/GiB) | **Chưa đo** | **Chưa đo** |
| Speedup / audible onset | **Không có bằng chứng** | **Không có bằng chứng** |

Không dùng startup time của provisioner, bộ nhớ exporter hoặc số stub ở Mục 4 làm số inference. Không lấy isolated-stage measurements cộng thành E2E. 100 turns lặp public samples sau khi unblock chỉ đo latency/coverage, không phải 100 ca y khoa độc lập. Cần báo cả số input thực được split nhiều clauses để so sánh TTS có ý nghĩa.

### Tests, integrity và checklist hiện tại

Lệnh cuối trong virtualenv `.venv/clean`:

```powershell
.venv/clean/Scripts/python.exe -m pytest tests/ -q -rs -p no:cacheprovider --basetemp .venv/test-provision-signoff-final
.venv/clean/Scripts/python.exe -m pip check
```

- **336 passed, 6 skipped**, **64.13 s**. Giữ toàn bộ 309 cases C; thêm **26** provisioner cases, SwiftF0 integration trước đây skip nay PASS. Không xóa hoặc weaken assertions. CLI missing-assets test tiếp tục chạy bằng temp config.
- `pip check`: **No broken requirements found**; runtime ORT vẫn **1.22.1**. Experiment ORT 1.23.2 chỉ ở `.venv/provision-ort-1.23.2`, không trên runtime import path.
- Sáu skips: golden/FLEURS, GTCRN frontend, noise data, hai offline integrations, real Branch A. Sample WAVs benchmark ở `results/inputs`, không giả tạo corpus/golden files để ép integration pass. Real model pack còn thiếu decoder, các integration chưa được nghiệm thu.
- Provisioner Python compile và `git diff --check`: PASS (cảnh báo LF/CRLF của file đã sửa từ trước được giữ).

Checklist bổ sung:

- [x] Script provisioning có source/revision chính thức, hai modes dùng cùng manifest/runtime config.
- [x] Các file đã cài đều khớp size và SHA-256; sai hash bị xóa và báo lỗi, không overwrite file tốt hoặc nới pin.
- [x] 369 full assets / 367 bench assets verified; SwiftF0 + band graph đúng pin.
- [x] Speech samples thật hợp lệ + baseline whole-utterance config + artifact diagnostics.
- [x] Giữ 309 tests trước provisioning; suite cuối 336 pass, 6 skip; pip check pass.
- [ ] Decoder NMT gốc đúng pin; **cả hai preflight exit 0 chưa đạt**.
- [ ] **100-turn real Alpha E2E chưa đạt**; cold/warm latency và RSS thực vẫn chưa đo.
- [ ] First-clause vs whole-utterance improvement và physical audible onset chưa nghiệm thu.

Cần đúng **một binary gốc** khớp hash `ea35145e…`, hoặc đầy đủ export environment/seed tạo ra binary đó. Importer hỗ trợ source directory chỉ chứa decoder thiếu tại đường dẫn models-relative, reuse mọi asset tốt hiện có. Runbook ở [MODEL_PROVISIONING.md](../docs/MODEL_PROVISIONING.md) chỉ rõ cách import, preflight và chạy lại bằng **fresh benchmark result paths**. Provisioning triển khai xong phần kiểm soát/download; **nghiệm thu Alpha inference vẫn BLOCKED_ARTIFACT_INTEGRITY**, không đánh dấu COMPLETE.

## 9. Đồng bộ GitHub main

Các kết quả 336/6 ở Mục 8 thuộc checkpoint trước đồng bộ. Đã giữ các thay đổi A–C cùng runtime bốn hướng của upstream; xem [MAIN_SYNC_2026-10-14.md](MAIN_SYNC_2026-10-14.md) cho giải quyết xung đột, nghiệm thu suite hợp nhất và trạng thái push. Blocker binary NMT historical vẫn giữ nguyên.
