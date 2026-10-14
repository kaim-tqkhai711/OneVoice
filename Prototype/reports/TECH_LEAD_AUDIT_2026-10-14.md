# Rà soát kỹ thuật ToneBridge / OneVoice — 14/10/2026

## 1. Kết luận và phạm vi

**Dự án hiện là research PoC trên laptop, xử lý từng WAV VI→EN. Có engine và thí nghiệm thực, nhưng chưa phải Android MVP, chưa có dịch hai chiều, chưa có prosody transfer và chưa đủ bằng chứng để phát tự động các bản dịch y khoa trong điều kiện nhiễu.**

Ước lượng theo mức độ đáp ứng chức năng, tích hợp và kiểm chứng:

- **Khoảng 40% (biên 35–45%) so với sản phẩm trong README:** còn thiếu hai chiều, prosody transfer, Zero-UI, NPU, dashboard và mesh/audit.
- **Khoảng 65% (biên 60–70%) so với PoC laptop VI→EN đã thu hẹp:** engine khá đầy đủ, còn thiếu khả năng tái lập từ checkout, safety theo quan hệ, quality gate hữu hiệu, urgency model và đánh giá độc lập.
- Phần Android/on-device mới ở mức chuẩn bị đo; chưa đủ để công nhận MVP của Technical Proposal v3.1.

Đây là ước lượng kỹ thuật, không phải phần trăm dòng code, test coverage hoặc xác suất an toàn. Các scope đã cắt không được tính là việc còn thiếu trong PoC laptop; chúng vẫn là khoảng trống nếu tiếp tục dùng README làm cam kết.

**Đối tượng audit:** repository Git nằm tại `D:/bachkhoa/OneVoice/OneVoice`; implementation tại `Prototype/`. Thư mục cha cũng có `.git`, đang có các file cũ bị xóa và repository con chưa được theo dõi. Trạng thái đó đã tồn tại trước audit; không khôi phục, commit hoặc sửa source. Báo cáo ngày 08/10 được đọc để tham khảo và các kết luận quan trọng được kiểm chứng lại.

Tài liệu đối chiếu: README, Technical Proposal v3.1 DOCX (trích xuất nội dung XML), PROGRESS, DEVIATIONS, Timeline v5, ADR-001/002, Branch B design, tài liệu offline/device/license, các báo cáo chất lượng và results/logs liên quan. Proposal sửa đổi đã bỏ định hướng prosody transfer/dashboard cũ; PROGRESS tiếp tục thu hẹp triển khai thành laptop-first.

## 2. Bằng chứng kiểm chứng trong lượt audit

| Kiểm tra | Kết quả hiện tại |
|---|---|
| Inventory và parse Python | 74 file: 32 source, 29 tool Python, 13 file test; không có syntax error. Có SyntaxWarning escape `\|` trong `make_nmt30_table.py` |
| Shell tooling | Có 5 script `.sh`; đã rà soát nội dung, chưa chạy trên điện thoại |
| Full test suite | Dừng ở collection: 5 lỗi import do thiếu `librosa`, `jiwer`, `soundfile`, `sherpa_onnx` |
| Nhóm test chạy được | **59 passed, 3 deselected**, gồm core, stub orchestration, safety, gate, glossary, noise synthetic, urgency training trên fake data và offline guard |
| Tái hiện safety | 9 cặp văn bản sai được đưa trực tiếp vào checker/gate; cả 9 nhận `passed=True` và `SPEAK` |
| Tái hiện audio/config | PCM uint8/int32 không được normalize; `max_utterance_s` không chặn input vượt giới hạn; thay pipeline threshold không thay quyết định gate thực |
| Kiểm tra assets | Manifest có 371 mục, 370 bắt buộc; **0/370 hiện diện**, tổng bắt buộc 288,148,313 byte, khoảng 274.8 MiB. `models/`, `data/`, `.venv/` không có |
| Tái tính kết quả lưu sẵn | WER khớp error counts; NMT slot rates khớp output; checker hiện tại tái tạo chính xác file safety v1.2 POSTHOC |
| ASR/NMT/TTS thật và hardware | **Không chạy lại** vì thiếu assets/dependencies; không đo Android, Snapdragon, NPU, thermal hoặc energy trong lượt này |

Lệnh nhóm test:

```powershell
# cwd: D:/bachkhoa/OneVoice/OneVoice/Prototype
python -m pytest -q -p no:cacheprovider tests/test_core.py tests/test_pipeline.py tests/test_safety.py tests/test_gate.py tests/test_glossary.py tests/test_noise_v2.py tests/test_urgency_train.py tests/test_offline.py -k 'not exact_snr_on_real and not full_pipeline_offline and not no_network_capable'
```

59 test pass không chứng minh chất lượng model hoặc độ an toàn E2E: nhiều test dùng stub và fake data. Môi trường dùng Python Anaconda 3.13.9; versions cài sẵn khác một số pin trong requirements-bench.

## 3. Code vs README / Proposal

| Cam kết tài liệu | Code / bằng chứng thực tế | Trạng thái |
|---|---|---|
| Dual-branch pipeline | `pipeline.py` fan-out hai branch từ cùng VAD segment; Branch A nhận denoise, Branch B nhận segment trước denoise | Đã có orchestration; file-based |
| Mic array, Zero-UI và playback trực tiếp | CLI đọc WAV, trả WAV và JSONL; không capture mic, session PTT thực, hotword hoặc playback | Chưa có; Proposal đã chuyển sang PTT |
| NAFNet trên NPU | Có GTCRN, OA mix và classical NS; default OFF theo ADR-001 | Đổi giải pháp; chưa có NAFNet/NPU |
| Whisper-small, dysarthria tuning | sherpa-onnx offline Zipformer VI INT8, CPU; chưa có fine-tuning dysarthria | Đổi model; chưa đáp ứng speech đặc thù |
| EN⇄VI trong README; VI⇄KO chính trong Proposal | Runtime thật chỉ VI→EN. Có bake-off KO nhưng không có KO runtime hoàn chỉnh hoặc EN→VI | Một chiều; ADR-002 Accepted |
| Opus-MT + clinical glossary | Marian ORT encoder/merged decoder, greedy KV-cache, lexical constraints. Runtime constrainer đọc safety lexicon; `glossary_vi_en.json` không phải nguồn trực tiếp của constrainer | Đã có; độ tin cậy hạn chế |
| Prosody envelope → prosody-transfer TTS | Branch B trích 21 acoustic features; `Tts.stream(text, lang)` không nhận pitch/energy/rate/urgency | Chưa code; đã cắt khỏi Proposal mới |
| Urgency classifier / cue | Có UNKNOWN rules, training/LOSO code và hàm export ONNX; `real_full()` tạo BranchB không MLP, nên UNKNOWN. `SPEAK_CUE` chưa có LED/beep/UI consumer | Feature extraction có; classifier/cue chưa hoàn thiện |
| Safety và 5 gate actions theo Proposal | Rules, FSM và tests đủ 5 action. Chưa có workflow người dùng xử lý confirm/repeat hay watchdog | Engine có; thiếu workflow và bảo toàn quan hệ |
| Piper synthesis | Voice EN, CPU. Sinh toàn câu rồi yield chunks; helper first-clause chưa nối vào pipeline | TTS có; không streaming synthesis thực |
| Qualcomm AI Hub, INT4/static quantization | Chưa có `edge/aihub/`, QNN/HTP provider hay compile/profile results. NMT đang dynamic INT8 trên CPU | Chưa có bằng chứng đáp ứng |
| <1 s/utterance trong README | Saved full smoke khoảng 3.00 s với audio 11.34 s, x86. Proposal mới dùng p50 ≤1.5 s/p95 <2 s nhưng chưa có ≥100-turn E2E on-device | Chưa đạt/kiểm chứng cam kết |
| 100% offline | Local inference architecture, asset manifest, OfflineGuard và tests. Các full-model offline tests chưa chạy lại trên checkout này | Có thiết kế và tooling; chưa xác minh release hiện tại |
| Logs chỉ metadata, không nội dung | TurnRecord giữ `asr_text`, `nmt_text`, `gate.speak_text`; CLI ghi toàn bộ JSONL và in stdout | Mâu thuẫn trực tiếp |
| Dashboard, tamper-evident audit, mesh sync | Không `app/`, web frontend, API server, DB/audit-chain hoặc mesh module | Chưa code; đã cắt trong scope mới |
| Getting Started chạy demo | `requirements.txt` và module `src.pipeline.run` được README nêu đều không tồn tại; package thật nằm dưới `Prototype/src/tonebridge` | Hướng dẫn không chạy được |
| Roadmap đánh dấu PoC Snapdragon/AI Hub hoàn thành | Device scripts mới chuẩn bị/dry-run; không có app Android hay profile NPU | Checkbox hoàn thành vượt bằng chứng |

Không cần API/Backend server để chạy engine offline trên thiết bị. Việc thiếu FastAPI/REST không tự động là lỗi kiến trúc; nếu dashboard/mesh quay lại scope thì cần thiết kế giao tiếp tương ứng.

## 4. Tiến độ theo module

Các phần trăm dưới đây tính theo **PoC laptop VI→EN**, gồm triển khai, tích hợp và kiểm chứng; chưa có assets trong checkout làm giảm khả năng bàn giao/tái lập nhưng không biến implementation thật thành mock.

| Module | Ước lượng | Đã làm | Còn thiếu |
|---|---:|---|---|
| Data / evaluation pipeline | 75% | SNR mixer, noise synthesis, fixed splits, WER/CER, bootstrap/binomial CI, safety templates, recording kit | Corpus thu âm thật, đánh giá domain độc lập, loại duplicate và audit protocol |
| Audio frontend / VAD / denoise | 70% | Silero adapter, GTCRN/OA/classical NS, sample alignment tests | Capture/SRC runtime, duration/input validation, no-speech short-circuit |
| ASR VI | 75% | Zipformer CPU adapter, confidence proxy, RTF và WER harness | Quality gate hiệu quả, data y khoa/dysarthria/accent, model provisioning |
| NMT / glossary | 70% | ORT decode/tokenizer, quantized pipeline, lexical bias, benchmark | Quan hệ thuốc–liều, translation độc lập, truncation status, chiều ngược/KO nếu khôi phục scope |
| Safety / fusion | 45% | Slot checker, 5 actions, TTS blocking tests | Scope phủ định, decimal, frequency/action/subject/symptom, OOV, upstream ASR evidence |
| Branch B / urgency | 40% | 21 features, ONNX band graph, UNKNOWN rules, training và LOSO scaffolding | Recordings, model đã train, artifact loader/softmax, calibration, per-SNR evaluation, cue |
| TTS EN | 70% | Piper adapter, chunk output, clause helper và bench | First-clause integration/playback, failure handling, release asset decision |
| Integration / offline reproducibility | 60% | CLI, stage factory, JSONL, offline guard/manifest | Dependency/runtime package, asset install/preflight, mode rõ ràng, privacy mode, effective config hash |
| Frontend / Android app | 0% | Chưa có code app | PTT/session/capture/confirm/cue/playback nếu mục tiêu là Android MVP |
| Device provisioning / measurements | 20% | adb probe/session scripts, protocol | Binaries/inputs, chạy thật, matching artifacts, RSS/thermal/network logs |
| QNN/NPU / AI Hub | 0% | Có kế hoạch trong docs | Export/compile/profile scripts và results |
| API/Backend | N/A | Engine dùng Python CLI | Chỉ cần khi scope có API/UI/mesh consumer |
| Docker / CI/CD / release packaging | 0% | Chưa thấy Dockerfile, workflow CI, package metadata | Container cho evaluation, CI, package/release automation nếu cần |

Ví dụ trọng số cho PoC laptop: Data 15%, Audio 10%, ASR 15%, NMT 15%, Safety 15%, Branch B 10%, TTS 10%, Integration 10%. Nhân các điểm module cho khoảng **63.75%, làm tròn 65%**. Frontend/NPU đã cắt không nằm trong phép tính này. README rộng hơn được đánh giá riêng với biên 35–45%.

## 5. Rủi ro và lỗi quan trọng

### P0: Safety pass không đồng nghĩa bản dịch giữ nguyên nghĩa

`safety.py:93` gom số EN thành set; `safety.py:120` kiểm tra từ/cụm từ trên toàn câu. Nó không ràng buộc số/liều với đúng thuốc hoặc phủ định với đúng mệnh đề. Các phản ví dụ audit đưa trực tiếp vào checker + Gate với ASR confidence 0.95:

| Lỗi | Source VI | Translation sai | Kết quả |
|---|---|---|---|
| Đảo liều giữa hai thuốc | uống hai miligam aspirin và năm miligam paracetamol | Take five milligrams of aspirin and two milligrams of paracetamol. | SPEAK |
| Đảo phạm vi phủ định | tôi không dị ứng penicillin nhưng dị ứng aspirin | I am allergic to penicillin but not allergic to aspirin. | SPEAK |
| Sai thập phân | uống 0,5 miligam paracetamol | Take 5 milligrams of paracetamol. | SPEAK |
| Thêm thuốc | tôi uống paracetamol | I take paracetamol and aspirin. | SPEAK |
| Đổi tần suất | uống hai viên paracetamol mỗi sáu giờ | Take two tablets of paracetamol every two hours. | SPEAK |
| Mất tính chưa chắc chắn | tôi có thể dị ứng penicillin | I am allergic to penicillin. | SPEAK |
| Đảo hành động | ngừng uống aspirin | Continue taking aspirin. | SPEAK |
| Đổi triệu chứng | tôi đau ngực | I have a headache. | SPEAK; 0 checks |
| Thuốc ngoài lexicon | tôi dị ứng clopidogrel | I am allergic to aspirin. | SPEAK; chỉ kiểm tra allergy |

Đây là lỗi logic có thể tái hiện, **không phải 9 lỗi NMT mới đo được**, không dùng để ước lượng unsafe-output rate. Counterexample control `warfarin→aspirin` bị chặn vì warfarin có trong lexicon, xác nhận checker vẫn hoạt động trong phạm vi keyword đã biết.

Ngoài ra checker chỉ đối chiếu ASR text với NMT output. Nếu ASR nghe sai và NMT dịch trung thành với transcript sai, checker không thể xác thực lời nói gốc.

### P0: Gate nhận evidence yếu và cấu hình không phản ánh thực thi

- Saved `asr_conf_gate_check_dev.json`: babble 0 dB, 161/200 utterance có WER >30%; threshold 0.5 bắt được **0%** nhóm sai. Confidence không đủ để bảo đảm safe output.
- `Gate.__call__` bỏ `_pipeline_cfg`, đọc `gate.json` riêng. Audit thay `pipeline.thresholds.asr_confidence_min` từ 0.5 lên 0.99: hash đổi nhưng confidence 0.9 vẫn SPEAK ở cả hai cấu hình.
- `PipelineConfig.models` chứa path/hash/version nhưng factory không dùng chúng để chọn/xác minh assets. Fingerprint không gồm gate/branch-B/lexicon/model bytes/bonus/decode settings thực sự.
- Gate cho `HIGH` với `voicing_quality=None` đi vào SPEAK_CUE; `score` và `GateConfig.urgency_high` không được kiểm tra ở quyết định này. Chưa gây cue trong factory hiện tại vì MLP chưa có, nhưng là lỗ hổng tích hợp classifier.

### P0: Khả năng chạy/tái lập và chế độ demo

- Thiếu assets; manifest chỉ liệt kê hash, chưa có quy trình provision đầy đủ cho tất cả active models.
- `requirements-bench.txt` không khai báo `sherpa-onnx`, `swift-f0`, `soundfile`, `jiwer`, `librosa`, `soxr`, dù code/test/tool dùng chúng. Không có runtime/dev requirements riêng hoặc `pyproject.toml`.
- CLI mặc định đi vào stub. `--real` vẫn AlwaysPassSafety + UnknownBranchB + v0 gate. Người dùng có thể tưởng WAV output chứng minh dịch thật/an toàn.
- `--full` chạy real adapters, nhưng urgency vẫn UNKNOWN do MLP chưa có; frontend và OFF denoise dùng passthrough. Tuyên bố “0 stubs” cần định nghĩa chức năng rõ ràng.
- Config path sai/thiếu tự fallback về mặc định ở `cli.py:53`, dễ chạy thí nghiệm với config ngoài ý định.

### P0: Input, failures, privacy

- `load_wav()` normalize chỉ int16. Audit PCM uint8 `[0,128,255]` trả `[0,128,255]`; PCM int32 trả cỡ ±2.147e9, trái contract [-1,1].
- `max_utterance_s` chưa được dùng. Audit config giới hạn 0.1 s vẫn xử lý WAV 1 s. Sample rate runtime nhiều stage cố định 16 kHz trong khi config cho thay đổi.
- VAD không speech vẫn gửi segment rỗng qua ASR→NMT; không có short-circuit trước model. Chưa chạy model thật để khẳng định adapter có crash, nhưng control flow thiếu bảo vệ.
- `fa.result()/fb.result()` không timeout; exception propagate, không tạo safe failure record. Branch B lỗi có thể làm mất cả bản dịch. Không watchdog, không stale-output policy.
- NMT đạt `max_new_tokens=48` không báo truncated/EOS thiếu cho gate.
- Logs mặc định chứa transcript, translation và approved text; không privacy mode, retention/rotation hoặc consent/session policy. CLI stdout cũng chứa nội dung. Evaluation cần nội dung nhưng phải tách rõ khỏi runtime mặc định.

### P1: Chất lượng đo và branch còn dở

- Test split safety có 612 rows nhưng chỉ **316 source VI duy nhất**; không trùng text giữa dev/test, song duplicate trong split và cùng template làm CI theo row dễ quá lạc quan. Nên deduplicate hoặc cluster theo template/utterance.
- Constraints, checker và gold forms cùng miền/nhóm tác giả. Slot presence có thể cao dù câu sai quan hệ hoặc mất ngữ nghĩa; không đủ thay thế translation review độc lập.
- `eval_safety.py` loại symptom khỏi CHECKABLE. Một số M2 primary metrics chỉ đo slot chính, không phải toàn bộ medication/allergen/symptom phụ.
- `urgency_train.loso()` chọn held-out rows `snr == clean`; chưa trả các fold noisy riêng như docstring hứa. Fake-data tests chỉ chứng minh training code chạy.
- `export_onnx()` có sẵn nhưng chưa có CLI train-final/export artifacts đầy đủ và factory loader. Export trả logits; runtime `mlp` cần P(HIGH), nên phải softmax và chuẩn hóa đúng feature order. Config `mlp/standardizer` hiện null và loader chưa dùng hai field đó.
- `bench_asr_rtf.py` dùng decoder FP32 `decoder-...onnx`; pipeline và phone script dùng decoder INT8. `device_session.sh` lấy laptop RTF 0.040 từ benchmark cũ để tính k: chưa cùng exact artifacts.
- `bench_rss_total.py` đo ASR/NMT/TTS/VAD, chưa bao gồm Branch B + safety + gate của `--full`. Tên “total pipeline” cần ghi rõ tập stage.
- `latency_projection.py` cộng percentile các stage, bỏ VAD (ghi 0 chưa đo), Branch B/safety/gate và playback. Tổng p95 không phải một upper bound p95 được bảo đảm; cần distribution E2E từng turn.
- `final_test.sh` không fail-fast, không kiểm status từng background PID; marker FINAL_TEST_DONE có thể được ghi sau subprocess lỗi. Marker không đủ chứng minh all-pass.
- Một số test integration import optional deps ở top level, nên skip-if-assets không cứu được collection; chưa chia marker unit/integration/model.
- License manifest có mục chưa chốt theo `LICENSES.md`; audit này chưa xác minh nguồn pháp lý mới và không kết luận nghĩa vụ phân phối.

## 6. TODO / FIXME / hard-code / mock còn lại

Không tìm thấy marker literal `TODO`, `FIXME`, `HACK`, `XXX` trong source/tools/tests. Việc thiếu chức năng thể hiện qua stub, config null, comments và docs; không được suy ra “hết TODO” từ grep rỗng.

| File | Nội dung còn dở/hard-code | Phân loại |
|---|---|---|
| `stages/stubs.py` | SidecarAsr fallback “xin chào”, confidence 0.9; TagNmt gắn nhãn; AlwaysPassSafety; UnknownBranchB; ToneTts sine 440 Hz; v0_gate | Mock có chủ đích, cần mode riêng |
| `cli.py`, `stages/factory.py` | Default stub; `--real` bypass checker thực; full frontend passthrough | Rủi ro nhầm mode |
| `branch_b.py`, `configs/branch_b.json` | `mlp=None`, `standardizer=null`, untuned thresholds; factory không load trained classifier | Chưa hoàn thành urgency |
| `urgency_train.py:synthetic_rows` | SEEDED FAKE DATA | Hợp lệ cho self-check, không phải kết quả thực |
| `stages/factory.py` | ASR_DECODER pinned, GLOSSARY_BONUS=5, threads=2, chỉ vi-en | Pin có chủ đích; đưa vào effective config/hash |
| `stages/nmt_ort.py` | max_new_tokens=48, CPU provider, fixed direction | Hạn chế scope; thêm truncation evidence |
| `stages/asr_sherpa.py`, `branch_b.py`, VAD | 16 kHz/256 hop, CPU provider, thresholds bắt đầu | Hợp lệ nếu validate và document contract |
| `configs/pipeline.json` | NMT/TTS/VAD version “unpinned”; metadata model không điều khiển loader | Debt reproducibility |
| `tools/device_session.sh` | DRY_RUN trả RTF giả 0.090, HWM giả; hard-code paths và k baseline | Tool simulation, không được dùng làm benchmark |
| `tools/latency_projection.py` | k=3/4, VAD_MS=0, cộng percentile | Estimate, không phải device result |
| `.sh` tools | `.venv/Scripts/python.exe` hoặc Windows path trong Bash | Portability debt |
| `offline_guard.py` | `except psutil.Error/AttributeError: pass` | Sampling failure không được phân biệt với zero traffic |
| Docs | PROGRESS còn ADR Proposed dù ADR Accepted; OFFLINE_FILES dẫn LAPTOP_RESULTS không tồn tại; PROGRESS nói AI Hub submit script nhưng chưa có | Tài liệu lỗi thời/thiếu artifact |

`pass` trong exception class và test clock là hợp lệ; `...` trong typing.Protocol là interface, không phải feature chưa code. OFF passthrough denoise là kết quả ADR có bằng chứng, không cần thay bằng denoiser chỉ để “loại stub”.

## 7. Benchmark hiện có nói được gì?

Các số dưới đây **đọc/tái tính từ artifacts lưu sẵn**, không phải inference benchmark mới:

| Chỉ số | Kết quả | Đánh giá |
|---|---|---|
| ASR clean test, n=300 | WER 19.21% | Không đạt target clean ≤15% của Proposal |
| ASR babble 0 dB test | WER 60.36% | Không đạt target ≤35%; cần quality reject hữu hiệu |
| ASR babble -5 dB test | WER 93.08% | Điều kiện rất khó, gần như mất khả năng hiểu |
| ASR demand 10/5/0 dB | 19.22 / 19.63 / 22.16% | Đạt các target tương ứng trên noise set đã đo |
| Denoise dev | 0/13 điều kiện có arm hơn OFF ≥2 điểm WER | Giữ default OFF là có căn cứ |
| NMT test medication / dose | 93.33 / 90.67% primary-slot presence | Chưa đạt target 95% term / 99% dose; metric presence còn yếu |
| NMT test negation | 98.75% primary-slot presence | Chưa đạt 99%; không đánh giá scope phủ định |
| Safety checker hiện tại trên saved test NMT | 39/43 detected = 90.7%; false-block 0/569 | Khớp v1.2 POSTHOC; không phải test set mới, chỉ kiểm checkable slots |
| Saved smoke full | ~3004 ms first audio, WAV 11.34 s; urgency UNKNOWN | Một lượt x86, không suy ra p50/p95 hoặc NPU performance |
| Saved RSS baseline stages | 1021.2 MiB; ≤7 s cohort 760.4 MiB | Không gồm toàn bộ full stages; cần chốt đơn vị GB/GiB và đo phone |

Unit lưu `*_mb` được code tính bằng 2^20, thực chất **MiB**. 1021.2 MiB vượt 1 GB thập phân nhưng thấp hơn 1 GiB; tài liệu ghi “1.0 GB” cần thống nhất để tránh kết luận pass/fail nhập nhằng.

## 8. Action plan có file và acceptance criteria

Base directory cho các file dưới đây: `D:/bachkhoa/OneVoice/OneVoice/Prototype/`; README nằm ở `D:/bachkhoa/OneVoice/OneVoice/README.md`. Các file ghi “tạo” là đề xuất, chưa được tạo trong audit.

### P0 — Cốt lõi để E2E thật, có thể bàn giao và chặn lỗi đã biết

- [ ] **P0-1: Reproducible runtime và asset preflight.** Sửa `requirements-bench.txt`; tạo `requirements-runtime.txt`, `requirements-dev.txt` hoặc `pyproject.toml`, `tools/provision_models.py`, `tools/check_runtime.py`; cập nhật `configs/runtime_files.json`, `cli.py`, README. Tách training deps khỏi inference, pin runtime/model URLs/revisions, verify SHA-256 trước load. Acceptance: checkout mới cài được, thiếu/sai asset báo rõ, `--full --assert-offline` chạy bằng model thật với package chuẩn.
- [ ] **P0-2: Explicit CLI modes và fail-closed config.** Sửa `cli.py`, `stages/factory.py`, `stages/stubs.py`, `tests/test_pipeline.py`; tạo test mode/preflight nếu cần. Dùng `--mode stub|branch-a-bench|full`, log mode và stage capabilities; full dùng safety gate thật, stub chỉ dùng khi chọn rõ. Config được chỉ định nhưng không tồn tại phải fail. Acceptance: không thể nhầm tone/stub với translation output.
- [ ] **P0-3: Safety theo quan hệ có scope.** Sửa `safety.py`, `contracts.py`, `gate.py`, `nmt_constraints.py`, `configs/safety_lexicon_vi_en.json`; tạo `clinical_relations.py`, `tests/test_safety_relations.py`. Parse từng clause/entity thành drug–amount–unit–frequency–action–negation–subject/modality; dùng Decimal cho 0,5/0.5; phát hiện target thêm entity, symptom/OOV và ambiguity. Không parse được critical relation thì CONFIRM/ABSTAIN. Acceptance: cả 9 counterexample bị chặn, các câu đúng tương ứng vẫn pass; test pipeline xác nhận TTS không được gọi.
- [ ] **P0-4: ASR evidence và noisy-input reject.** Sửa `stages/asr_sherpa.py`, `contracts.py`, `gate.py`, `tools/asr_conf_gate_check.py`, `tools/snr_gate_probe.py`; tạo `quality_gate.py`, evaluation cho unsafe-output/coverage. So sánh quality signals trên dev, calibrate rồi freeze; không coi token confidence là xác suất transcript đúng. Acceptance: báo bad-transcript recall, false reject và coverage theo noise/SNR; lớp chưa kiểm chứng dùng confirm/repeat policy bảo thủ.
- [ ] **P0-5: Runtime failure và audio validation.** Sửa `cli.py`, `pipeline.py`, `config.py`, `stages/vad_silero.py`, `stages/nmt_ort.py`, `contracts.py`; tạo `tests/test_audio_input.py`, `tests/test_runtime_failures.py`. Normalize PCM đúng dtype, validate finite/mono/16 kHz/duration; no-speech short-circuit; expose truncation; timeout/watchdog có khả năng kết thúc native job, safe error state và không phát output cũ. Acceptance: invalid/empty/silent/too-long/model-error không đi đến TTS hoặc treo turn.
- [ ] **P0-6: Metadata-only logging mặc định.** Sửa `contracts.py`, `telemetry.py`, `cli.py`, README; tạo `tests/test_privacy_logging.py`. Dùng runtime record loại transcript/translation/speak_text khỏi log lẫn stdout; nội dung chỉ bật trong evaluation mode có retention rõ ràng. Acceptance: quét output runtime không có nội dung hội thoại; error path cũng không leak input.
- [ ] **P0-7: Một effective config có fingerprint thật.** Sửa `config.py`, `gate.py`, `branch_b.py`, `stages/factory.py`, `configs/*.json`; mở rộng `tests/test_core.py`, `tests/test_gate.py`. Resolve và validate model paths/hashes, gate/branch thresholds, lexicon, bonus, decode caps và threads vào snapshot; reject inconsistent settings. HIGH phải có quality/score hợp lệ. Acceptance: đổi config vừa đổi hành vi mong đợi vừa đổi hash; runtime assets khớp manifest.
- [ ] **P0-8 nếu deliverable là Android MVP: device baseline trước claim.** Sửa `tools/bench_asr_rtf.py`, `tools/device_session.sh`, `docs/DEVICE_SESSION.md`; tạo Android harness tối thiểu trong `android/` nếu cần. Đo exact INT8 decoder trên cả laptop/phone, chuẩn bị ORT inputs KV-cache hợp lệ; giữ composed estimate riêng với E2E. Acceptance: raw logs hardware thật, artifact hashes/runtime versions, latency/RSS/load/thermal/network; không dùng dry-run hay k chung thay measurement. Nếu deliverable chỉ laptop PoC, chuyển task này sang P1 và sửa rõ cam kết README/Proposal.

### P1 — Chất lượng, model còn thiếu và kiểm thử bàn giao

- [ ] **P1-1: Thu âm và đóng vòng urgency.** Sửa `recording_kit/`, `tools/train_urgency.py`, `urgency_train.py`, `branch_b.py`, `stages/factory.py`, `configs/branch_b.json`; tạo artifact loader và `tests/test_urgency_model.py`. Thu recordings theo protocol, speaker-disjoint LOSO và từng SNR, train-final/export ONNX + standardizer + feature schema; logits→probability đúng, calibrate threshold. Acceptance: F1/Recall/UNKNOWN/abstain theo speaker/SNR, factory thật có LOW/HIGH/UNKNOWN; model chưa đạt thì disable cue.
- [ ] **P1-2: Validation độc lập ASR/NMT/safety.** Sửa `tools/make_safety_set.py`, `evalkit/slotgold.py`, `tools/eval_safety.py`, `docs/nmt_ref_template.csv`; tạo held-out relation/OOV/multi-clause corpus do người review độc lập. Deduplicate/group bootstrap, kiểm full semantic relations và subject/symptom; sau sửa post-hoc cần holdout mới. Acceptance: không báo slot presence như translation correctness; các target Proposal và fail slices được công bố đầy đủ.
- [ ] **P1-3: Integration tests chạy được trong môi trường sạch.** Sửa `pytest.ini`, các test optional deps; tạo integration fixtures và full safety pipeline tests. Marker unit/integration/model, kiểm availability trước imports optional; test fault injection/privacy/config/unsupported language. Acceptance: suite unit không phụ thuộc model assets; job có assets chạy full integration thực, có skip reason chính xác.
- [ ] **P1-4: E2E latency/RSS thực và TTS output.** Sửa `stages/tts_piper.py`, `pipeline.py`, `tools/bench_latency_options.py`, `tools/bench_rss_total.py`, `tools/latency_projection.py`; tạo `tools/bench_e2e.py`. Đo ≥100 turns toàn stage, cold/warm riêng, hợp nhất TTS clause synthesis nếu không giảm chất lượng; first-buffer readiness và audible playback đo riêng. Chuẩn hóa MiB/GiB. Acceptance: p50/p95 từ samples E2E, RSS full stages và phone measured; projections ghi estimate, không gọi tổng p95 là bound bảo đảm.
- [ ] **P1-5: Workflow PTT/confirm/cue/playback nếu giữ Android scope.** Tạo `android/app/` session state, capture/SRC, permission/consent state, repeat/confirm, urgency indicator và playback; nối engine qua ORT/sherpa/JNI. Acceptance: user có thể hoàn tất một turn và xử lý đủ 5 actions trên thiết bị. API web không phải điều kiện bắt buộc.
- [ ] **P1-6: Benchmark integrity.** Sửa `tools/final_test.sh`, `tools/run_adr001_v2.py`, benchmark default decoder và result metadata. Theo dõi từng PID/exit code, ghi marker chỉ khi toàn bộ outputs hợp lệ; resume kiểm config fingerprint trước skip. Sửa empty-hyp metric để đo đúng deletion thay literal `<empty>` nếu jiwer version cho phép. Acceptance: injected subprocess failure không tạo marker hoàn thành; mismatched config không tái dùng cells cũ.
- [ ] **P1-7: Chuẩn hóa docs và release asset decisions.** Sửa README, PROGRESS, ADR-002, OFFLINE_FILES, LICENSES; tạo `docs/STATUS.md`, `docs/RUNBOOK.md`, quyết định runtime/voice asset cụ thể. Phân biệt done/proposed/measured/projected, bỏ dead links và claims ngoài bằng chứng. AI Hub là requirement challenge thì tạo `tools/aihub_submit.py` và lưu job/profile artifacts khi đã có quyền/token; token cấu hình qua env, không commit. Acceptance: hướng dẫn chạy được; runtime license/asset status được chốt trước phân phối, không coi license data là license toàn runtime.

### P2 — Refactor, vận hành và mở rộng

- [ ] CI chạy unit + integration có assets riêng, static checks và package validation; lưu artifacts/report theo revision.
- [ ] Docker cho môi trường benchmark/training CPU có pinned deps; Android deployment dùng APK/AAB/assets, không dùng Docker để thay đo phone.
- [ ] Observability metadata-only: error taxonomy, stage timeout/load metrics, RSS trend, log rotation và release manifest.
- [ ] Refactor tools tránh top-level execution, hard-coded Windows interpreter và duplicated config; giảm per-turn ThreadPool creation sau khi profiling.
- [ ] Watchdog/device soak tests, process isolation/native cancellation, model loading/memory arenas sau đo thực.
- [ ] QNN/HTP/AI Hub optimization sau CPU baseline, kiểm operator/fixed-shape/artifact compatibility từng model.
- [ ] EN→VI/KO, prosody transfer, dashboard, audit-chain/mesh và glossary OTA chỉ mở lại khi có scope được chốt; mỗi feature cần protocol riêng.

## 9. Thứ tự thực hiện và mốc chuyển giai đoạn

1. **Milestone A — PoC tái lập:** P0-1/2/5/7; full WAV→WAV chạy từ môi trường sạch, explicit modes, validated input/config.
2. **Milestone B — PoC có output gate được kiểm chứng:** P0-3/4/6 và P1-2/3; relation counterexamples bị chặn, privacy default, có unsafe-output/coverage report trên holdout độc lập.
3. **Milestone C — Alpha trên laptop:** urgency real-data hoặc disabled cue công khai; ≥100-turn full E2E/RSS benchmark; fault tests và runbook hoàn chỉnh.
4. **Milestone D — Android MVP:** P0-8/P1-5 và measurements thật; complete user workflow, không còn cần laptop/adb để thực hiện turn sử dụng thông thường.

Chưa có cơ sở gọi production-ready: chưa hoàn tất các mốc trên, validation domain/clinical còn thiếu và release/runtime/license decisions chưa chốt. Ưu tiên tăng khả năng tái lập và chặn lỗi output trước khi tối ưu NPU hoặc mở rộng ngôn ngữ.

Audit chỉ bổ sung báo cáo này; không sửa logic ứng dụng, tải model, cài dependency, commit/push hoặc tác động điện thoại.
