# Kiểm chứng tích hợp laptop bốn chiều — 09/10/2026

Base runtime `05b71b9` + branch NMT/safety `ef221c5`, trên `codex/laptop-four-direction-integration`. Windows 11, Python 3.12 CPU, 2 threads/model, urgency UNKNOWN. [Setup hiện tại](../docs/LAPTOP_FOUR_DIRECTION_INTEGRATION.md), [summary có số liệu đầy đủ](LAPTOP_FOUR_DIRECTION_SUMMARY_2026-10-09.json).

## Đã thực hiện

- Ghép branch của bạn ấy, giữ audio/runtime và bổ sung factory theo bốn chiều. Nhận schema evidence mới, kiểm EOS/truncation/UNK/constraints trước TTS, lưu evidence và safety status/reasons trong telemetry.
- Sửa test fixture có EOS thật và test dùng tín hiệu thay silence; thêm regression bốn chiều, Korean block, evidence không được bypass bởi custom gate, log không lộ diagnostic content, budget stop với pretrained models.
- Pin EN→VI/KO→EN, kiểm hash local trước load; chuyển checkpoint `.bin` sang safetensors bằng weights-only + strict state_dict. Giữ vocabulary ID đã công bố và chặn mọi lượt có UNK.
- Audit thật Helsinki EN→KO: 25.330 source pieces thiếu mapping và `Hello.` chứa UNK. Giữ ứng viên này cách ly. Đăng ký Argos EN→KO 1.1 / CTranslate2 INT8 riêng sau hash/tokenizer/EOS smoke; license package chưa rõ và Korean clinical review chưa xong.
- Sửa định dạng đầu câu/dấu kết thúc cho ASR tiếng Anh. Giữ nguyên transcript và từ/số trong câu; evidence ghi thao tác. Trước sửa, `hello` ra `Có thể` / `뚱 베어`; với `Hello.` ra greeting đúng trong các smoke này. Đây chưa phải punctuation restoration tổng quát.
- Watchdog ghi timeout/crash vào log; CLI/worker dùng UTF-8. Đổi log mặc định sang vùng local ignored.
- Sửa RAM tăng do tạo thread branch mới mỗi lượt: giữ worker cố định, thêm `close()`/context manager. Chạy lại benchmark xác nhận plateau trên mẫu lặp.

## Validation

`pytest`: **204 passed, 3 skipped**, 1 warning từ generation config của tiny random test fixture. Skip: Branch B golden/FLEURS thiếu, GTCRN thiếu, noise corpus thiếu. `pip check` không có dependency lỗi; `git diff --check` sạch. **452 assets** khớp size/hash; manifest SHA256 `8cee471dde353ea35a80ced7f5f2f87a5e00f249272a6cfdf681acf86e538d4a`.

Real-model smoke: 13 lượt audio tổng cộng, 20 câu text dev mỗi chiều. Không lỗi runtime; Python network attempts = 0, OS connections seen = 0. VI→EN và EN→VI greeting đều SPEAK và sinh WAV thật; Korean dừng CONFIRM. Một mẫu tiếng Anh vượt thời lượng cho phép được REPEAT. Raw logs/WAV chỉ giữ local, không đưa lên Git.

Benchmark sau sửa worker: **608 lượt đo** + 12 warmup, riêng process từng chiều. Không error/crash, offline audit 0/0 ở cả bốn chiều.

| Chiều | Lượt / WAV độc lập | SPEAK / CONFIRM / REPEAT | p95 first sample (ms) | p95 text (ms) | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| VI→EN | 104 / 4 | 26 / 78 / 0 | 234 | 459 | 846 |
| EN→VI | 201 / 3 | 67 / 67 / 67 | 1163 | 798 | 1182 |
| EN→KO | 102 / 3 | 0 / 68 / 34 | Không có audio | 506 | 699 |
| KO→EN | 201 / 3 | 0 / 201 / 0 | Không có audio | 562 | 999 |

`first sample` chỉ tính lượt SPEAK, không phải first audible. Không so sánh audio latency của Korean vì chưa được phát. Mẫu là greeting TTS và WAV của nhà cung cấp ASR, không phải 100 câu độc lập hay corpus clinical/noise.

Trước sửa worker, EN→VI tăng RSS 1149→1316 MiB trong 102 lượt; KO→EN 967→1140 MiB. Sau sửa, EN→VI 1158→1180 MiB trong 201 lượt, ổn định quanh 1179 MiB sau khoảng lượt 20; KO→EN 973→997 MiB, ổn định quanh 997 MiB. Đây là bằng chứng trên mẫu lặp ngắn, chưa phải kiểm chứng qua đêm. EN→VI còn vượt mốc 1 GiB, cần tối ưu/export INT8 nếu mốc đó là release gate.

Hardware VI→EN: người dùng sẵn sàng đọc `Xin chào`; lượt đầu chưa đọc kịp nên REPEAT, lượt thứ hai capture 15 giây, nhận speech, safety PASS, gate SPEAK, tạo PCM16 WAV và hoàn tất playback API. Người dùng xác nhận **nghe rõ “Hello”**. Endpoint→first sample khoảng **405 ms**; raw microphone audio không lưu. Chưa đo first audible bằng loopback và chưa thử hardware trực tiếp ba chiều còn lại.

## NMT/safety còn hạn chế

Kết quả 20 **draft dev** mỗi chiều, chưa có nhãn clinical độc lập:

| Chiều | PASS / CONFIRM / FAIL | chrF theo draft reference |
|---|---:|---:|
| VI→EN | 4 / 14 / 2 | 65,52 |
| EN→VI | 4 / 4 / 12 | 33,55 |
| EN→KO | 0 / 8 / 12 | 23,07 |
| KO→EN | 0 / 16 / 4 | 64,93 |

Các số này không phải clinical accuracy, false accept/false block hay safety recall. Ví dụ lỗi cần xem: EN→VI `Take two tablets of paracetamol.` → `Lấy hai khẩu đại tá.` bị FAIL; VI→EN `uống prednisolone` → chỉ `prednisolone`, mất action và bị FAIL. Một false block rõ về coverage: `I have chest pains.` không khớp alias `chest pain`. Không nới gate để làm các câu này phát.

[Gói review 80 câu](LAPTOP_DEV_REVIEW_PACKET_2026-10-09.jsonl) có source, output thật, draft reference, lý do checker và model revision. `clinical_correct=null`, `reviewer` trống đến khi người biết ngôn ngữ review; không dùng checker tự gán gold. Holdout chưa được chạy hoặc dùng chỉnh code.

Việc còn cần dữ liệu/người review: sửa/chọn model cho các lỗi clinical, mở rộng checker có regression và đo false block; review bilingual/Korean lexicon và license EN→KO; corpus tối thiểu 100 WAV độc lập mỗi chiều, ASR confidence/noise calibration, memory dài hạn với câu đa dạng và hardware các chiều còn lại. Android/NPU, urgency MLP, nút PTT và incremental TTS chưa thuộc phần đã kiểm chứng.
