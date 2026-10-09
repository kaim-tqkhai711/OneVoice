# ToneBridge — OneVoice

Prototype dịch giọng nói dựa trên **TechProposal v3.1**. Ưu tiên hiện tại là chạy ổn trên laptop CPU với bốn chiều **VI→EN, EN→VI, EN→KO, KO→EN**. Mã tiếng Hàn trong cấu hình là `ko`.

Cập nhật ngày **09/10/2026**, theo branch `codex/laptop-four-direction-integration`. Hướng dẫn dưới đây mô tả bản tích hợp hiện tại; Android/NPU chưa được kiểm chứng trong bản này.

## Bắt đầu

- [README Prototype: setup, tải model và chạy](Prototype/README.md).
- [Hướng dẫn tích hợp và chia việc](Prototype/docs/LAPTOP_FOUR_DIRECTION_INTEGRATION.md).
- [Báo cáo kiểm chứng ngày 09/10/2026](Prototype/reports/LAPTOP_FOUR_DIRECTION_VERIFICATION_2026-10-09.md).
- [Gói 80 câu dev để review NMT/safety](Prototype/reports/LAPTOP_DEV_REVIEW_PACKET_2026-10-09.jsonl).
- [TechProposal v3.1](ToneBridge_TechProposal_v3_1_Revised_Template.docx).

Clone bản tích hợp:

```powershell
git clone --branch codex/laptop-four-direction-integration https://github.com/kaim-tqkhai711/OneVoice.git
cd OneVoice/Prototype
```

Sau đó làm theo [README Prototype](Prototype/README.md). Python 3.12 trên Windows là môi trường đã kiểm chứng. Virtualenv, model weights, audio và log cục bộ không nằm trong Git; máy mới cần tải model trước khi chạy.

## Trạng thái bốn chiều

| Chiều | NMT đang dùng trên laptop | Trạng thái |
|---|---|---|
| VI→EN | OPUS-MT ONNX INT8 | Greeting đã qua safety và phát “Hello” bằng micro/loa thật. |
| EN→VI | Helsinki OPUS-MT Marian CPU | Greeting sinh WAV thật; chưa thử trực tiếp micro/loa. |
| EN→KO | Argos 1.1 / CTranslate2 INT8, thử nghiệm | Safety tiếng Hàn chưa review; dừng CONFIRM, không phát audio. |
| KO→EN | Helsinki OPUS-MT Marian CPU | Safety tiếng Hàn chưa review; dừng CONFIRM, không phát audio. |

Checkpoint Helsinki EN→KO ban đầu được cách ly vì lỗi mapping vocabulary và UNK. Ứng viên Argos hiện chưa có license cụ thể trong package; cần review trước khi chọn cho bản phát hành. Bốn đường xử lý đã nối vào runtime, nhưng chất lượng dịch clinical còn lỗi và chưa được xác nhận bằng nhãn độc lập.

## Luồng xử lý

```text
WAV / microphone → mono 16 kHz → VAD / endpoint
                                  ├─ Branch A: ASR → NMT → clinical checker
                                  └─ Branch B: urgency (hiện UNKNOWN)
                               → Gate → TTS → WAV / loa
```

Các preset laptop hiện tắt denoise và urgency classifier. Urgency MLP chưa được huấn luyện/kiểm chứng. TTS dùng Piper cho tiếng Anh và Supertonic cho tiếng Việt/tiếng Hàn; TTS hiện tổng hợp cả câu rồi mới chia chunk/phát.

Gate có năm hành động: `SPEAK`, `SPEAK_CUE`, `REPEAT`, `CONFIRM`, `ABSTAIN`. Chỉ hai hành động SPEAK tạo audio. Evidence thiếu EOS, truncation, UNK hoặc constraints không hợp lệ sẽ chặn TTS. Clinical checker dùng grammar/lexicon có giới hạn; PASS không chứng minh ASR đúng hay câu phù hợp sử dụng clinical.

Setup cần mạng; inference dùng assets local được kiểm hash. Các lượt smoke/benchmark đã ghi nhận 0 network attempts và 0 OS connections. Log mặc định loại transcript/translation và nội dung diagnostic; kết quả này chỉ áp dụng cho phạm vi đo đã thực hiện.

## Đã kiểm chứng

- Regression: **204 passed, 3 skipped**; `pip check` sạch. Ba nhóm còn thiếu dữ liệu là Branch B golden/FLEURS, GTCRN và noise corpus.
- **452 assets** trong môi trường kiểm chứng khớp size/hash.
- **608 lượt benchmark + 12 warmup** trên bốn chiều, không error/crash. Mỗi chiều chỉ dùng 3–4 WAV độc lập rồi lặp lại.
- Người dùng đã đọc “Xin chào” và xác nhận nghe rõ “Hello” trong một lượt VI→EN thật.
- RAM đã ổn định trên mẫu lặp sau khi giữ worker cố định. Peak EN→VI còn **1182 MiB**, chưa có kiểm chứng qua đêm với câu đa dạng.

Xem [báo cáo đầy đủ](Prototype/reports/LAPTOP_FOUR_DIRECTION_VERIFICATION_2026-10-09.md) để biết latency, RAM, lỗi dịch và giới hạn của phép đo. Các số trên không thay thế đánh giá clinical hay kiểm thử 100 câu độc lập mỗi chiều.

## Chia việc tiếp theo

| Người phụ trách | Việc cần làm | Phạm vi sửa chính |
|---|---|---|
| Bạn — runtime/audio | Corpus ít nhất 100 WAV độc lập mỗi chiều; ASR/noise calibration; đo RAM dài hạn; thử hardware các chiều còn lại khi safety cho phép. | Pipeline, audio, CLI, benchmark và tests liên quan. |
| Bạn của bạn — NMT/safety | Review 80 câu dev bằng người biết ngôn ngữ; xử lý lỗi model và coverage checker; review lexicon/reference tiếng Hàn và license EN→KO; đo false accept/false block từ nhãn độc lập. | Adapter NMT, lexicon/checker, tools text và tests liên quan. |

Cả hai tạo branch riêng từ `codex/laptop-four-direction-integration`. Nếu cần đổi contract, config, factory hoặc model registry, thống nhất trước vì đây là điểm dùng chung. Giữ holdout nguyên trạng; không đổi `korean_reviewed` chỉ để phát demo.

Android/NPU, urgency MLP, nút PTT và incremental TTS là các hạng mục tiếp theo. Runtime laptop hiện có PyTorch CPU cho Marian; cần kế hoạch export/tối ưu riêng trước khi chuyển sang mobile.

## Cấu trúc repository

```text
README.md
ToneBridge_TechProposal_v3_1_Revised_Template.docx
Prototype/
  README.md
  configs/       # Preset bốn chiều, registry và manifest smoke
  src/           # Runtime, model adapter, safety, audio
  tests/         # Regression
  tools/         # Provision, CLI, smoke, benchmark, review
  docs/          # Hướng dẫn và audit
  reports/       # Kết quả kiểm chứng, gói review
  models/        # Assets local, không đưa vào Git
  results/       # Audio/log/benchmark local
onevoice/
paper research/
```

## Tài liệu lịch sử

[Audit ngày 08/10/2026](Prototype/docs/ToneBridge_Deep_Codebase_Audit_2026-10-08.md) và các handoff cũ hữu ích để hiểu quá trình phát triển; trạng thái sử dụng hiện tại nằm trong README và báo cáo tích hợp ở trên.

Nội dung README trước lần cập nhật này được giữ tại [README root cũ](Prototype/docs/README_ROOT_LEGACY_2026-10-09.md) và [README Prototype cũ](Prototype/docs/README_PROTOTYPE_LEGACY_2026-10-09.md). Các claim, số liệu và đường dẫn trong bản lưu thuộc thời điểm cũ.
