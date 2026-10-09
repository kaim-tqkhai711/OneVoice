# Bàn giao phần NMT / safety

**Branch NMT/safety đã được tích hợp:** xem [hướng dẫn hiện tại và việc còn lại](LAPTOP_FOUR_DIRECTION_INTEGRATION.md), [báo cáo kiểm chứng](../reports/LAPTOP_FOUR_DIRECTION_VERIFICATION_2026-10-09.md) và [gói review 80 câu dev](../reports/LAPTOP_DEV_REVIEW_PACKET_2026-10-09.jsonl). Danh sách bên dưới là phạm vi bàn giao ban đầu, không còn là trạng thái thiếu adapter hiện tại.

Scope leader đã chốt: **laptop CPU trước**, bốn chiều **VI→EN, EN→VI, EN→KO, KO→EN**. Tiếng Hàn dùng mã `ko`, không dùng `kr` trong code. Android/NPU chưa thuộc vòng công việc này.

## Đọc trước

1. [Hợp đồng runtime, setup và lệnh chạy](LAPTOP_RUNTIME_HANDOFF.md).
2. [Technical Proposal v3.1](../../ToneBridge_TechProposal_v3_1_Revised_Template.docx).
3. [Audit ngày 08/10/2026](ToneBridge_Deep_Codebase_Audit_2026-10-08.md). Đây là snapshot cũ; các đường dẫn máy cũ và nhận xét thiếu file không phản ánh đầy đủ branch hiện tại.
4. [Kết quả kiểm chứng runtime](../reports/LAPTOP_RUNTIME_VERIFICATION_2026-10-09.md).

README root và ADR VI→EN-only là tài liệu lịch sử. Scope ở trên là phạm vi bàn giao hiện tại; chưa có quyết định thay thế toàn bộ yêu cầu product dài hạn của proposal.

## Công việc của người phụ trách text

- Giữ VI→EN baseline, bổ sung NMT EN→VI / EN→KO / KO→EN. Khảo sát EN→KO sớm vì candidate trong bake-off cũ gặp lỗi tokenizer/output. Không suy ra mọi model EN→KO đều thất bại từ bake-off đó.
- Kiểm tra text/tokenizer trước khi export hoặc quantize; pin checkpoint, revision, license và SHA256 assets.
- Sửa safety VI→EN: drug–dose–unit association, decimal và alias đơn vị, negation scope, thuốc bị thêm, stop/continue, frequency và uncertainty. Không chỉ kiểm sự hiện diện từ khóa.
- Làm checker/lexicon đúng ngôn ngữ cho các chiều mới. Critical construction chưa hỗ trợ → CONFIRM hoặc ABSTAIN; không dùng `AlwaysPassSafety` để hoàn tất demo.
- Trả EOS/truncation/constraint evidence cho runtime. VI→EN legacy vẫn cần được bổ sung evidence; runtime không tự suy ra EOS khi adapter không báo.
- Tạo bộ text/reference riêng cho từng chiều; có cặp đúng/sai, thuốc/liều/phủ định, nhiều mệnh đề và nội dung ngoài lexicon. Tách dev và holdout; output tiếng Hàn cần người đọc được tiếng Hàn review. Smoke test không thay thế independent evaluation.

## Ranh giới file để làm song song

Người phụ trách text sửa:

- `src/tonebridge/stages/nmt_ort.py` và adapter NMT mới.
- `src/tonebridge/safety.py`, `glossary.py`, `nmt_constraints.py` và lexicon tương ứng.
- Factory text riêng, config text riêng, tools/tests NMT và safety.

Runtime owner giữ `contracts.py`, `config.py`, `pipeline.py`, `gate.py`, `cli.py`, `stages/base.py`, `stages/factory.py`, các ASR/TTS adapter và config audio. Nếu cần đổi interface, thống nhất trước; không sửa đồng thời các file này.

Factory bàn giao:

```python
def build_text_stages(cfg, threads=2):
    # Build theo cfg.direction, đọc assets local.
    return nmt, safety
```

Module riêng có thể đăng ký qua CLI `--text-factory tonebridge.partner_text:build_text_stages`; không cần sửa `stages/factory.py`. Xem hợp đồng chi tiết trong `LAPTOP_RUNTIME_HANDOFF.md`.

## Bàn giao lại cho runtime owner

- Code adapter/factory cho cả bốn chiều và test đi kèm.
- Requirements riêng cho export/training; inference dependencies được ghi rõ.
- Tool lấy/export model, checkpoint revision, license, hashes và đường dẫn local.
- Dataset text/reference, protocol review, kết quả từng chiều và danh sách lỗi còn tồn tại.
- Câu đúng được checker chấp nhận; regression errors bị chặn; output truncated/unsupported không tự phát. Báo coverage và false-block, không đạt detection recall bằng cách chặn tất cả.

Sau đó runtime owner ghép audio E2E, thử microphone/playback thật và benchmark ít nhất 100 câu độc lập mỗi chiều. Ba chiều mới hiện chưa có NMT/safety được đăng ký nên runtime báo thiếu adapter và không tạo audio.

Model assets, `.venv`, audio và log mới không được đưa lên branch. Clone mới cần setup/provision theo hướng dẫn; kết quả thử cục bộ được tóm tắt trong verification report.
