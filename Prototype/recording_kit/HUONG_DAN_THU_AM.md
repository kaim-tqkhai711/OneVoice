# Hướng dẫn thu âm cho ToneBridge (≈ 12 phút mỗi người)

Mục đích: lấy giọng nói **diễn theo kịch bản** để huấn luyện/đánh giá bộ ước lượng "vocal urgency" (độ gấp của giọng) và đo WER.
Đây là dữ liệu **diễn xuất**, không phải bệnh nhân thật và không dùng để chẩn đoán. Tuyệt đối **không đọc thông tin bệnh nhân thật**.

## 1. Việc cần làm
- Đọc **30 câu** trong `script_vi.csv`, mỗi câu **2 lần**: một lần **NEUTRAL** (bình thường) và một lần **URGENT** (gấp).
- Tổng: 60 file/người. Đọc xong toàn bộ khối NEUTRAL trước, rồi mới sang khối URGENT.
- Điền 1 dòng trong `speaker_metadata_template.csv` (đổi `spk02` thành mã của bạn: Chi = `spk02`, Trí = `spk03`, Thịnh = `spk04`, Tân = `spk05`, Bảo = `spk06`; Khải = `spk01`).

## 2. Cách đọc
- **NEUTRAL:** giọng nói bình thường, như hỏi/nói chuyện trong phòng khám yên tĩnh.
- **URGENT:** giọng **gấp, căng**, như đang đau hoặc khó thở thật: nhanh hơn, cao hơn, to hơn một chút, có thể hơi hụt hơi. Vẫn phải **nói rõ chữ**. **Không hét, không khóc, không thì thào.**
- Mỗi câu một lần, không lặp lại để chọn bản đẹp. Nếu đọc sai rõ ràng (vấp, ho) thì thu lại ngay câu đó và **xóa bản sai**.
- Đọc đúng chữ trong kịch bản, không thêm bớt.

## 3. Thiết bị và môi trường
- Điện thoại bất kỳ, ghi bằng ứng dụng ghi âm có xuất **WAV hoặc FLAC** (không chất lượng AAC/MP3 nếu tránh được; nếu chỉ có M4A thì ghi chú vào `recording_app`).
- **Tắt** khử ồn / chế độ "phỏng vấn" / tự động tăng âm nếu máy có tùy chọn.
- Phòng yên tĩnh (tắt quạt, điều hòa, TV). Không đeo tai nghe có mic.
- Khoảng cách miệng đến mic: **20–30 cm**, giữ cố định cả buổi, mic không bị che.
- **Mono, giữ sample rate gốc của máy** (thường 44.1 hoặc 48 kHz). Đừng tự đổi sang 16 kHz, nhóm sẽ chuyển đổi một lần duy nhất.
- Đừng chỉnh sửa, cắt gọt, lọc ồn sau khi thu.

## 4. Đặt tên file (bắt buộc, không dấu, không khoảng trắng)

`<speaker_id>_<NEU|URG>_<sentence_id>.wav`

Ví dụ: `spk03_NEU_s07.wav`, `spk03_URG_s07.wav`

- `NEU` = NEUTRAL, `URG` = URGENT, `sentence_id` lấy từ cột đầu của `script_vi.csv` (s01 … s30).
- Giữ nguyên đuôi (`.wav` hoặc `.flac`).

## 5. Gửi file
- Nén tất cả vào một file `spkXX.zip` gồm: 60 file âm thanh + `speaker_metadata_template.csv` đã điền.
- Gửi cho Khải qua kênh nội bộ của nhóm. Không đăng lên nơi công khai.

## 6. Đồng ý sử dụng (consent)
Trong cột `consent_adult_yes` ghi `yes` nếu bạn xác nhận:
1. Bạn là người trưởng thành.
2. Bạn đồng ý cho nhóm dùng các bản ghi này **chỉ** cho nghiên cứu/prototype ToneBridge (huấn luyện và đánh giá), và nhóm không công khai âm thanh gốc.
3. Bạn có thể yêu cầu xóa bản ghi của mình bất cứ lúc nào.

Điền thêm ngày vào `consent_date`.

## 7. Kiểm tra nhanh trước khi gửi
- [ ] Đủ 60 file, tên đúng mẫu.
- [ ] Nghe thử 3 file NEUTRAL và 3 file URGENT, thấy rõ khác nhau.
- [ ] Không có tiếng ồn nền lớn, không bị rè/vỡ tiếng.
- [ ] Đã điền metadata và consent.
