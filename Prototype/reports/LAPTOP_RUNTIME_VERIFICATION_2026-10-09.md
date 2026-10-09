# Laptop runtime verification — 09/10/2026

Phạm vi: CPU laptop Windows, runtime/audio và hợp đồng bàn giao text. Không đánh giá medical accuracy hoặc Android/NPU.

| Check | Kết quả |
|---|---|
| Full pytest suite | **114 passed, 3 skipped**, 46,07 giây |
| Test skipped | FLEURS/golden data thiếu; GTCRN thiếu; noise test data thiếu |
| Dependencies | `pip check`: no broken requirements |
| Asset manifest | 401 files khớp size/SHA256 |
| ASR VI / EN / KO | Adapter thật chạy trên model-provided samples, transcript không rỗng |
| TTS VI / EN / KO | Adapter thật tạo WAV finite, không rỗng; chưa có human listening assessment |
| Offline component smoke | 0 Python network attempts, 0 OS connections được ghi nhận |
| New direction without text adapter | EN→KO trả ABSTAIN/error, không tạo WAV |
| Microphone/playback hardware | Chưa xác minh; microphone boundary test dùng thiết bị giả |

## VI→EN repeated-sample stability smoke

Preset `configs/laptop_vi-en.json`: urgency disabled/UNKNOWN, denoise OFF, 2 CPU threads/model, 3 warm-up không tính timing. **3 WAV độc lập, lặp 34 lần = 102 lượt**, 102 OK/SPEAK, 0 error/crash.

| Measure | Kết quả |
|---|---|
| Endpoint → first TTS sample p50 | 514 ms |
| Endpoint → first TTS sample p95 | 791 ms |
| Peak RSS | 895 MiB |
| RSS first / last measured turn | 871 / 895 MiB |
| Network evidence | 0 Python attempts / 0 sampled OS connections |

Đây không phải holdout 100 câu độc lập. First sample không phải first audible. RAM tăng khoảng 24 MiB; chưa chứng minh leak hoặc stability dài hạn. Metrics gốc được giữ cục bộ trong `results/laptop_current/`, không đẩy audio/content logs lên branch.

Assets hiện phục hồi khác bản export cục bộ đã dùng trong audit trước. Không kế thừa kết luận accuracy/latency cũ cho assets mới. Manifest SHA256 đã kiểm tra: `d77056b2df8a04de49892b8e75bfab170fb95117e431ce69f5027c823bd0280c`.

## Còn phải làm khi nghiệm thu

- NMT/safety EN→VI, EN→KO, KO→EN và cải tiến checker VI→EN: người phụ trách text bàn giao.
- Tích hợp E2E bốn chiều: runtime owner.
- Corpus thực, noise slices, phát âm tên thuốc/số/phủ định, microphone/loa và 100 câu độc lập mỗi chiều: chưa xác minh.
- Model/license freeze: chưa hoàn tất; TTS VI/KO laptop dùng Supertonic thay MeloTTS candidate trong proposal.
