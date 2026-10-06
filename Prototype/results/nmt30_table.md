| # | Source (vi) | fp32 | INT8 arm64 | INT8 enc + fp32 dec | INT8 beam4 | flags (check v1) | error belongs to |
|---|---|---|---|---|---|---|---|
| 1 | Bạn có bị đau ngực không? | Do you have chest pains? | Do you have chest pains? | Do you have chest pains? | Do you have chest pains? | - | none |
| 2 | Bạn đau ở đâu? | Where do you hurt? | Where do you hurt? | Where do you hurt? | Where do you hurt? | - | none |
| 3 | Bạn bị đau này bao lâu rồi? | How long have you been in this pain? | How long have you been in this pain? | How long have you been in this pain? | How long have you been in this pain? | - | none |
| 4 | Bạn có dị ứng với loại thuốc nào không? | Are you allergic to any drugs? | Are you allergic to any drugs? | Are you allergic to any drugs? | Are you allergic to any drugs? | - | none |
| 5 | Tôi bị dị ứng với penicillin. | I'm allergic to penicillin. | I'm allergic to penicillin. | I'm allergic to penicillin. | I'm allergic to penicillin. | - | none |
| 6 | Uống hai viên paracetamol mỗi sáu giờ. | Take two paracetamols every six hours. | Take two paracetamols every six hours. | Take two paracetamols every six hours. | Take two paracetamols every six hours. | fp32/greedy:SAFETY; int8/greedy:SAFETY; int8enc_fp32dec/greedy:SAFETY; int8/beam4:SAFETY; fp32/beam4:SAFETY | both |
| 7 | Không uống thuốc này cùng với rượu bia. | Don't take this medicine with the wine. | Don't take this with the alcohol. | Don't take this medicine with the wine. | Don't take it with alcohol. | - | none |
| 8 | Cơn đau có đang nặng hơn không? | Is the pain getting worse? | Is the pain getting worse? | Is the pain getting worse? | Is the pain getting worse? | - | none |
| 9 | Bạn có thở bình thường được không? | Can you breathe normally? | Can you breathe normally? | Can you breathe normally? | Can you breathe normally? | - | none |
| 10 | Tôi không thở được. | I can't breathe. | I can't breathe. | I can't breathe. | I can't breathe. | - | none |
| 11 | Bệnh nhân chưa ăn gì từ sáng nay. | Patient hasn't eaten since this morning. | Patient hasn't eaten since this morning. | Patient hasn't eaten since this morning. | Patient hasn't eaten since this morning. | - | none |
| 12 | Xin cho biết tên và ngày sinh của bạn. | Please contact your name and date of birth. | Please state your name and date of birth. | Please state your name and date of birth. | Please state your name and date of birth. | - | none |
| 13 | Bạn có bị sốt hoặc ho không? | Do you have a fever or a cough? | Do you have a fever or a cough? | Do you have a fever or a cough? | Do you have a fever or a cough? | - | none |
| 14 | Tôi thấy chóng mặt và tức ngực. | I felt dizzy and hurt by my chest. | I felt dizzy and hurt by my chest. | I felt dizzy and hurt by my chest. | I felt dizzy and hurt by my chest. | - | none |
| 15 | Trước đây bạn đã từng phẫu thuật chưa? | Have you had surgery before? | Have you had surgery before? | Have you had surgery before? | Have you had surgery before? | - | none |
| 16 | Bạn có đang dùng thuốc chống đông máu không? | Are you on blood thinners? | Are you on blood thinners? | Are you on anti-septic medication? | Are you on blood thinners? | int8enc_fp32dec/greedy:SAFETY | none |
| 17 | Ngừng uống ibuprofen và uống nhiều nước hơn. | Stop drinking ibuprofen and drink more water. | Stop drinking medrine and drink more water. | Stop drinking medrine and drink more water. | Stop drinking ibuprofen and drink more water. | int8/greedy:SAFETY; int8enc_fp32dec/greedy:SAFETY | int8_only |
| 18 | Huyết áp rất cao. | High blood pressure. | High blood pressure. | High blood pressure. | High blood pressure. | fp32/greedy:CONFIRM; int8/greedy:CONFIRM; int8enc_fp32dec/greedy:CONFIRM; int8/beam4:CONFIRM; fp32/beam4:CONFIRM | both |
| 19 | Hãy chỉ cho tôi chỗ bắt đầu đau. | Show me where it started. | Show me where it started. | Show me where it started. | Show me where it started. | - | none |
| 20 | Tôi bị tiểu đường mười năm rồi. | I've been diabetes for ten years. | I've been diabetes for ten years. | I've been diabetes for ten years. | I've had diabetes for ten years. | - | none |
| 21 | Bạn có thấy tê ở cánh tay không? | Do you feel numb in your arm? | Do you feel numb in your arm? | Do you feel numb in your arm? | Do you feel numb in your arm? | - | none |
| 22 | Chúng tôi sẽ tiêm cho bạn năm miligam. | We'll give you five millimetres. | We'll give you five millimetres. | We'll give you five millimetres. | We're going to inject you with five milligrams. | fp32/greedy:SAFETY; int8/greedy:SAFETY; int8enc_fp32dec/greedy:SAFETY | both |
| 23 | Xin hãy nằm xuống và giữ bình tĩnh. | Please lie down and remain calm. | Please lie down and remain calm. | Please lie down and remain calm. | Please lie down and remain calm. | - | none |
| 24 | Cơn đau có lan ra lưng hoặc hàm không? | Does the pain spread to the back or to the jaw? | Does the pain spread to the back or to the jaw? | Does the pain spread to the back or to the jaw? | Does the pain spread to the back or to the jaw? | - | none |
| 25 | Tôi đau bụng và buồn nôn. | I'm sick of stomach pains and nausea. | I have stomach pains and nausea. | I'm sick of stomach and nausea. | I have stomach pains and nausea. | - | none |
| 26 | Hôm nay bạn đã nôn mấy lần? | How many times did you throw up today? | How many times have you puked today? | How many times did you throw up today? | How many times did you throw up today? | - | none |
| 27 | Cô ấy không có dị ứng thuốc nào được biết. | She has no allergies to know about. | She has no allergies to know about. | She has no allergies to know about. | She's not allergic to drugs. | - | none |
| 28 | Uống một viên sau bữa ăn, hai lần một ngày. | Take one after dinner, twice a day. | Take one after dinner, twice a day. | Take one after dinner, twice a day. | Take one after dinner, twice a day. | fp32/greedy:SAFETY; int8/greedy:SAFETY; int8enc_fp32dec/greedy:SAFETY; int8/beam4:SAFETY; fp32/beam4:SAFETY | both |
| 29 | Hãy gọi bác sĩ nếu máu không ngừng chảy. | Call a doctor if the blood won't stop flowing. | Call the doctor if the blood doesn't stop bleeding. | Call the doctor if the blood won't stop bleeding. | Call the doctor if the blood won't stop bleeding. | - | none |
| 30 | Tôi cần giúp đỡ ngay bây giờ. | I need help right now. | I need help right now. | I need help right now. | I need help right now. | - | none |

Flagged sentences per build (critical or confirm): {'fp32|greedy': 4, 'int8|greedy': 5, 'int8enc_fp32dec|greedy': 6, 'int8|beam4': 3, 'fp32|beam4': 3}
Attribution fp32-vs-INT8 greedy: {'both': 4, 'int8_only': 1, 'fp32_only': 0, 'none': 25}
