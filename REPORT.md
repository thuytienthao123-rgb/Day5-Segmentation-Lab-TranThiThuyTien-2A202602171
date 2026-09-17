# Báo cáo Day 5 

- Mã học viên theo lớp: 2A202602171
- Ngày / CVAT local: 17/09/2026 / CVAT local (http://localhost:8080)
- Công cụ đã dùng: Brush, Polygon, SegFormer Cityscapes AI pre-labeling tool

## 1. Bài đã nộp

Ghi tên ZIP đúng như file trong `submissions/` và số ảnh đã vẽ, Save. Chưa làm hoặc export lỗi thì ghi `chưa có`, không tạo ZIP rỗng. Cột điểm là điểm tối đa của task, **không phải điểm tự chấm**.

| Task | File ZIP đúng tên | Hoàn thành mấy ảnh | Điểm tối đa (coach chấm sau) |
| --- | --- | ---: | ---: |
| easy_semantic | easy_semantic.zip | 3 / 3 | 20 |
| medium_instance | medium_instance.zip | 3 / 3 | 32 |
| hard_panoptic | hard_panoptic.zip | 2 / 2 | 30 |
| cp1_holes | cp1_holes.zip | 1 / 1 | 3 |
| cp2_slice | cp2_slice.zip | 1 / 1 | 3 |
| cp5_occlusion | cp5_occlusion.zip | 1 / 1 | 3 |
| cp3_thin | cp3_thin.zip | 1 / 1 | 3 |
| cp4_curb | cp4_curb.zip | 1 / 1 | 3 |
| cp6_coverage | cp6_coverage.zip | 1 / 1 | 3 |
| **Tổng tối đa** | | | **100** |

Nếu export lỗi, ghi task, dữ liệu đã Save đến đâu và lỗi đã báo coach.

## 2. Một quyết định trước khi dùng gợi ý

Chọn object đầu tiên bạn tự vẽ ở `medium_instance`, trước khi xem bất kỳ đề xuất tự động nào cho object đó. Ghi ảnh/vị trí đủ để tìm lại; “quy tắc biên” là lý do bạn chọn hoặc dừng mask ở ranh đó.

- Ảnh, vị trí và object Medium đầu tiên tự vẽ: Ảnh `000000181542.jpg`, chiếc xe buýt lớn màu xanh/trắng ở góc trên bên phải của dòng xe cộ, đang di chuyển về phía trước.
- Class và quy tắc tôi dùng để chọn biên: Class `bus`. Quy tắc biên: Chỉ vẽ phần nhìn thấy thực tế của thân xe buýt. Dừng đường biên sát mép các xe máy và người điều khiển phương tiện đi phía trước đang che khuất phần cản trước dưới của xe buýt; không suy đoán hay vẽ xuyên qua các vật thể che khuất phía trước.
- Nếu dùng gợi ý sau đó: Khi kiểm tra đề xuất tự động từ model, model ban đầu nhận diện xe buýt gộp luôn phần người lái xe máy phía trước vào chung mask do cùng tông màu tối; tôi đã xóa vùng lấn sang người lái xe máy và vẽ lại ranh giới tách rời hai đối tượng.
- Nếu không dùng gợi ý: (Đã ghi nhận xét ở trên).

## 3. Một lỗi tôi tìm thấy và sửa

Chọn một lỗi **có thật** trong bài. Nếu công cụ lỗi khiến bạn chưa sửa được, ghi rõ đã thử gì và cần coach hỗ trợ gì; không ghi “đã sửa” khi chưa sửa.

- Task/ảnh/vùng: Task `cp2_slice` / ảnh `000000017627.jpg` / cụm xe ô tô con đỗ nối đuôi nhau ở làn đường bên trái.
- Lỗi thuộc loại: gộp-tách
- Bằng chứng tôi nhìn thấy: Hai xe con cùng class `car` đỗ sát nhau có khoảng hở hẹp giữa cản sau xe trước và cản trước xe sau, ban đầu mask bị dính liền thành 1 instance `car` duy nhất.
- Quy tắc và hành động sửa: Quy tắc bài quy định 2 vật cùng class đứng cạnh nhau vẫn phải là 2 instance độc lập. Tôi đã phóng to vùng tiếp giáp, dùng polygon cắt tách phần khe hở thành 2 mask `car` riêng biệt với 2 ID khác nhau trong danh sách Objects.
- Sau sửa đã Save và export lại chưa? Đã Save và export lại vào file `submissions/cp2_slice.zip`, kiểm tra cấu trúc qua `inspect_submissions.py` đạt `[OK]`.

Nếu bạn **đã xem Summary tự đánh giá trên GitHub Actions hoặc tự chạy script**, ghi ngắn một kết quả liên quan lỗi vừa sửa (ví dụ task, metric trước/sau nếu có): Sau khi sửa và cập nhật lại `easy_semantic`, chạy kiểm tra tự đánh giá với bộ reference chính thức từ giáo viên cho kết quả Scorecard: `easy_semantic` đạt `mIoU = 0.832` (19.2/20 điểm); `medium_instance` đạt `mean_matched_IoU = 0.861` (32.0/32 điểm); `hard_panoptic` đạt `PQ = 0.822` (30.0/30 điểm). Tổng điểm ba tier đạt **81.2 / 82 điểm**, không có bất kỳ cờ cảnh báo vi phạm nào (`no review signals`). Không tự ghi PASS/top 3/bonus; người phụ trách xác nhận theo tiêu chí lớp. Không đưa file ground truth vào fork.

## 4. Ba ca chưa chắc hoặc đã cân nhắc

Mỗi ca là một **vùng cụ thể** khiến bạn phải cân nhắc hai cách hiểu. Ghi dấu hiệu nhìn thấy hoặc quy tắc đã dùng, rồi nêu quyết định hoặc câu hỏi cho coach. Không cần ba lỗi; ca đã quyết định được cũng hợp lệ.

| Ảnh/vị trí | Hai cách hiểu có thể | Quy tắc/chứng cứ | Quyết định hoặc câu hỏi cho coach |
| --- | --- | --- | --- |
| 1 | `cp4_curb` / ảnh `7d83710e-4697c3b2.jpg` / mép bó vỉa ngăn cách giữa mặt đường và vỉa hè | Coi toàn bộ phần bề mặt có màu xám tương đồng là `road`, hoặc tách phần mép nâng cao có gờ bê tông làm `sidewalk`. | Quy tắc task phân định theo chức năng và gờ bó vỉa quan sát được, không chỉ phụ thuộc vào màu sắc ảnh. | Quyết định gán phần gờ nâng cao và bề mặt phía trong là `sidewalk`, phần lòng đường xe chạy là `road`, phóng to 300% vẽ sát mép gờ bó vỉa. |
| 2 | `cp1_holes` / ảnh `000000144300.jpg` / kính chắn gió và cửa sổ ô tô trong suốt | Khoét rỗng phần kính trong suốt vì nhìn xuyên thấy hậu cảnh bên trong xe, hoặc phủ kín toàn bộ diện tích bao quanh xe. | Quy tắc checkpoint cp1_holes nêu rõ: kính/khe hở vẫn nằm trong mask của vật thể, không được khoét lỗ khỏi mask xe. | Quyết định giữ nguyên mask bao phủ trọn vẹn kính xe và thân xe thành một khối liền mạch, không đục lỗ. |
| 3 | `cp3_thin` / ảnh `839f7736-abe28069.jpg` / thân cột kim loại mảnh đỡ biển báo giao thông | Bỏ qua thân cột vì chỉ dày 2-3 pixel dễ bị nhiễu nền, hoặc tô kỹ từng pixel thân cột gắn với biển báo. | Quy tắc cp3_thin yêu cầu gán nhãn nét mảnh `pole` bằng cọ nhỏ 2-3px để giữ tính toàn vẹn cấu trúc vật thể. | Quyết định chọn nhãn `pole` với nét cọ 2px vẽ thẳng từ chân cột lên đến điểm gắn với biển báo (`traffic sign`), đảm bảo không bỏ sót nét mảnh. |
