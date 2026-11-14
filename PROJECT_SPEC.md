# RAV25 Occupancy Lab

## Mục tiêu
Xây lab phân tích dữ liệu xe tự hành offline từ camera–LiDAR–radar.
Kỹ sư dùng công cụ để chạy mô hình, xem occupancy, so sánh cấu hình
và đánh giá lỗi. Công cụ không điều khiển xe.

## Bản cơ bản
1. Chọn và phát lại frame/sequence dữ liệu thật.
2. Chạy checkpoint OccFusion qua worker GPU.
3. Hiển thị BEV, occupancy 3D, ảnh camera, point cloud và detection boxes.
4. Đồng bộ các lớp hiển thị theo timestamp và hệ tọa độ.
5. Tính IoU/mIoU occupancy và detection mAP khi có GT tương ứng.
6. Đo latency của model và toàn pipeline.
7. Có vai trò Engineer và Reviewer.
8. Lưu nhận xét, trạng thái duyệt và lịch sử thí nghiệm.
9. Ẩn danh thông tin nhận dạng trong dữ liệu hiển thị/xuất theo yêu cầu.

## Bản nâng cao
1. So sánh C+R, C+L, C+L+R trên cùng giao thức đánh giá.
2. Bổ sung temporal fusion và occupancy flow.
3. Định nghĩa flow, valid mask, EPE và cách tính ADE/FDE.
4. Xây kiểm chứng hình học–động học độc lập với mô hình.
5. Hiệu chuẩn và đánh giá độ bất định.
6. Phân tích FP/FN theo điều kiện vận hành và khai thác ca khó.
7. Tạo vùng rủi ro dựa trên occupancy và quỹ đạo tham chiếu.
8. Đo đánh đổi chất lượng–latency–VRAM–chi phí GPU.

## Quy tắc triển khai
- Ghi rõ nguồn dữ liệu, config, checkpoint và phiên bản mã nguồn.
- Phân biệt ground truth, prediction và dữ liệu mô phỏng.
- Không tạo số liệu đánh giá giả.
- Kết quả lưu sẵn phải được ghi rõ là kết quả phát lại.
- Tách inference khỏi evaluation; inference không bắt buộc có GT.
- Metadata phải mô tả hệ tọa độ, timestamp, voxel size,
  class map, visibility mask và đơn vị flow.
- Phát lại các frame độc lập chưa được coi là temporal fusion.
- Mỗi mốc phải có hướng dẫn chạy và bằng chứng kiểm tra.
