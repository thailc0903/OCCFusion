# Suy luận một frame nuScenes mini, không cần occupancy GT

Chạy từ web qua queue/worker riêng: [WEB_INFERENCE_JOBS.md](WEB_INFERENCE_JOBS.md).
CLI và baseline bên dưới được giữ nguyên.

Ngày thực thi: **2026-11-13**. Áp dụng [PROJECT_SPEC.md](../PROJECT_SPEC.md).
[BASELINE_READINESS.md](BASELINE_READINESS.md) là khảo sát trước khi môi trường,
dataset và checkpoint hiện tại được chuẩn bị; các kết quả thực thi mới nằm ở đây
và trong `work_dirs/infer_mini_*/report.json`.

**Kết quả: check-only CPU đạt và inference FP32 một frame đã thành công trên
GPU.** Artifact thật nằm tại
[`work_dirs/infer_mini_cuda_v2`](../work_dirs/infer_mini_cuda_v2). Không phát sinh
CUDA OOM trong lượt thành công; không cần đổi grid hoặc độ phân giải baseline.

Entry point là [tools/infer_nuscenes_mini.py](../tools/infer_nuscenes_mini.py), dùng
[configs/OccFusion_infer_mini.py](../configs/OccFusion_infer_mini.py). Model kế thừa
nguyên `configs/OccFusion.py`: camera + LiDAR + radar, ResNet-101/DCNv2, 17 lớp,
bounds `[-50,-50,-5,50,50,3]`, grid đầu ra `[200,200,16]`, voxel `[0.5,0.5,0.5]` m.
CLI kiểm tra config model bằng baseline và từ chối nếu khác. Không gọi
`init_weights()` để tải backbone pretrain trước khi nạp checkpoint đầy đủ.

## Các lệnh đã chạy

Chạy trong `/mnt/d/New folder/OCCFusion`, bằng Python rav25:

```bash
/home/hatis/.venvs/rav25/bin/python -m py_compile \
  configs/OccFusion_infer_mini.py tools/infer_nuscenes_mini.py

/home/hatis/.venvs/rav25/bin/python tools/infer_nuscenes_mini.py --help

/home/hatis/.venvs/rav25/bin/python tools/infer_nuscenes_mini.py \
  --scene-index 0 --frame-index 1 --check-only \
  --out-dir work_dirs/infer_mini_check

/home/hatis/.venvs/rav25/bin/python tools/infer_nuscenes_mini.py \
  --scene-index 0 --frame-index 1 --device cuda:0 \
  --out-dir work_dirs/infer_mini_cuda

/home/hatis/.venvs/rav25/bin/python tools/infer_nuscenes_mini.py \
  --scene-index 0 --frame-index 1 --device cuda:0 \
  --out-dir work_dirs/infer_mini_cuda_v2
```

Đường dẫn mặc định được tính từ gốc repo, kể cả khi gọi CLI từ thư mục khác:
dataset `data/nuscenes`, version `v1.0-mini`, checkpoint
`ckpt/Cam_Lidar_Radar_Atten3D.pth`. Có thể chỉ định `--config`, `--data-root`,
`--checkpoint`. `--out-dir` là bắt buộc; dùng thư mục mới cho từng lần chạy.
CLI từ chối ghi đè artifact cũ để một lần lỗi không bị nhầm với kết quả trước đó.

`--scene-index` là chỉ số từ 0 trong `NuScenes.scene`; `--frame-index` là chỉ số
từ 0 khi đi từ `scene.first_sample_token` qua `sample.next`. Giá trị mặc định
đều là 0, không tự đổi sang frame khác khi đầu vào không hợp lệ.

CLI hiện hỗ trợ `--scene-name scene-0103` thay cho `--scene-index`; hai lựa chọn
loại trừ nhau. Lệnh cũ vẫn tương thích. Metadata/report của run mới lưu split
mini_train/mini_val lấy từ nuscenes-devkit. Đánh giá GT SurroundOcc thực hiện
riêng theo [OCCUPANCY_EVALUATION.md](OCCUPANCY_EVALUATION.md), không sửa metric
trong bốn artifact inference hoặc mở rộng phạm vi approval cũ.

`--check-only` luôn dùng CPU: kiểm tra bảng JSON và file cảm biến, dựng model,
đối chiếu/nạp checkpoint strict, đọc ảnh/điểm và chạy preprocessor. **Không gọi
forward**, không tạo `occupancy.npz` hoặc `bev.png`.

Trong phiên Codex này, sandbox không cho thấy GPU: `torch.cuda.is_available()`
trả `False`, `nvidia-smi` báo GPU access blocked. Kiểm tra ngoài sandbox trả
`True` và đúng RTX 3050 Laptop; hai lệnh CUDA trên được chạy ngoài sandbox.
CLI không tự chuyển sang CPU nếu CUDA không khả dụng.

## Dữ liệu, calibration và sweep

CLI xác nhận checkpoint và đủ 13 bảng JSON của nuScenes mini:
`attribute`, `calibrated_sensor`, `category`, `ego_pose`, `instance`, `log`,
`map`, `sample`, `sample_annotation`, `sample_data`, `scene`, `sensor`,
`visibility`. `nuscenes-devkit` tự đọc các bảng chuẩn và dựng chỉ mục. Dữ liệu
annotation có trong API nhưng không được dùng làm đầu vào mạng hoặc metric.
Không yêu cầu PKL trainval, `occ_samples`, Occ3D, lidarseg hay mapping nhãn GT.

Thứ tự camera theo converter của repo:

```text
CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT,
CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
```

Cảm biến điểm: `LIDAR_TOP`, `RADAR_FRONT`, `RADAR_FRONT_LEFT`,
`RADAR_FRONT_RIGHT`, `RADAR_BACK_LEFT`, `RADAR_BACK_RIGHT`.
Mọi file keyframe của 12 cảm biến đều được kiểm tra, cùng tất cả sweep được dùng.
File thiếu/rỗng được báo kèm channel và đường dẫn; không tải hoặc tạo file thay thế.

Hệ tham chiếu là **LIDAR_TOP tại timestamp của LiDAR keyframe**, đơn vị mét.
Với ma trận tác động lên vector cột:

```text
T_ref_from_sensor = inverse(T_global_from_ref_lidar)
                    @ T_global_from_ego_at_sensor_time
                    @ T_ego_from_sensor
```

Mỗi record dùng đúng `calibrated_sensor_token`, `ego_pose_token` và timestamp
của nó, kể cả camera. Camera sử dụng
`lidar2img = intrinsic_4x4 @ inverse(T_ref_from_camera)`. Radar quay cả vận tốc
ego-compensated về LiDAR; chỉ bù pose của ego, không ngoại suy chuyển động vật thể.
Metadata giữ toàn bộ ma trận, timestamp và độ trễ so với LiDAR tham chiếu.

LiDAR dùng keyframe và tối đa **10 sweep trước**, radar tối đa **6 record mỗi
channel**, gồm record mới nhất đủ điều kiện. Chỉ đi qua `sample_data.prev`,
timestamp sweep phải `<=` LiDAR tham chiếu; kiểm tra chu trình, timestamp không
giảm và biên scene. Radar keyframe nằm sau mốc tham chiếu bị loại, rồi tìm record
trước đó. Nếu một radar không có record quá khứ thì dừng và yêu cầu chọn frame
khác. Nếu có ít hơn số sweep mong muốn thì dùng số thật, ghi warning và số lượng,
không lặp keyframe để bù.

Frame `0` của `scene-0061` không có lịch sử cho ba radar có timestamp sau LiDAR:
`RADAR_FRONT`, `RADAR_FRONT_LEFT`, `RADAR_BACK_LEFT`. Vì vậy lệnh mẫu chỉ định
**frame 1**; đây là lựa chọn tường minh, không phải fallback ngầm.

Tái sử dụng `LoadPointsFromFile`, loader radar, image loader, `PointsRangeFilter`,
`Custom3DPack` và `OccFusionDataPreprocessor`. Có hai điều chỉnh được công bố:

- Tích lũy LiDAR bằng adapter áp dụng `xyz @ R.T + t`. Trong MMDetection3D 1.4.0
  được cài, `LoadPointsFromMultiSweeps` áp dụng phép quay rồi trừ translation của
  ma trận nghịch đảo; cách đó không tương đương phép nghịch đảo SE(3) tổng quát.
  Adapter dùng trực tiếp ma trận sensor→LiDAR đã tính bằng API, giữ bỏ điểm gần
  trong hình vuông 1 m của sweep, intensity và age tính bằng giây. Không gọi
  loader sweeps đó và không sửa thư viện đã cài.
- Không lặp LiDAR keyframe khi thiếu lịch sử dù val pipeline cũ đặt
  `pad_empty_sweeps=True`. Số sweep yêu cầu, kích thước ảnh và cấu hình model/grid
  baseline vẫn giữ nguyên. Time feature radar giữ quy ước baseline: tuổi so với
  record mới nhất **được dùng của từng radar**; `age_to_reference_s` thực tế nằm
  riêng trong metadata, không được gọi là temporal occupancy flow.

Ảnh giữ nguyên độ phân giải file, BGR, trừ mean `[103.530,116.280,123.675]`, chia
std `[1,1,1]`, pad theo bội 32. Loader cũ ghi `scale_factor=0.2` dù không resize;
CLI không pack/dùng trường đó và ghi scale thực tế `1.0`. Preprocessor của repo
chạy trên CPU, giữ nguyên thứ tự điểm, lấy tối đa 35 điểm/voxel và cách tạo 8/11
feature LiDAR/radar; batch sau đó được chuyển sang GPU. Cách này tránh việc
preprocessor cấp phát `torch.zeros` trên CPU trong vòng lặp với điểm trên GPU.

Sweep không có điểm sau devkit filters được ghi warning và count 0. LiDAR hoặc
radar tổng hợp rỗng sau lọc range sẽ dừng trước voxelization; không thay bằng
tensor giả. NaN/Inf trong point cloud, feature, ảnh, projection hoặc logits là lỗi.

## Checkpoint và tensor forward

Checkpoint thực tế có kích thước **1.785.062.775 byte**, SHA-256:

```text
d0e70f1cb172cb94cdd17075d13e102014b6c62cb753f0997e8f1c01932b8bcf
```

Đối chiếu toàn bộ `state_dict` trên CPU, báo các danh sách `missing_keys`,
`unexpected_keys`, `shape_mismatches` và giá trị không phải tensor vào console
và `report.json`. Chỉ hỗ trợ bỏ prefix DDP `module.` khi tất cả key đều có prefix
đó. Bất kỳ sai khác nào đều dừng; chỉ gọi `load_state_dict(..., strict=True)`
khi toàn bộ khớp. Không lọc bỏ trọng số backbone/fusion/head để chạy tiếp.

Đã đọc mã API cục bộ của MMDetection3D 1.4.0:
`Base3DSegmentor.forward(inputs, data_samples, mode='tensor')` gọi `_forward`.
CLI kiểm tra signature của framework, dùng `model.eval()`,
`torch.inference_mode()`, batch 1 và FP32, không bật autocast. Không gọi
`predict()`, `test_step()` hoặc evaluator vì `OccFusion.predict()` vẫn đọc GT.

Lượt CUDA đầu (`infer_mini_cuda`) dừng ở `forward` với lỗi CPU/CUDA khác thiết bị.
EfficientViT tạo cache `ab` trong `train(False)` dưới dạng tensor thường;
`model.to(cuda)` không chuyển cache này. Entry point hiện gọi lại `eval()` sau
khi chuyển model sang CUDA để tạo cache từ đúng trọng số trên đúng thiết bị.
Không sửa kiến trúc, checkpoint, sampling rate, grid hay độ phân giải để xử lý lỗi.
Traceback lần lỗi được giữ trong `work_dirs/infer_mini_cuda/report.json`.

## Hợp đồng đầu ra

Mỗi lần chạy ghi `report.json` (trạng thái, lỗi/giai đoạn nếu có, phiên bản thư
viện, so sánh checkpoint, timings, VRAM), `resolved_config.py`, và
`metadata.json` khi metadata cảm biến đã hợp lệ.

Chỉ khi forward trả logits hữu hạn có shape `[1,200,200,16,17]` mới xuất:

| File | Nội dung |
| --- | --- |
| `occupancy.npz` | `labels`: `uint8 [X,Y,Z]`, lớp `0..16`; `max_softmax_probability`: `float32 [X,Y,Z]`, softmax theo 17 lớp rồi lấy max |
| `metadata.json` | Sample/scene/timestamp, hệ tọa độ và transforms, bounds/grid/voxel size, camera order, sweeps/counts, preprocessing, class map/palette, config đã resolve, hash checkpoint/config/code, môi trường và kết quả đo |
| `bev.png` | Mỗi cột `(x,y)` lấy lớp của voxel không-free cao nhất theo z; cột toàn free giữ 0. Cột ảnh đi theo x tăng, hàng ảnh theo y giảm. Ảnh gốc `200×200`, không nội suy |

Grid có thứ tự `[x,y,z]`, flatten C (z thay đổi nhanh nhất), tâm voxel là
`bounds_min + (index + 0.5) * voxel_size`. Class 0 là **free**, không dùng chuỗi
`IoU` trong metadata evaluator làm tên lớp. Không tạo visibility mask;
metadata ghi `visibility_mask: null`. Max softmax probability **chưa phải độ bất
định đã hiệu chuẩn**. Không có GT occupancy thì IoU/mIoU là `null`, lý do không
khả dụng được ghi rõ, không có số liệu đánh giá thay thế.

Ví dụ đọc kết quả thành công:

```python
import json
import numpy as np

output_dir = 'work_dirs/infer_mini_cuda_v2'
with np.load(f'{output_dir}/occupancy.npz') as result:
    labels = result['labels']
    max_probability = result['max_softmax_probability']
with open(f'{output_dir}/metadata.json', encoding='utf-8') as stream:
    metadata = json.load(stream)
```

## Đo thời gian, bộ nhớ và lỗi

`preprocessing` bao gồm I/O ảnh/điểm, transform, packing, CPU preprocessor và
chuyển batch sang thiết bị. `forward` chỉ đo tensor forward. `postprocessing_export`
bao gồm chuyển logits về CPU, softmax FP32, argmax và ghi NPZ/PNG. `pipeline_total`
bao trùm ba phần đó, loại trừ import, đọc/index JSON, hash/nạp checkpoint,
dựng/chuyển model và ghi metadata/report cuối cùng; các phần startup có timing
riêng và `process_until_report` để tránh nhầm với tổng thời gian khởi động CLI.

CUDA được synchronize sau device setup, sau chuyển batch, trước/sau forward và
ở nhánh lỗi. Reset peak stats trước khi đưa model lên GPU; lưu
`max_memory_allocated`/`max_memory_reserved` dạng byte, phạm vi gồm model setup và
một frame. Precision và cờ TF32 được ghi trong report. Trên CPU/check-only, peak
CUDA là `null` vì không dùng CUDA. Đây là **phép đo một frame không warm-up, chưa
phải benchmark ổn định**.

Nếu CUDA OOM, exit code khác 0, report có `status='cuda_out_of_memory'`,
`error.stage`, thông báo/traceback, thời gian đến lỗi và peak allocator. Không có
`occupancy.npz`/`bev.png` từ forward lỗi; không thử lại bằng grid/ảnh thấp hơn.
Không hoàn tất forward thì không có latency forward thành công hoặc pipeline
hoàn chỉnh để công bố.

## Bằng chứng kiểm tra

Python **3.10.12**, PyTorch **2.1.1+cu121**, torchvision **0.16.1+cu121**,
MMEngine **0.10.5**, MMCV **2.1.0**, MMDetection **3.3.0**, MMDetection3D **1.4.0**,
nuscenes-devkit **1.1.11**. GPU được nhận là NVIDIA GeForce RTX 3050 Laptop.

Lệnh CUDA cuối (`infer_mini_cuda_v2`) exit 0, `status='success'`:

| Số đo một frame | Kết quả thực tế |
| --- | --- |
| Precision / batch | FP32 / 1; autocast tắt; cuDNN cho phép TF32 theo baseline |
| Tiền xử lý và chuyển batch | `2.368699 s` |
| Tensor forward | `12.921675 s` |
| CPU softmax và xuất NPZ/PNG | `0.433491 s` |
| Tổng pipeline frame | `15.723976 s` |
| Tiến trình đến trước ghi metadata/report cuối | `64.774631 s` |
| Peak allocated (PyTorch) | `5,716,246,016 byte` = `5.323669 GiB` |
| Peak reserved (PyTorch) | `6,245,318,656 byte` = `5.816406 GiB` |
| Bộ nhớ GPU do CUDA báo | `4,294,443,008 byte` (xấp xỉ 4 GiB) |
| Logits thật | `[1,200,200,16,17]`, hữu hạn |
| Max softmax probability | Min `0.2468943894`, max `1.0` |

Peak allocator **cao hơn tổng bộ nhớ GPU vật lý được báo** trong môi trường này.
Đây là số đo từ `torch.cuda` cho allocation/reservation; chưa đo residency/paging
của driver, nên không kết luận baseline nằm gọn trong 4 GiB VRAM vật lý. Kết quả
không bảo đảm một máy/driver khác với cùng dung lượng sẽ chạy được. Nhánh báo cáo
CUDA OOM đã triển khai nhưng chưa gặp lỗi cấp phát OOM thực trong lần thử này.

Đã mở lại `occupancy.npz` để xác nhận đủ hai array, đúng shape/dtype, nhãn
`0..16`, probability hữu hạn trong `[1/17,1]`; kiểm tra metadata khớp hash script,
grid, voxel size và thứ tự cảm biến. Đã kiểm tra BEV RGB `200×200` và xem ảnh
trực tiếp. Output thật có 640.000 voxel, trong đó 44.856 voxel dự đoán không-free;
đây là thống kê prediction, không phải metric chất lượng. Các lần check-only và
lỗi trước đó không có `occupancy.npz`/`bev.png`.

Không còn chặn đường inference một frame đã thử. IoU/mIoU vẫn không khả dụng
do không có GT occupancy; chưa có benchmark nhiều frame, hiệu chuẩn độ bất định,
hoặc phép đo VRAM residency độc lập.

CPU check-only exit 0, `check_only_passed`:

- Scene `scene-0061`, sample `39586f9d59004284a7114a68825e8eec`, frame index `1`;
  timestamp LiDAR `1532402928147847` microsecond.
- Kiểm tra 50 file cảm biến, dùng 10 LiDAR sweep trước và 6 record/radar.
- Khớp toàn bộ **1.814 tensor**, **148.726.103 tham số**; missing/unexpected/shape
  mismatch đều rỗng, strict load thành công.
- Ảnh sau padding `[1,6,3,928,1600]`; 274.392 điểm LiDAR và 846 điểm radar sau
  lọc range; 4.384 voxel LiDAR và 398 voxel radar.
- Tiền xử lý CPU `3.150 s`; thời gian tiến trình đến khi ghi report `51.331 s`.
  Không gọi forward, không có artifact prediction.

Bộ kiểm tra bổ sung đã chạy bằng:

```bash
/home/hatis/.venvs/rav25/bin/python /tmp/occfusion_mini_checks.py
```

Script kiểm tra tạm không phải entry point sản phẩm. Kết quả 12 ca được lưu tại
`work_dirs/infer_mini_validation.json`: 52 ma trận sensor/sweep đối chiếu với
chuỗi transform của devkit (sai số tuyệt đối tối đa khoảng `2.3e-13`), chỉ sweep
quá khứ, từ chối frame 0 thiếu radar quá khứ, báo tất cả đường dẫn file bị thiếu,
checkpoint không tồn tại, sweep thiếu không lặp/pad, strict checkpoint
match/missing/unexpected/shape mismatch, tiền xử lý dữ liệu thật khi cấm
`np.load` đọc GT, từ chối LiDAR/radar rỗng. Các ca lỗi mô phỏng bằng monkeypatch
hoặc slice rỗng chỉ kiểm tra nhánh từ chối; không thay file dataset gốc và không
forward bằng dữ liệu thay thế.

`py_compile` và `--help` đạt. Kiểm tra whitespace riêng ba file mới không có lỗi.
`git diff --check` toàn checkout báo whitespace/CRLF trong các thay đổi có sẵn
ngoài nhiệm vụ; không chỉnh các file đó.
