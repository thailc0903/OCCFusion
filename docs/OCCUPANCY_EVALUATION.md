# Đánh giá occupancy bằng GT SurroundOcc

Job inference từ web gọi evaluator CPU này sau khi prediction hợp lệ, theo
[WEB_INFERENCE_JOBS.md](WEB_INFERENCE_JOBS.md); output evaluation riêng không
tự thay đổi approval.

**Tích hợp viewer/review:** [WEB_EVALUATION_REVIEW.md](WEB_EVALUATION_REVIEW.md)
mô tả ghép bằng token/hash và snapshot manifest v2. Bốn artifact inference và
evaluation gốc giữ nguyên; approval v1 không tự mở rộng phạm vi.

Áp dụng [PROJECT_SPEC.md](../PROJECT_SPEC.md). Entry point:
[`tools/evaluate_occupancy.py`](../tools/evaluate_occupancy.py). Evaluator chỉ đọc
prediction đã lưu và chạy NumPy trên CPU; không dựng model, nạp checkpoint hoặc
chạy inference. Inference vẫn dùng CLI riêng, theo
[INFERENCE_MINI.md](INFERENCE_MINI.md).

## Nguồn GT và cấu trúc thư mục

GT dùng **SurroundOcc nuScenes, 200×200×16, voxel 0.5 m**, không phải Occ3D,
OpenOccupancy, lidarseg hoặc occupancy tạo từ box. Nguồn:

- [SurroundOcc data.md](https://github.com/weiyithu/SurroundOcc/blob/main/docs/data.md):
  hàng `200x200x16 / val` để lấy GT validation đã tạo sẵn.
- [Config tạo GT chính thức](https://github.com/weiyithu/SurroundOcc/blob/main/tools/generate_occupancy_nuscenes/config.yaml):
  bounds `[-50,-50,-5,50,50,3]`, grid `[200,200,16]`, voxel size 0.5 m.
- [nuScenes devkit splits](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py):
  split được lấy bằng `create_splits_scenes()`, không suy ra từ scene index.

Dữ liệu đã có trong workspace này:

```text
data/
├── downloads/val.zip
└── nuscenes/
    ├── v1.0-mini/*.json
    ├── samples/LIDAR_TOP/<lidar_basename>.pcd.bin
    ├── samples/<camera, radar>/...
    ├── sweeps/...
    └── occ_samples/
        └── nuscenes_occ/
            └── samples/
                └── <lidar_basename>.pcd.bin.npy
```

`--gt-root` nhận thư mục đã giải nén, có thể là `data/nuscenes/occ_samples` hoặc
thư mục lá `.../nuscenes_occ/samples`. Công cụ lập index các file `.npy` bên dưới,
ghép **tên file LiDAR lấy từ nuScenes API + `.npy`**, rồi đối chiếu token/tên file
với metadata/report. Nhiều file cùng basename gây lỗi; không tự chọn một bản.
Không cần PKL trainval, không tự tải hoặc tạo GT thay thế. `--gt-root` không tồn
tại vẫn được ghi vào báo cáo với coverage thiếu GT.

Sparse NPY không chứa sample token, bounds hoặc mô tả hệ tọa độ riêng. Giao thức
trên được khai báo tường minh theo nguồn SurroundOcc, còn sample được liên kết
bằng nuScenes và basename. SHA-256 chứng minh nội dung file đã dùng; không phải
chứng nhận nguồn tải của bên phát hành. Không giả định prediction NPZ có token:
token prediction lấy từ `metadata.json` và `report.json`.

## Chọn split và frame

CLI đọc `v1.0-mini` bằng **nuscenes-devkit**. `mini_val` chính thức gồm
`scene-0103`, `scene-0916`; metadata mới còn lưu split lấy từ devkit. Evaluator
không tin split do CLI/client khai báo nếu nó khác API.

`tools/infer_nuscenes_mini.py` bổ sung `--scene-name`; chọn scene bằng tên chính
xác trong API. `--scene-name` và `--scene-index` loại trừ nhau. Lệnh cũ dùng
`--scene-index` vẫn hoạt động, và không truyền lựa chọn scene vẫn dùng index 0.
`--frame-index` tiếp tục bắt đầu từ 0 theo chuỗi `sample.next`.

Run `infer_mini_cuda_v2` cũ là `scene-0061`, frame 1, thuộc **mini_train**; bộ GT
hiện có không chứa file của sample đó. Không được dùng run này để báo mini_val.
Một số scene mini_train có GT trong archive val đầy đủ; sự hiện diện của GT cũng
không làm chúng trở thành mini_val.

## Lệnh chạy trong WSL

Không thêm dependencies. Dùng Python môi trường mô hình đã cài NumPy và
nuscenes-devkit; evaluator không cài thêm gì vào môi trường web:

```bash
cd '/mnt/d/New folder/OCCFusion'

# Coverage theo official split, chưa chạy inference hoặc tính IoU.
/home/hatis/.venvs/rav25/bin/python tools/evaluate_occupancy.py \
  --gt-root data/nuscenes/occ_samples --split mini_val --inventory-only \
  --out work_dirs/eval_mini_val_inventory/evaluation.json

# Chỉ tạo prediction cho một frame mini_val, giữ nguyên baseline FP32/grid.
/home/hatis/.venvs/rav25/bin/python tools/infer_nuscenes_mini.py \
  --scene-name scene-0103 --frame-index 1 --device cuda:0 \
  --out-dir work_dirs/infer_mini_val_scene0103_f1

# Đánh giá prediction đã lưu; chạy hoàn toàn trên CPU.
/home/hatis/.venvs/rav25/bin/python tools/evaluate_occupancy.py \
  --run-dir work_dirs/infer_mini_val_scene0103_f1 \
  --gt-root data/nuscenes/occ_samples/nuscenes_occ/samples \
  --split mini_val --verify-reference \
  --out work_dirs/eval_mini_val_scene0103_f1/evaluation.json
```

Các đường dẫn truyền tường minh cho evaluator tính từ thư mục làm việc hiện
tại; chạy các lệnh trên tại gốc repo. Output phải tên `evaluation.json` trong
thư mục riêng; công cụ từ chối ghi đè file đã có. Các lệnh đã chạy ở đây để lại
artifact, vì vậy khi chạy lại hãy chọn thư mục mới cho inference/evaluation.

Nhánh thiếu GT được kiểm tra với run cũ, chọn đúng mini_train:

```bash
/home/hatis/.venvs/rav25/bin/python tools/evaluate_occupancy.py \
  --run-dir work_dirs/infer_mini_cuda_v2 \
  --gt-root data/nuscenes/occ_samples/nuscenes_occ/samples \
  --split mini_train \
  --out work_dirs/eval_existing_missing_gt/evaluation.json
```

Exit code: **0** khi mọi cặp prediction/GT yêu cầu được đánh giá thành công;
**2** khi thiếu GT/sai split hoặc coverage inventory thiếu; **1** cho input/lỗi
khác. `--inventory-only` đầy đủ trả 0 nhưng không tạo metric. Lỗi đối số argparse
cũng trả 2, với thông báo cụ thể.

Chỉ sau khi frame đầu hợp lệ, có thể truyền nhiều `--run-dir` để pooled:

```bash
/home/hatis/.venvs/rav25/bin/python tools/evaluate_occupancy.py \
  --run-dir work_dirs/<prediction_frame_A> \
  --run-dir work_dirs/<prediction_frame_B> \
  --gt-root data/nuscenes/occ_samples --split mini_val --verify-reference \
  --out work_dirs/<evaluation_moi>/evaluation.json
```

Lệnh nhiều frame trên là mẫu, chưa được chạy với nhiều prediction thật trong
mốc này. Sample token lặp bị từ chối để tránh đếm hai lần. Nếu một phần thiếu
GT/lỗi, kết quả là `partial`; pooled chỉ gồm frame đã đánh giá, có count/coverage
rõ ràng và exit code khác 0. Không gọi đó là metric của toàn split.

## Dựng dense GT và kiểm chứng hình học

Đọc đúng NPY **N×4 `[index_x,index_y,index_z,label]`**, không hiểu tọa độ này là
điểm mét. Tái hiện hai bước trong repo:

1. [`LoadOccupancy`](../occfusion/loading.py): nhãn 0 ở **các hàng được liệt kê**
   đổi thành ignore 255. Nhãn 255 có sẵn vẫn được bỏ qua.
2. [`multiscale_supervision`](../occfusion/main.py), mức đầu ra và ratio `[1,1,1]`:
   tạo dense zeros `[200,200,16]`, rồi gán nhãn vào index. Ô không được liệt kê
   là **empty=0**, khác với hàng sparse có nhãn gốc 0.

Đọc dữ liệu với `allow_pickle=False`. Chặn NaN/Inf, số không nguyên, sai N×4,
index âm/ngoài grid, nhãn ngoài `0..16/255`, và voxel trùng có nhãn sau chuẩn hóa
xung đột. Hàng trùng tương đương được chấp nhận, ghi count và chỉ chiếm một ô.
Không có cắt index, đổi nhãn không biết, đổi grid hoặc rasterize box.

Prediction phải có đúng bounds/voxel/grid/class map và trục mảng **XYZ** của
giao thức. Bản này từ chối trục khác, không tự transpose/resample. Tâm voxel:

```text
xyz = (-50,-50,-5) + ((ix,iy,iz) + 0.5) * (0.5,0.5,0.5)
index (10,30,4) -> (-44.75,-34.75,-2.75) m
```

Kiểm tra metadata/report token, scene/frame/precision, dataset version và split
với API; keyframe `LIDAR_TOP`, tên file, timestamp, calibrated_sensor, ego_pose
và các ma trận LiDAR→ego/global phải khớp nuScenes. Transform của LiDAR keyframe
về hệ tham chiếu phải là identity. Không quay GT sang ego hoặc dùng trục chiều
cao Y. Shape/range nhãn, probability hữu hạn trong `[0,1]` và histogram prediction
phải khớp report trước khi tính metric. Confidence không tham gia trọng số IoU.

## Định nghĩa metric và pooling

Mask hợp lệ là **GT != 255**. Tất cả thống kê GT/prediction đều dùng mask này;
không dùng visibility/camera mask bổ sung. Confusion matrix có hàng là GT, cột
là prediction và lưu count nguyên.

- **Occupancy IoU**: occupied là nhãn khác 0. Intersection đếm vị trí cả GT và
  prediction occupied, dù sai lớp semantic. Union là `GT_occupied +
  predicted_occupied - intersection`. Empty không phải occupied.
- **IoU lớp c**, `c=1..16`: `TP_c / (GT_count_c + predicted_count_c - TP_c)`.
- **Semantic mIoU**: trung bình cộng **đủ 16 lớp 1..16**, không gồm empty 0 hoặc
  ignore 255. Union bằng 0 cho IoU **0**, vẫn đưa lớp đó vào mean, đúng
  `np.nan_to_num` trong [`EvalMetric`](../occfusion/evaluate.py). Vì vậy perfect
  prediction của frame chỉ có hai lớp cho mIoU `2/16`, không phải 100%.
- Occupancy union bằng 0 cũng cho 0 theo evaluator gốc. Nếu mọi voxel là ignore,
  valid voxel count là 0 và các tỷ số theo quy ước này là 0; đó không phải nhánh
  thiếu file GT. Thiếu GT cho metric **null**, không phải 0.

Nhiều frame: **cộng confusion/count intersection/union**, rồi mới chia và lấy
mean semantic. Không lấy trung bình IoU của các frame. Ví dụ một frame đúng 9/9
voxel occupied và một frame sai 0/1: pooled IoU là **9/10=0.9**, khác mean frame
IoU **0.5**. Evaluator gốc `.mean(0)` các count trước khi chia; tỷ lệ tương đương
với cộng count, không phải mean IoU.

Metric trong JSON là tỷ lệ `[0,1]`; nhân 100 khi trình bày phần trăm. Mỗi lớp
còn có intersection, union, số GT/prediction để kiểm toán lớp vắng mặt.

## Báo cáo và phạm vi approval

`evaluation.json` ghi protocol/version, source URL và hash mã loader/model-helper/
evaluator, command/môi trường CPU, split chính thức và coverage; từng frame ghi
sample/scene/frame/timestamp, LiDAR token/filename, GT path/hash, hash prediction
và cả bốn artifact nguồn, thống kê sparse/ignore, confusion và metric. Có count
frame yêu cầu/đã đánh giá cùng lý do từng frame không đánh giá được.

`--verify-reference` thực thi nguyên hàm `EvalMetric.evaluation_semantic` lấy qua
AST từ checkout hiện tại, trên cùng dense GT/prediction và NumPy CPU. Không chép
lại hàm để làm oracle và không import framework/dựng model. Báo cáo ghi hash mã
gốc và sai số tuyệt đối lớn nhất, yêu cầu `<=1e-12`, từng frame và pooled.

Bốn artifact inference không bị cập nhật để chèn metric. `report.json` cũ vẫn
ghi IoU không khả dụng do lượt inference không tải GT; metric mới ở evaluation
riêng. Approval/ZIP v1 đã duyệt chỉ gắn với bốn artifact cũ. Mốc viewer mới chỉ
thêm evaluation qua revision manifest v2 được duyệt theo tài liệu liên kết ở
đầu trang; quyết định review cũ không tự mở rộng phạm vi.

## Kiểm thử

```bash
MPLCONFIGDIR=/tmp/occfusion-matplotlib \
  /home/hatis/.venvs/rav25/bin/python -m unittest discover \
  -s tests -p 'test_occupancy_evaluation.py' -v

/home/hatis/.venvs/rav25/bin/python -m py_compile \
  tools/evaluate_occupancy.py tools/infer_nuscenes_mini.py
```

[`tests/test_occupancy_evaluation.py`](../tests/test_occupancy_evaluation.py) có
ca tính tay đúng/FP/FN/sai semantic/ignore/lớp vắng mặt/union-zero/pooled khác mean
frame. Thực thi trực tiếp hàm gốc từ ba file yêu cầu để đối chiếu loader, dense
supervision và metric; PyTorch của test dense chạy CPU. Fixture tổng hợp chỉ ở
thư mục tạm, không được dùng làm kết quả dữ liệu thật.

Các ca khác kiểm tra index/nhãn/tọa độ trùng xung đột, geometry/trục/class map,
token/tên LiDAR/timestamp/calibration/frame, thiếu GT, sample lặp, GT basename
mơ hồ, probability NaN và bảo vệ file gốc/output đã có. CLI chọn tên scene còn
được thử với scene order khác để phát hiện hardcode index.

## Kết quả thực tế ngày 2026-11-13

Các lệnh coverage, inference một frame, evaluation với reference, nhánh thiếu
GT và unittest ở trên đã được chạy bằng Python rav25 **3.10.12**. Không giảm
grid/ảnh/sweep hoặc thay checkpoint. Không mở rộng inference sang frame thứ hai.

| Kiểm tra | Kết quả |
| --- | --- |
| GT `.npy` dưới root đã giải nén | **6.019 file**, không có basename trùng |
| mini_val từ devkit | `scene-0103`: **40/40** file; `scene-0916`: **41/41** file |
| Unit tests | **25 passed**, 7.355 s; ca tính tay, ba hàm gốc và CLI compatibility |
| `py_compile` và `--help` hai CLI | Đạt |
| Inference đầu tiên | `scene-0103`, frame **1**, success, FP32, 49 file cảm biến |
| Sample token | `3950bd41f74548429c0f7700ff3d8269` |
| GT sparse | **32.545** hàng, không trùng; 88 hàng nhãn 0 → ignore |
| Dense GT | **639.912** voxel hợp lệ; **32.457** occupied; **607.455** ô empty không được liệt kê |
| Occupancy intersection / union | **17.945 / 44.567** |
| Occupancy IoU | **0.40265218659546304 = 40.265219%** |
| Semantic mIoU, đủ 16 lớp | **0.12446913030529963 = 12.446913%** |
| So với `EvalMetric` gốc trên dữ liệu thật | Đạt cả frame/pooled; sai số tuyệt đối lớn nhất **0.0** |
| Forward / pipeline của inference một frame | 14.433268 s / 18.584325 s, chưa phải benchmark |
| Bốn artifact run cũ và prediction mới | SHA-256 sau đánh giá khớp bộ nguồn; không bị sửa |

IoU từng lớp của frame đầu (phần trăm):

| ID | Lớp | Intersection | Union | IoU % |
| --- | --- | --- | --- | --- |
| 1 | barrier | 0 | 0 | 0 |
| 2 | bicycle | 0 | 23 | 0 |
| 3 | bus | 0 | 0 | 0 |
| 4 | car | 310 | 1180 | 26.271186 |
| 5 | construction_vehicle | 0 | 0 | 0 |
| 6 | motorcycle | 0 | 0 | 0 |
| 7 | pedestrian | 349 | 1511 | 23.097287 |
| 8 | traffic_cone | 0 | 30 | 0 |
| 9 | trailer | 0 | 0 | 0 |
| 10 | truck | 0 | 0 | 0 |
| 11 | driveable_surface | 6038 | 14475 | 41.713299 |
| 12 | other_flat | 0 | 0 | 0 |
| 13 | sidewalk | 2596 | 7882 | 32.935803 |
| 14 | terrain | 0 | 0 | 0 |
| 15 | manmade | 5638 | 19991 | 28.202691 |
| 16 | vegetation | 795 | 1694 | 46.930342 |

Prediction NPZ SHA-256:

```text
bee3c38b064df61ff0a63f62e298f903f001a942e600fc0eca1d6c503346d425
```

GT SHA-256:

```text
c6cba8ec0514dec08cc6ae6cd44647229564ec456b67934c4d16565215aba059
```

GT của frame này khớp nguyên byte với entry tương ứng trong `data/downloads/val.zip`
local (627.182.445 byte, SHA-256
`fb2b5a2f2c23290be83de85f0b7be46513b44d698c76863e6a006c47a206aeae`).
Không có hash của bên phát hành để xác thực độc lập archive này.

Bằng chứng local:

- [`evaluation.json` frame đầu](../work_dirs/eval_mini_val_scene0103_f1/evaluation.json):
  đầy đủ protocol, metric/count từng lớp, hashes và reference checks.
- [`data_provenance.json`](../work_dirs/eval_mini_val_scene0103_f1/data_provenance.json):
  archive/entry hash, xác nhận hai bộ artifact không đổi và hash evaluation.
- [`prediction report`](../work_dirs/infer_mini_val_scene0103_f1/report.json) và
  [`metadata`](../work_dirs/infer_mini_val_scene0103_f1/metadata.json).
- [`GT inventory`](../work_dirs/eval_mini_val_inventory/evaluation.json).
- [`evaluation thiếu GT`](../work_dirs/eval_existing_missing_gt/evaluation.json):
  run cũ scene-0061, mini_train, `missing_gt`, metric null, exit 2.

Đủ dữ liệu để đánh giá frame đầu. **80 frame mini_val còn lại chưa được chạy
inference/đánh giá**; mới xác nhận file GT có mặt. Pooled nhiều frame đã đối chiếu
trên các ca tính tay, chưa chạy trên nhiều prediction thật. Ở mốc evaluator ban
đầu chưa gộp vào viewer/review; kết quả tích hợp tiếp theo ghi riêng tại
WEB_EVALUATION_REVIEW.

## Giới hạn

Mini_val chỉ có hai scene; một frame không đại diện toàn mini_val hoặc full
nuScenes validation. Coverage inventory chỉ xác nhận file có mặt; không có nghĩa
mọi GT đã được kiểm tra nội dung hoặc mọi frame đã có prediction. Số lớp vắng mặt
trong một frame ảnh hưởng mạnh đến macro mIoU theo quy ước 16 lớp. Chưa xác minh
độc lập lịch sử dữ liệu huấn luyện của checkpoint để kết luận metric là kiểm tra
held-out không trùng dữ liệu. Đây là giao thức voxel IoU/mIoU OccFusion, không
phải RayIoU, đánh giá flow, hiệu chuẩn confidence hoặc benchmark latency.
