# Occupancy forecasting v1 — sequence dataset và persistence CPU

Nhánh nghiên cứu offline tiếp tục checkout WSL f3270651, sau nghiệm thu bản cơ
bản 175 test. Đây là **persistence baseline**, chưa có temporal model học được
hoặc learned 4D flow. Không sửa viewer, model/config/profile/checkpoint,
database, benchmark, inference/evaluation hoặc approval cũ. Không bật worker,
không xử lý ba job queued, không tải dữ liệu/weights hoặc huấn luyện.

## Code và dữ liệu

- Entry point: [tools/forecast_persistence.py](../tools/forecast_persistence.py).
- [tools/forecasting](../tools/forecasting): protocol, sequence dataset,
  full SE(3), persistence, evaluator và kiểm chứng độc lập.
- [tests/test_forecasting.py](../tests/test_forecasting.py): fixture CPU,
  không dùng fixture làm forecast/prediction thật.
- NuScenes mini đọc bằng nuscenes-devkit; mini_val chính thức gồm
  scene-0103/scene-0916. Manifest benchmark occupancy đủ 81 token là nguồn
  membership, không lấy danh sách 79 prediction thành công làm sequence scope.
- Prediction thật lấy từ work_dirs/benchmark_mini_val qua progress.json,
  prediction_commit.json và SHA-256 từng artifact/resolved config. Không
  đọc file .pending, backfill frame hoặc tự chạy inference khi cache lỗi.
- GT SurroundOcc N×4 lấy từ root đã giải nén, nối bằng nuScenes LiDAR
  basename + .npy. Không thay bằng lidarseg hoặc box rasterization.

Kiểm tra NPZ thực tế: labels uint8 XYZ 200×200×16 và
max_softmax_probability float32 cùng shape. NPZ không chứa token, logits
hoặc features đầy đủ. Token lấy từ metadata/report và kiểm tra với API.
Confidence không dùng để chọn window, tạo prediction hoặc mask đánh giá.

## Protocol cố định v1

Timestamp dùng số nguyên microsecond của sample_data LIDAR_TOP, không lấy
frame index làm đồng hồ. Mỗi sample mini_val là một candidate anchor.

| Thành phần | Quy tắc |
| --- | --- |
| History | 3 frame, nominal offset −1,0 / −0,5 / 0 giây, gồm anchor |
| Forecast horizon | +0,5 / +1,0 / +1,5 giây |
| Matching | Timestamp gần nominal nhất trong **cùng scene**, tolerance ±50.000 µs |
| Tie | Timestamp sớm hơn, rồi token; không phụ thuộc prediction/metric |
| Continuity | Sáu sample chọn phải phân biệt, liên tiếp theo sample chain; mỗi khoảng liền kề 0,5 s ±50 ms |
| Input cutoff | Chính timestamp LiDAR anchor t; mọi sensor/ego-pose thực dùng trong history phải ≤ t |
| Future | GT và ego pose tương lai chỉ dùng xây target/evaluation offline |

Không bỏ qua sample giữa window, vượt scene, padding, lặp frame hoặc lấy
prediction bù. Sai tolerance vẫn bị loại dù có file. Quy tắc 50 ms cố định
trước khi tính metric; không nới tolerance theo kết quả. JSON ghi desired
timestamp, actual timestamp, sai lệch có dấu và lý do từng candidate.

Sensor audit đối chiếu từng camera/LiDAR/radar đã dùng với metadata API,
file hash, calibration, ego pose và full transform. Kiểm tra lại prev-chain
và sweep selection của adapter. Radar keyframe có thể ở sau LiDAR nhưng đã
bị adapter loại: vẫn audit, đánh dấu used_in_prediction=false, không coi là
input. Camera muộn hơn LiDAR của history cũ chỉ được dùng khi không vượt
**cutoff anchor hiện tại**. Anchor có camera sau cutoff bị loại, không sửa
baseline hay timestamp để hợp thức hóa.

Cutoff là giao thức acquisition timestamp, chưa đo transport/processing
latency hoặc chứng minh đáp ứng deadline thời gian thực. Không tự suy ra
thời điểm từng điểm trong một LiDAR scan.

## Hình học và masks

Mọi history/output/target dùng native LIDAR_TOP tại anchor t, mét, full
SE(3), không thay trục LiDAR bằng trục ego hoặc three.js:

~~~text
global_from_lidar_s = global_from_ego_s @ ego_from_lidar_s
anchor_from_source = inv(global_from_lidar_t) @ global_from_lidar_s
p_anchor = anchor_from_source @ [x_source, y_source, z_source, 1]
~~~

Calibration và ego pose riêng tại mỗi timestamp. History alignment chỉ
đọc các pose history/anchor. Future pose được đọc bởi load_targets sau khi
forecast đã được tạo. Predictor nhận HistoryInputs, không nhận dataset,
future token/pose/GT hoặc probability.

Giữ XYZ [200,200,16], bounds [−50,−50,−5,50,50,3], voxel [0,5;0,5;0,5] m,
free=0, ignore=255 và đủ 16 semantic class của baseline.

~~~text
centre_anchor = bounds_min + (index_xyz + 0.5) * voxel_size
(10,30,4) -> (-44.75,-34.75,-2.75) m
point_source = inv(anchor_from_source) @ centre_anchor
source_index = floor((point_source - bounds_min) / voxel_size)
~~~

Semantic dùng pull cell lookup/nearest centre, tie tại biên chọn index cao
hơn; source bounds nửa mở [min,max). Không nội suy tuyến tính nhãn, không
clip index ngoài bounds vào biên. Output OOB/unknown là 255, có in_bounds,
known và valid mask riêng. History-valid và future-target masks tách biệt.
Không lấp OOB bằng free hoặc dùng history mask để loại lỗi forecast.

GT decoder tái sử dụng sparse_to_dense của evaluator đã kiểm chứng: hàng
sparse nhãn 0 → ignore 255, ô không được liệt kê → empty 0. Kiểm tra index,
nhãn, NaN/Inf và duplicate xung đột. Đây là convention của SurroundOcc,
không chứng nhận unlisted empty là free space đã thực sự được quan sát.

Kiểm chứng real geometry so ma trận SDK với chuỗi Quaternion.rotate độc lập
trên các tâm voxel cố định, thêm roundtrip. Báo sai số float64 thực đo; sai
số số học này khác lượng tử hóa cell 0,5 m và không phải flow/EPE.

## Persistence và evaluator

O(t+h) = saved prediction O(t), giữ grid anchor cho mọi horizon. Hai history
cũ được căn chỉnh/cấp vào dataset nhưng baseline này chỉ copy frame cuối.
Không dùng prediction hoặc GT tương lai để tạo forecast. Các example còn
kiểm tra thay toàn bộ future GT thành class 7 **trong RAM**, khôi phục ngay,
rồi đối chiếu hash byte forecast; không sửa file GT.

Evaluation mask là future source in-bounds AND decoded GT !=255. Nó chỉ phụ
thuộc target/protocol, không confidence, nhãn dự đoán hoặc đúng/sai.
Anchor prediction phải phủ đủ grid, không có hidden mask theo prediction.

- Occupancy IoU: occupied !=0, bỏ target ignore/OOB.
- Semantic IoU từng lớp 1..16; mIoU trung bình đủ **16 lớp**.
- Union=0 nhận IoU 0, vẫn đưa lớp vắng vào mean.
- Mỗi horizon cộng integer confusion matrix 17×17 (hàng GT, cột prediction)
  qua tất cả window hợp lệ, sau đó mới chia; không mean IoU từng window.
- JSON metric là fraction [0,1]. Per-class/count/window/valid voxel được lưu.
- Mỗi window/horizon đối chiếu nguyên hàm EvalMetric gốc qua AST, NumPy CPU,
  không import OccFusion/Torch. Pooled cộng các count đã đối chiếu đó.
- Flow, EPE, ADE và FDE để null, status=not_implemented.

Window chồng lấn có tương quan; không xem các window là sample độc lập.
Không có visibility/occlusion mask, object-motion model hoặc volume-conservative
splatting. Metric của window hợp lệ không phải metric của toàn bộ 81 anchor.

## Lệnh chạy CPU trên WSL

Giữ đúng cách viết root/config đã nghiệm thu; không source env QA vào web
production, không gọi launcher/worker. Không cài thêm dependency vào rav25.

~~~bash
cd '/mnt/d/New folder/OCCFusion'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''

/home/hatis/.venvs/rav25/bin/python -m unittest discover \
  -s tests -p 'test_forecasting.py' -v

# Một sequence thật đầu tiên theo manifest; không chọn theo metric:
/home/hatis/.venvs/rav25/bin/python tools/forecast_persistence.py \
  --pilot --out-dir work/forecasting/persistence_v1_pilot

# Chỉ sau khi pilot/reference/geometry hợp lệ:
/home/hatis/.venvs/rav25/bin/python tools/forecast_persistence.py \
  --out-dir work/forecasting/persistence_v1
~~~

Các lệnh thật lưu log tại work/forecasting/qa_v1. --audit-only chỉ ghi audit/
manifest, không metric. --data-root/--benchmark-root/--gt-root nhận root nguồn
đã cấu hình. --out-dir bắt buộc là thư mục con mới của work/forecasting,
không được work_dirs, data hoặc state. Output có sẵn bị từ chối; không resume
inference, không ghi đè hoặc tự retry. Nếu có lỗi, failure.json được giữ;
thư mục .pending/không có report/commit thành công không là kết quả hoàn tất.

CLI chặn import Torch/torchvision/OpenMMLab/OccFusion và báo modules đã load.
Checkpoint SHA trong provenance được **đọc từ manifest cũ**, không mở file
checkpoint hoặc dựng model. NumPy/SDK/Matplotlib chạy CPU.

## Cấu trúc output và audit train

~~~text
work/forecasting/persistence_v1/
  protocol.json
  sequence_manifest.json
  input_audit.json
  gt_audit.json
  metrics_by_horizon.json
  geometry_validation.json
  forecasting_report.json
  windows/<anchor_token>/
    forecast.npz                    # persistence labels [horizon,x,y,z]
    window.json                     # nguồn/hash/masks/count/metric
    commit.json
    aligned_history.npz             # chỉ vài example
    offline_targets.npz              # GT tách riêng, chỉ vài example
    forecast_vs_offline_target.png   # BEV phân biệt forecast và GT
~~~

Manifest ghi mọi candidate, history/anchor/future token/time/hash/source,
primary exclusion counts (cộng bằng số bị loại) và any-reason counts (có
thể chồng lấn). Coverage sequence được đếm riêng, không lấy 79/81 frame.
Mỗi horizon có số window và voxel target-valid thực tế.

gt_audit.json kiểm tra toàn bộ official mini_train và mini_val, hash/giải mã
từng GT có mặt, danh sách thiếu theo scene. GT mini_train còn thiếu không
được thay bằng mini_val để tự huấn luyện rồi báo test độc lập. Không huấn
luyện ở đây; lịch sử data training checkpoint cũ chưa được xác minh độc lập.

## Kết quả thực thi

Ngày 2026-11-16, chạy bằng /home/hatis/.venvs/rav25/bin/python, NumPy 1.26.4,
nuscenes-devkit 1.1.11, OPENBLAS_NUM_THREADS=1 / OMP_NUM_THREADS=1 và
CUDA_VISIBLE_DEVICES rỗng. Không thay dependencies.

**37 test CPU đạt**, 0,488 s, không skip; py_compile đạt. Lượt đầu 34 test
có 33 đạt/1 lỗi vì fixture receipt thiếu reference_timestamp_us; bổ sung
đúng field trong fixture, thêm ca GT hash/missing và GT perturbation restore.
Không bỏ assertion để qua test. Giữ unit_initial.log, unit_after.log và
unit_final.log trong work/forecasting/qa_v1. Không chạy lại full backend 175
test hoặc frontend ở mốc này vì không sửa web; 175 là bằng chứng checkout
nghiệm thu trước đó, không cộng vào số test mới.

Pilot chọn sequence hợp lệ đầu tiên theo thứ tự manifest: scene-0103 frame
3, anchor 700c1a25559b4433be532de3475e58a9, history frame 1/2/3, future 4/5/6.
Pilot hoàn tất trước khi chạy full CPU; sai số geometry max 3,890678e-13 m.
Thay ba future GT thành class 7 trong RAM không đổi SHA-256 byte forecast:
e60e91e2c4c09c5a06dad660c97571c8f8039d726791d1cb31c103d6a00ecd9d.

### Coverage sequence

| Scene | Candidate anchor | Window hợp lệ | Bị loại |
| --- | ---: | ---: | ---: |
| scene-0103 | 40 | 9 | 31 |
| scene-0916 | 41 | 17 | 24 |
| Tổng | **81** | **26** | **55** |

**Tất cả 26 window hợp lệ đã được đánh giá**, status
complete_for_valid_windows. Đây không phải coverage 79/81 frame và không
phải complete forecasting trên 81 anchor. Thứ tự/coverage không chọn theo
IoU hoặc confidence.

| Lý do | Primary windows, cộng bằng 55 | Any-reason windows, có chồng lấn |
| --- | ---: | ---: |
| history_scene_boundary | 4 | 4 |
| future_scene_boundary | 5 | 6 |
| history_timestamp_tolerance | 13 | 13 |
| future_timestamp_tolerance | 14 | 19 |
| missing_prediction | 1 | 6 |
| input_after_cutoff | 18 | 31 |

Hai source frame thiếu radar vẫn missing_prediction như benchmark cũ. Chúng
ảnh hưởng sáu candidate windows; chỉ một window có missing_prediction là
lý do đầu tiên, các window khác còn vi phạm boundary/timestamp. Không tạo
prediction, sweep tương lai hoặc frame thay thế.

Audit 79 prediction hợp lệ ghi **3.713 observation cảm biến thực dùng** và
197 radar keyframe không dùng (số observation theo frame, không phải số
file vật lý duy nhất). Trong đó có 474 camera, 869 LiDAR keyframe/sweep và
2.370 radar sweep observations. CAM_BACK_LEFT muộn hơn LiDAR ở **31 frame**,
muộn nhất **1.204 µs = 1,204 ms**. Tất cả window có input vượt cutoff bị
loại; 197 radar keyframe đã bị adapter loại không bị tính thành input.

Trong 26 window hợp lệ, latest-used timestamp bằng anchor LiDAR cutoff;
lookback sensor thực tế 1,499039–1,502464 s do mỗi history prediction vốn
đã dùng sweep. Sai lệch future nominal lớn nhất theo horizon:
**1.001 / 1.433 / 1.511 µs**; history lớn nhất 1.767 µs. Không resample thời
gian hoặc coi mọi interval là chính xác 0,5 s.

### Pooled metrics thực tế

| Horizon danh nghĩa | Window | Voxel target-valid | Occupancy IoU % | Semantic mIoU %, đủ 16 lớp |
| --- | ---: | ---: | ---: | ---: |
| +0,5 s | 26 | 15.380.503 | **43,722891** | **21,002409** |
| +1,0 s | 26 | 14.329.399 | **43,531356** | **19,730614** |
| +1,5 s | 26 | 13.490.977 | **43,600755** | **19,094839** |

Metric JSON giữ fraction; bảng trên chỉ đổi sang % một lần. Mỗi horizon có
confusion 17×17 và per-class intersection/union/IoU/count đầy đủ trong
metrics_by_horizon.json. Target in-bounds/ignore mask thay đổi theo horizon,
vì thế không diễn giải chênh lệch IoU như so sánh cùng một tập voxel cố định.
Flow/EPE/ADE/FDE đều null/not_implemented.

Sai số hình học full 26 window: **max 6,913201073447074e-13 m**, roundtrip
max **2,5622831388630473e-14 m**, float64. Đây là kiểm tra số học transform,
không phải sai số calibration thực hoặc độ phân giải hình học của nhãn.

Lượt kiểm tra CPU độc lập từ artifact đã lưu thực sự đã chạy:

~~~bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  /home/hatis/.venvs/rav25/bin/python work/forecasting/qa_v1/verify_saved.py \
  work/forecasting/persistence_v1
~~~

Lượt này kiểm tra 61 committed files, đọc lại forecast, đối chiếu từng
forecast bằng source anchor prediction, hash GT và dựng lại target từ
calibration/ego pose thật. Tổng **43.200.879 voxel comparisons** qua ba
horizon. Từng confusion và pooled counts khớp; pooled nguyên EvalMetric gốc
khác tối đa **1,1102230246251565e-16**. Không gọi predictor/model/forward mới.
Bằng chứng: persistence_v1/saved_replay_verification.json và qa_v1/saved_replay.log.

### GT mini_train và giới hạn training

| Official mini_train scene | Số frame | GT hợp lệ | GT thiếu |
| --- | ---: | ---: | ---: |
| scene-0061 | 39 | 0 | 39 |
| scene-0553 | 41 | 41 | 0 |
| scene-0655 | 41 | 0 | 41 |
| scene-0757 | 41 | 0 | 41 |
| scene-0796 | 40 | 40 | 0 |
| scene-1077 | 41 | 0 | 41 |
| scene-1094 | 40 | 0 | 40 |
| scene-1100 | 40 | 0 | 40 |
| Tổng | **323** | **81** | **242** |

GT có mặt đã hash và giải mã, không chỉ kiểm tra tên file. Mini_val có GT
hợp lệ 81/81. Danh sách token/frame/expected filename bị thiếu theo từng
scene nằm trong gt_audit.json. Nguồn local vẫn là SurroundOcc val archive
đã có; việc vài scene thuộc mini_train có file không đổi official split.
Chưa đủ GT cho toàn mini_train; benchmark nguồn cũng không cung cấp lịch
sử prediction mini_train để huấn luyện. Không huấn luyện trên mini_val rồi
báo chính nó là test độc lập, không tải dữ liệu bổ sung ở lượt này.

### Bằng chứng và bảo toàn

- work/forecasting/persistence_v1_pilot: pilot CPU thật, giữ nguyên.
- work/forecasting/persistence_v1: đủ 26 forecast, metric/count/mask, ba
  example có aligned history + offline target và hình BEV phân biệt rõ.
- Example anchor: 700c1a25559b4433be532de3475e58a9 (scene-0103/3),
  048a45dd2cf54aa5808d8ccc85731d44 (scene-0916/10),
  841dd6709e9b4d7c9d6bf888f1fe6d7e (scene-0916/33). Cả ba có GT perturbation
  check pass và GT được khôi phục trong RAM, file nguồn không sửa.
- work/forecasting/preservation_before.json / preservation_after.json:
  hash source/artifact cũ, hash từng row database read-only và checkpoint
  stat (không đọc bytes checkpoint). Log runtime production không dùng làm
  tiêu chí bất biến vì có thể có traffic người dùng ngoài lượt này.
  Kết quả **1.618 file không đổi**, **76 row trong 14 bảng không đổi**,
  checkpoint stat không đổi, 3 queued/0 running; không restart production.
- work/forecasting/completion_report.json: danh sách lệnh, file và kết quả
  cuối. Không đưa bất kỳ kết quả forecasting mới vào approval/export cũ.

Pilot mất 144,175 s; full CPU 134,795 s, gồm load/audit/hash/alignment/
evaluation/export, không phải latency inference GPU hoặc benchmark model.
Import guard báo không có Torch/OpenMMLab/OccFusion được load; model load,
checkpoint load, model forward, GPU và training đều **0**. Production không
restart, worker không bật, ba job queued không được nhận.
