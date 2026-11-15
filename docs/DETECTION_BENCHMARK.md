# CenterPoint detection benchmark: nuScenes mini_val

Benchmark đánh giá toàn bộ split **mini_val** từ metadata và
nuscenes.utils.splits.create_splits_scenes. Không dùng danh sách 79 prediction
occupancy. CLI/profile CenterPoint cũ, CSRF, frontend, production, artifact và
approval hiện có được giữ nguyên.

## Dữ liệu và preprocessing

Dataset data/nuscenes, v1.0-mini; thứ tự scene name tăng dần rồi ordinal trong
chuỗi sample. Kiểm tra count, token duy nhất, prev/next, timestamp, scene
boundary và đối chiếu toàn bộ sample của split trong metadata.

Thực tế **81 sample**: scene-0103 **40**, scene-0916 **41**. Cả hai scene cũng
thuộc val trong split trainval của devkit. Đây vẫn chỉ là hai scene mini_val,
không phải toàn nuScenes validation.

LiDAR-only: frame không có radar vẫn nằm trong manifest. Cả hai frame 0 không
có lịch sử LiDAR; loader chính thức pad thêm **9 bản keyframe** với
remove_close=True. Chỉ lịch sử hoàn toàn rỗng được pad; 1..8 sweep không tự
pad đủ 9. Không lấy sweep tương lai hoặc thay file thiếu.

Mỗi frame audit file hash/size, điểm hữu hạn, timestamp, calibration, ego pose,
sweep chain và scene. Dùng nguyên frame_info trong
[infer_centerpoint_mini.py](../tools/infer_centerpoint_mini.py), rồi nguyên
cfg.test_pipeline của MMDetection3D v1.4.0; không viết loader LiDAR thứ hai.

Giữ config/checkpoint CenterPoint PointPillars 0.2 m/circle NMS đã kiểm chứng
trong [CENTERPOINT_DETECTION.md](CENTERPOINT_DETECTION.md). Checkpoint SHA-256:
191a38226d2ce990f26cc3f78c454310cc562d81d29e86d8ca8fa39c0a44215d.
FP32, no autocast, batch 1, TF32 matmul false/cuDNN true. Manifest ghi hash
config/resolved config/checkpoint, adapter, benchmark, pipeline/model cài đặt,
package versions và 13 bảng JSON.

## Giới hạn loader trên toàn manifest

Không sửa loader v1.4 hoặc đổi baseline để cải thiện metric. Audit **711 sweep
quá khứ** đo norm chênh lệch phần tịnh tiến giữa loader row_xyz @ R - t và
biến đổi SE(3) vật lý:

| Thống kê sweep | Mét |
| --- | ---: |
| Median | 0,007184933 |
| p95, NumPy linear | 0,193105039 |
| Max | **0,428603971** |

Frame có max: **scene-0916 / 18**, token
8092909473464f80b9f791a4d31ddcb8. 6,33 cm của smoke chỉ là max trong frame
smoke. input_audit.json và manifest lưu sai khác từng sweep/frame, timestamp
và padding. Kết quả benchmark phản ánh baseline có giới hạn này.

## Box và evaluator CPU

[evaluate_detection.py](../tools/evaluate_detection.py) dùng trực tiếp
**nuscenes-devkit 1.1.11**, DetectionEval với detection_cvpr_2019. Evaluator
không xây model hoặc gọi forward. Nguồn:
[giao thức detection chính thức](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md),
[formatter MMDetection3D v1.4.0](https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/mmdet3d/evaluation/metrics/nuscenes_metric.py).

Prediction gốc dùng geometric center trong LIDAR_TOP, dimensions [dx,dy,dz],
yaw radian CCW quanh +Z, velocity [vx,vy] m/s. Khi xuất:

- Center: global_from_lidar @ [center,1]; không cộng nửa height lần nữa.
- Size nuScenes: **[dy,dx,dz] = width/length/height**.
- Orientation: global rotation nhân yaw LiDAR, quaternion **w,x,y,z**.
- Velocity: chỉ xoay [vx,vy,0], không cộng translation; lấy global XY.

Attribute chỉ dùng class và vận tốc prediction theo policy v1.4: global XY
speed **> 0,2 m/s**. Xe chạy dùng vehicle.moving, cycle chạy dùng
cycle.with_rider. Xe chậm dùng defaults parked, riêng bus stopped;
pedestrian chậm standing. Defaults đầy đủ và policy nằm trong protocol.
Không lấy attribute/velocity GT làm prediction.

Đưa **toàn bộ box sau postprocessing baseline** vào JSON. Không lọc score
0,35 hoặc thêm lọc range riêng. Devkit áp dụng đúng range, point-count và
bicycle-rack filters chính thức. GT lấy từ annotation nuScenes, không thay
bằng occupancy, lidarseg hoặc rasterized box.

Token manifest phải bằng token GT chính thức; prediction phải đủ và không
dư. Missing/failed frame không được thêm key chứa mảng rỗng. Chỉ frame inference
thành công, hash đạt, zero box mới được dùng mảng rỗng. Thiếu prediction:
**partial, metrics null**, không công bố mAP toàn mini_val.

mAP trung bình đủ 10 lớp và ngưỡng center-distance 0,5/1/2/4 m; TP error dùng
match 2 m. NDS do devkit tính, trọng số mAP 5 và năm TP scores. Không lấy
trung bình AP từng frame. mAP/NDS/AP là **fraction**, đổi sang phần trăm chỉ
nhân 100 một lần. TP error không áp dụng có NaN trong file official; JSON
normalized dùng null, giữ nguyên file devkit gốc và hash. Không clamp metric
chính thức vì round-off nhỏ.

## Checkpoint, cache, resume và GPU lane

[benchmark_detection.py](../tools/benchmark_detection.py) nạp dataset/config
một lần mỗi invocation; khi có frame cần chạy mới nạp model/checkpoint một
lần. Model eval(), inference_mode(), batch 1, tuần tự. Không giữ logits/tensor
GPU toàn tập; giải phóng tham chiếu mỗi frame và ghi allocated sau giải phóng.

Dùng cùng flock/inode **work/inference-worker/gpu.lock** với worker hiện có.
Không start worker, tạo SQLite job hoặc claim queue production. Output có
khóa riêng chặn hai invocation trùng.

Mỗi frame ghi bundle vào .pending-UUID, kiểm tra artifact, ghi
prediction_commit.json, rename rồi atomic-save progress. File pending chưa
có commit đầy đủ không phải prediction thành công. Resume hash lại artifact,
mọi LiDAR/sweep, token, input metadata, checkpoint/config/code, framework,
TF32 và pipeline. Cache hỏng bị invalid_cache, không tự inference lại.
Launch dở bị interrupted; commit hoàn chỉnh nhưng progress chưa cập nhật
được recovery mà không forward.

Smoke thật có thể nhập qua --smoke-run: kiểm tra real publication receipt,
token/timestamp/poses, config/checkpoint/CLI/framework/test pipeline và input
hash; chép nguyên byte vào benchmark, origin=external_smoke. Không dùng
fixture/check-only. Single-frame report/metrics cũ vẫn pending; không bị ghi
thêm metric toàn tập.

Forward lỗi/OOM hoặc export/publication lỗi: lưu sample, stage, traceback và
dừng các model call trong invocation. Không tự retry, đổi resolution/model;
resume không retry failed/interrupted. CSV ghi dở bị chặn trước model load
để giữ bằng chứng; không đoán latency bị mất.

Cache evaluation cần prediction JSON/artifact SHA, GT table SHA và protocol
khớp. Official summary/details và snapshot cũng phải còn nguyên hash. Cache
evaluation hỏng chỉ được đánh giá CPU lại vào thư mục mới.

## Timing

Không warm-up ngoài manifest. Forward mới đầu tiên mỗi invocation gắn
latency_warmup=true, vẫn tham gia accuracy. Median/p95 chỉ dùng forward mới
hoàn tất; không dùng thời gian đọc cache hoặc timing smoke nhập ngoài invocation.

Forward synchronize CUDA, gồm predict/decode/circle NMS. End-to-end frame gồm
input audit/hash, preprocessing, forward, CPU export, validation và commit;
tách dataset/config/audit chung và model/device setup. Có số mẫu all/không
warm-up, scope và NumPy linear p95. Peak allocated/reserved là **PyTorch
allocator**, không phải VRAM vật lý đo độc lập. Report từng invocation được
giữ riêng; resume/evaluate-only không thay report GPU gốc bằng timing cache.

## Chạy trong WSL

Không thêm dependency. Benchmark/evaluator/tests dùng rav25; hồi quy web
dùng .venv-web và state QA riêng.

~~~bash
cd '/mnt/d/New folder/OCCFusion'
export MPLCONFIGDIR="$PWD/.cache/matplotlib"

# CPU, không model load/forward.
/home/hatis/.venvs/rav25/bin/python -u tools/benchmark_detection.py \
  --prepare-only --out-dir work_dirs/benchmark_detection_mini_val

# Nhập smoke thật bằng CPU; mới có 1/81 nên partial là kết quả đúng.
/home/hatis/.venvs/rav25/bin/python -u tools/benchmark_detection.py \
  --resume --evaluate-only --out-dir work_dirs/benchmark_detection_mini_val \
  --smoke-run work/centerpoint-smoke-wyp1nl__/runs/infer_job_65e1b30833e343c3b434da8dc71423f9

# Sau CPU tests đạt: inference chỉ các frame chưa chạy/cache hợp lệ.
/home/hatis/.venvs/rav25/bin/python -u tools/benchmark_detection.py \
  --resume --out-dir work_dirs/benchmark_detection_mini_val

# Khi đã complete, hai lệnh sau chỉ dùng cache, không forward mới.
/home/hatis/.venvs/rav25/bin/python -u tools/benchmark_detection.py \
  --resume --out-dir work_dirs/benchmark_detection_mini_val
/home/hatis/.venvs/rav25/bin/python -u tools/benchmark_detection.py \
  --resume --evaluate-only --out-dir work_dirs/benchmark_detection_mini_val

# Evaluator độc lập: chỉ CPU, không import Torch/OCCFusion.
/home/hatis/.venvs/rav25/bin/python tools/evaluate_detection.py \
  --benchmark-dir work_dirs/benchmark_detection_mini_val
~~~

Không dùng output mới chỉ để đọc lại kết quả. Resume thường có thể inference
frame chưa từng chạy; dùng --evaluate-only để bảo đảm không chạy model.
Exit 0: prepared/complete; exit 2: partial/preparation_failed. Lỗi provenance,
lock/input bất hợp lệ trả lỗi rõ ràng.

~~~text
work_dirs/benchmark_detection_mini_val/
    manifest.json, input_audit.json, resolved_config.py
    progress.json
    frames/<sample_token>/<attempt_uuid>/
        detections.json, metadata.json, report.json, resolved_config.py
        prediction_commit.json
        job_result.json                  # chỉ bản copy smoke
    timings.csv
    predictions_nuscenes.json             # complete coverage
    predictions_nuscenes_partial.json     # chỉ token thực sự có
    evaluation.json
    evaluations/<uuid>/
        evaluation.json, evaluation_commit.json
        official/metrics_summary.json, official/metrics_details.json
    reports/<invocation_id>.json          # giữ report GPU gốc
    benchmark_report.json                 # invocation gần nhất
~~~

Test CPU:

~~~bash
/home/hatis/.venvs/rav25/bin/python -m unittest discover \
  -s tests -p 'test_detection_benchmark.py' -v
VIEWER_STATE_ROOT="$PWD/work/detection-benchmark-qa/state" \
  .venv-web/bin/python -m pytest web/backend/tests/test_detection.py \
  web/backend/tests/test_csrf_bootstrap.py -q \
  --junitxml=work_dirs/detection_benchmark_preparation/web_regression.xml \
  -o faulthandler_timeout=60
~~~

Tests dùng manifest/audit CPU đã chuẩn bị ở work_dirs/benchmark_detection_mini_val
(có thể chọn bằng DETECTION_TEST_AUDIT) và bản copy smoke thật. Fixture metric tổng hợp chỉ nằm trong thư mục test
tạm, không phải prediction benchmark. TestClient cần native ngoài sandbox
nếu asyncio bị chặn; không đổi dependency để vượt lỗi môi trường.

Benchmark là kết quả **cấp tập frame**, không tự approved, không mở rộng
approval v1/v2/v3 cũ và chưa ghép vào review một frame. Đây vẫn là prediction
frame độc lập, chưa phải temporal fusion/learned 4D flow.

## Kết quả thực tế ngày 2026-11-15

Invocation GPU **367b21268aed4eb4852c6b4d221a59c5** đã đạt **complete, 81/81**:
80 forward mới, 1 cache smoke thật, **1 model/checkpoint load**, không retry,
không lỗi/OOM và không có forward warm-up ngoài manifest. Cả hai frame 0
từng lỗi radar ở benchmark occupancy đều có prediction detection hợp lệ.

Report GPU gốc:
[367b21268aed4eb4852c6b4d221a59c5.json](../work_dirs/benchmark_detection_mini_val/reports/367b21268aed4eb4852c6b4d221a59c5.json).
Report gần nhất:
[benchmark_report.json](../work_dirs/benchmark_detection_mini_val/benchmark_report.json).
Evaluation và các hash:
[evaluation.json](../work_dirs/benchmark_detection_mini_val/evaluation.json).

| Đại lượng | Kết quả thực tế |
| --- | ---: |
| mAP, fraction | **0.41280115564259257** |
| mAP, phần trăm | **41,280116%** |
| NDS, fraction | **0.47759091555233457** |
| NDS, phần trăm | **47,759092%** |
| Prediction tokens / GT tokens | **81 / 81**, không thiếu/dư |
| Box lưu / box sau official filters | **16.086 / 13.317** |
| Box có score dưới 0,35 vẫn đưa vào evaluation | **11.900** |
| GT box trước / sau official filters | **4.441 / 3.393** |

| Lớp | AP (%) | GT sau official filters |
| --- | ---: | ---: |
| car | 87,486081 | 1.913 |
| truck | 67,419766 | 95 |
| bus | 97,693840 | 33 |
| trailer | 0 | 0 |
| construction_vehicle | 0 | 0 |
| pedestrian | 87,257408 | 1.067 |
| motorcycle | 49,040602 | 214 |
| bicycle | 23,884493 | 36 |
| traffic_cone | 0,018966 | 35 |
| barrier | 0 | 0 |

Ba lớp không có GT vẫn nằm trong trung bình 10 lớp chính thức. Mini chỉ có
hai scene liên tiếp, mất cân bằng lớp; không dùng các số này thay kết quả
model zoo trên toàn validation.

| Phép đo GPU gốc | Số mẫu | Median | p95 |
| --- | ---: | ---: | ---: |
| Forward bỏ warm-up | **79** | **64,663 ms** | **80,473 ms** |
| End-to-end frame bỏ warm-up | **79** | **1,649535 s** | **1,741010 s** |
| Forward gồm cả warm-up | **80** | **64,738 ms** | **80,585 ms** |

Warm-up được đánh dấu là scene-0103/frame 0, token
3e8750f331d7499e9b5123e9eb70f2e2: forward **3,223129 s**, prediction vẫn tính
accuracy. Common CPU setup/audit **27,804307 s**, model/checkpoint CPU setup
**2,679241 s**, device setup **0,217304 s**. Invocation execute wall
**252,211141 s**, gồm chạy frame, validation và đánh giá CPU; không phải
forward latency hoặc process time đã bao gồm common setup.

Peak PyTorch allocated **547.854.848 bytes = 522,475 MiB**; reserved
**3.795.845.120 bytes = 3.620 MiB**. Đây là allocator, không phải VRAM vật lý.
Allocated sau giải phóng **32.588.800 bytes** không đổi qua 80 frame.

Kiểm chứng đã chạy:

- **25 tests CPU** đạt, gồm transform xác định, so 151 box smoke với devkit
  Box.rotate/translate, preprocessing count, zero-history padding, lỗi
  timestamp/file/scene, hash/code/framework/pipeline mismatch, cache hỏng,
  restart/recovery, lỗi forward giả, token coverage, official AP/NDS,
  undefined TP, evaluation cache và tách common setup khỏi latency frame đầu.
- **46 tests detection/CSRF** đạt bằng SQLite QA/native. Lượt sandbox trước
  đó bị kẹt TestClient/asyncio, đã dừng riêng process test và chạy lại native;
  không sửa dependency hoặc source auth.
- CPU import ban đầu: **1/81 prediction**, GT **81**, metric **null**, 0 model
  load/forward; có snapshot bằng chứng partial.
- Resume thực tế: **81 cache, 0 model load, 0 forward**, metrics bằng report
  GPU; prediction JSON, CSV và report GPU gốc giữ nguyên hash.
- Evaluate-only thực tế: **81 cache, 0 model load, 0 forward**; mAP/NDS không
  đổi. Evaluator độc lập cũng đạt, **torch_imported=false**,
  **model_modules_loaded=false**. Kiểm tra cache/hash CPU có thể mất vài phút
  và không được dùng làm inference latency.
- Raw official summary bằng metric normalized; đối chiếu công thức NDS từ
  mAP/TP errors có absolute difference **0,0**.

Bằng chứng/log nằm trong
[detection_benchmark_preparation/](../work_dirs/detection_benchmark_preparation/),
gồm cpu_readiness.json, cpu_tests.log, web_regression.xml, gpu_benchmark.log,
gpu_original_evidence.json, resume_report.json, evaluate_only_report.json
và standalone_evaluation.json.

Audit preservation_after.json xác nhận **883 file cũ không đổi/mất**, gồm
CSRF, source/profile OCCFusion và CenterPoint smoke, checkpoint, benchmark
occupancy, inference/evaluation và manifest approval cũ. Các hàng tài khoản,
run/revision/review/approval/job production không đổi; không restart dịch vụ.
Không chạy lại occupancy benchmark, huấn luyện, toàn bộ nuScenes val hoặc
frontend build/browser trong mốc này. Frontend không được sửa.

File mới: tools/benchmark_detection.py, tools/evaluate_detection.py,
tools/detection_benchmark_common.py, tests/test_detection_benchmark.py,
docs/DETECTION_BENCHMARK.md. Không đổi dependency hoặc lockfile.
