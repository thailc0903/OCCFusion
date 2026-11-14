# CenterPoint LiDAR-only và box 3D trong viewer

Mốc này bổ sung task `detection` vào queue, validator, viewer và review hiện có.
OCCFusion vẫn là task `occupancy`; profile/job không có trường `task` giữ cách
hiểu và fingerprint cũ. Không thay đổi model, config, preprocessing, checkpoint
hoặc CLI/evaluator OCCFusion. Bản sửa CSRF giữ nguyên.

## File thay đổi trong mốc này

| Phần | File |
| --- | --- |
| Assets, CLI, QA | `tools/prepare_centerpoint.py`, `tools/infer_centerpoint_mini.py`, `tools/check_web_detection.py`, `web/centerpoint_profiles.json` |
| Dispatch, validation, publication | `web/backend/app/task_types.py`, `detections.py`, `jobs.py`, `job_worker.py`, `job_results.py`, `store.py` |
| API, review | `web/backend/app/main.py`, `review.py` |
| Tests | `web/backend/tests/detection_support.py`, `test_detection.py` |
| Trang và schema frontend | `web/frontend/app/detections/page.tsx`, `lib/detections.ts`, `lib/jobs.ts`, `app/globals.css` |
| Components | `web/frontend/components/DetectionExplorer.tsx`, `DetectionOverlay.tsx`, `OccupancyScene.tsx`, `Viewer.tsx`, `JobsViewer.tsx`, `ReviewPanel.tsx`, `AuthGate.tsx` |
| Tài liệu | `docs/CENTERPOINT_DETECTION.md`, liên kết bổ sung trong `docs/WEB_INFERENCE_JOBS.md` |

Các tên rút gọn trong cùng hàng nằm ở thư mục của file đầu tiên. Diff
`docs/CSRF_BOOTSTRAP_FIX.md` đã có trước mốc này và được giữ nguyên; không sửa
`auth.py`, `database.py` hoặc lockfile/dependency.

## Assets và môi trường

Dùng đúng cặp trong [model zoo MMDetection3D v1.4.0](https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/configs/centerpoint/metafile.yml):

- Config: [`centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py`](https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py).
- [Checkpoint chính thức](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth):
  **24.154.629 bytes**, SHA-256
  `191a38226d2ce990f26cc3f78c454310cc562d81d29e86d8ca8fa39c0a44215d`.
- Config SHA-256:
  `f65e622570896f391a4b7f30e8baf81b7a21cbc34e62c5c27e621af57a51839d`.

`tools/prepare_centerpoint.py` tải config cùng các `_base_` và checkpoint vào
`ckpt/centerpoint_pillar02_v140/`, ghi URL/SHA/size trong `assets.json`. Thư mục
này độc lập với checkpoint OCCFusion. Không đổi dependency của rav25 hoặc web.
Web/worker điều phối dùng `.venv-web`; CLI model dùng
`/home/hatis/.venvs/rav25/bin/python` với MMDetection3D **1.4.0**,
Torch **2.1.1+cu121**. Backend không import Torch/OpenMMLab.

CPU check-only đã strict load **414 tensor, 5.982.854 parameters**; không có
missing keys, unexpected keys hoặc tensor sai kích thước. Không bỏ qua trọng
số lỗi. `--check-only` chạy kiểm tra input/model/test pipeline trên CPU, không
forward và không khởi tạo CUDA.

## Tiền xử lý và hình học

CLI lấy sample từ nuScenes API theo scene name/frame và split chính thức;
không cần PKL, GT, camera hoặc radar. Tái sử dụng nguyên `cfg.test_pipeline`:
đọc LiDAR 5 chiều, tối đa 9 sweep quá khứ, test augmentation identity và
`Pack3DDetInputs`. `LoadPointsFromMultiSweeps` giữ `pad_empty_sweeps=True`,
`remove_close=True`. Không có lịch sử thì dùng đúng padding của loader chính
thức, được ghi trong metadata; thiếu file, timestamp tương lai/không giảm,
chuỗi vòng hoặc khác scene gây lỗi. Không tìm sweep tương lai để bù.

Metadata lưu calibration, ego pose, timestamp, file SHA/size và biến đổi vật lý
của từng sweep về LiDAR tham chiếu. `lidar2sensor` được dựng như converter
v1.4.0, rồi chạy loader cài đặt trong môi trường.

**Giới hạn đã kiểm tra:** loader v1.4 đang cài áp dụng `row_xyz @ R - t` với
metadata inverse `lidar2sensor`. Phần tịnh tiến khác phép nghịch đảo SE(3)
đầy đủ. Audit CPU frame smoke đo chênh lệch lớn nhất **0,063285865 m** trong
9 sweep. Metadata ghi `official_loader_translation_difference_m`, biến đổi
vật lý và mô tả này. Giữ nguyên preprocessing chính thức theo yêu cầu; chưa
âm thầm sửa loader. Vì vậy không khẳng định sweep fusion đã trùng phép đổi
SE(3) vật lý tuyệt đối. Box và point cloud hiển thị dùng native `LIDAR_TOP`
của keyframe, không đổi trục để che sai khác này.

Artifact detection riêng, không tạo `occupancy.npz` hoặc BEV thay thế:

```text
infer_job_<uuid>/
    detections.json      # schema_version=1, task=detection, boxes/provenance
    metadata.json        # sample/timestamps/poses/sweeps/config/classes
    report.json          # status/timing/allocator/strict checkpoint check
    resolved_config.py
    job_result.json      # publication receipt, không phải approval
```

`detections.json` chứa token, timestamp sample và LiDAR, class map, source
config/checkpoint/code SHA. Box có `center=[x,y,z]` **geometric center**, mét;
`dimensions=[dx,dy,dz]`; yaw radian, CCW quanh +Z, zero dọc +X;
`velocity=[vx,vy]` m/s; `score` và `label` 0..9. CLI dùng `gravity_center` của
MMDetection3D để đổi từ tensor bottom center. Các lớp theo config: car, truck,
construction_vehicle, bus, trailer, barrier, motorcycle, bicycle, pedestrian,
traffic_cone. Bounds detector `[-51.2,-51.2,-5,51.2,51.2,3]`, pillar
`[0.2,0.2,8]` m; không giả làm grid occupancy.

Validator kiểm tra schema, count (kể cả zero box hợp lệ), số hữu hạn, kích
thước dương, score/label, provenance, resolved config hash và strict checkpoint.
LiDAR keyframe được đối chiếu lại nuScenes sample/sample_data, timestamp,
calibration, ego pose, file hash trong dataset root cho phép.

Viewer giữ native XYZ, camera up `(0,0,1)`, box quay quanh +Z. Test xác định:
center `(10,20,3)`, dimensions `(4,2,6)`, yaw `pi/2` có AABB min `(9,18,0)`,
max `(11,22,6)`; đối chiếu CPU với `LiDARInstance3DBoxes` và hàm TypeScript.

## Queue, API, viewer và review

`web/centerpoint_profiles.json` là file opt-in gồm profile OCCFusion cũ nguyên
nội dung và `centerpoint_pillar02_circle` mới. `web/inference_profiles.json`
mặc định không đổi. Dùng `VIEWER_JOB_PROFILES` để chọn file mới cho **cả web
và worker**. Job snapshot pin checkpoint/config/source/data SHA, FP32,
batch 1, autocast false, CUDA:0, TF32 matmul false/cuDNN true.

Worker dispatch CLI/validator/publication theo task. Detection không gọi
occupancy evaluator hoặc đòi GT root. Cùng SQLite queue và khóa GPU liên tiến
trình hiện có: model ở subprocess riêng, một job trong lane; không giữ cả hai
model trong web/worker. Restart/recovery không tự launch lại job đã chạy.

API chỉ nhận run ID, giữ auth/CSRF/role/origin như cũ:

```text
GET /api/detections
GET /api/detections/{run_id}
GET /api/detections/{run_id}/lidar
GET /api/runs/{occupancy_run_id}/detections
```

Trang `/jobs` chọn profile detector; `/detections?run=...` mở box + LiDAR.
Occupancy explorer có selector overlay. Trước ghép, backend kiểm tra token,
sample/reference timestamp, dataset version, LiDAR sample_data token,
coordinate system, đơn vị, calibration/global pose và hash bundle. Bounds
hai model khác nhau là bình thường; không resample box sang voxel. Run sai
token/hình học/hash không được ghép. Toggle box, class và score chỉ lọc hiển
thị; không ghi lại prediction, chạy model hoặc thay dữ liệu đánh giá.

Detection review có **manifest schema v3/task=detection**, fingerprint chứa
SHA/size bốn file detection ở trên. Reviewer chỉ thấy run đã submitted;
Engineer không được approve. Export cần revision approved, hash vẫn đạt;
ZIP gồm bốn file, `manifest.json` và `review.json`. Không đưa overlay vào
approval/export occupancy v1/v2 cũ. Không cần migration database mới.

Job mới thành công vẫn **unregistered, chưa approved**. `mAP` và `NDS` là
`null`, trạng thái **pending_dataset_evaluation**; sample smoke có detection
GT annotations nhưng chưa chạy đánh giá detection toàn tập. Không ghi “thiếu
GT” hoặc 0 mAP. Score detection và max softmax occupancy chưa hiệu chuẩn.

## Kiểm thử đã chạy

Các lệnh dưới đây là bằng chứng thực thi; hai lệnh GPU đã dùng hết số lượt
được cho phép, **không chạy lại để mở/chụp UI**.

```bash
cd '/mnt/d/New folder/OCCFusion'
.venv-web/bin/python tools/prepare_centerpoint.py
/home/hatis/.venvs/rav25/bin/python -u tools/infer_centerpoint_mini.py \
  --config ckpt/centerpoint_pillar02_v140/configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
  --checkpoint ckpt/centerpoint_pillar02_v140/model.pth \
  --scene-name scene-0103 --frame-index 1 --check-only \
  --out-dir work_dirs/centerpoint_check_cpu

VIEWER_STATE_ROOT="$PWD/work/centerpoint-test-default/state" \
  .venv-web/bin/python -m pytest web/backend/tests -q \
  --junitxml=work_dirs/centerpoint_milestone/backend_all.xml \
  -o faulthandler_timeout=60
VIEWER_STATE_ROOT="$PWD/work/centerpoint-test-default/state" \
  .venv-web/bin/python -m pytest web/backend/tests/test_detection.py -q \
  --junitxml=work_dirs/centerpoint_milestone/detection_after_init_fix.xml \
  -o faulthandler_timeout=60
PATH="$PWD/.cache/toolchains/node/bin:$PATH" npm --prefix web/frontend run typecheck
PATH="$PWD/.cache/toolchains/node/bin:$PATH" \
  VIEWER_API_ORIGIN=http://127.0.0.1:8005 VIEWER_NEXT_DIST_DIR=.next-qa \
  NEXT_TELEMETRY_DISABLED=1 npm --prefix web/frontend run build
.venv-web/bin/python tools/check_web_detection.py

# Lịch sử hai lần launch, không phải hướng dẫn chạy thêm:
.venv-web/bin/python tools/check_web_detection.py --gpu-smoke
.venv-web/bin/python tools/check_web_detection.py --gpu-smoke-after-init-fix
```

| Kiểm tra | Kết quả |
| --- | --- |
| Toàn backend trước GPU | **127 passed**, 479,51 s, không skip; auth/CSRF, viewer, revision, collections, jobs và detection |
| Detection sau sửa init CUDA | **14 passed**, 31,08 s; thêm mock kiểm tra init trước allocator reset, không dùng GPU |
| Typecheck / Next.js QA build | Đạt; routes `/`, `/jobs`, `/benchmarks`, `/detections` |
| CPU metadata/preprocessing/box audit | 9 sweep quá khứ; không truy cập radar/GT làm input; thiếu lịch sử xử lý rõ; geometric center/yaw đạt |
| Chromium worker fixture | Đạt; 2 box **tổng hợp**, gắn nhãn fixture; chưa phải bằng chứng model |
| Chromium GPU thật | Đạt; tạo job qua web → CLI → publication → box + LiDAR → occupancy overlay; không page error |
| Box filtering | Bật/tắt, class/score/reset camera đạt; hash prediction và launch count không đổi |
| Auth/review/export | Anonymous/reviewer bị chặn đúng scope; Engineer không approve; detector v3 approve/export/restart đạt bằng fixture; sửa hash bị chặn |

Backend regression đầy đủ và lần targeted cuối là hai lượt riêng, không cộng
thành số test độc lập mới. Bộ đầy đủ chưa chạy lại sau thêm mock init; targeted
và Chromium thật đã chạy sau bản sửa. Các cảnh queue/restart/lock lỗi dùng
fixture trong bộ jobs cũ, không cần inference GPU.

## Hai lần launch thực tế

**Lượt 1:** job `75c38bb4e7c24fd88340d89af18a1ad8`, workspace
`work/centerpoint-smoke-u9u513bj/`. Lỗi `Invalid device argument 0: did you
call init?` tại `device_setup`, vì gọi reset peak allocator trước khởi tạo
CUDA. **1 launch, 0 forward**, không box, không có phép đo forward/bộ nhớ
hợp lệ. Giữ nguyên report/log và
[`gpu_attempt1_failed.json`](../work_dirs/centerpoint_milestone/gpu_attempt1_failed.json).
Đã sửa thứ tự `set_device → synchronize → reset_peak_memory_stats`, kiểm tra
bằng mock CPU trước khi xin và nhận quyền cho lượt bổ sung duy nhất.

**Lượt bổ sung được cho phép:** job `65e1b30833e343c3b434da8dc71423f9`,
workspace `work/centerpoint-smoke-wyp1nl__/`, output mới:
[`infer_job_65e1b30833e343c3b434da8dc71423f9/`](../work/centerpoint-smoke-wyp1nl__/runs/infer_job_65e1b30833e343c3b434da8dc71423f9/).

| Trường | Giá trị thực |
| --- | --- |
| Scene/frame/split | scene-0103 / 1 / mini_val |
| Sample token | `3950bd41f74548429c0f7700ff3d8269` |
| Reference LiDAR timestamp | `1533151604048025` microseconds |
| Launch / forward hoàn tất / warm-up | **1 / 1 / 0**; không retry |
| Box lưu / qua display score ≥ 0,35 | **151 / 43** |
| Điểm sau test preprocessing | **219.704** (keyframe + sweeps); viewer LiDAR hiển thị keyframe riêng |
| Precision / TF32 | FP32, no autocast; matmul false / cuDNN true |
| Input validation / model+checkpoint CPU / device setup | **2,020540 / 0,512847 / 0,237303 s** |
| Preprocessing / forward / export | **0,493178 / 5,760913 / 0,009136 s** |
| Pipeline / process đến trước report | **6,263331 / 17,688451 s** |
| Peak PyTorch allocated / reserved | **355.857.408 / 387.973.120 bytes** = **339,372 / 370 MiB** |
| Detection prediction SHA-256 | `0ac4f3772d87b6e13ba4c6e83dc9310489bba8bd77f95d8d23592adbcd51b95f` |
| CLI SHA-256 | `8e9655b6d5fe4c324443a90e7b0136a823126ecb87cc1d49c1ac1fdb61569beb` |

Forward được synchronize CUDA trước/sau và gồm decode/circle NMS của
`mode='predict'`. Pipeline gồm test preprocessing, forward, CPU export;
không gồm imports, khởi tạo dataset/model/device hoặc queue. Thống kê allocator
gồm model setup và pipeline, **không phải VRAM vật lý đo độc lập**. Một frame
không warm-up chưa phải benchmark latency ổn định.

Tổng hai lần: **2 launch CLI, 1 forward thực tế, 0 warm-up**. Không chạy lại
model sau forward thành công. Sentinel của cả hai lượt còn nguyên.
[`report.json`](../work_dirs/centerpoint_milestone/report.json) tổng hợp file
thay đổi, lệnh launch nguyên argv, test, timing, allocator và giới hạn.
[`gpu_smoke_evidence.json`](../work_dirs/centerpoint_milestone/gpu_smoke_evidence.json)
và [`browser-evidence.json`](../work/centerpoint-smoke-wyp1nl__/browser-evidence.json)
lưu job/events/argv, box count, schema, hash và kiểm tra WebGL.
Ảnh thật: [Box + LiDAR](../work/centerpoint-smoke-wyp1nl__/detection.png),
[occupancy overlay](../work/centerpoint-smoke-wyp1nl__/occupancy-overlay.png).

## Mở lại kết quả đã lưu, không chạy GPU

QA dùng cổng **8005/3005**, state/root/profile riêng; harness đã dừng cả web
và worker khi kết thúc. Bản đang phục vụ production ở 8000/3000 chưa được
restart hoặc đổi profile trong mốc này. Chạy lại QA để phát lại artifact thật:

```bash
# Terminal A, từ repo. Không chạy worker.
export VIEWER_RUNS_ROOT="$PWD/work/centerpoint-smoke-wyp1nl__/runs"
export VIEWER_STATE_ROOT="$PWD/work/centerpoint-smoke-wyp1nl__/state"
export VIEWER_JOB_PROFILES="$PWD/web/centerpoint_profiles.json"
export VIEWER_EVALUATION_ROOTS="[\"$VIEWER_RUNS_ROOT\"]"
export VIEWER_BENCHMARK_ROOTS='[]'
export VIEWER_ALLOWED_ORIGINS='["http://127.0.0.1:8005","http://127.0.0.1:3005"]'

# Tạo tài khoản cá nhân trong QA một lần; nhập mật khẩu tương tác.
# qa_engineer của harness dùng mật khẩu ngẫu nhiên, không lưu trong Git.
.venv-web/bin/python tools/create_viewer_user.py --username myengineer --role engineer
.venv-web/bin/python -m uvicorn web.backend.app.main:app --host 127.0.0.1 --port 8005
```

```bash
# Terminal B, từ repo; dùng bản .next-qa đã build cho API 8005.
PATH="$PWD/.cache/toolchains/node/bin:$PATH" VIEWER_NEXT_DIST_DIR=.next-qa \
  VIEWER_API_ORIGIN=http://127.0.0.1:8005 \
  NEXT_TELEMETRY_DISABLED=1 node web/frontend/node_modules/next/dist/bin/next \
  start web/frontend --hostname 127.0.0.1 --port 3005
```

Mở `http://127.0.0.1:3005/detections?run=infer_job_65e1b30833e343c3b434da8dc71423f9`
hoặc `http://127.0.0.1:3005/?run=occupancy`, đăng nhập tài khoản QA vừa tạo.
Mở trang/đổi bộ lọc chỉ GET artifact; không có worker nên job mới sẽ chờ.
Không dùng launcher production mặc định để xem QA.

## Bảo toàn và giới hạn

Audit [`preservation_after.json`](../work_dirs/centerpoint_milestone/preservation_after.json)
xác nhận **669 file được bảo vệ** không đổi/mất, gồm model/config/CLI
OCCFusion, bản sửa CSRF, profile cũ, benchmark, inference/evaluation và manifest
approval cũ. Fingerprint profile occupancy vẫn
`459b5877d44853c3e7e09cd4c7a075e718be1f8b45bd231ad1c8346488f4956d`.
Production DB chỉ mở đọc: tài khoản và các hàng run/revision/review cũ còn
nguyên. Production có thêm hoạt động/session/job trong lúc làm việc; không
khẳng định toàn database giống từng byte và không khôi phục đè những thay đổi
đó. Worker production vẫn tắt; hai job đang queued chưa được nhận.

Chưa chạy detection mini_val benchmark, mAP/NDS hoặc kiểm chứng chất lượng
box trên toàn tập. Chưa triển khai CenterPoint vào tiến trình production đang
chạy; đã kiểm chứng QA độc lập. Giới hạn sweep loader nêu trên còn nguyên.
Đây là phát lại prediction một frame độc lập, **chưa phải learned 4D flow**.
