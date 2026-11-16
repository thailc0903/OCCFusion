# Camera viewer, projection và provenance ẩn danh

Mốc này chỉ dùng ảnh/LiDAR nuScenes và prediction đã lưu. Backend không gọi
model, checkpoint, CUDA hoặc evaluator. Worker không khởi động; production
không restart. Giữ nguyên CSRF, profile, model/config, metric và artifact cũ.

## Nguồn ảnh và trạng thái

[nuScenes: Privacy protection](https://www.nuscenes.org/data-format) mô tả
việc nhà phát hành làm mờ mặt và biển số trong ảnh release. Đây là nguồn
provenance của ảnh dataset, không phải kết quả detector PII của RAV25 và
không bảo đảm mọi PII đã được che. Policy nằm ở web/camera_policy.json.

Ba lớp kiểm tra được tách rõ:

1. Source registration: CLI quản trị xác minh camera và ghi hash ảnh release,
   calibration/pose, metadata tables, dimensions, policy/source vào sidecar
   bất biến dưới state web. Chưa phải kiểm tra privacy bởi Engineer.
2. Hiển thị: camera phải khớp sidecar, dataset root cho phép và policy hiện
   tại; mặc định human review là not_reviewed. Thiếu provenance, file sai
   hash/calibration/dimensions, policy thiếu hoặc derivative hỏng sẽ bị chặn,
   có lý do. Không có fallback phục vụ ảnh không xác minh.
3. Engineer kiểm tra đúng từng ảnh, xác nhận bằng checkbox và nhận xét. Mask
   bổ sung là rectangle theo pixel gốc, không suy ra tự động. Tạo PNG RGB giữ
   nguyên dimensions, bỏ metadata nguồn và che đen vùng đã chọn. Ảnh inference
   gốc luôn read-only. Trạng thái này không chứng nhận PII completeness.

Không tự đánh dấu toàn sample/dataset đã kiểm tra. Privacy event ghi người,
thời gian, nhận xét, source/policy hash, masks, derivative hash và revision
trước đó. Dùng khóa file và expected revision chống ghi chồng. Sidecar/hash
hỏng không tự sửa hoặc đăng ký lại. Source registration chỉ pin file thực tế
trong root do người dùng cung cấp; không thay thế việc kiểm chứng archive
release từ nhà phát hành.

## Hình học

Mỗi camera chọn đúng sample_data keyframe của sample/channel trong metadata
nuScenes, không chọn sweep gần nhất hoặc đường dẫn do browser cung cấp.
Kiểm tra token, file, ảnh decode, intrinsic K, extrinsic, image dimensions,
camera ego_pose timestamp, LiDAR sample/frame/reference timestamp và pose.
Nếu metadata inference có camera, đối chiếu cả các trường đã lưu ở đó.
Detector LiDAR-only được bổ sung camera context từ dataset, không cần giả
camera là input của model.

Đặt G_L là global_from_lidar tại thời điểm LiDAR, G_C là global_from_camera
tại thời điểm camera:

~~~text
G_L = global_from_ego(t_lidar) @ ego_from_lidar
G_C = global_from_ego(t_camera) @ ego_from_camera
p_camera = inverse(G_C) @ G_L @ p_lidar
pixel = (K @ p_camera) / camera_depth
~~~

Native LiDAR không được coi là ego axes. Camera optical có X sang phải,
Y xuống dưới, Z ra trước. Box dùng geometric center, dimensions dx/dy/dz,
yaw CCW quanh LiDAR +Z và đúng prediction lưu sẵn; không đọc annotation GT
để tạo overlay.

Projection dùng float64, kích thước ảnh release và intrinsics gốc. Không dùng
ảnh normalized/resized/padded của model. Point source là toàn bộ LiDAR
keyframe, không sweep; depth > 1 m và margin ảnh 1 px như devkit. Box được cắt
từng cạnh ở near-plane 0,1 m trước phép chia depth, rồi cắt Liang-Barsky trong
biên ảnh. Điểm phía sau camera không được chiếu. Không nối cạnh qua depth âm.

SVG viewBox cùng dimensions ảnh, đặt đúng lên ảnh và cùng scroll/zoom. Mask
được đổi từ pointer qua bounding rectangle sang pixel gốc rồi floor/ceil;
không lưu tọa độ CSS. Box score/toggle chỉ lọc hiển thị; không đổi prediction,
metric hoặc gọi inference.

Giới hạn: hai ego poses bù ego-motion giữa timestamp, chưa truyền vị trí box
theo velocity của vật thể, chưa bù chuyển động trong vòng quét LiDAR.
Projection hình học không chứng minh occlusion/visibility. Mốc này không
chiếu occupancy hoặc khẳng định mọi voxel trong frustum nhìn thấy được.
Playback frame độc lập chưa phải learned 4D flow.

## Review/export riêng

Camera nằm dưới quyền xem parent run hoặc benchmark frame. Reviewer cần
parent đã submitted như trước. Quyền xem parent không phải camera approval.
Mask/review ảnh chỉ Engineer được ghi; CSRF/origin/session áp dụng như cũ.

Camera export dùng collection/revision/event hiện có, không migration
database. Tạo scope riêng camera-run-... hoặc camera-frame-...; manifest kind
rav25_camera_frame_collection, task camera, schema 1. Không thay manifest
v1/v2/v3 hoặc collection detection/occupancy cũ.

Chỉ đăng ký camera collection khi cả sáu ảnh đã được Engineer kiểm tra riêng,
có derivative hợp lệ và source pin còn khớp. Trình tự draft → submitted →
approved/rejected/needs_changes giữ vai trò cũ. Export cần đúng current
revision approved, sidecar và hash đạt; kiểm tra lại bytes/revision khi đóng ZIP.

Manifest mới chứa hashes của prediction nguồn, sáu PNG derivative, từng source
provenance, privacy event, calibration/projection JSON và source policy. Source
JPEG không nằm trong ZIP. Metric camera bundle là null, không tạo metric mới.
Đổi mask, privacy event, source, prediction hoặc projection làm approval
không còn áp dụng; cần revision mới. Lịch sử approval cũ được giữ.

Source registration không chỉnh inference artifact và không approve.
Collection xuất từ QA chỉ là bằng chứng workflow QA, không approval production.

## API

~~~text
GET  /api/runs/{id}/cameras
GET  /api/benchmarks/{id}/frames/{sample_token}/cameras
GET  {camera_base}/{channel}/image?version={image_fingerprint}
GET  {camera_base}/{channel}/projection?version={image_fingerprint}
POST {camera_base}/{channel}/privacy
GET  {camera_base}/review
POST {camera_base}/register
POST {camera_base}/review/{submit,decision,revise,notes}
POST {camera_base}/export
~~~

Browser chỉ gửi ID/token/channel, version, masks và review fields. Không có
file path input. Stale image/privacy version bị 409, hash/calibration sai bị
từ chối và UI hiện lý do. Đổi run/frame unmount ảnh/overlay; mở camera là một
thao tác riêng để không tải ảnh vào luồng viewer cũ khi chưa cần.

## Chạy WSL trên QA riêng

Dependency mới chỉ ở web: Pillow 12.3.0, hỗ trợ Python 3.10; requirements.in
và requirements.lock được pin. Không thay dependency rav25.

~~~bash
cd '/mnt/d/New folder/OCCFusion'
.venv-web/bin/python -m pip install -r web/backend/requirements.lock

# State riêng, không dùng launcher production hoặc worker.
export VIEWER_RUNS_ROOT="$PWD/work_dirs"
export VIEWER_STATE_ROOT="$PWD/work/camera-manual/state"
export VIEWER_BENCHMARK_ROOTS="[\"$VIEWER_RUNS_ROOT\"]"
export VIEWER_EVALUATION_ROOTS="[\"$VIEWER_RUNS_ROOT\"]"
export VIEWER_ALLOWED_ORIGINS='["http://127.0.0.1:8007","http://127.0.0.1:3007"]'

# Xác minh và pin sáu source; không đánh dấu Engineer đã kiểm tra.
.venv-web/bin/python tools/register_camera_source.py \
  --benchmark-id benchmark_detection_mini_val \
  --sample-token 3950bd41f74548429c0f7700ff3d8269 \
  --state-root "$VIEWER_STATE_ROOT"

.venv-web/bin/python tools/create_viewer_user.py --username myengineer --role engineer
.venv-web/bin/python -m uvicorn web.backend.app.main:app --host 127.0.0.1 --port 8007
~~~

~~~bash
# Terminal B, build/state/port riêng.
PATH="$PWD/.cache/toolchains/node/bin:$PATH" \
  VIEWER_API_ORIGIN=http://127.0.0.1:8007 VIEWER_NEXT_DIST_DIR=.next-camera-qa \
  NEXT_TELEMETRY_DISABLED=1 npm --prefix web/frontend run build
PATH="$PWD/.cache/toolchains/node/bin:$PATH" \
  VIEWER_API_ORIGIN=http://127.0.0.1:8007 VIEWER_NEXT_DIST_DIR=.next-camera-qa \
  NEXT_TELEMETRY_DISABLED=1 node web/frontend/node_modules/next/dist/bin/next \
  start web/frontend --hostname 127.0.0.1 --port 3007
~~~

Mở http://127.0.0.1:3007/benchmarks?benchmark=benchmark_detection_mini_val,
chọn scene-0103 / frame 1 rồi “Mở sáu camera”. Chọn thumbnail, zoom hoặc
phóng to camera; bật/tắt LiDAR/box. Frame khác cần source registration riêng
khi file/provenance đủ. Với run riêng, CLI dùng --run-id thay vì benchmark/
sample-token. Không có worker nên thao tác hiển thị không nhận job GPU.

## Kiểm chứng projection thực tế

Đối chiếu nuScenes-devkit 1.1.11, map_pointcloud_to_image, LidarPointCloud,
Box, Quaternion và view_points bằng CPU. So sánh toàn bộ điểm cùng nằm trong
frustum và 151 box prediction của smoke thật; không dùng GT box làm prediction.
Web float64 được so với đường PointCloud float32 chính thức và một chuỗi
devkit float64 độc lập.

| Camera | Camera − LiDAR (ms) | Max error so SDK float32 (px) |
| --- | ---: | ---: |
| CAM_FRONT | −35,621 | 0,018956783 |
| CAM_FRONT_RIGHT | −27,543 | 0,027752399 |
| CAM_BACK_RIGHT | −19,655 | 0,040566366 |
| CAM_BACK | −10,467 | 0,035000641 |
| CAM_BACK_LEFT | −0,620 | 0,026957642 |
| CAM_FRONT_LEFT | −43,226 | 0,033431857 |

LiDAR timestamp 1533151604048025 µs. Tất cả ảnh 1600×900 gốc.
Max điểm float64 so SDK float64: 1,2576811635656108e-10 px.
Max box corner float64: 6,953326946095085e-7 px (gồm corner phía trước
near-plane nhưng có thể ngoài ảnh, trước clipping).

Không tuyên bố sai số bằng 0 hoặc frustum membership float32 giống tuyệt đối:
CAM_FRONT có một điểm index 11244, web u=1599,007435 px và SDK float32
u=1598,995342 px, hai phía của biên width−1=1599. Web 3039, SDK32 3040
điểm; 3039 điểm chung dùng tính error. Các camera khác không lệch membership.
Chuỗi devkit float64 có membership trùng web cho cả sáu camera.

projection_audit.json ghi median/p95/max, precision, counts, source hashes,
timestamps và điểm sát biên. Đây là audit hình học, không phải evaluator
chất lượng detection và không đánh giá mAP/NDS lại.

## Các ảnh đã được quan sát trong QA

Chỉ sáu camera của sample 3950bd41f74548429c0f7700ff3d8269,
scene-0103/frame 1 được Codex quan sát bằng view_image ở kích thước gốc,
ghi kế hoạch mask bổ sung quanh vùng mặt/đầu người và biển số nhìn thấy.
manual_inspection_plan.json chứa chính xác filenames, SD tokens/source hashes
và rectangles; không áp dụng tự động cho frame khác.

Các xác nhận trong tài khoản qa_engineer là kiểm tra workflow ở QA riêng,
không thay thế chứng nhận kiểm tra PII đầy đủ của Engineer thực địa.
Không gán mọi ảnh nuScenes là đã được kiểm tra. Frame 0 dùng kiểm tra đổi
frame vẫn chưa đăng ký provenance/chưa được review privacy.

## Lệnh kiểm thử và bằng chứng

~~~bash
VIEWER_STATE_ROOT="$PWD/work/camera-tests-default/state" \
  .venv-web/bin/python -m pytest web/backend/tests/test_cameras.py -q -x \
  -o faulthandler_timeout=120 \
  --junitxml=work_dirs/web_camera_qa/camera_after_png.xml
VIEWER_STATE_ROOT="$PWD/work/camera-regression-default/state" \
  .venv-web/bin/python -m pytest web/backend/tests -q \
  -o faulthandler_timeout=180 \
  --junitxml=work_dirs/web_camera_qa/backend_all.xml

# Chỉ CPU/devkit, không model/torch/evaluator.
MPLCONFIGDIR="$PWD/.cache/matplotlib" PYTHONDONTWRITEBYTECODE=1 \
  /home/hatis/.venvs/rav25/bin/python tools/check_camera_projection.py
PATH="$PWD/.cache/toolchains/node/bin:$PATH" npm --prefix web/frontend run typecheck
# Build QA như trên, rồi harness yêu cầu cổng 8007/3007 trống.
.venv-web/bin/python tools/check_web_cameras.py
git diff --check
~~~

Harness yêu cầu manual_inspection_plan.json có trước và kiểm tra đúng từng
source hash trước khi mô phỏng thao tác Engineer trên QA. Nó tạo tài khoản
ngẫu nhiên/state/artifact copy riêng, không lưu mật khẩu Git, không khởi động
worker; Chromium dùng SwiftShader CPU. Không sửa ảnh nguồn dataset để test lỗi.
Kiểm thử hash/missing image dùng bản sao dataset trong pytest; browser sửa
tạm derivative trong QA rồi phục hồi đúng bytes.

Kết quả thực tế:

| Kiểm tra | Kết quả |
| --- | --- |
| Camera sau sửa PNG | 22 passed, 75,19 s |
| Hồi quy toàn backend | 163 passed, 918,10 s, không fail/skip |
| Camera cuối, JSON hỏng và EXIF/ICC stripping | 26 passed, 84,20 s, không fail/skip |
| TypeScript / Next.js QA build cuối | Đạt; .next-camera-qa/API 8007 |
| Chromium sáu camera thật | Đạt: chọn camera, zoom/phóng to, LiDAR/box, mask, chuyển benchmark frame |
| Quyền và integrity | Anonymous 401, Engineer approve 403, unreviewed/unapproved export 409; hash hỏng bị chặn |
| Review/export/restart QA | Đạt; source/policy/derivative/projection hashes, account/history/approval giữ qua restart |
| Model/evaluator | 0 lần gọi thực; health không có Torch/OpenMMLab; 0 job camera QA, worker tắt |

Hai suite full/targeted có phần lặp lại; không cộng thành số test độc lập.
Full 163 chạy trước ba test JSON malformed, kiểm tra timestamp LiDAR và
stripping metadata cuối; 26 targeted chạy sau các thay đổi đó. Chromium cuối
chạy với guard giao diện và backend stripping metadata đã hoàn thiện.
Regression jobs dùng CLI fixture copy-prediction/copy-evaluation; riêng
test_missing_gt gọi evaluator CPU với GT rỗng và metric null, không tính lại
benchmark hoặc chạy model/GPU. Camera browser/readiness không gọi evaluator.
Một warning Starlette/httpx deprecation có sẵn.

Lượt đầu 15 passed/1 failed trước sửa thứ tự PNG verify/getexif được giữ trong
camera_initial.xml. Chromium đầu đọc state quá sớm sau save; đã thêm chờ POST.
Khi quan sát screenshot của lượt tiếp theo, projection đang tải sau đổi camera;
đã thêm guard identity/ảnh đã load, readiness assertion và build/Chromium lại.
Ảnh overlay cuối xác nhận có cả circle LiDAR và line box, không chỉ ảnh nền.

Nguồn bằng chứng cuối: work/camera-web-0akayoop/:

- [Sáu camera đã có mask QA](../work/camera-web-0akayoop/six-cameras-reviewed.png).
- [CAM_FRONT: box prediction và LiDAR](../work/camera-web-0akayoop/camera-front-projection.png).
- [Review camera approved trong QA](../work/camera-web-0akayoop/camera-review-approved.png).
- [Derivative hash mismatch bị chặn](../work/camera-web-0akayoop/camera-hash-blocked.png).
- [ZIP export QA](../work/camera-web-0akayoop/camera-collection-approved.zip).

ZIP có 31 entries: 29 file nguồn đã kiểm tra SHA/size và manifest/review.
Fingerprint camera revision:
bfd26bf82c542190a31f547fa33e6a7308e78d5f93741776986a5c3942a05756.
ZIP SHA-256:
d8e4ddebe7b0dbf7c08044f0710db2fa509c3629eee0e82739c467ddffd5138d.
Mask được vẽ qua UI ở viewport 1000×1000, zoom 1,5:
yêu cầu [78,428,144,494], lưu [78,427,144,494], trong sai số quantization
floor/ceil 1 pixel gốc. Các nguồn ảnh vẫn nguyên hash.

26 targeted cuối thực sự chạy bằng:

~~~bash
VIEWER_STATE_ROOT="$PWD/work/camera-tests-metadata-final/state" \
  .venv-web/bin/python -m pytest web/backend/tests/test_cameras.py -q -x \
  -o faulthandler_timeout=120 \
  --junitxml=work_dirs/web_camera_qa/camera_metadata_final.xml
# Dependency mới thực sự cài riêng web, không upgrade thư viện khác:
.venv-web/bin/python -m pip install --no-deps --cache-dir .cache/pip Pillow==12.3.0
~~~

Audit bảo toàn: 1.332 file được hash trước, 1.313 giữ nguyên; 19 thay đổi là
source web/.gitignore/dependency web trong phạm vi mốc này, không mất file.
Toàn bộ 1.134 artifact trong tập bảo vệ và sáu source JPEG giữ nguyên hash.
auth.py, database.py, CLI/evaluator/model/config/profile/checkpoint không đổi.
Production SQLite mở read-only, schema 4/quick_check ok; 2 account, 3 run/
revision, 6 review event, 9 job/50 job event cùng nội dung hàng như trước.
Approval/review mới chỉ nằm ở state QA. Cổng QA được dừng khi harness kết thúc.

File thêm: camera_projection.py, camera_sensors.py, cameras.py, camera_routes.py
trong backend; CameraViewer.tsx, lib/cameras.ts; web/camera_policy.json;
test_cameras.py; tools/register_camera_source.py, check_camera_projection.py,
check_web_cameras.py và tài liệu này. Tích hợp vào SensorCatalog, config,
main/collections, private frame metadata ở hai benchmark stores, bốn viewer
và panel cha, CollectionReviewPanel, API/types/CSS. Khóa Pillow ở web/backend/
requirements.in và requirements.lock; .gitignore bỏ qua build QA mới.

Các log/XML, source_registration.json, six_sources.json, projection_audit.json,
manual_inspection_plan.json và preservation audits ở work_dirs/web_camera_qa/.
completion_report.json tổng hợp lệnh, source hashes, test scopes, ảnh và export.
Production chưa nạp tính năng mới; dùng hướng dẫn QA riêng ở trên để mở.
