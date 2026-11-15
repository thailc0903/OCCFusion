# Detection benchmark viewer và collection review

Mốc này chỉ đọc prediction/evaluation đã lưu trong
work_dirs/benchmark_detection_mini_val. Không chạy inference, GPU, huấn luyện
hoặc evaluator trong HTTP request. Backend chỉ dùng môi trường web,
NumPy và các validator hiện có; không import Torch/OpenMMLab, CLI hoặc evaluator.
Không sửa baseline/preprocessing, CSRF, profile, checkpoint, artifact gốc
hoặc approval trước đây. Không restart production.

## Dữ liệu và đơn vị

Trang /benchmarks có bộ chọn task Occupancy / Detection. Benchmark occupancy
cũ giữ fingerprint, protocol, revision và export cũ. Detection dùng
CenterPoint PointPillars 0.2 m, circle NMS, LiDAR-only, FP32/batch 1.

Detection hiện có đủ mini_val: scene-0103 (40 sample) và scene-0916 (41 sample).
Danh sách được đối chiếu lại chuỗi sample, LiDAR keyframe và metadata nuScenes;
không lấy danh sách 79 prediction occupancy. Hai frame đầu scene không có
lịch sử radar vẫn có prediction detection thật.

Metric lấy từ JSON đã lưu, không hardcode trong UI:

| Bằng chứng hiện tại | Giá trị |
| --- | ---: |
| Coverage | Complete, 81/81 |
| mAP | 0.41280115564259257 → 41,2801% |
| NDS | 0.47759091555233457 → 47,7591% |
| Box đã lưu | 16.086 |
| Box sau official filters | 13.317 |
| GT sau official filters | 3.393 |

API giữ fraction, frontend dùng Intl.NumberFormat với style percent đúng một
lần. Đủ 10 lớp, kể cả trailer, construction_vehicle và barrier không có GT
sau official filters trong hai scene này. AP chính thức của lớp đó vẫn được
hiển thị cùng nhãn “Không có GT lớp”. TP error không áp dụng giữ null và hiển
thị “Không áp dụng”; không thay bằng 0. Đơn vị TP error riêng: translation m,
orientation rad, velocity m/s, scale/attribute ratio.

Protocol là detection_cvpr_2019, nguồn
nuscenes.eval.detection.evaluate.DetectionEval, devkit 1.1.11 theo artifact.
Trang hiển thị cấu hình protocol và provenance. Đây là phạm vi mini_val,
không phải toàn bộ nuScenes validation. Score hiển thị chưa được hiệu chuẩn.

## Timing và giới hạn loader

Nguồn timing được ghim là report GPU gốc:
reports/367b21268aed4eb4852c6b4d221a59c5.json.
Đối chiếu invocation, token, origin frame và timings.csv trước khi hiển thị.

| Scope | Median | p95 | Số mẫu |
| --- | ---: | ---: | ---: |
| Forward mới, bỏ forward đầu invocation | 0,064662924 s | 0,080472662 s | 79 |
| End-to-end frame tương ứng | 1,649535009 s | 1,741009826 s | 79 |

Invocation có 80 forward mới, 1 prediction smoke dùng cache, không warm-up
ngoài manifest. Forward mới đầu tiên đánh dấu warm-up vẫn tính accuracy.
Scope, khởi tạo CPU/model/device, invocation wall time và source SHA xuất
hiện trong phần mở rộng. Không dùng latency đọc cache hoặc report resume.
Peak allocated/reserved là thống kê PyTorch allocator, không phải VRAM vật lý.

Audit 711 sweep quá khứ giữ nguyên giới hạn loader đã đo: median
0,0071849328593709344 m, p95 0,19310503901559473 m, max
0,42860397132690353 m. Frame lớn nhất scene-0916 / 18, sample
8092909473464f80b9f791a4d31ddcb8. Có nút mở frame này. Viewer không sửa
loader hoặc metric để che sai khác với SE(3).

## Kiểm tra trước khi ghép

DetectionBenchmarkStore kiểm tra:

- Manifest/provenance/protocol fingerprint; bộ 13 bảng nuScenes được hash
  lại trong dataset root cấu hình; membership và thứ tự toàn mini_val.
- Report GPU thật có forward/model load; không dùng invocation cache hoặc
  evaluate-only để suy ra timing inference.
- Evaluation snapshot khớp đúng SHA/size trong report GPU, evaluation commit
  và hash official summary/details; normalized metrics phải khớp summary.
- Token coverage của prediction global, evaluation và toàn bộ frame;
  tất cả bundle có prediction_commit hoàn chỉnh và hash còn nguyên.
- Input keyframe/sweep, timestamp/scene/prev chain, padding, calibration,
  ego pose, SE(3), loader audit, config/checkpoint/code/pipeline provenance.
- Box native-LiDAR khớp JSON global: geometric center, w/l/h, quaternion,
  velocity, score và policy attribute prediction. Không gọi evaluator lại.
- AP đủ lớp/ngưỡng distance, metric finite, support, warm-up/timing/allocator
  khớp report và CSV.

Cache kiểm tra hình học dùng hash nội dung artifact/input/manifest/commit/
dataset/protocol/global prediction. Mỗi lần mở/export vẫn đọc và hash lại
file; không dùng mtime làm quyền xuất. File ngoài root, symlink thoát root,
pending/missing/ambiguous snapshot hoặc hash/protocol/token sai bị từ chối.
UI xóa metric cũ khi đổi task, đổi benchmark hoặc xác minh thất bại.

## Frame playback và occupancy overlay

Chọn scene/frame để đọc detections.json và LiDAR keyframe thật, trong
LIDAR_TOP, mét, +Z hướng lên. Dùng lại box transform và WebGL renderer hiện
có. Score mặc định 0,35, class, toggle box/LiDAR và reset camera chỉ thay
hiển thị; prediction, metric và fingerprint không thay đổi.

Các frame benchmark không cần publication receipt của queue. Không tạo job,
không giả receipt, không tạo occupancy.npz. Chỉ frame smoke nhập trước đây
mới giữ nguyên job_result.json thật trong bundle.

Overlay có thể lấy từ run occupancy riêng hoặc frame benchmark occupancy
đã lưu. Backend kiểm tra quyền nguồn overlay, hash, sample token, sample/
reference timestamp, LiDAR sample_data token, calibration, hệ tọa độ và
dataset. Khác bounds giữa pillar và occupancy không làm đổi box; voxel
occupancy dùng grid/bounds/voxel center đã kiểm chứng. Frame occupancy thiếu/
lỗi không thành overlay. Đổi frame xóa overlay cũ.

Overlay phục vụ hiển thị, nằm ngoài approval/export detection. Approval
collection không cấp quyền xem nguồn occupancy chưa gửi review.

## Collection và export riêng

Không cần migration database. Dùng lại bảng collection/revision/event và
phân quyền Engineer đăng ký/gửi, Reviewer quyết định. CSRF/origin/session
giữ nguyên. Job/single-frame approval/occupancy collection không cấp approval
cho detection collection.

Manifest collection mới: schema_version 2, task detection, kind
centerpoint_detection_benchmark_collection. Fingerprint chứa:

- Manifest benchmark, resolved config, input audit và metadata table hashes.
- predictions_nuscenes.json, protocol cùng protocol fingerprint.
- Evaluation snapshot bất biến, evaluation_commit và official summary/details.
- Report GPU gốc, timings.csv.
- Toàn bộ 81 frame: detection/metadata/report/resolved config, prediction
  commit; receipt smoke thật nếu có.

Hiện có 416 file nguồn được hash. progress.json, benchmark_report.json và
evaluation.json pointer ở root không thuộc snapshot được duyệt. Lựa chọn
evaluation dựa trên hash report GPU ghim, không dựa pointer invocation
cache mới nhất. Thêm/sửa artifact làm fingerprint/kiểm tra integrity thay đổi;
cần revision mới và approval tương ứng. Revision cũ giữ lịch sử.

Export yêu cầu current revision approved, sidecar đúng và toàn bộ hash đạt.
Từng file được hash rồi ghi đúng bytes đã kiểm tra vào ZIP; kiểm tra lại
revision/sidecar trước khi hoàn tất. ZIP gồm 416 file nguồn, manifest.json
và review.json; review ghi nguồn metric/timing, protocol fingerprint,
người duyệt và lịch sử. Không chứa raw dataset LiDAR hoặc overlay occupancy.
Không tự approve kết quả mới trong production.

## API

Giữ nguyên endpoint collection, bổ sung dispatch theo kind manifest:

~~~text
GET  /api/benchmarks
GET  /api/benchmarks/{benchmark_id}
GET  /api/benchmarks/{benchmark_id}/frames/{sample_token}
GET  /api/benchmarks/{benchmark_id}/frames/{sample_token}/lidar?bundle=...
GET  /api/benchmarks/{benchmark_id}/frames/{sample_token}/overlays
GET  /api/benchmarks/{benchmark_id}/frames/{sample_token}?occupancy_run_id=...
GET  /api/benchmarks/{benchmark_id}/frames/{sample_token}?occupancy_benchmark_id=...
GET  /api/collections/{benchmark_id}/review
POST /api/collections/{benchmark_id}/register
POST /api/collections/{benchmark_id}/review/submit
POST /api/collections/{benchmark_id}/review/decision
POST /api/collections/{benchmark_id}/review/revise
POST /api/collections/{benchmark_id}/export
~~~

Chỉ nhận ID/token trong root cấu hình, không nhận file path từ browser.
Anonymous bị chặn; Reviewer chỉ xem collection đã submitted.
Engineer không được gọi decision. Ghi dữ liệu cần CSRF/origin như trước.

## Chạy QA trong WSL

Không dùng launcher/worker production cho mốc này. Build QA riêng:

~~~bash
cd '/mnt/d/New folder/OCCFusion'
PATH="$PWD/.cache/toolchains/node/bin:$PATH" \
  VIEWER_API_ORIGIN=http://127.0.0.1:8006 \
  VIEWER_NEXT_DIST_DIR=.next-detection-benchmark-qa \
  NEXT_TELEMETRY_DISABLED=1 npm --prefix web/frontend run build

# Harness sao chép artifact, tạo tài khoản ngẫu nhiên trong SQLite QA riêng,
# chạy Chromium và restart chỉ backend QA; dừng cả hai server khi kết thúc.
.venv-web/bin/python tools/check_web_detection_benchmark.py
~~~

Harness yêu cầu cổng 8006/3006 trống, dùng state/root mới trong
work/detection-benchmark-web-*/. Không ghi mật khẩu trong Git và không tạo
tài khoản QA trong production. Worker không được khởi động; không gọi model
hoặc evaluator. Chromium dùng SwiftShader CPU; không dùng GPU inference.

Để tự mở trang, chọn state riêng và tài khoản cá nhân:

~~~bash
# Terminal A; không chạy worker.
export VIEWER_RUNS_ROOT="$PWD/work_dirs"
export VIEWER_STATE_ROOT="$PWD/work/detection-benchmark-manual/state"
export VIEWER_EVALUATION_ROOTS="[\"$VIEWER_RUNS_ROOT\"]"
export VIEWER_BENCHMARK_ROOTS="[\"$VIEWER_RUNS_ROOT\"]"
export VIEWER_ALLOWED_ORIGINS='["http://127.0.0.1:8006","http://127.0.0.1:3006"]'
.venv-web/bin/python tools/create_viewer_user.py --username myengineer --role engineer
.venv-web/bin/python -m uvicorn web.backend.app.main:app --host 127.0.0.1 --port 8006
~~~

~~~bash
# Terminal B; dùng QA build phía trên, không thay .next production.
PATH="$PWD/.cache/toolchains/node/bin:$PATH" \
  VIEWER_API_ORIGIN=http://127.0.0.1:8006 \
  VIEWER_NEXT_DIST_DIR=.next-detection-benchmark-qa \
  NEXT_TELEMETRY_DISABLED=1 node web/frontend/node_modules/next/dist/bin/next \
  start web/frontend --hostname 127.0.0.1 --port 3006
~~~

Mở http://127.0.0.1:3006/benchmarks?benchmark=benchmark_detection_mini_val.
Dừng QA trước khi chạy harness. Thay đổi mật khẩu/tài khoản chỉ trong state
đã chọn. Không chạy lệnh benchmark/evaluator để mở trang.

## Bằng chứng và giới hạn

Các lệnh thực sự đã chạy (không cần chạy model để lặp kiểm tra viewer):

~~~bash
VIEWER_STATE_ROOT="$PWD/work/web-detection-benchmark-regression/default" \
  .venv-web/bin/python -m pytest web/backend/tests -q \
  -o faulthandler_timeout=180 \
  --junitxml=work_dirs/web_detection_benchmark_qa/backend_all.xml
VIEWER_STATE_ROOT="$PWD/work/web-detection-benchmark-focused/default" \
  .venv-web/bin/python -m pytest \
  web/backend/tests/test_detection_benchmark_collections.py -q -x \
  -o faulthandler_timeout=180 \
  --junitxml=work_dirs/web_detection_benchmark_qa/targeted_backend_final.xml
PATH="$PWD/.cache/toolchains/node/bin:$PATH" npm --prefix web/frontend run typecheck
# Build QA và Chromium: đúng các lệnh trong phần trên.
git diff --check
~~~

| Kiểm tra thực tế | Kết quả |
| --- | --- |
| Hồi quy toàn backend | 140 passed, 838,44 s, không failed/skip |
| Detection collection sau bổ sung overlay benchmark/symlink | 13 passed, 443,67 s, không failed/skip |
| TypeScript / Next.js QA build | Đạt, routes cũ và mới giữ nguyên |
| Chromium | Đạt: metric units, task/frame switch, box/LiDAR, hai overlay, filters, WebGL |
| Auth/review/export | Engineer approve 403; draft export 409; Reviewer approve/export đạt |
| Artifact bị sửa | API metric 422, export 409, metric cũ được xóa khỏi UI |
| Restart backend QA | Tài khoản, approval, fingerprint và export history còn nguyên |
| Không chạy model | Health model_modules_loaded=false; 0 job QA, 0 request tạo job; worker tắt |

Hai suite backend là các lượt riêng, có test lặp lại; không cộng thành 153
test độc lập. Lượt đầy đủ chạy trước bổ sung overlay benchmark/symlink cuối;
suite targeted và Chromium kiểm tra lại các đường bị ảnh hưởng. Một warning
Starlette/httpx deprecation có sẵn; không đổi dependency để xử lý ngoài phạm vi.
Python compile và diff whitespace check đạt.

Lượt Chromium đầu dừng vì bộ chọn role alert trùng vùng thông báo mặc định
của Next.js; lúc đó API đã trả 422 đúng. Đã sửa selector trong harness và chạy
lại đạt. Build trong sandbox ban đầu bị lỗi đọc TypeScript showConfig;
build native vào thư mục QA riêng đạt. Không có kiểm tra GPU/evaluator nào
được chạy để sửa UI.

Ảnh từ Chromium đạt:

- [Toàn trang detection](../work/detection-benchmark-web-krgpn3yw/detection-overview.png).
- [Box + LiDAR + occupancy benchmark](../work/detection-benchmark-web-krgpn3yw/detection-occupancy-overlay.png).
- [Reviewer approved](../work/detection-benchmark-web-krgpn3yw/detection-collection-approved.png).
- [Hash mismatch bị chặn](../work/detection-benchmark-web-krgpn3yw/detection-hash-mismatch.png).

[ZIP export QA](../work/detection-benchmark-web-krgpn3yw/detection-collection-approved.zip)
có 418 entries, 416 file nguồn đã đối chiếu SHA/size, không có occupancy.npz.
Fingerprint revision:
77c603a8282d4a446276cb327160b44606f124fa9ea4c9e282daf976a6155892.
ZIP SHA-256:
e5f88659fdc92ea707abd5c8ef3dd5b6f098ab7faca113355540cc37d28aaa70.
Approval này chỉ ở SQLite QA; không approve hoặc sửa approval production.

Audit preservation_after.json: 1.264/1.265 file bảo vệ giữ nguyên, không mất
file. Thay đổi duy nhất trong nhóm bảo vệ là .gitignore để bỏ qua build QA
mới. Toàn bộ 1.156 artifact cũ, CLI/model/config/checkpoint/profile/CSRF và
dependency lockfile được giữ nguyên. Các hàng production: 2 account,
3 run/revision, 6 review event, 8 job/49 job event không đổi. Database mở
read-only để so sánh hash nội dung hàng; không dùng làm fixture.
PID launcher/backend/frontend production 96561/96565/96566 giữ nguyên.
Cổng QA 8006/3006 đã dừng, worker production vẫn tắt.

File đổi: backend detection_benchmarks.py mới và benchmarks.py,
collections.py, detections.py, main.py; frontend DetectionBenchmarkPanel.tsx,
DetectionBenchmarkFramePlayer.tsx, lib/detectionBenchmarks.ts mới và
BenchmarkViewer.tsx, CollectionReviewPanel.tsx, lib/types.ts, app/globals.css;
test_detection_benchmark_collections.py, tools/check_web_detection_benchmark.py,
tài liệu này và .gitignore. Không đổi auth/database/dependency/schema cũ.

Log, XML, ảnh, bản export QA và audit bảo toàn nằm trong
work_dirs/web_detection_benchmark_qa/ và thư mục QA được browser_evidence.json
ghi lại. Các bước kiểm tra bao gồm auth/CSRF, token/protocol/hash mismatch,
snapshot/revision/export/restart, task/frame switch, phần trăm, TP null,
WebGL và overlay từ run/benchmark occupancy.

Đọc docs/DETECTION_BENCHMARK.md về nguồn chất lượng và giới hạn hai scene.
Các kiểm tra ở mốc viewer không phải chạy lại evaluation hoặc benchmark.
Hash đảm bảo bộ file được duyệt giữ nguyên, không chứng nhận chất lượng
thực địa. Lần mở đầu có thể mất vài chục giây do hash input và kiểm tra
81 frame; mạng/ổ WSL chậm có thể tăng thời gian này.

Production giữ tiến trình/build/state/approval trước đó; cần đợt triển khai
riêng mới phục vụ UI này ở cổng production. Playback frame độc lập
**chưa phải learned 4D flow**.
