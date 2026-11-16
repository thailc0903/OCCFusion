# RAV25 Occupancy Lab — nghiệm thu bản cơ bản WSL

Ngày nghiệm thu: 2026-11-16 (Asia/Saigon). Mã gốc lúc bắt đầu:
5c977bc5, working tree sạch; tiếp tục từ mốc camera đã hoàn thành.
Không checkout/reset. Các thay đổi ở mốc này dành cho vận hành/readiness/
QA/tài liệu, không thay model, config, checkpoint, profile, protocol hoặc metric.

Kết luận phạm vi: bản lab offline WSL có thể bàn giao **đạt có giới hạn**.
Chọn/duyệt frame đã lưu, các lớp hiển thị, metric và review/export có bằng
chứng; chưa nghiệm thu playback sequence tự động/liền mạch, privacy toàn
dataset, TLS/Docker deployment hoặc learned 4D flow. Nếu các mục này là
điều kiện bắt buộc thì chưa thể gọi toàn bộ PROJECT_SPEC đã hoàn tất.

## Bảng nghiệm thu

“Đạt” là chức năng/phạm vi có bằng chứng, không có nghĩa toàn nuScenes val
hoặc mọi tình huống vận hành đã được kiểm chứng. Metric lấy từ artifact gốc,
không chạy lại model/evaluator để nghiệm thu.

| Yêu cầu | Tính năng và dữ liệu | Lệnh/thao tác kiểm chứng | Bằng chứng | Trạng thái |
| --- | --- | --- | --- | --- |
| Dữ liệu/scene/frame | nuScenes mini, mini_val scene-0103/0916, 81 sample đúng metadata/split | Readiness --full; chọn scene/frame | benchmark manifest, metadata table hashes, readiness report | Đạt trong mini |
| Fusion pretrained | OCCFusion C+L+R, baseline FP32, batch 1, checkpoint strict match | Đọc smoke/benchmark đã lưu; test profile/dispatch, không forward mới | docs/WEB_INFERENCE_JOBS.md; web_job_gpu_smoke_evidence.json; benchmark_mini_val/initial_report.json | Đạt có giới hạn, lịch sử GPU đã kiểm chứng |
| Web → worker | Queue bền vững, một GPU lane, không giữ model trong web, không retry job dở | Full backend fixture: queue/lock/failure/cancel/restart/publication | backend_final.xml; test_inference_jobs.py; smoke cũ | Đạt; worker OFF lúc bàn giao |
| Frame/sequence playback | Duyệt scene/frame theo thứ tự metadata, các prediction độc lập | Chromium đổi frame benchmark, không POST job | integrated_qa.json; ảnh benchmark/camera | Đạt có giới hạn; chưa autoplay sequence hoặc temporal fusion |
| BEV/occupancy 3D | XYZ 200×200×16, voxel 0,5 m, native LIDAR_TOP, bỏ free | Chromium occupancy/BEV/WebGL; test tâm voxel và shape/probability | acceptance-occupancy-metric.png; test_viewer.py | Đạt trên artifact đã lưu |
| Point cloud | Đúng LiDAR keyframe/token/pose; ROI ở 3D, raw keyframe cho projection | API payload, kiểm tra finite/hash, overlay UI | SensorCatalog; camera projection audit; Chromium | Đạt; không sweep cloud giả |
| Detection 3D | CenterPoint Pillar 0,2 m/circle NMS LiDAR-only, geometric center/yaw/dimensions | Xem 81 frame lưu; lọc score/class chỉ hiển thị | detection benchmark; tests detection/box transform | Đạt có giới hạn loader đã công bố |
| Sáu camera/projection | Scene-0103/frame 1, 1600×900, camera/LiDAR ego poses riêng, clipping near-plane/ảnh | Chromium sáu ảnh, zoom/mask; audit devkit đã lưu | docs/CAMERA_VIEWER.md; projection_audit.json; ảnh QA mới | Đạt có giới hạn ego-motion/visibility |
| Occupancy IoU/mIoU | Pooled 79/81, đủ 16 lớp, ignore/empty đúng loader, union=0 → 0 | Kiểm tra saved hashes/statistics/protocol; UI metric frame đã ghép | metrics_pooled.json; OCCUPANCY_EVALUATION.md; reference_check | Đạt có giới hạn, PARTIAL |
| Detection mAP/NDS | Official detection_cvpr_2019, 81/81, 10 lớp, GT support và TP null | Hash-verified immutable evaluation/report, UI fraction → % một lần | DETECTION_BENCHMARK.md; GPU report 367b...; detection-metric-evidence.json | Đạt trong mini_val, COMPLETE |
| Latency/bộ nhớ | Forward GPU gốc và end-to-end, warm-up/count/source riêng | Đối chiếu report/CSV đã lưu; không lấy cache timing | initial_report.json; reports/367b21268aed4eb4852c6b4d221a59c5.json | Đạt có giới hạn, allocator không phải VRAM vật lý |
| Hai vai trò/session | Engineer/Reviewer; hash password, TTL/logout, CSRF/origin, quyền backend | Full backend auth/CSRF/owner; hai Chromium contexts | backend_final.xml; CSRF_BOOTSTRAP_FIX.md; integrated QA | Đạt local; chưa load test đông người |
| Review run/evaluation | Revision v1/v2/v3 và history, approval gắn hash | Fixture approval/export/restart; Engineerapprove bị chặn | test_auth_review.py; test_evaluation_review.py; test_detection.py | Đạt; approval cũ không mở rộng |
| Review collection | Occupancy/detection set và camera snapshot riêng | Full tests permission/hash/export; Chromium camera submit → Reviewer approve | manifest/review trong ZIP QA; collection tests | Đạt; pooled set không thuộc frame approval |
| Privacy | Publisher release provenance; explicit per-image review/mask; derivative PNG giữ dimensions, bỏ EXIF/ICC | Test source/token/calibration/hash; mask UI ở zoom 1,5; export/restart | manual_inspection_plan.json; camera QA ZIP; test_cameras.py | Đạt có giới hạn: chỉ sáu ảnh được quan sát ở QA |
| Vận hành WSL | Env chung backend/worker, hai mode, guard build production, readiness read-only | check_web_readiness.py; launcher tests; localhost deploy verification | WSL_OPERATIONS.md; readiness_after.json; deployment.json; production_http.json | Đạt trên WSL hiện tại, worker OFF |
| Docker | Dockerfile upstream cũ, chưa gồm nghiệm thu web/worker stack | Không build Docker ở mốc này | Dockerfile giữ nguyên | Chưa nghiệm thu |
| TLS/public multi-user | Local HTTP loopback; Secure flags đã unit test HTTPS | Không reverse proxy/TLS/certificate/load test thật | auth/CSRF tests | Chưa nghiệm thu ngoài local |
| Temporal/4D | Chưa model temporal/flow, không flow EPE/ADE/FDE | Không triển khai ở bản cơ bản | PROJECT_SPEC.md, UI playback label | Còn thiếu ở bản nâng cao |

Các path bằng chứng không có prefix trong bảng thuộc work_dirs/basic_acceptance/
cho QA lần này; path benchmark/evaluation gốc thuộc work_dirs tương ứng.
Tài liệu chi tiết liên kết trong README giữ bằng chứng lịch sử của từng mốc.

## Kết quả chất lượng/thời gian gốc được giữ

| Task / scope | Coverage | Metric | Forward median/p95 không warm-up |
| --- | --- | --- | --- |
| OCCFusion mini_val subset thực sự có prediction/evaluation | 79/81, partial | IoU 44,1789%; mIoU 23,0672%, 16 lớp | 32,510879 / 36,368349 s, 78 mẫu từ 79 forward |
| CenterPoint toàn mini_val | 81/81, complete | mAP 41,2801%; NDS 47,7591%, 10 lớp | 0,064662924 / 0,080472662 s, 79 mẫu từ 80 forward mới |

Detection có 1 smoke cache, không đưa cache latency vào thống kê. End-to-end
79 mẫu median/p95 1,649535009/1,741009826 s; common setup tách riêng.
Nguồn occupancy là initial_report.json, detection là report GPU gốc
367b21268aed4eb4852c6b4d221a59c5.json, không phải report resume gần nhất.

Hai scene mini_val chưa đại diện validation đầy đủ. Ba lớp detection không
có filtered GT vẫn hiển thị AP/GT support theo protocol; TP không áp dụng
giữ null. Không dùng display score 0,35 để lọc dữ liệu đánh giá. Không có GT
thì UI “Chưa có GT”, không 0%. Confidence/max softmax chưa hiệu chuẩn.

Hai occupancy frame đầu scene bị từ chối do không có radar quá khứ hợp lệ,
không tự thêm prediction hoặc dùng sweep tương lai. Xem radar_startup_audit.json.
Loader detection khác SE(3) tịnh tiến: max 0,428603971 m, median 0,007184933,
p95 0,193105039, frame max scene-0916/18. Giữ nguyên preprocessing/metric.
Allocator allocated/reserved không phải đo độc lập VRAM vật lý.

Projection CPU đã đối chiếu SDK: max 0,040566366 px so float32 và
1,257681e-10 px so chuỗi SDK float64; một điểm CAM_FRONT sát biên frustum
khác membership float32/64 đã ghi rõ. Offset camera−LiDAR:
front −35,621; front-right −27,543; back-right −19,655; back −10,467;
back-left −0,620; front-left −43,226 ms. Ego-motion compensation chưa bù
chuyển động vật thể/intra-scan, projection không chứng minh visibility.

## Truy vết “Bad Request”

Audit: work_dirs/basic_acceptance/bad_request_audit.json.
Tại 2026-11-15T11:42:00.386Z (18:42:00 Asia/Saigon), session Codex kết thúc
với event task_complete, error.message={"detail":"Bad Request"},
codex_error_info=other, sau output công cụ cuối thành công. Log không gắn
endpoint RAV25 hoặc server traceback. Không tìm thấy HTTP 400/Bad Request
trong log camera backend/frontend và production đã quét.

Kết luận được hỗ trợ là lỗi completion của phiên công cụ, không phải một
request camera API đã xác định. Endpoint/root cause dịch vụ bên ngoài không
được log cung cấp; không suy ra lỗi auth/projection và không sửa ứng dụng
theo phỏng đoán. Lỗi automatic approval review tương tự ngày 13/11 là sự
kiện khác, không lấy làm nguyên nhân của phiên camera.

## Mã và vận hành mới

README mở đầu RAV25, giữ nguyên nội dung/citation/Model Zoo OCCFusion, phân
biệt số upstream với số dự án. Thêm hướng dẫn WSL, env example, readiness
read-only và integrated QA. Launcher mặc định worker OFF, chặn mode/build
QA sai production API. Không đổi backend model/auth/review/projection logic.

Đường dẫn chuẩn giữ nguyên fingerprint occupancy
459b5877d44853c3e7e09cd4c7a075e718be1f8b45bd231ad1c8346488f4956d.
Lượt test ban đầu chạy từ alias lower-case có 5 lỗi OutsideRoot tại
evaluation_validation, mặc dù hai root cùng inode; không sửa hash/artifact
hoặc nới containment. Full suite cuối chạy đúng canonical production path.
Đây là giới hạn portability của absolute-path provenance đã lưu.

Readiness chỉ đọc package metadata, config, dataset/profile/artifact hashes,
build và queue/cổng; không import main.py, tạo/migrate DB, user, job, model,
worker hoặc tự download. --full không chạy evaluator. Kết quả không thay
thế thử CUDA/driver mới; mốc này không chạy thêm GPU.

## Kiểm chứng/bàn giao cuối

Full backend sau camera cuối, readiness tests, TypeScript/build QA và
Chromium tích hợp dùng state/build/port riêng. Fixture jobs copy
prediction/evaluation cũ; riêng test_missing_gt gọi CLI evaluator CPU với
GT rỗng, trả metric null, không tính lại benchmark. Không model/GPU.
Chromium/readiness không gọi evaluator. Production không dùng
làm fixture và không có account/approval QA được seed/chuyển sang.

Chi tiết lệnh, exit code, số test, URL/PID/build, gói nguồn và bảo toàn được
ghi trong work_dirs/basic_acceptance/completion_report.json khi chốt bàn giao.
Các giới hạn/thiếu dữ liệu trong bảng không tự chuyển thành “đạt” vì test
phần mềm đã pass.

Kết quả thực tế trên mã cuối:

- Full backend: **175 passed, 1 warning**, 913,35 s; warning có sẵn về
  Starlette/httpx. Tám test readiness thuộc 175 test này, không cộng hai lần.
  [Log](work_dirs/basic_acceptance/backend_final.log),
  [JUnit](work_dirs/basic_acceptance/backend_final.xml).
- Lượt full đầu từ alias lower-case: **162 passed, 5 failed**, giữ nguyên
  [log lỗi](work_dirs/basic_acceptance/backend_all.log). Cả năm lỗi là
  OutsideRoot của GT path khác cách viết canonical, không phải metric giả
  hoặc lỗi CSRF. Fixture readiness ban đầu 7/8 do thiếu cột created_at trong
  DB giả; bổ sung đúng schema fixture, lượt cuối 8/8.
- TypeScript typecheck, build QA API 8007 và build production API 8000 đều
  thành công. Không sửa frontend ứng dụng hoặc dependency lockfile.
- Chromium QA: occupancy/BEV/WebGL/metric → detection benchmark/camera →
  Engineer submitted → Reviewer approved → export → restart; sai quyền/hash
  vẫn bị chặn, không POST job. State/build/port QA tách riêng.
  [Báo cáo](work_dirs/basic_acceptance/integrated_qa.json),
  [occupancy](work_dirs/basic_acceptance/screenshots/acceptance-occupancy-metric.png),
  [sáu camera](work_dirs/basic_acceptance/screenshots/six-cameras-reviewed.png),
  [projection](work_dirs/basic_acceptance/screenshots/camera-front-projection.png),
  [approval QA](work_dirs/basic_acceptance/screenshots/camera-review-approved.png),
  [hash bị chặn](work_dirs/basic_acceptance/screenshots/camera-hash-blocked.png).
- ZIP **chỉ QA** có 31 entries, 29 file nguồn được kiểm tra hash, fingerprint
  60c14855506013b5ad7bcf60076b5c5b6d4290353af4d3429661e1975998794e,
  SHA-256 58b3f828983aa62db52b47430f21126cb9cf589897df527e2e027656322b4bd7.
  [ZIP bằng chứng](work_dirs/basic_acceptance/camera-collection-approved-QA.zip),
  không có source JPEG, không chuyển approval sang production.
- Production localhost:3000 và API 8000 chạy cùng state thật, build mới
  fKCMQIKsJA429zBWRm3hJ, workspace 2a8ee3391278f125. Có đúng một backend,
  worker OFF, **3 queued / 0 running**. HTTP trang 200/API chưa đăng nhập 401.
  Không có session trình duyệt người dùng để thử login production; kiểm tra
  đăng nhập/review hai vai trò ở QA, giữ nguyên account/session thật.
- Readiness cuối: **ready, 15/15 kiểm tra pass**; Windows PowerShell xác nhận
  localhost 3000 trả 200 và health 8000 đúng workspace. Chromium production
  mới xác nhận trang login/API 401, không tạo account. Lần mở Chromium trực
  tiếp đầu thiếu LD_LIBRARY_PATH cho libnspr4.so; dùng đúng browser-libs cache
  đã có của QA thì đạt, không sửa/cài dependency ứng dụng.
- Sáu source camera được đăng ký vào production với human_review=not_reviewed;
  không có privacy derivative, Engineer annotation hoặc camera approval mới.
  Người dùng phải thực hiện review thật bên dưới.
- Đối chiếu bảo toàn: 1.576 file baseline, 1.569 file không đổi; năm file
  source/tài liệu thay có chủ đích và hai file runtime log/PID đổi khi restart.
  Không có file thiếu hoặc thay đổi ngoài dự kiến. Sáu ảnh gốc và **75 bản
  ghi trong 12 bảng production**, gồm account/session/history/approval/jobs,
  giữ nguyên hash. Sáu provenance sidecar mới không chứa human review.

Lệnh kiểm chứng chính (từ canonical root; log trong thư mục bằng chứng):

~~~bash
VIEWER_CONFIG="$PWD/web/viewer.config.json" \
  VIEWER_STATE_ROOT="$PWD/work/basic-final-canonical/state" \
  .venv-web/bin/python -m pytest web/backend/tests -q \
  -o faulthandler_timeout=180 \
  --junitxml=work_dirs/basic_acceptance/backend_final.xml
npm --prefix web/frontend run typecheck
VIEWER_API_ORIGIN=http://127.0.0.1:8007 VIEWER_NEXT_DIST_DIR=.next-camera-qa \
  NEXT_TELEMETRY_DISABLED=1 npm --prefix web/frontend run build
.venv-web/bin/python tools/check_basic_acceptance.py
VIEWER_API_ORIGIN=http://127.0.0.1:8000 VIEWER_NEXT_DIST_DIR=.next \
  NEXT_TELEMETRY_DISABLED=1 npm --prefix web/frontend run build
# Sau khi đã source web/viewer.env.example và khởi động production:
.venv-web/bin/python -B tools/check_web_readiness.py --mode view --full --ports serving
~~~

Không chạy inference/GPU/huấn luyện. Hai lượt full suite gọi nhánh evaluator
CPU missing_gt với GT fixture rỗng, metric null; không chạy lại evaluation
benchmark. Readiness và Chromium chỉ đọc kết quả đã lưu.

## Gói bàn giao

- Mã nguồn hiện tại và lockfiles:
  work_dirs/basic_acceptance/rav25-basic-wsl-2026-11-16.tar.gz.
- Danh sách file/SHA-256, commit gốc và deployment build:
  work_dirs/basic_acceptance/release_manifest.json. Đây là working tree đã
  kiểm chứng dựa trên 5c977bc5, không giả là một release Git đã commit/tag.
- Bằng chứng lệnh/test/ảnh/export QA và bảo toàn:
  work_dirs/basic_acceptance/completion_report.json,
  preservation_before.json và preservation_after.json.

Gói nguồn không chứa dataset, GT, weights, build/node_modules, môi trường
Python, SQLite/account/session, ảnh nguồn hoặc mật khẩu. Giữ dataset/model/
state thật riêng, cấu hình đúng root rồi cài từ lockfile/build theo hướng
dẫn WSL. Các liên kết bằng chứng work_dirs là bằng chứng local đi kèm trong
workspace, không phải dữ liệu được nhúng vào source archive.

Hai lệnh chạy từ cùng environment ở docs/WSL_OPERATIONS.md:

~~~bash
VIEWER_START_WORKER=0 bash tools/run_web_viewer.sh
VIEWER_START_WORKER=1 bash tools/run_web_viewer.sh
~~~

Lượt bàn giao chỉ dùng lệnh thứ nhất. Không khởi động đồng thời hai launcher.
Worker OFF giữ nguyên queue; bật worker sau này có thể nhận job queued ngay.
Không tạo job để thử startup hoặc xóa queue.

Người dùng còn cần đăng nhập bằng account thật, kiểm tra/gửi parent
run/collection, kiểm tra từng ảnh camera và tạo derivative nếu cần, gửi
camera revision riêng; Reviewer quyết định trên UI rồi mới export.
Source registration demo không đồng nghĩa human review/approval.
