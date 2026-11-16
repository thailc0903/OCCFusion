# Inference một frame từ web

Mốc này bổ sung job vào viewer RAV25 hiện có. Dùng lại CLI
[infer_nuscenes_mini.py](../tools/infer_nuscenes_mini.py) và
[evaluate_occupancy.py](../tools/evaluate_occupancy.py); không thay model,
checkpoint, benchmark viewer hoặc giao thức review/export.

Task CenterPoint LiDAR-only được mô tả tại
[CENTERPOINT_DETECTION.md](CENTERPOINT_DETECTION.md), dùng profile opt-in riêng,
CLI/validator/artifact detection riêng và cùng GPU lane. Profile/job cũ không
có `task` vẫn là occupancy và giữ fingerprint cũ. Các bước evaluator và
artifact occupancy dưới đây áp dụng cho task occupancy.

## Chạy trong WSL

Không thêm dependency. Backend/worker điều phối dùng `.venv-web`, frontend
dùng npm lockfile hiện có; tiến trình CLI dùng Python rav25 riêng.

```bash
cd '/mnt/d/New folder/OCCFusion'
export PATH="$PWD/.cache/toolchains/node/bin:$PATH"
NEXT_TELEMETRY_DISABLED=1 npm --prefix web/frontend run build
VIEWER_START_WORKER=1 bash tools/run_web_viewer.sh
```

Mở **http://localhost:3000/jobs**, đăng nhập bằng tài khoản Engineer hiện có.
Khi VIEWER_START_WORKER=1, launcher chạy FastAPI, Next.js và worker. Mặc định
launcher hiện tại tắt worker để chỉ phát lại artifact. Worker chờ queue
không nạp model. Ctrl+C gửi shutdown cho cả ba. Có thể tách terminal:
Worker dừng ngoài dự kiến không làm tắt web; heartbeat/queue thể hiện việc
không có worker. Khởi động worker lại bằng terminal riêng khi cần.

```bash
# Terminal web: không tự khởi động worker
VIEWER_START_WORKER=0 bash tools/run_web_viewer.sh

# Terminal worker: dùng cùng VIEWER_* config/state/root với backend
.venv-web/bin/python -u tools/run_inference_worker.py

# Vận hành giới hạn, xử lý tối đa một job rồi thoát
.venv-web/bin/python -u tools/run_inference_worker.py --max-jobs 1

# Thử lấy queue một lần; không có job thì thoát
.venv-web/bin/python -u tools/run_inference_worker.py --once
```

Các lệnh chạy worker thật có thể xử lý job queued; dùng công cụ fake QA bên
dưới khi chỉ kiểm thử phần mềm. Không có mật khẩu mặc định; tạo tài khoản theo
[AUTH_REVIEW.md](AUTH_REVIEW.md). Reviewer chỉ xem run được gửi review, không
tạo/xem/hủy job của Engineer. Engineer chỉ truy cập job của mình.

## Luồng sử dụng

1. Engineer chọn profile, scene và frame. Danh sách frame/token/timestamp lấy
   từ chuỗi sample trong metadata mini; membership mini_train/mini_val được
   chốt từ split chính thức nuscenes-devkit, không suy từ scene index.
2. Bấm **Tạo job inference** gửi POST có CSRF. API ghi job + event trong SQLite,
   trả **202** ngay; không khởi chạy model trong request. UI GET polling đọc
   queue/log, không tạo job mới. Request key chống tạo trùng khi gửi lại cùng
   request; cùng key nhưng khác tham số/profile trả 409.
3. Worker claim job theo thời gian tạo, gọi CLI trong Python rav25 qua argv cố
   định, không qua shell. Log và các phase được lưu; có Chờ, Đang chạy,
   Thành công, Thất bại, Đã hủy. Không có worker thì job giữ trạng thái chờ.
4. Kiểm tra prediction rồi chạy evaluator CPU riêng. Có GT → đối chiếu và
   ghép evaluation. Thiếu GT → job vẫn thành công với `missing_gt`, metric
   **null / Chưa có GT**. Evaluation lỗi/hash/token/hình học sai → job thất bại;
   prediction giữ ở staging để chẩn đoán, chưa công bố bundle chưa hợp lệ.
5. **Mở kết quả trong viewer** chọn đúng run qua ID trong URL. BEV, occupancy
   3D, LiDAR và evaluation đọc artifact đã lưu. Đổi bộ lọc/reset camera không
   chạy CLI. Thời gian forward/pipeline/khởi tạo/tiến trình của CLI vẫn tách
   riêng; thời gian chờ job/đánh giá/kiểm tra không phải latency inference.
6. Run mới **chưa đăng ký review, chưa submitted, chưa approved**. Engineer
   đăng ký snapshot v1 rồi thêm evaluation bằng revision v2 như
   [WEB_EVALUATION_REVIEW.md](WEB_EVALUATION_REVIEW.md). Export kèm evaluation
   cần đúng revision v2 được Reviewer duyệt và kiểm tra lại hash. Approval cũ
   không tự mở rộng; collection benchmark có namespace riêng như trước.

## Profile baseline và phân tách môi trường

Admin config [web/inference_profiles.json](../web/inference_profiles.json)
chỉ có `occfusion_clr_fp32`: `configs/OccFusion_infer_mini.py` kế thừa baseline,
`Cam_Lidar_Radar_Atten3D.pth`, C+L+R, CUDA:0, batch 1, FP32/autocast false,
TF32 matmul **false**, cuDNN **true**. Giữ ảnh BGR không resize,
`[1,6,3,928,1600]`, bounds `[-50,-50,-5,50,50,3]`, XYZ `[200,200,16]`,
voxel 0.5 m và chính sách chỉ dùng sweep timestamp ≤ LiDAR như CLI đã kiểm chứng.

Snapshot profile trong mỗi job có fingerprint, checkpoint SHA/size, hash
config/CLI/model source và 13 bảng metadata. Worker kiểm tra byte trước khi
launch; thay code/config/checkpoint không được âm thầm dùng profile cũ.
Giữ nguyên đường dẫn `rav25/bin/python` khi thực thi: resolve symlink sang
`/usr/bin/python` sẽ mất môi trường virtualenv.

Web backend và worker điều phối không import Torch/OCCFusion/OpenMMLab hoặc
CLI. Chỉ subprocess inference nạp model; evaluator subprocess dùng NumPy/CPU,
không dùng GT làm đầu vào inference. Browser chỉ gửi profile ID, scene, frame,
request key; không được truyền đường dẫn, command, device hoặc precision.

`GET /api/health` có `job_submission_enabled=true` và
`inference_enabled=false`: cờ sau chỉ mô tả **tiến trình web**, không phủ nhận
worker riêng. `model_modules_loaded=false` kiểm tra các module trong web.

Có thể cấu hình bằng `VIEWER_JOB_PROFILES` và `VIEWER_GPU_LOCK`; các root/state/
origin vẫn dùng config cũ. Backend và worker phải trỏ cùng SQLite và root.
Mọi worker dùng chung GPU lane lock mặc định
`work/inference-worker/gpu.lock`, kể cả khi chạy hai terminal/state khác nhau.
Khóa điều phối các worker này; CLI chạy thủ công không tham gia queue.

## Lưu bền vững, hủy và restart

Migration SQLite **v3 → v4** chỉ thêm `inference_jobs`,
`inference_job_events`, `inference_workers` và unique index một job running.
Users, sessions, run/collection revisions, approvals, manifest và lịch sử cũ
giữ nguyên. Không seed tài khoản hoặc tự tạo review khi startup.

Ngoài SQLite transaction/unique index, `flock` liên tiến trình khóa toàn job,
kể cả validation/evaluation. CLI kế thừa lock FD; Linux shim dùng parent-death
signal trước exec. Worker chết thì CLI bị TERM; lane chỉ rảnh khi process thực
sự thoát. PID kèm `/proc` start ticks tránh nhầm PID tái sử dụng. Backend restart
không giết worker đang chạy; worker mới không nhận lại job do worker còn sống.

- Hủy job chờ: canceled ngay, launch count 0.
- Hủy job chạy: ghi cờ; worker TERM process group, đợi, cần thì KILL và chỉ
  xác nhận canceled sau khi process thoát. Partial output/log giữ riêng.
- Shutdown worker đang chạy: dừng CLI, ghi failed (`WorkerStopped`), không
  chuyển job về queued.
- Worker bị kill/crash: worker kế tiếp chờ lock/process, xác nhận owner chết,
  ghi failed/canceled cho job dở. **Không tự retry inference**. Launch intent
  lưu trước Popen và mỗi job chỉ được launch tối đa một lần.
- Crash sau publication hoàn chỉnh nhưng trước commit DB: worker mới chỉ
  khôi phục trạng thái success sau khi xác minh receipt, hash, profile,
  prediction/evaluation; không gọi CLI thêm lần nữa.
- Muốn thử lại job lỗi cần Engineer tạo job mới có UUID/output mới. Không
  sửa baseline để tự vượt lỗi/OOM hoặc dùng sweep tương lai/điểm thay thế.

Mỗi job dùng output riêng:

```text
work_dirs/.inference-jobs/<job_id>/
    inference.log
    evaluation.log
    prediction/                 # private trong lúc chạy/lỗi
    evaluation/
work_dirs/infer_job_<job_id>/    # chỉ publish sau kiểm tra toàn bundle
    occupancy.npz, metadata.json, report.json, bev.png
    resolved_config.py
    job_result.json             # receipt riêng, không sửa 4 artifact inference
work_dirs/eval_job_<job_id>/evaluation.json
```

Stage ẩn bị loại khỏi danh sách run/evaluation. Publish dùng rename trong
cùng root, không ghi đè thư mục cũ; `job_result.json` chứa token, profile,
mode thực thi, hash/size 5 file prediction và evaluation, fingerprint toàn
receipt. Run có tiền tố `infer_job_` cần receipt complete và hash còn đúng mỗi
khi mở. Đây là bằng chứng publication, **không phải manifest approval**.
Command/run_dir trong artifact vẫn ghi đúng vị trí staging tại thời điểm CLI
thực thi. Publication chỉ chuyển thư mục, không viết lại JSON; liên kết sau
chuyển dựa trên receipt, token và hash byte.

Kiểm tra trước publication gồm status success, đủ file, NPZ shape/nhãn/
probability hữu hạn/histogram, đúng sample/scene/frame, ảnh/cảm biến/bounds/XYZ,
LiDAR calibration/ego pose từ dataset, config/checkpoint/source SHA,
strict checkpoint match, precision/TF32. Evaluation phải CPU, đúng protocol
16 lớp/union-zero=0, geometry, token và hash bốn artifact; GT basename/hash/size
được kiểm tra lại khi có GT. Không giả định NPZ có sample token.

## API

Mọi API cần đăng nhập, POST cần cookie + CSRF + origin như luồng cũ.

```text
GET  /api/jobs/options                  # profile + scene/frame mini metadata
GET  /api/jobs                          # chỉ job của owner; heartbeat worker
POST /api/jobs                         # {profile_id,scene_name,frame_index,request_key}
GET  /api/jobs/{id}
POST /api/jobs/{id}/cancel
GET  /api/jobs/{id}/log?kind=inference|evaluation&offset=0
```

Log trả tối đa 64 KiB/lần với next_offset. Job record/event/log giữ bền vững;
API không nhận log path. Precision và TF32 ghi riêng; allocator allocated/
reserved vẫn không được gọi là VRAM vật lý thực đo. Max softmax probability
chưa hiệu chuẩn. Một frame smoke không phải benchmark latency ổn định.

## Kiểm thử và bằng chứng

```bash
.venv-web/bin/python -m pytest web/backend/tests -q \
  -o faulthandler_timeout=60 --junitxml=work_dirs/web_jobs_all_backend_pytest.xml
PATH="$PWD/.cache/toolchains/node/bin:$PATH" npm --prefix web/frontend run typecheck
.venv-web/bin/python tools/check_web_jobs.py

# Chỉ sau fake QA đạt, đúng một GPU smoke test đã được yêu cầu:
.venv-web/bin/python tools/check_web_jobs.py --skip-build --real-smoke
```

Fake QA chạy tại cổng **8003/3003**, `.next-qa`, SQLite/root riêng. Fake CLI chỉ
chép prediction/evaluation thật đã lưu làm fixture, ghi mode fixture rõ ràng;
không tạo occupancy/metric thay thế. Backend tests gồm CLI lỗi/thiếu output,
sai token/hash, CSRF/quyền/owner, idempotency, queue, lock giữa process, cancel,
worker kill/restart không lặp inference, phục hồi publication, thiếu GT bằng
evaluator CPU thật, viewer và review/export revision. Chromium kiểm tra form,
cancel queued/running, queue hai worker, lỗi/log và mở artifact đã publish.

Real smoke tạo QA Engineer/session/database riêng để giữ tài khoản/approval
production; output UUID mới trong work_dirs nên Run explorer production đọc
được sau reload. Lịch sử job smoke thuộc SQLite QA, không chuyển ownership sang
tài khoản production. Harness chặn chạy smoke lặp bằng sentinel
`work_dirs/web_job_gpu_smoke_intent.json`, worker `--max-jobs 1`, Playwright
không retry hoặc POST lần hai. Nếu UI assertion lỗi sau inference thì kiểm tra
job/artifact đã lưu, không chạy lại GPU để chụp ảnh.

Kết quả thực thi và ảnh được ghi vào `work_dirs/web_jobs_fixture_evidence.json`,
`web_job_gpu_smoke_evidence.json` và thư mục QA nêu trong từng báo cáo.
Lệnh nguyên argv do worker thực thi nằm ở
[web_jobs_execution_commands.json](../work_dirs/web_jobs_execution_commands.json)
và hai event `inference_launch`/`evaluation_launch` trong SQLite QA.

## Kết quả thực tế ngày 2026-11-14

Smoke thành công là job **`414c1a8312064803b5b855ced5082182`**, QA workspace
`work/job-smoke-rcp5n9tw/`; Engineer thao tác form bằng Chromium.
Prediction mới:
[`infer_job_414c1a8312064803b5b855ced5082182/`](../work_dirs/infer_job_414c1a8312064803b5b855ced5082182/).
Evaluation:
[`eval_job_414c1a8312064803b5b855ced5082182/evaluation.json`](../work_dirs/eval_job_414c1a8312064803b5b855ced5082182/evaluation.json).

| Kiểm tra | Kết quả thực tế |
| --- | --- |
| Toàn bộ backend cuối | **82 passed**, 418.25 s; không skip; gồm auth, viewer, review v1/v2, collection và jobs |
| Production build / TypeScript | Đạt; routes `/`, `/benchmarks`, `/jobs`; build cuối exit 0 |
| Backend queue/worker/review job | **16 passed**, 156.77 s; không inference GPU |
| Bổ sung cross-state lock + backend restart khi fake CLI chạy | **1 passed**, 14.16 s |
| Hồi quy benchmark collection | **19 passed**, 274.60 s; chỉ artifact/CPU |
| Diagnostic auth/review v1/v2 | **27 passed**, 61.78 s |
| Stress CSRF | **1.500** signed tokens + POST idempotent đạt; một job queued trong SQLite tạm, không worker |
| Chromium fake worker | **4 passed**, 38.3 s; queue hai worker, cancel, lỗi/log, viewer/evaluation và backend restart |
| Chromium một GPU smoke | **1 passed**, khoảng 2.9 phút; không retry/POST job thứ hai |
| Scene/frame/split | **scene-0103 / 1 / mini_val**, một frame |
| Sample token | `3950bd41f74548429c0f7700ff3d8269` |
| Baseline | FP32, batch 1, autocast false; TF32 matmul false/cuDNN true; ảnh/grid không đổi |
| Checkpoint | Strict load **1.814 tensor**; missing/unexpected/shape mismatch đều rỗng |
| Occupancy IoU | **0.40266122155903605 = 40.266122%** |
| Semantic mIoU | **0.12447001208028746 = 12.447001%**, đủ 16 lớp; union=0 cho IoU=0 |
| Evaluator gốc | Frame và pooled một frame đều passed; max absolute difference **0.0** |
| Preprocessing / forward / export | **1.957866 / 27.281115 / 0.588720 s** |
| Pipeline / tiến trình CLI đến report | **29.829619 / 66.562060 s** |
| Khởi tạo suy ra (tiến trình trừ pipeline) | **36.732441 s** |
| Job wall claimed → succeeded | Khoảng **150.536 s**, gồm preflight/CLI/validation/evaluation; không phải latency forward |
| Peak PyTorch allocated / reserved | **5.719.002.112 / 6.245.318.656 byte**; không phải VRAM vật lý đo độc lập |
| Publication/viewer | BEV/3D/LiDAR đúng sample, evaluation matched, số launch **1** |
| Review/export | Job vẫn **unregistered**, export trả **409**; không tự approved |
| Restart backend smoke | Tài khoản/session/job/events còn, byte artifacts không đổi |
| Localhost production | Chromium `/jobs` **200**, chưa đăng nhập API jobs **401**, không page error; health không import model |
| Windows localhost | PowerShell `Invoke-WebRequest http://localhost:3000/jobs` trả **200** |
| Bảo toàn | **666** file nguồn cũ không đổi; bảng account/review/approval cũ giữ nguyên; schema v4, SQLite quick_check ok |

SHA-256 mới:

```text
occupancy.npz   d41a7e5e328c07f973bc2f21fb9ab06249ea15af20fc6033511be67d2ca460f0
evaluation     c62f263f3f2110a981ed471439851840977738fd2d00d57e32822ddd82f6519d
GT             c6cba8ec0514dec08cc6ae6cd44647229564ec456b67934c4d16565215aba059
```

Metric trên UI đọc evaluation mới; không sao chép số của run/benchmark cũ.

Bằng chứng:

- [Fake QA](../work_dirs/web_jobs_fixture_evidence.json),
  [real smoke QA](../work_dirs/web_job_gpu_smoke_evidence.json).
- [Ảnh job thật](../work/job-smoke-rcp5n9tw/real-job-completed.png),
  [viewer prediction + LiDAR + evaluation thật](../work/job-smoke-rcp5n9tw/real-viewer.png).
- `work/job-smoke-rcp5n9tw/browser-evidence.json`, `browser-results.json`;
  CLI logs tại `work_dirs/.inference-jobs/414c1a8312064803b5b855ced5082182/`.
- [Job backend tests](../work_dirs/web_jobs_pytest.xml),
  [cross-state](../work_dirs/web_jobs_cross_state_pytest.xml),
  [collection regression](../work_dirs/web_jobs_collection_regression_pytest.xml).
- [Toàn bộ 82 backend tests](../work_dirs/web_jobs_all_backend_pytest.xml),
  [build/test commands và exit code](../work_dirs/web_jobs_final_checks.json),
  [log cuối](../work_dirs/web_jobs_final_checks.log).
- [Trước migration](../work_dirs/web_jobs_preservation_before.json),
  [audit cuối](../work_dirs/web_jobs_preservation_after.json),
  backup `work_dirs/web_jobs_pre_migration.sqlite3`.
- [Localhost/Windows/worker đang phục vụ](../work_dirs/web_jobs_localhost.json),
  [ảnh login production](../work_dirs/web_jobs_local_login.png).

Production đã restart bằng `bash tools/run_web_viewer.sh`, launcher PID ghi
trong `work_dirs/web_viewer_server.pid`, log ở `web_viewer_server.log`.
Mở `http://localhost:3000/jobs` bằng account Engineer hiện có để tạo job của
mình; kết quả smoke có thể mở trong Run explorer bằng
`/?run=infer_job_414c1a8312064803b5b855ced5082182` sau đăng nhập.

Trong thời gian kiểm tra cuối, tài khoản production `engineer` đã tạo thêm
**4 job** (09:18–09:24 UTC): hai success (`scene-0103/1`, `scene-0916/32`),
hai failed ở input_validation (`scene-0757/0`, `scene-0916/0`) do thiếu radar
past sweep. Đây là job từ account thật sau restart, không phải các POST của
harness QA. Giữ nguyên toàn bộ output/log/history này; audit ghi riêng chúng.
Chỉ **một** job GPU smoke do harness QA tạo, không chạy lại benchmark 79 frame.
Session production đã xoay khi account thật đăng nhập; audit tách thay đổi
runtime này và giữ phiên hiện tại, không restore session cũ từ backup.

Trong một lượt hồi quy TestClient đã gặp 403 CSRF gián đoạn ở luồng review cũ;
không tái hiện trong diagnostic 27 ca và stress 1.500 token. Không giảm bảo vệ
CSRF hoặc thay session/account production để làm test qua; nguyên nhân hiện
tượng TestClient gián đoạn chưa được xác định. Lượt cuối đủ 82 ca đạt. Các lượt build/
pytest foreground bị SIGTERM (143) được chạy lại qua tiến trình tách riêng,
có log/status tại `work_dirs/web_jobs_final_checks.*`.

Race CSRF này đã được tái hiện bằng lịch xác định và sửa ở mốc tiếp theo:
[CSRF_BOOTSTRAP_FIX.md](CSRF_BOOTSTRAP_FIX.md). Bootstrap authenticated không
còn xoay hash; không đổi frontend jobs hoặc chạy lại GPU smoke để kiểm tra.

## File thuộc mốc này

| Nhóm | File |
| --- | --- |
| Profile/queue/validation/worker | `web/inference_profiles.json`; `web/backend/app/jobs.py`, `job_results.py`, `job_worker.py` |
| API/config/migration/publication gate | `web/backend/app/main.py`, `config.py`, `database.py`, `store.py`, `evaluations.py` |
| Worker CLI/shim/launcher | `tools/run_inference_worker.py`, `job_subprocess.py`, `run_web_viewer.sh` |
| Trang jobs/tích hợp viewer | `web/frontend/app/jobs/page.tsx`; `components/JobsViewer.tsx`, `AuthGate.tsx`, `Viewer.tsx`; `lib/jobs.ts`; `app/globals.css` |
| Test/QA | `web/backend/tests/test_inference_jobs.py`, `job_support.py`; schema assertion `test_evaluation_review.py`; `web/frontend/e2e/jobs.spec.ts`, `jobs-smoke.spec.ts`; `tools/check_web_jobs.py` |
| Docs | File này; liên kết trong `INFERENCE_MINI.md`, `OCCUPANCY_EVALUATION.md`, `WEB_BENCHMARK_COLLECTIONS.md`, `WEB_VIEWER.md` |

Không thay CLI inference/evaluator, config/model source, dependency hoặc
lockfile. `.gitignore` hiện có đã bao phủ state/log/output/cache mới.

Phạm vi vẫn là một frame OCCFusion. Viewer phát lại prediction độc lập, **chưa
phải learned 4D flow**. Mốc này chưa bổ sung temporal model, detector hay MP4.
