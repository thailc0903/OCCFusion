# Vận hành RAV25 Occupancy Lab trên WSL

Bản được bàn giao là ứng dụng offline localhost, đọc dataset/prediction đã
lưu và điều phối inference qua worker riêng khi được bật chủ động. Không có
temporal/learned 4D flow. Xem [bảng nghiệm thu](../BASIC_ACCEPTANCE.md).

## Thư mục và môi trường

Deployment đã kiểm chứng dùng /mnt/d/New folder/OCCFusion. Alias viết thường
trên ổ Windows trỏ cùng inode nhưng đường dẫn là một phần của profile
fingerprint và giới hạn GT đã lưu. Vì vậy giữ nguyên cách viết production
khi source cấu hình, build, chạy backend và worker; không sửa metadata/job
hoặc nới kiểm tra file để vượt khác biệt đường dẫn.

~~~bash
cd '/mnt/d/New folder/OCCFusion'
export PATH="$PWD/.cache/toolchains/node/bin:$PATH"
source web/viewer.env.example
~~~

web/viewer.env.example không có credential. Cả hai terminal nếu tách worker
phải source cùng file từ cùng root. Các biến quan trọng:

| Biến | Ý nghĩa |
| --- | --- |
| VIEWER_CONFIG | web/viewer.config.json, base của các đường dẫn relative |
| VIEWER_STATE_ROOT | SQLite account/session/review/job và camera sidecars |
| VIEWER_RUNS_ROOT | Prediction/evaluation đã lưu và staging job |
| VIEWER_DATA_ROOTS | JSON list dataset root cho phép |
| VIEWER_EVALUATION_ROOTS / VIEWER_BENCHMARK_ROOTS | JSON list root được đọc |
| VIEWER_JOB_PROFILES | Profile dùng chung cho backend/worker |
| VIEWER_GPU_LOCK | Cùng file/inode GPU lane giữa mọi worker |
| VIEWER_CAMERA_POLICY | Policy nguồn ảnh và quy tắc derivative/export |
| VIEWER_API_ORIGIN / VIEWER_NEXT_DIST_DIR | Production dùng 127.0.0.1:8000 / .next |

Backend/worker điều phối: .venv-web Python 3.10, lock
web/backend/requirements.lock, gồm Pillow 12.3.0. Frontend: Node >=24 <27,
web/frontend/package-lock.json. Inference: Python
/home/hatis/.venvs/rav25/bin/python riêng. Không resolve bin/python sang
/usr/bin/python hoặc cài web dependency vào rav25. Model/package versions
đã có trong requirements/occfusion-py310-cu121.txt,
requirements/environment-installed.txt và các báo cáo inference.

Dataset cần data/nuscenes/v1.0-mini/*.json cùng samples/sweeps thật. Mini_val
có scene-0103 và scene-0916, tổng 81 sample. Các benchmark lưu trong
work_dirs/benchmark_mini_val và benchmark_detection_mini_val.
SurroundOcc GT nằm riêng dưới data/nuscenes/occ_samples/nuscenes_occ/samples,
không bắt buộc để phát lại prediction hoặc inference không GT.

OCCFusion dùng ckpt/Cam_Lidar_Radar_Atten3D.pth. CenterPoint dùng cặp assets
đã pin dưới ckpt/centerpoint_pillar02_v140; xem nguồn/SHA trong
[CENTERPOINT_DETECTION.md](CENTERPOINT_DETECTION.md). Readiness không tải
weights hoặc sửa dependency để ép chạy. Dataset, weights, artifacts, SQLite
và mật khẩu không nằm trong source package/Git.

## Chuẩn bị và kiểm tra

Các lệnh cài/build dưới đây chỉ dành cho lúc chuẩn bị môi trường hoặc nâng
cấp có kiểm soát; không chạy lại để xem kết quả đã lưu:

~~~bash
python3 -m venv .venv-web
.venv-web/bin/python -m pip install -r web/backend/requirements.lock
npm --prefix web/frontend ci
VIEWER_API_ORIGIN=http://127.0.0.1:8000 VIEWER_NEXT_DIST_DIR=.next \
  NEXT_TELEMETRY_DISABLED=1 npm --prefix web/frontend run build
~~~

Không build đè .next trong lúc Next đang phục vụ. Đợi job đang chạy hoàn tất,
dừng đúng launcher, giữ bản build trước để có thể khôi phục, build đúng
production API, kiểm tra rồi khởi động lại. Không dùng build QA trỏ 8007.
Launcher từ chối VIEWER_NEXT_DIST_DIR/API origin sai hoặc rewrite sai.

~~~bash
# Chỉ đọc config/dependency/metadata/queue/build/cổng, không import main.py:
.venv-web/bin/python -B tools/check_web_readiness.py --mode view

# Đọc/hash đầy đủ model/profile và snapshot benchmark CPU; không xây model:
.venv-web/bin/python -B tools/check_web_readiness.py --mode view --full

# Khi dịch vụ đã chạy, kiểm tra cả workspace_id đúng state:
.venv-web/bin/python -B tools/check_web_readiness.py --mode view --ports serving
~~~

Readiness dùng importlib.metadata cho rav25, không import Torch/OpenMMLab,
không tạo/migrate DB, user, job, camera pin hoặc worker. SQLite mở mode=ro.
--full chỉ xác minh artifact/count/protocol/hash đã lưu, không gọi evaluator.
Report JSON ra stdout; có thể redirect vào thư mục báo cáo riêng.
Exit 1 là not_ready; ready_with_warnings cần đọc từng cảnh báo. Nó không thay
thế kiểm thử GPU/driver; mốc này dùng bằng chứng model/GPU đã lưu.

## Hai chế độ khởi động

~~~bash
# Playback: mặc định an toàn, job queued không được nhận.
VIEWER_START_WORKER=0 bash tools/run_web_viewer.sh
~~~

~~~bash
# Chạy inference: CHỈ dùng khi chủ động muốn xử lý queue hiện có.
VIEWER_START_WORKER=1 bash tools/run_web_viewer.sh
~~~

Mỗi lần chỉ một launcher trên 3000/8000. Hai chế độ dùng cùng state, roots,
profile và lane. Launcher mặc định VIEWER_START_WORKER=0; bật worker có thể
nhận ngay job đang queued, không đợi bạn tạo job mới. Không dùng --once,
--max-jobs 1 hoặc POST tạo job như một phép thử startup.

Nếu tách terminal:

~~~bash
# Terminal A, với environment chung phía trên:
VIEWER_START_WORKER=0 bash tools/run_web_viewer.sh

# Terminal B: cd đúng root, source cùng env; lệnh này xử lý queue thật.
.venv-web/bin/python -u tools/run_inference_worker.py
~~~

Profile mặc định occfusion_clr_fp32 không đổi. Nếu muốn opt-in CenterPoint,
đặt VIEWER_JOB_PROFILES="$PWD/web/centerpoint_profiles.json" trong cả web và
worker trước khi restart; file chứa cả baseline occupancy cũ. Kiểm tra
readiness --mode inference để phát hiện profile/path khác job queued.
Không sửa fingerprint/snapshot của job đã lưu.

## URL, đăng nhập và trạng thái job

- http://localhost:3000/ — occupancy, BEV, point cloud, evaluation và review run.
- http://localhost:3000/benchmarks — occupancy partial/detection complete và review collection.
- http://localhost:3000/detections — detection run đã lưu trong configured root.
- http://localhost:3000/jobs — Engineer tạo/xem job của mình.
- http://127.0.0.1:8000/api/health — health công khai, không cần session.

Đăng nhập bằng tài khoản thật hiện có. Không có mật khẩu mặc định; không
seed user khi startup. Quản trị có thể dùng tools/create_viewer_user.py khi
thực sự cần tài khoản mới; nhập tương tác, không ghi password vào command,
Git hoặc log. Không tạo tài khoản QA trong state production.

Queued nghĩa là đang chờ worker/lane. Với worker tắt, đây là trạng thái đúng.
API jobs hiển thị worker heartbeat; quá 30 s không có heartbeat active thì
không coi worker đó còn sẵn sàng. Không xóa queue hoặc lockfile để xử lý.
Trước khi restart/khởi động worker khác, kiểm tra process/job thực tế; nếu
còn inference đang chạy, chờ hoàn tất. Worker recovery xác minh owner/process,
không tự retry forward job dở. Muốn thử lại cần Engineer tạo job mới riêng.
Ctrl+C khi worker đang chạy có thể dừng CLI; không dùng làm thao tác restart
vô hại giữa job. Không đổi baseline để vượt lỗi/OOM.

Log:

~~~bash
tail -n 80 work_dirs/web_viewer_server.log
# Chọn đúng job_id từ UI, không truyền file path qua browser.
tail -n 80 work_dirs/.inference-jobs/JOB_ID/inference.log
tail -n 80 work_dirs/.inference-jobs/JOB_ID/evaluation.log
~~~

UI job log có quyền owner và offset. Không đưa cookie/header Authorization/
API key/password vào log chẩn đoán. Không sửa SQLite để chuyển failed về
queued. Giữ report/staging lỗi để truy vết.

## Review thật sau bàn giao

1. Engineer chọn run hoặc benchmark đã lưu, xem token, scene/frame, metric,
   protocol, latency source và giới hạn coverage. Job success chưa approved.
2. Đăng ký/gửi parent run hoặc collection để Reviewer được quyền xem.
   Occupancy evaluation mới cần revision v2 tương ứng, detection/pooled
   collection có scope riêng; approval cũ không tự bao gồm dữ liệu mới.
3. Demo camera: chọn detection benchmark → scene-0103 → frame 1 → “Mở sáu
   camera”, sample 3950bd41f74548429c0f7700ff3d8269. Source provenance có
   thể đã được admin đăng ký, nhưng vẫn chưa được Engineer kiểm tra.
4. Engineer kiểm tra từng ảnh ở đúng channel/zoom. Nếu cần, vẽ mask, xác nhận
   đúng ảnh và nhận xét, lưu derivative. Không giả định publisher processing
   hoặc QA là chứng nhận loại hết PII. Các frame khác mặc định chưa kiểm tra.
5. Sau đủ sáu ảnh, đăng ký/gửi camera collection riêng. Reviewer đăng nhập
   tài khoản của mình, xem parent đã submitted và camera revision/fingerprint,
   rồi approved/rejected/needs_changes. Phải gửi parent trước để Reviewer
   được quyền truy cập camera scope.
6. Export chỉ revision approved còn đúng hash. Thay artifact/mask/evaluation
   thì tạo revision mới và duyệt lại; không tự mở rộng approval. Camera ZIP
   chỉ có derivative đã review, calibration/projection/source hashes; không
   có source JPEG. Pooled mAP không tự thuộc approval một frame.

Nếu source chưa được đăng ký, admin có thể chạy đúng thao tác provenance:

~~~bash
.venv-web/bin/python tools/register_camera_source.py \
  --benchmark-id benchmark_detection_mini_val \
  --sample-token 3950bd41f74548429c0f7700ff3d8269 \
  --state-root "$VIEWER_STATE_ROOT"
~~~

Lệnh này không gán human review/approval. Không sao chép SQLite, privacy event
hoặc approval từ QA sang production. Giữ nguồn ảnh inference nguyên trạng.

## Restart, bảo toàn và khắc phục

Trước restart: GET health, kiểm tra queue/heartbeat/process thuộc ứng dụng;
chờ running=0. Chỉ dừng launcher/process có đúng cwd và PID/start-time đã
xác minh; không kill mọi Python/Node. Giữ queue/history/manifest và ảnh nguồn.
Backup state phải tính cả SQLite/WAL hoặc dùng SQLite backup nhất quán;
không restore đè state đang sử dụng. Backup chứa dữ liệu nhạy cảm, không
commit/publish cùng gói mã.

Failed to Load Page: kiểm tra launcher còn chạy, cổng 3000/8000, log startup
và rewrite .next. Trong Windows có thể dùng:

~~~powershell
Invoke-WebRequest http://localhost:3000/ -UseBasicParsing
~~~

Chỉ HTTP 200 của trang login không chứng minh login/review thành công.
QA tích hợp phải kiểm tra bằng hai session riêng. Lỗi CSRF cần chẩn đoán
cookie/header/origin/session; không tắt bảo vệ hoặc retry POST vô điều kiện.
“Bad Request” ở sự kiện Codex task_complete khác HTTP lỗi của ứng dụng;
xem audit trong BASIC_ACCEPTANCE.md trước khi sửa server theo phỏng đoán.

## Phạm vi được và chưa được kiểm chứng

| Môi trường | Trạng thái |
| --- | --- |
| WSL hiện tại, loopback 3000/8000, SQLite, saved artifacts | Được nghiệm thu với logs/QA của bản bàn giao |
| Hai vai trò/session QA, ownership/CSRF/hash/restart | Đã kiểm tra; không phải load test đông người |
| HTTPS cookie flags qua TestClient | Có regression; chưa kiểm tra TLS reverse proxy/certificate thật |
| Dockerfile cũ | Giữ tham khảo upstream; chưa build/test web+worker stack bằng Docker |
| Public Internet, multi-host, SSO, nhiều GPU | Chưa nghiệm thu |
| Dataset validation đầy đủ / PII toàn dataset / learned 4D | Chưa có |

Đây là lab offline, không dùng để điều khiển xe. Các giới hạn metric/loader/
privacy/latency và thao tác review thật còn lại nằm trong bảng nghiệm thu.
