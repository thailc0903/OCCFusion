# Benchmark mini_val trên viewer và review collection

Mốc tiếp theo [WEB_INFERENCE_JOBS.md](WEB_INFERENCE_JOBS.md) thêm trang `/jobs`
và worker một frame. Trang benchmark/collection trong tài liệu này vẫn chỉ
đọc artifact đã lưu; không khởi chạy lại benchmark khi mở trang.

Áp dụng [PROJECT_SPEC.md](../PROJECT_SPEC.md),
[BENCHMARK_MINI_VAL.md](BENCHMARK_MINI_VAL.md) và
[WEB_EVALUATION_REVIEW.md](WEB_EVALUATION_REVIEW.md).
Mốc này chỉ phát lại artifact có sẵn. Backend web không import model,
checkpoint, Torch/OpenMMLab, CLI inference hoặc GT evaluator. Playback các
prediction độc lập **chưa phải learned 4D flow**.

## Chạy và sử dụng trong WSL

Không thêm dependency; dùng `.venv-web` và npm lockfile hiện có, tách rav25.

```bash
cd '/mnt/d/New folder/OCCFusion'
export PATH="$PWD/.cache/toolchains/node/bin:$PATH"
NEXT_TELEMETRY_DISABLED=1 npm --prefix web/frontend run build
bash tools/run_web_viewer.sh
```

Mở **http://localhost:3000/benchmarks**, đăng nhập account hiện có. Run explorer
có liên kết sang trang Benchmark. Không có tài khoản/mật khẩu mới mặc định;
cách cấp account vẫn theo [AUTH_REVIEW.md](AUTH_REVIEW.md).

1. Engineer chọn `benchmark_mini_val`. Lượt mở đầu xác minh toàn bộ bundle,
   có thể mất vài chục giây. Màn hình hiển thị partial **79/81**, hai lỗi radar,
   accuracy pooled, bảng 16 lớp và confusion matrix 17×17 gồm empty/free.
2. Mở phần audit để xem token, LiDAR timestamp, độ lệch từng radar, prev rỗng,
   file có sẵn và số record hợp lệ. Hai frame lỗi vẫn ở danh sách chọn frame.
3. Chọn scene/frame để xem BEV, occupancy 3D, LiDAR đúng sample và evaluation
   của chính frame đó. Lớp nền bỏ khỏi InstancedMesh; tọa độ và tâm voxel dùng
   helper đã kiểm chứng của viewer. Bộ lọc lớp/Z và reset camera chạy ở client.
4. Frame lỗi không có prediction/evaluation; metric hiện không khả dụng, không
   lấy metric frame trước, không tạo dữ liệu thay thế hoặc kích hoạt inference.
5. Engineer nhập nhận xét và đăng ký **collection**, sau đó gửi review.
   Reviewer chỉ thấy collection đã được gửi, có thể nhận xét và chọn approved,
   rejected hoặc needs_changes. Engineer không có quyền gọi API quyết định.
6. Export chỉ mở khi revision collection hiện tại approved và hash/sidecar
   còn đúng. Partial có thể được review với phạm vi **79 frame**, không biến
   coverage thành complete. Revision draft mới vẫn phải được duyệt, kể cả khi
   fingerprint trùng bản cũ vì byte chưa đổi.

## Nguồn dữ liệu và latency

Nguồn mặc định là `work_dirs/benchmark_mini_val/` và
`work_dirs/radar_startup_audit.json`. Browser chỉ gửi benchmark ID, collection
ID và sample token; không gửi đường dẫn hoặc tên file tùy ý.

`benchmark_roots` là danh sách thư mục **cha** chứa benchmark; mặc định bằng
runs_root. Có thể cấu hình trong web/viewer.config.json hoặc môi trường:

```bash
export VIEWER_BENCHMARK_ROOTS='["/mnt/d/New folder/OCCFusion/work_dirs"]'
```

Chỉ mở manifest kind `occfusion_occupancy_benchmark`, scope mini_val, với đủ
membership theo sample chain của metadata nuScenes đã hash. Pilot hai frame
không được trình bày thành full mini_val. Trùng ID giữa các root gây lỗi.
Mọi file đều bị giới hạn trong root, kể cả symlink. Path tuyệt đối trong
progress chỉ cung cấp tên attempt: server tự dựng
`<benchmark>/frames|evaluations/<token>/<attempt>`. Path bên trong JSON không
được dùng để mở file ngoài root, nên bản sao QA vẫn giữ byte artifact gốc.

| Thông tin | Nguồn và quy tắc |
| --- | --- |
| Coverage/status | Manifest + progress + metrics_pooled + benchmark_report phải khớp |
| Pooled accuracy | metrics_pooled.json; cộng counts của đúng các token success trước tính tỷ lệ |
| Semantic mIoU | Đủ 16 lớp, union=0 nhận IoU=0; tỷ lệ [0,1] format percent đúng một lần |
| Confusion matrix | 17×17, hàng GT/cột prediction; bỏ GT ignore, có empty/free |
| Latency inference | **initial_report.json**, latency_current_invocation.excluding_marked_warmup |
| Chứng cứ latency | CSV lọc đúng session GPU gốc, prediction_source=new, forward đã hoàn tất |
| Khởi tạo/tiến trình | bootstrap, setup, model initialization và invocation wall tách riêng |
| Allocator | Peak allocated/reserved từ initial_report; không phải VRAM vật lý thực đo |
| Frame accuracy | evaluation.json cùng token và prediction hash, không lấy pooled làm metric của frame |
| Confidence | Max softmax probability chưa hiệu chuẩn |

Current benchmark_report.json hiện là lượt evaluate-only: latency invocation
đó có 0 mẫu. Viewer **không dùng** nó thay inference latency. Báo nguồn, SHA,
session, số mẫu, warm-up và phạm vi đo cạnh median/p95. Thiếu hoặc sai báo cáo
gốc/CSV gây trạng thái dữ liệu không hợp lệ, không fallback sang timing cache.

## Kiểm tra trước khi ghép

- Hash manifest/identity/provenance và metadata tables; membership/token/order
  theo nuScenes sample chain. Không tin scene index hoặc đường dẫn do browser gửi.
- Receipt prediction và evaluation: status, token, hash/size, preprocessing
  fingerprint, checkpoint/config/precision/TF32, GT hash và protocol fingerprint.
- NPZ keys/shape/nhãn/probability hữu hạn, report histogram/logits shape, class
  map/palette, bounds/voxel/XYZ và reference LiDAR. Không giả định NPZ có token.
- LiDAR sample, filename, timestamps, calibration, ego pose và hash byte phải
  khớp dataset được cấu hình và provenance prediction.
- Protocol từng evaluation khớp pooled; counts/IoU/mIoU nhất quán; tổng confusion
  của các token success bằng pooled. Không nạp GT hay tính metric mới từ GT.
- Audit radar ghép bằng token/timestamp/metadata source hash, kiểm tra lại file
  hash và số record cùng scene/channel. Không kết luận có bug từ thiếu history.

JSON được đọc lại/hash byte, không dựa mtime/size để chấp nhận bundle. Cache
validation theo nội dung (tối đa 256), packed render LRU theo cấu hình. URL
payload chứa fingerprint bundle; request cũ khi bundle thay đổi trả 409. Bản
đang ghi dở .pending không được nhận là success. Schema/hash/token/geometry sai
không được ghép; UI bỏ metric/prediction cũ và hiện lý do.

## Migration và manifest collection

SQLite nâng **v2 → v3**, chỉ thêm `collections`, `collection_revisions`,
`collection_events`. Users, sessions, runs, revisions, review_events,
revision_evaluations và sidecar manifest v1/v2 giữ nguyên. Không đăng ký,
submit hoặc approve collection khi startup. Account/quyền/CSRF dùng cơ chế cũ.

Sidecar mới nằm riêng tại:

```text
work/viewer/collection-manifests/<collection_id>/<fingerprint>.json
```

Manifest kind **occfusion_benchmark_collection**, schema_version=1 thuộc
namespace riêng, không phải manifest run v1. Core chứa:

- Membership 81 frame, trạng thái từng frame, danh sách 79 token đã đánh giá,
  prediction/evaluation SHA từng frame; frame lỗi không có hash prediction giả.
- Hash/size của bốn artifact, resolved config và hai receipt từng frame;
  evaluation.json; manifest/progress/pooled/original/latest report, timings CSV
  và radar audit. Tất cả được đặt tên tương đối trong archive.
- Scope partial, protocol fingerprint, token sources từ metadata/report/
  evaluation, npz_contains_sample_token=false và nguồn latency gốc.
- Fingerprint SHA-256 của canonical JSON core, chưa có chính fingerprint.

Register/revise tạo draft và ghi actor/time/comment. Submit/decision/export
kiểm tra lại bundle và sidecar so DB. Byte hoặc membership thay đổi làm approval
cũ không áp dụng cho bộ mới. Lịch sử không bị sửa; Engineer owner tạo revision
mới khi cần. Approval run v1/v2 không cấp quyền export collection và ngược lại.

## API và export

Tất cả route cần đăng nhập; POST cần session + CSRF như API run. Reviewer chỉ
được xem collection có current revision đã submitted. Chỉ owner Engineer sửa/
gửi revision; Reviewer quyết định. Stale revision/fingerprint trả 409.

```text
GET  /api/benchmarks
GET  /api/benchmarks/{id}
GET  /api/benchmarks/{id}/frames/{sample_token}
GET  /api/benchmarks/{id}/frames/{sample_token}/{occupancy|lidar|bev}
GET  /api/collections/{id}/review
POST /api/collections/{id}/register                     {comment}
POST /api/collections/{id}/review/notes                 {comment}
POST /api/collections/{id}/review/revise                {comment,revision_id,fingerprint}
POST /api/collections/{id}/review/submit                {comment,revision_id,fingerprint}
POST /api/collections/{id}/review/decision              {comment,revision_id,fingerprint,decision}
POST /api/collections/{id}/export
```

Frame lỗi trả detail riêng; payload route trả 409. Hash/schema không hợp lệ
trả 422 cho detail, còn workflow/export trả 409. Collection export ZIP:

```text
manifest.json                 # collection sidecar đúng fingerprint được duyệt
review.json                   # actor/history/approval + scope pooled và nguồn latency
benchmark/manifest.json       # manifest membership gốc
benchmark/metrics_pooled.json
benchmark/initial_report.json
benchmark/benchmark_report.json
benchmark/progress.json
benchmark/timings.csv
radar_startup_audit.json
frames/<success_token>/occupancy.npz, metadata.json, report.json, bev.png
frames/<success_token>/resolved_config.py, prediction_commit.json
frames/<success_token>/evaluation.json, evaluation_commit.json
```

ZIP được dựng trong state directory riêng, từng file được đọc/hash rồi ghi
chính byte đó; bộ nhớ không giữ toàn bộ 79 bundle. Chỉ trả ZIP sau khi đóng
archive và xác nhận revision hiện tại còn approved/sidecar khớp. File tạm được
xóa sau response hoặc lỗi. Không có prediction giả của hai frame lỗi. Gói không
chép GT/sensor dataset/checkpoint, không thay đổi artifact nguồn.

## Kiểm thử và bằng chứng

```bash
.venv-web/bin/python -m pytest web/backend/tests -q \
  --junitxml=work_dirs/web_benchmark_pytest.xml -o faulthandler_timeout=60
PATH="$PWD/.cache/toolchains/node/bin:$PATH" npm --prefix web/frontend run typecheck
.venv-web/bin/python tools/check_web_benchmark.py
```

Trong Codex, ASGI/Chromium/build dùng cùng chế độ ngoài sandbox như các mốc
trước. QA tạo bản sao benchmark và SQLite riêng tại `work/benchmark-qa-*/`,
cổng **8002/3002**, build `.next-qa`; không dừng viewer production. Mật khẩu
QA ngẫu nhiên, chỉ truyền qua stdin/environment và được redact khỏi context
lỗi Playwright. Trace tắt; screenshot không chứa mật khẩu. SwiftShader dùng
software WebGL; không phải GPU benchmark hoặc learned motion/flow.

QA build phải nhận VIEWER_API_ORIGIN=8002 khi **build**, không chỉ lúc start.
`--skip-build` chỉ chấp nhận routes-manifest trỏ đúng QA; tool còn đối chiếu
health workspace_id giữa proxy/frontend và backend trước khi gửi credentials.
Next.js đang cài có proxy timeout mặc định 30 s; cấu hình web đặt 300 s cho
request xác minh/đóng ZIP collection. Đây là thời gian phục vụ export, không
được trộn vào latency inference. Lượt export QA quan sát được mất 29.934 s,
gồm 15.084 s xác minh và phần còn lại đọc/hash/đóng gói. Request frame chỉ kiểm
tra bundle được chọn; mở collection/review/export vẫn kiểm tra toàn tập.

Backend tests kiểm tra cả v1/v2 approval cũ, auth/quyền/CSRF, membership,
token/geometry/protocol mismatch, byte sửa dù giữ mtime/size, source/sidecar,
stale revision, restart và ZIP. Chromium kiểm tra metric thực, cả hai scene,
frame lỗi, LiDAR/filter, review/approve/download và lỗi hash trên bản sao QA.
Tool restart backend thật, kiểm tra account/session/history/export và đối
chiếu lại hash artifact nguồn. Không chạy CLI inference hoặc evaluator.

## Kết quả thực tế ngày 2026-11-14

| Lệnh / trường hợp | Kết quả |
| --- | --- |
| pytest web/backend/tests | **66 passed**, 296.60 s |
| pytest riêng test_benchmark_collections.py sau sửa schema/đường đọc frame | **19 passed**, 225.20 s |
| pytest -k 'timings or actual_partial or collection_approval_export' sau sửa parser CSV | **3 passed**, 69.39 s |
| npm run typecheck | Đạt |
| QA build Next.js + TypeScript | Đạt, route /benchmarks |
| tools/check_web_benchmark.py → Chromium | **4 passed**, khoảng 3.4 phút, không skip/flaky |
| Scope/metric trên UI | partial **79/81**; occupancy **44.1789%**, semantic mIoU **23.0672%**, đủ 16 lớp |
| Latency UI | **78** mẫu; median **32.5109 s**, p95 **36.3683 s**, từ initial_report.json |
| Chuyển scene/frame, hai lỗi, BEV/3D/LiDAR/filter/reset | Đạt; không giữ metric cũ, không gọi inference/evaluator |
| Engineer quyết định approve; chưa approved export | **403**; **409** tương ứng |
| Reviewer approve collection submitted | Đạt; benchmark vẫn partial |
| Byte sửa cùng size/mtime, token/geometry/protocol sai, sidecar sai | Chặn ghép/approval/export, không tạo prediction thay thế |
| CSV bị hỏng | Integrity/export thất bại có kiểm soát, không lỗi parser chưa xử lý |
| Restart backend QA thật | Account, session và **4/4** event cũ còn; export lại đạt |
| ZIP từ UI và sau restart | **641 entries**; hash/size đúng cho **639** file nguồn, cộng collection manifest và review |
| Legacy v1/v2 trong backend tests | Approval và lịch sử giữ nguyên khi review/revise/export collection |
| Migration production v2→v3 | Tám bảng legacy giữ nguyên digest/count tại thời điểm migration; hai user, session và lịch sử cũ còn; không tự tạo collection |
| Hash nguồn | **666 file** benchmark/inference/evaluation/audit/code/manifest cũ không đổi |
| Model | Không import Torch/OpenMMLab, không nạp checkpoint/model, không inference/GPU; Chromium SwiftShader CPU |

QA thành công nằm tại `work/benchmark-qa-uvs9p_5r/`; approval/nhận xét QA chỉ
ở SQLite riêng. Database dùng thật vẫn chưa đăng ký collection tự động; hai
vai trò thực hiện review bằng account của mình. Backend tests có một cảnh báo
deprecation TestClient/httpx từ dependencies hiện có, không làm test skip/lỗi.

Bằng chứng local (ignored):

- [web_benchmark_pytest.xml](../work_dirs/web_benchmark_pytest.xml),
  [web_benchmark_collection_pytest.xml](../work_dirs/web_benchmark_collection_pytest.xml),
  [web_benchmark_targeted_pytest.xml](../work_dirs/web_benchmark_targeted_pytest.xml).
- [web_benchmark_evidence.json](../work_dirs/web_benchmark_evidence.json):
  restart, source hashes, fingerprint, export, source latency và screenshot paths.
- [web_benchmark_preservation_before.json](../work_dirs/web_benchmark_preservation_before.json),
  [web_benchmark_preservation_after.json](../work_dirs/web_benchmark_preservation_after.json),
  [web_benchmark_migration_after.json](../work_dirs/web_benchmark_migration_after.json);
  SQLite backup trước migration ở `work_dirs/web_benchmark_pre_migration.sqlite3`.
- [Overview](../work_dirs/web_benchmark_overview.png),
  [occupancy 3D + LiDAR + BEV](../work_dirs/web_benchmark_frame3d.png),
  [frame lỗi](../work_dirs/web_benchmark_failed_frame.png),
  [collection approved](../work_dirs/web_benchmark_approved.png),
  [hash mismatch](../work_dirs/web_benchmark_hash_mismatch.png).

Production cuối đã build và restart tại **http://localhost:3000/benchmarks**.
Chromium mở trang login HTTP 200, không page error; API benchmark chưa đăng
nhập trả 401, health xác nhận không import model và inference disabled.
Windows PowerShell gọi cùng URL localhost trả **200**. Xem
[web_benchmark_localhost.json](../work_dirs/web_benchmark_localhost.json) và
[ảnh login production](../work_dirs/web_benchmark_local_login.png).

Audit cuối xác nhận 666 file nguồn và các hàng users/app_settings/runs/
revisions/review_events/revision_evaluations không đổi. Session hiện tại đã
xoay sau một lượt production logout/login HTTP 200 sau migration: cùng user,
TTL 8 giờ, không đổi tài khoản hay approval. Audit ghi riêng thay đổi runtime
này, giữ phiên hiện tại, không khôi phục session cũ từ backup. Production có
0 collection tự đăng ký; approval QA không được chuyển vào database thật.

## File triển khai

| Nhóm | File thay đổi/thêm |
| --- | --- |
| Backend collection mới | web/backend/app/benchmarks.py, collections.py |
| Cấu hình, migration, API, kiểm tra pooled count | web/backend/app/config.py, database.py, main.py, evaluations.py |
| Trang và component mới | web/frontend/app/benchmarks/page.tsx; components/BenchmarkViewer.tsx, BenchmarkFramePlayer.tsx, CollectionReviewPanel.tsx |
| Tích hợp viewer/auth/evaluation | web/frontend/components/Viewer.tsx, AuthGate.tsx, EvaluationPanel.tsx |
| Types/API/style/proxy | web/frontend/lib/types.ts, api.ts; app/globals.css; next.config.mjs |
| Kiểm thử | web/backend/tests/test_benchmark_collections.py, test_evaluation_review.py; web/frontend/e2e/benchmark.spec.ts |
| QA CLI | tools/check_web_benchmark.py |
| Tài liệu | docs/WEB_BENCHMARK_COLLECTIONS.md, BENCHMARK_MINI_VAL.md, WEB_EVALUATION_REVIEW.md |

Không thêm dependencies/đổi lockfile hoặc quy tắc ignore. `next-env.d.ts` được
Next.js sinh lại khi build như các mốc trước.
Danh sách 24 file triển khai/tài liệu/kiểm thử của mốc này cũng được lưu tại
[web_benchmark_changed_files.json](../work_dirs/web_benchmark_changed_files.json).
Rà soát cuối đã sửa nền popup lọc lớp sang màu tối để giữ tương phản với chữ;
production build và TypeScript sau chỉnh sửa đạt, không đổi logic/data.

Giới hạn: vẫn thiếu prediction cho hai frame đầu scene theo baseline, coverage
partial 79/81. Lần mở đầu và review/export toàn tập phải đọc/hash nhiều file,
có thể mất vài chục giây trên WSL/NTFS. Chưa kiểm chứng HTTPS/reverse proxy khác,
nhiều người dùng đồng thời hoặc render bằng GPU của trình duyệt Windows.
Đây là playback frame độc lập, **chưa phải learned 4D flow**.
