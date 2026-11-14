# Evaluation trong viewer và revision review v2

**Mốc collection mini_val:** xem [WEB_BENCHMARK_COLLECTIONS.md](WEB_BENCHMARK_COLLECTIONS.md).
Trang `/benchmarks` và manifest collection có namespace riêng; không thay đổi
approval v1/v2 hoặc dùng pooled metric làm metric của một frame.

Viewer phát lại báo cáo từ [OCCUPANCY_EVALUATION.md](OCCUPANCY_EVALUATION.md),
giữ đăng nhập/phân quyền của [AUTH_REVIEW.md](AUTH_REVIEW.md). Không chạy model,
GT loader hoặc CLI evaluator khi xem trang, đổi run hay lọc voxel.

## Chạy và sử dụng

Không thêm dependencies; lockfile Python/npm hiện có tiếp tục áp dụng.

```bash
cd '/mnt/d/New folder/OCCFusion'
export PATH="$PWD/.cache/toolchains/node/bin:$PATH"
cd web/frontend
NEXT_TELEMETRY_DISABLED=1 npm run build
cd ../..
bash tools/run_web_viewer.sh
```

Đăng nhập tại **http://localhost:3000** bằng account hiện có. Chọn
`infer_mini_val_scene0103_f1` để xem evaluation thật. Chọn `infer_mini_cuda_v2`
để xem scene-0061 thiếu GT; hai metric tiếp tục hiện **Chưa có GT**.

Engineer bổ sung evaluation vào review:

1. Chọn scene-0103, kiểm tra `matched`, split, frame, class IoU và hash ở mục
   **Protocol & bằng chứng hash**.
2. Nếu chưa đăng ký run, nhập nhận xét và **Đăng ký run**. Revision đầu giữ
   snapshot bốn artifact, làm rõ phạm vi review.
3. Nhập nhận xét, chọn **Thêm evaluation vào revision mới**. Backend tạo draft
   mới với manifest v2; revision cũ giữ nguyên.
4. Gửi review. Reviewer đăng nhập, kiểm tra snapshot mới và phê duyệt. Export
   chỉ thêm evaluation sau khi revision tương ứng được duyệt.

UI ghi rõ evaluation đã ghép có thuộc revision hiện tại hay chưa. Lịch sử
revision hiển thị trạng thái, manifest version và phạm vi `4 artifact` hoặc
`có evaluation`. Approval cũ không tự chuyển sang revision mới.

## Schema và đơn vị

Báo cáo thật: `work_dirs/eval_mini_val_scene0103_f1/evaluation.json`, schema **1**.

| Trường nguồn | Cách dùng |
| --- | --- |
| `frames[0].sample_token` | Đối chiếu metadata/report; không tìm token trong NPZ |
| `frames[0].prediction_sha256` | Đối chiếu byte occupancy.npz hiện tại |
| `frames[0].artifact_hashes` | Đối chiếu cả bốn file inference |
| `frames[0].metrics.occupancy_iou` | Tỷ lệ [0,1], format phần trăm |
| `frames[0].metrics.semantic_miou` | Tỷ lệ [0,1], format phần trăm |
| `frames[0].metrics.per_class_semantic` | IoU, intersection/union từng lớp |
| split, scene/frame, frame_count_evaluated | Phạm vi kết quả đã lưu |
| protocol.semantic_mean_class_count, union_zero_iou | Quy ước mean/union-zero |

Frontend dùng `Intl.NumberFormat('vi-VN', {style:'percent'})` trực tiếp trên tỷ
lệ, không nhân thêm 100. Không hardcode IoU/mIoU trong UI. Hiện tại là **một frame
mini_val**, không phải toàn split. Mean đủ **16 lớp**, union=0 nhận IoU=0.
Max softmax probability vẫn ghi **Chưa hiệu chuẩn**.

Backend kiểm tra tỷ lệ hữu hạn [0,1], protocol/class map/geometry,
scene/frame/timestamp/LiDAR và các count/confusion đã lưu. Tỷ lệ phải khớp count,
mIoU phải khớp mean lớp. Viewer không nạp lại GT để tính metric mới.

## Ghép và giới hạn truy cập file

Mặc định tìm `evaluation.json` dưới runs_root (`work_dirs`). Có thể cấu hình
`evaluation_roots` trong web/viewer.config.json hoặc biến môi trường:

```bash
export VIEWER_EVALUATION_ROOTS='["/mnt/d/New folder/OCCFusion/work_dirs"]'
```

Path config tương đối tính từ thư mục file config. Browser chỉ gửi run_id và
**evaluation_id opaque 64 ký tự hex**, không gửi đường dẫn. ID ánh xạ tới file
trong root được cấu hình; symlink thoát root bị chặn. Path GT/run trong nội dung
JSON là bằng chứng, không dùng để mở file tùy ý.

Mỗi lần đọc/kiểm tra đều hash byte hiện tại, không tin mtime/size. Cache JSON
parse theo SHA-256, tối đa 16 nội dung. Thay byte nhưng giữ mtime/size vẫn bị
phát hiện. Không sửa artifact inference hoặc evaluation nguồn.

`GET /api/runs/{run_id}` bổ sung evaluation gồm status/reason, matched (hoặc
null), candidates (ID/nhãn/hash/trạng thái/lý do các nguồn liên quan).

| Status | Ý nghĩa |
| --- | --- |
| matched | Token/hash bốn artifact khớp, schema/protocol/count hợp lệ |
| missing_gt | Báo cáo khớp sample/prediction nhưng thiếu GT, không có metric |
| token_mismatch | Token khác |
| hash_mismatch | Hash prediction hoặc artifact nguồn khác không khớp |
| invalid | Schema/protocol/count/metric không hợp lệ |
| unmatched | Chưa ghép được hoặc nguồn không còn trong root |
| ambiguous | Nhiều báo cáo khác nội dung cùng khớp; cần thu hẹp root cấu hình |
| unsupported_scope | Pooled nhiều frame chưa hỗ trợ làm snapshot một run |
| unavailable | Run không có prediction hoặc evaluation chưa thành công |

Không có matched thì UI không giữ metric của run trước. Báo cáo không parse
được/không xác định được run liên quan không có quyền ghép; run hiện unmatched
và không có metric. Bản này chỉ gắn báo cáo **success một frame**, không dùng
pooled metric làm metric của một frame.

## Migration và snapshot có phiên bản

SQLite nâng user_version **1 → 2**, chỉ thêm bảng revision_evaluations. Giữ
nguyên users/password hashes, session/CSRF secret, runs, revisions, events và
manifest v1. Không tự đăng ký/duyệt run hoặc tạo revision khi startup.

Manifest v1 giữ cấu trúc/fingerprint và bốn entries: occupancy.npz,
metadata.json, report.json, bev.png. Manifest v2 thêm **evaluation.json** và
liên kết evaluation ID/sample token/prediction SHA-256/scope một frame/đơn vị.
Fingerprint vẫn là SHA-256 của canonical JSON core, nhưng core v2 chứa cả năm
hash/size. Bảng liên kết và sidecar được kiểm tra với manifest.

Endpoint mới, cần Engineer owner + session/CSRF:

```text
POST /api/runs/{run_id}/review/evaluation
{
  "comment": "Lý do bổ sung bằng chứng",
  "revision_id": <revision hiện tại>,
  "fingerprint": "<fingerprint hiện tại>",
  "evaluation_id": "<ID từ evaluation.matched.id>"
}
```

Revision hiện tại không được đang submitted, ID/fingerprint phải còn đúng,
evaluation phải ghép được. Luôn tạo draft mới, ghi actor/time/comment.
Reviewer không được gọi API này. Tạo draft thông thường từ v2 tiếp tục mang
evaluation vào snapshot mới, không âm thầm hạ phạm vi về bốn artifact.

Tìm thấy evaluation trên ổ đĩa không làm ZIP v1 tự thêm file. Sau khi tạo v2,
export revision hiện tại bị chặn đến khi v2 được duyệt; approval v1 còn trong
lịch sử.

## Export

Giữ POST /api/runs/{run_id}/export. V1 được duyệt xuất sáu file như trước. V2
được duyệt xuất bảy file: bốn artifact inference, **evaluation.json**,
manifest.json và review.json. Năm artifact nguồn giữ nguyên byte.

review.json v2 ghi metric từ evaluation.json và metric_unit=ratio_0_1.
report.json gốc giữ metric null của lượt inference không tải GT. Trước export,
đọc/hash lại nguồn, kiểm tra sidecar và liên kết DB. ZIP dùng chính byte vừa
kiểm tra; sửa/mất một trong năm file hoặc sidecar khiến export trả 409.

## Kiểm thử

```bash
.venv-web/bin/python -m pytest web/backend/tests -q \
  --junitxml=work_dirs/web_evaluation_pytest.xml -o faulthandler_timeout=45
.venv-web/bin/python -m pytest web/backend/tests/test_evaluation_review.py -q \
  --junitxml=work_dirs/web_evaluation_revision_pytest.xml -o faulthandler_timeout=45
.venv-web/bin/python tools/check_web_review.py
```

Trong Codex chạy FastAPI/Chromium ngoài sandbox như mốc trước. QA dùng bản sao
artifact/evaluation và account ngẫu nhiên trong SQLite riêng, cổng 8001/3001.
Không ghi approval QA vào database sử dụng thật, không chạy inference/evaluator.
Chỉ sửa byte trên bản sao, khôi phục trước kiểm tra cuối; không ghi trace chứa
credentials.

## File triển khai

- Backend: web/backend/app/evaluations.py, review.py, database.py, config.py, main.py.
- Frontend: components/EvaluationPanel.tsx, ReviewPanel.tsx, Viewer.tsx,
  lib/types.ts, app/globals.css dưới web/frontend.
- QA: web/backend/tests/test_evaluation_review.py,
  web/frontend/e2e/evaluation.spec.ts, tools/check_web_review.py.
- Tài liệu: file này và các liên kết trong AUTH_REVIEW, OCCUPANCY_EVALUATION, WEB_VIEWER.

Chưa kiểm chứng HTTPS qua reverse proxy, tải nhiều người dùng hoặc pooled
nhiều frame trong viewer. Chromium QA dùng SwiftShader, không phải benchmark
render GPU thật.

## Kết quả thực tế ngày 2026-11-13

Dùng bản sao nguyên byte của hai run và evaluation thật; không chạy inference
hoặc evaluator trong mốc tích hợp này.

| Lệnh / trường hợp | Kết quả |
| --- | --- |
| Backend toàn bộ trước khi thêm ca malformed-frame | **46 passed**, 83.95 s, không skip |
| Chạy lại riêng test_evaluation_review.py sau bổ sung ca malformed-frame | **14 passed**, 48.89 s, không skip |
| npm run typecheck | Đạt |
| QA production build, gồm TypeScript | Đạt |
| tools/check_web_review.py → Chromium | **10 passed**, 86.64 s, không flaky/skip |
| Ghép scene-0103/frame 1 | matched; IoU **40,2652%**, mIoU **12,4469%** |
| Scene-0061 thiếu GT; đổi run hai chiều | Chưa có GT; không giữ metric trước đó |
| Evaluation sai token/hash, mất file hoặc sai schema/count | Không có matched metric; trạng thái/lý do rõ |
| Approval v1 khi có evaluation mới trên ổ đĩa | Vẫn chỉ bốn artifact; ZIP không tự thêm evaluation |
| Thêm evaluation vào revision mới | r2 draft, manifest v2; r1 approved còn trong lịch sử |
| Export r2 chưa approved | HTTP 409; nút UI bị khóa |
| Sửa từng file trong năm nguồn, giữ size/mtime | Approval không còn hợp lệ; export 409 |
| Reviewer duyệt r2; export sau restart backend thật | ZIP đủ bảy file, evaluation và bốn artifact đúng byte/hash |
| Migration database sử dụng thật | v1→v2; hash các hàng users/app_settings/sessions/runs/revisions/events giữ nguyên |
| Artifact inference/evaluation gốc | SHA-256 trước/sau QA không đổi |

Bằng chứng local (ignored):

- [web_evaluation_pytest.xml](../work_dirs/web_evaluation_pytest.xml),
  [web_evaluation_revision_pytest.xml](../work_dirs/web_evaluation_revision_pytest.xml).
- [web_evaluation_review_evidence.json](../work_dirs/web_evaluation_review_evidence.json):
  root QA, v1/v2 fingerprint, source hashes, byte-exact export và restart.
- [web_evaluation_migration_before.json](../work_dirs/web_evaluation_migration_before.json),
  [web_evaluation_migration_after.json](../work_dirs/web_evaluation_migration_after.json):
  digest hàng DB và đường dẫn backup SQLite trước migration, không ghi password/token.
- [web_evaluation_viewer.png](../work_dirs/web_evaluation_viewer.png),
  [web_evaluation_approved.png](../work_dirs/web_evaluation_approved.png).
- web/frontend/test-results/browser-results.json: 10 ca Chromium.

Approval và nhận xét QA nằm trong database riêng. Database sử dụng thật giữ
nguyên tài khoản/lịch sử, không tự tạo approval evaluation. Engineer và Reviewer
thực hiện thao tác bằng account của mình theo hướng dẫn ở đầu tài liệu.

Production đã build và khởi động lại tại **http://localhost:3000**. Chromium
mở trang login HTTP 200, không có page error, tải stylesheet mới; API dữ liệu
chưa đăng nhập trả 401. Windows PowerShell kiểm tra localhost trả HTTP 200.
Kiểm tra root cấu hình production ghép scene-0103 là matched và scene-0061 là
missing_gt; không import torch. Xem
[web_evaluation_localhost.json](../work_dirs/web_evaluation_localhost.json).
Server được để chạy; không cần tạo lại tài khoản sau migration.
