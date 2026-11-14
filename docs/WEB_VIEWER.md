# Web viewer local cho kết quả OccFusion đã lưu

**Inference từ web:** [WEB_INFERENCE_JOBS.md](WEB_INFERENCE_JOBS.md) bổ sung
trang `/jobs`, queue lưu bền vững và worker riêng; các trang viewer/benchmark
vẫn phát lại artifact mà không chạy lại model khi mở hoặc lọc dữ liệu.

**Evaluation và revision v2:** xem [WEB_EVALUATION_REVIEW.md](WEB_EVALUATION_REVIEW.md)
để hiển thị IoU/mIoU đã ghép và đưa evaluation vào snapshot review mới.

**Mốc đăng nhập/review:** viewer hiện yêu cầu tài khoản trước khi truy cập dữ
liệu. Xem [AUTH_REVIEW.md](AUTH_REVIEW.md) để tạo Engineer/Reviewer, gửi review,
duyệt fingerprint và export gói đã kiểm tra hash. Các kết quả kiểm thử viewer
bên dưới mô tả các mốc trước; test hiện tại đăng nhập trước khi đọc prediction.

Viewer đọc trực tiếp run `work_dirs/infer_mini_cuda_v2/`. Đây là **phát lại
prediction đã lưu**, không chạy inference, không nạp checkpoint, không cần CUDA
hoặc occupancy GT. Áp dụng [PROJECT_SPEC.md](../PROJECT_SPEC.md) và schema đã xuất
trong [INFERENCE_MINI.md](INFERENCE_MINI.md).

Backend FastAPI nằm tại `web/backend/`; frontend Next.js + three.js tại
`web/frontend/`. Backend chỉ phụ thuộc NumPy và thư viện web; không import
`torch`, `occfusion`, OpenMMLab hoặc nuscenes-devkit. Môi trường `.venv-web` tách
biệt hoàn toàn với `/home/hatis/.venvs/rav25`.

## Chạy trong WSL

Yêu cầu đã kiểm tra: Python **3.10.12**, Node.js **24.22.3**, npm **11.19.1**.
Phiên bản Node nằm trong `web/frontend/.nvmrc`. Phiên làm việc này cài Node local
tại `.cache/toolchains/node/`; không thay Node hệ thống. Nếu dùng nvm hoặc Node
hệ thống phù hợp thì không cần thêm đường dẫn cache vào `PATH`.

Chuẩn bị một lần, tại gốc repo:

```bash
cd '/mnt/d/New folder/OCCFusion'
/usr/bin/python3 -m venv .venv-web
.venv-web/bin/python -m pip install --cache-dir .cache/pip \
  -r web/backend/requirements.lock

export PATH="$PWD/.cache/toolchains/node/bin:$PATH"
cd web/frontend
npm ci
NEXT_TELEMETRY_DISABLED=1 npm run build
```

Terminal 1, backend:

```bash
cd '/mnt/d/New folder/OCCFusion'
.venv-web/bin/python -m uvicorn web.backend.app.main:app \
  --host 127.0.0.1 --port 8000
```

Terminal 2, frontend production:

```bash
cd '/mnt/d/New folder/OCCFusion'
export PATH="$PWD/.cache/toolchains/node/bin:$PATH"
cd web/frontend
NEXT_TELEMETRY_DISABLED=1 npm run start -- --port 3000
```

Sau khi cài/build xong, có thể thay hai terminal bằng một lệnh tại gốc repo:

```bash
bash tools/run_web_viewer.sh
```

Script dùng Node local nếu có, chạy hai service trên loopback và dừng cả hai
khi nhấn `Ctrl+C` hoặc khi một service thoát.

Tạo account bằng `tools/create_viewer_user.py` theo tài liệu đăng nhập. Mở
**http://localhost:3000**, đăng nhập rồi chọn `infer_mini_cuda_v2`. BEV, occupancy 3D và
LiDAR của sample thật được tải từ backend. Chọn run `infer_mini_check` để xem
báo cáo check-only: không có canvas prediction, BEV hoặc request point cloud.
Run `infer_mini_cuda` được nhận diện là run lỗi trước đó, không bị dùng như
prediction thành công. `Ctrl+C` ở hai terminal để dừng ứng dụng.

Trong lúc phát triển có thể dùng `npm run dev -- --port 3000` thay cho
`npm run start`. Các kiểm tra trình duyệt của mốc này dùng bản production.
Backend API docs tại http://127.0.0.1:8000/docs.

Next.js proxy `/api/*` về `http://127.0.0.1:8000`, vì vậy trình duyệt dùng cùng
origin, không cần CORS wildcard. Nếu đổi cổng backend, đặt `VIEWER_API_ORIGIN`
trước khi build và chạy frontend. Cả hai service mặc định chỉ bind loopback.

## Dependencies và cấu hình thư mục

- `web/backend/requirements.in`: pin các dependency trực tiếp.
- `web/backend/requirements.lock`: khóa toàn bộ 25 dependency của runtime và bộ
  kiểm tra trên Linux/Python 3.10 bằng phiên bản chính xác. Dùng lock này để cài.
- `web/frontend/package.json`: pin phiên bản trực tiếp, gồm Next.js **16.4.6**,
  React **19.3.1**, three.js **0.187.0** và Playwright **1.65.0**.
- `web/frontend/package-lock.json`: khóa dependency gián tiếp và integrity của
  npm; dùng `npm ci`, không dùng môi trường model để cài web.

`web/viewer.config.json` cấu hình từ phía server:

```json
{
  "runs_root": "../work_dirs",
  "dataset_roots": ["../data/nuscenes"],
  "cache_entries": 8
}
```

Đường dẫn tương đối tính từ thư mục chứa file config. Có thể chọn file config
bằng `VIEWER_CONFIG`, hoặc ghi đè `VIEWER_RUNS_ROOT` và `VIEWER_DATA_ROOTS` ở môi
trường backend. `VIEWER_DATA_ROOTS` là JSON array đường dẫn, ví dụ:

```bash
VIEWER_RUNS_ROOT='/mnt/d/New folder/OCCFusion/work_dirs' \
VIEWER_DATA_ROOTS='["/mnt/d/New folder/OCCFusion/data/nuscenes"]' \
.venv-web/bin/python -m uvicorn web.backend.app.main:app \
  --host 127.0.0.1 --port 8000
```

API nhận **run_id**, không nhận đường dẫn file từ trình duyệt. Run là thư mục
con trực tiếp của `runs_root`. Kiểm tra tên ID, `Path.resolve()` và containment
chặn `..`, directory traversal và symlink trỏ ra ngoài root/run. Artifact có tên
cố định: `report.json`, `metadata.json`, `occupancy.npz`, `bev.png`.

File LiDAR được chọn từ `sample_data.filename` trong bảng nuScenes dưới dataset
root được cấu hình. Không mở đường dẫn tuyệt đối do metadata khai báo. Metadata
path chỉ được đối chiếu với filename catalog; cơ chế này cũng cho phép chuyển
dataset sang một root được cấu hình khác mà vẫn kiểm tra đúng sample.
Symlink cảm biến hoặc bảng JSON thoát dataset root bị từ chối.

Backend cache trong RAM theo run, LRU tối đa 8 run mặc định; mtime/size các
artifact là khóa invalidation. Index JSON nuScenes cũng được cache. Dữ liệu
cảm biến đã đọc được giữ trong cache của run; nếu thay file dataset trong lúc
server đang chạy, khởi động lại backend để kiểm chứng lại dữ liệu đó. Nút làm
mới đọc lại danh sách/artifact; không chạy mô hình.

Trình duyệt cache payload occupancy và LiDAR. Bộ lọc lớp, chiều cao, reset
camera và bật/tắt LiDAR sau lần tải đầu chỉ đổi state/geometry local, không gửi
request inference. Backend không có route chạy model, không spawn worker GPU.

`.gitignore` root trước đó chưa tồn tại; đã thêm các thư mục `data/`, `ckpt/`,
`work/`, `work_dirs/`, môi trường Python, cache, `node_modules/`, `.next/` và
artifact kiểm thử. Không thay đổi quy tắc ignore ở vị trí khác.

## Schema thực tế và dữ liệu truyền cho viewer

Đã mở trực tiếp NPZ, metadata và report của run đầu vào để xác nhận:

| Trường | Giá trị thực tế |
| --- | --- |
| NPZ keys | `labels`, `max_softmax_probability` |
| Shape hai array | `[200,200,16]` |
| Dtype | `uint8`, `float32` |
| Thứ tự trục mảng | `grid_axis_order = ["x","y","z"]` |
| Bounds XYZ | `[-50,-50,-5,50,50,3]` m |
| Voxel size XYZ | `[0.5,0.5,0.5]` m |
| Class nền | `free_label=0`, `class_names[0]="free"` |
| Nhãn dự đoán | Trong `0..16`; 595.144 voxel free, 44.856 không-free |
| Probability | Hữu hạn, min `0.2468943894`, max `1.0` |
| Sample | `39586f9d59004284a7114a68825e8eec` |
| Scene/frame | `scene-0061` / frame index `1` |
| LiDAR sample_data token | `4f792c8da81e4cb7aca1790654da1c27` |
| Timestamp LiDAR | `1532402928147847` microsecond |

Backend xác nhận shape, integer label range, probability hữu hạn trong `[0,1]`,
đủ class palette, bounds/voxel size/grid/axis order nhất quán, shape logits và
histogram nhãn khớp report. Dữ liệu không hợp lệ trả lỗi 422 thay vì hiển thị
prediction thay thế. Run check-only/lỗi trả 409 khi gọi payload prediction.

Sample token, scene, frame index, status và precision trong metadata/report được
đối chiếu. LiDAR được kiểm tra độc lập bằng sáu bảng JSON nuScenes: `sample`,
`sample_data`, `scene`, `sensor`, `calibrated_sensor`, `ego_pose`. Kiểm tra đúng
keyframe, channel `LIDAR_TOP`, timestamp, filename và calibration/pose. Transform
của LiDAR keyframe về chính LiDAR tham chiếu phải là identity; transform ego và
global phải khớp quaternion/translation từ dataset.

**NPZ không có sample token nhúng.** Không thể kiểm tra token trực tiếp bên trong
NPZ hoặc chứng minh nguồn gốc bằng mật mã với schema hiện tại. Viewer đối chiếu
shape và histogram NPZ với report; metadata/report/sensor được kiểm tra token.
Giao diện và API ghi rõ giới hạn này, không coi histogram là hash chứng thực.

## API phát lại (cần đăng nhập)

| Route | Nội dung |
| --- | --- |
| `GET /api/health` | Chế độ replay, `inference_enabled=false`, hit/miss cache |
| `GET /api/runs` | Danh sách theo quyền, status, kind `prediction/check_only/failed` |
| `GET /api/runs/{run_id}` | Chi tiết, geometry, palette/counts, timing, confidence, nguồn gốc, kiểm chứng |
| `GET /api/runs/{run_id}/occupancy` | Chỉ các voxel không-free: mỗi record 5 số float32 little-endian `[x,y,z,label,max_softmax_probability]` |
| `GET /api/runs/{run_id}/lidar` | Keyframe thật: mỗi record 4 số float32 little-endian `[x,y,z,intensity]` |
| `GET /api/runs/{run_id}/bev` | Byte ảnh BEV PNG đã xuất |

Số record, stride, dtype và tên field được khai báo trong response detail.
Frontend kiểm tra byte length và đọc bằng `DataView.getFloat32(..., true)`.
Không chuyển toàn lưới free sang JSON hoặc tạo 640.000 mesh riêng lẻ.

## Quy ước hiển thị và ý nghĩa số liệu

`voxel_centers()` chuyển index mảng sang XYZ theo `grid_axis_order`, sau đó dùng:

```text
center_xyz = bounds_min_xyz + (index_xyz + 0.5) * voxel_size_xyz
```

Bốn ca baseline đã kiểm tra xác định:

| Index `[x,y,z]` | Tâm XYZ, mét |
| --- | --- |
| `[0,0,0]` | `[-49.75,-49.75,-4.75]` |
| `[199,199,15]` | `[49.75,49.75,2.75]` |
| `[100,100,10]` | `[0.25,0.25,0.25]` |
| `[10,30,4]` | `[-44.75,-34.75,-2.75]` |

Bộ kiểm tra còn dùng trục hoán vị `[z,x,y]`, bounds có offset, voxel size không
đồng đều và index ngoài grid để phát hiện lỗi tráo trục/thiếu half-voxel.

Frontend dùng một `THREE.InstancedMesh` cho voxel không-free, màu từ palette
thật. Kích thước khối vẽ là 97% voxel để tạo khe nhìn; tâm và khoảng cách ô giữ
đúng metadata. Camera và OrbitControls đặt Z up, trục X/Y/Z có chú giải. Reset
camera xóa quán tính đang còn sau thao tác kéo. Lọc chiều cao theo tâm voxel.

LiDAR overlay dùng **một keyframe**, không cộng sweeps. File thật có 34.720 điểm;
32.483 điểm nằm trong bounds XYZ occupancy được truyền cho viewer. Điểm này đã
ở hệ LiDAR tham chiếu; không quay sang ego hay coi Y là chiều cao. Bộ lọc chiều
cao áp dụng lên z của cả voxel và point cloud; count hiển thị cập nhật theo lọc.

BEV là PNG gốc, toàn bộ lớp, không vẽ lại khi thay bộ lọc 3D. Hướng hàng/cột
lấy từ metadata BEV. Class 0 hiển thị trong chú giải là free/nền và luôn được bỏ
khỏi InstancedMesh.

Thông tin run lấy từ artifact: scene/frame/token, checkpoint/hash, precision.
Forward, pipeline frame, khởi tạo và tiến trình được trình bày riêng. Khởi tạo
được **suy ra** từ `process_until_report - pipeline_total`; pipeline không gồm
startup và ghi metadata/report cuối. Đây vẫn là số đo cũ của một frame, không
phải latency của trang web và chưa phải benchmark ổn định.

IoU/mIoU `null` hiển thị **“Chưa có GT”**, không đổi thành 0%. Max softmax
probability ghi rõ **confidence chưa hiệu chuẩn**, chưa phải độ bất định.
Allocated/reserved được gắn nhãn **bộ cấp phát PyTorch**, không phải VRAM vật lý
đo độc lập; các số được phát lại từ report, không phải mức sử dụng của viewer.

## Lệnh kiểm thử

Backend (chạy tại gốc repo):

```bash
.venv-web/bin/python -m pip check
.venv-web/bin/python -m pytest web/backend/tests -q
```

Frontend:

```bash
cd web/frontend
npm run typecheck
npm run build
cd ../..
.venv-web/bin/python tools/check_web_review.py
```

Mốc đăng nhập dùng công cụ QA để cấp hai tài khoản tạm trong SQLite riêng, chạy
backend/frontend QA ở 8001/3001 và truyền credentials cho Playwright qua process
environment. Chạy `npm run test:e2e` riêng sẽ cần các biến QA đó; không dùng mật
khẩu thật hoặc seed tài khoản QA vào database đang sử dụng. Xem
[AUTH_REVIEW.md](AUTH_REVIEW.md) cho quy trình và kết quả mới nhất.

Nếu chưa có browser cho Playwright:

```bash
npx playwright install --with-deps chromium
```

Trong phiên WSL này Chromium được tải vào `.cache/ms-playwright`. Ba package
Ubuntu `libnspr4`, `libnss3`, `libasound2` còn thiếu được tải bằng `apt-get download`
và giải nén bằng `dpkg-deb -x` vào `.cache/browser-libs`, không cài đè hệ thống.
`playwright.config.ts` nhận biết cache này và nối thư mục thư viện vào
`LD_LIBRARY_PATH`; nếu dùng thư viện/browser hệ thống thì không cần cách đó.
Browser test dùng SwiftShader software WebGL để có thể kiểm tra InstancedMesh
trong headless Chromium; không đo hiệu năng render GPU thật.

Trong Codex sandbox, TestClient bị kẹt ở asyncio/thread bridge; cùng bộ kiểm tra
chạy ngoài sandbox đạt. Server, tải dependencies và browser QA cũng được chạy
ngoài sandbox để có kết nối/bind loopback. Lệnh bình thường trong terminal WSL
không cần các API permission của Codex.

Nguồn API tham khảo:
[Next.js installation](https://nextjs.org/docs/app/getting-started/installation),
[three.js InstancedMesh](https://threejs.org/docs/pages/InstancedMesh.html),
[FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/).

## Kết quả thực tế ngày 2026-11-13

Đã khởi động backend `127.0.0.1:8000`, frontend production
`127.0.0.1:3000`, mở trang bằng Chromium và xem ảnh chụp desktop/mobile.
Sau đó đã kiểm tra lại launcher `bash tools/run_web_viewer.sh` qua HTTP.

| Kiểm tra | Kết quả |
| --- | --- |
| Backend pytest | **19 passed**, `3.49 s` |
| Frontend `npm run typecheck` | Đạt |
| Production `npm run build` | Đạt, gồm kiểm tra TypeScript của bản cuối |
| Playwright / Chromium | **3 passed**, `12.5 s` |
| Run thật | `infer_mini_cuda_v2`, đúng sample token đã nêu trên |
| WebGL thực sự vẽ | **44.856 instance voxel**, **32.483 điểm LiDAR** |
| BEV | HTTP 200, PNG gốc `200×200`, bytes giống file gốc |
| Payload occupancy / LiDAR | `897120` / `519728` byte, khớp count × stride × 4 |
| Lọc lớp | Bỏ chọn → 0 voxel; chỉ `car` → 611 voxel |
| Lọc chiều cao | Khoảng không chứa tâm voxel → 0; mở lại → đúng số voxel |
| Bật/tắt LiDAR | Số điểm WebGL đổi 0 ↔ 32.483 |
| Camera | Kéo làm đổi vị trí; reset trả về đúng vị trí ban đầu |
| Run check-only/lỗi | Không có canvas/payload prediction; forward check-only hiển thị `—` |
| IoU/mIoU thiếu GT | Hiển thị `Chưa có GT` |
| Thao tác lọc/reset/toggle sau tải đầu | **0 request API bổ sung** |
| Inference | `/api/health`: `inference_enabled=false`; backend không import torch/occfusion |
| Artifact inference gốc | SHA-256 của NPZ, metadata, report, BEV không đổi trước/sau thao tác |
| Mobile | Không tràn ngang ở `390×844`; xem được prediction thật |
| Lỗi JavaScript/console trong ca replay | Không có |

Ca pytest dùng fixture riêng trong thư mục tạm để kiểm tra nhánh lỗi và bảo vệ
đường dẫn; không thay dataset/run gốc. Bao gồm shape/nhãn/NaN/probability sai,
histogram lệch, token/timestamp/calibration/frame sai, symlink run/artifact/cảm
biến thoát root, cache invalidation, run check-only và các phép tính tọa độ xác
định. Pytest có một deprecation warning từ Starlette TestClient dùng HTTPX;
warning không làm hỏng kiểm thử hoặc ứng dụng.

Bằng chứng local được giữ ở:

- [`work_dirs/web_viewer_browser_evidence.json`](../work_dirs/web_viewer_browser_evidence.json):
  sample, số instance/điểm, request bộ lọc, SHA-256 bốn artifact và console errors.
- [`work_dirs/web_viewer_http_evidence.json`](../work_dirs/web_viewer_http_evidence.json):
  launcher, URL, payload bytes và cache hit/miss của kiểm tra HTTP.
- [`work_dirs/web_viewer_desktop.png`](../work_dirs/web_viewer_desktop.png),
  [`work_dirs/web_viewer_mobile.png`](../work_dirs/web_viewer_mobile.png).
- `web/frontend/test-results/browser-results.json`: kết quả ba ca Playwright.

Không còn phần bị chặn cho mục tiêu xem run đã cho. Giới hạn còn lại: NPZ chưa
có token/hash provenance nhúng; LiDAR overlay hiện là keyframe trong bounds,
không cộng sweeps; browser QA dùng software WebGL, chưa phải benchmark render
GPU hoặc kiểm tra bằng trình duyệt Windows của người dùng. Các phiên QA được
dừng sau kiểm tra; khởi động lại bằng launcher ở trên để tiếp tục sử dụng.

## Kiểm chứng bổ sung: index `(10,30,4)`, three.js và LiDAR

Đã đọc lại metadata thật của `infer_mini_cuda_v2`: bounds
`[-50,-50,-5,50,50,3]`, voxel size `[0.5,0.5,0.5]` m, thứ tự trục mảng
`[x,y,z]`. Hàm `voxel_centers()` trả về chính xác:

```text
(-50,-50,-5) + ((10,30,4) + 0.5) * (0.5,0.5,0.5)
= (-44.75,-34.75,-2.75) m
```

Ca này trước đó chưa có. Các ca baseline cũ đều có `x=y`; đã thêm điểm bất đối
xứng trên để phát hiện lỗi tráo X/Y. Fixture kiểm tra voxel/LiDAR cũng đổi sang
điểm `[0.5,1.5,-0.5]`, có ba tọa độ khác nhau và Z âm, để bắt lỗi tráo trục/đổi dấu.

Đường chuyển sang three.js hiện tại giữ nguyên `(x,y,z)` theo mét. Backend đã
chuyển index thành tâm XYZ; frontend dùng `Matrix4.makeTranslation(x,y,z)` và
`InstancedMesh.setMatrixAt()`, không cộng bounds hoặc half-voxel thêm lần nữa.
Camera dùng `up=(0,0,1)`. Phép quay `grid.rotation.x = PI/2` chỉ đưa **grid helper**
từ mặt phẳng XZ sang XY, không quay voxel hoặc point cloud. LiDAR vốn đã là XYZ
trong hệ keyframe tham chiếu nên không áp dụng công thức đổi index cho điểm.

Kết quả thực tế của lượt kiểm tra bổ sung:

| Kiểm tra | Kết quả |
| --- | --- |
| Index yêu cầu `(10,30,4)` | Tâm đúng `(-44.75,-34.75,-2.75)` m |
| Tất cả 44.856 voxel không-free | Payload backend khớp tâm tính độc lập từ NPZ/metadata; sai số lớn nhất **0 m** |
| 32.483 điểm LiDAR trong bounds | Payload khớp dữ liệu keyframe gốc, gồm XYZ/intensity; sai số lớn nhất **0** |
| Transform keyframe→LiDAR tham chiếu | Khớp identity; sai số phần tử lớn nhất `8.79296635503124e-14` |
| three.js control point | Tâm instance và vị trí point geometry cùng bằng tọa độ yêu cầu; world matrices identity; phép chiếu camera trùng nhau |
| Buffer WebGL production | Translation của **44.856 instance** khớp XYZ API; sai số lớn nhất **0 m**, phần tuyến tính của ma trận instance là identity |
| Buffer vị trí LiDAR WebGL | **32.483 điểm** khớp XYZ API; sai số lớn nhất **0 m** |
| Voxel thật đầu tiên trong payload | Index `[1,43,13]` → `[-49.25,-28.25,1.75]` m; khớp buffer WebGL |
| Test backend | **20 passed**, `3.13 s` (không skip trên dataset hiện tại) |
| Test three.js/WebGL bổ sung | **2 passed**, `5.3 s` |
| TypeScript | `npm run typecheck` đạt |

Trong prediction thật, ô `[10,30,4]` có nhãn **`free=0`**, nên đúng quy tắc viewer
nó không tạo instance để vẽ. Test three.js dùng một control point riêng trong
bộ nhớ để kiểm tra ô này; không thêm voxel giả vào prediction. Kiểm tra WebGL
chạy trên trang production thật, bắt các mảng gửi vào `bufferData/bufferSubData`
trong browser context riêng và đối chiếu toàn bộ dữ liệu với API.

Các lệnh đã chạy:

```bash
# Tại gốc repo; ASGI tests chạy ngoài Codex sandbox như lần trước.
.venv-web/bin/python -m pytest web/backend/tests -q \
  --junitxml=work_dirs/web_viewer_coordinate_pytest.xml \
  -o faulthandler_timeout=20

cd web/frontend
npm run typecheck
npm run test:e2e -- e2e/coordinates.spec.ts \
  --reporter=list --output=test-results/coordinates
```

Test dữ liệu thật trong pytest là integration test tùy chọn: máy không có run
hoặc nuScenes mini sẽ skip ca này. Bằng chứng của lượt chạy hiện tại nằm ở
[`web_viewer_coordinate_pytest.xml`](../work_dirs/web_viewer_coordinate_pytest.xml)
và [`web_viewer_coordinate_browser_evidence.json`](../work_dirs/web_viewer_coordinate_browser_evidence.json).
Có warning deprecation của Starlette/HTTPX và bộ nạp three.js trong test runner;
không có test lỗi. Không phát hiện lỗi đổi tọa độ trong code viewer hiện tại;
chỉ bổ sung test và tài liệu, giữ dịch vụ viewer đang chạy.
