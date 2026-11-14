# Sửa race CSRF trong authenticated bootstrap

Repo bắt đầu sạch tại commit `2313c13b`; không reset/checkout. Mốc này chỉ sửa
auth và thêm regression/QA. Không đổi frontend ứng dụng, database schema,
model/config/CLI/checkpoint/profile, dependencies hoặc artifact inference.

## Nguyên nhân và regression trước sửa

`Identity.bootstrap` trước đây kiểm tra cookie theo snapshot session rồi mint
token ngẫu nhiên và UPDATE `sessions.csrf_hash`. Hai request có thể cùng đọc
hash ban đầu, cùng thay token. Ghi DB theo A → B nhưng browser nhận B → A làm
cookie/header giữ token A và DB giữ hash B. Chữ ký token vẫn đúng; POST bị 403
`CSRF token does not belong to this session`.

`web/backend/tests/test_csrf_bootstrap.py` dùng hai Identity/connection cùng
SQLite QA, `Barrier(2)` buộc cả hai authenticate trước khi bootstrap, Event buộc
A hoàn tất trước B. Sau đó test áp dụng Set-Cookie/body B trước A rồi POST logout
qua API thật, kiểm tra DELETE session. Không có sleep ngẫu nhiên hoặc stress
POST để suy ra race.

| Lịch xác định | Trước sửa | Sau sửa, cùng hai regression |
| --- | --- | --- |
| Cùng cookie cũ, snapshot giống nhau; response B → A | 403, fail | POST 200, pass |
| Thiếu cookie, snapshot giống nhau; response B → A | 403, fail | POST 200, pass |

Mã auth chưa sửa được chạy trước khi áp dụng patch: **2 failed**, đúng thông
báo session binding, 8.39 s. Nguyên hai test sau sửa: **2 passed**, 5.61 s.
Bằng chứng nằm trong `work_dirs/csrf_fix/regression_before.xml` và
`regression_after_initial.xml`. Mã trước sửa tương ứng auth.py tại commit trên.

## Cách sửa

Authenticated session luôn có một token ổn định, suy ra từ khóa HMAC trong
`app_settings`, `sessions.token_hash`, `sessions.csrf_hash` và `expires_at`.
Bootstrap không ghi session hoặc lưu token plaintext; các process/restart đọc
cùng dữ liệu sẽ trả cùng token dù cookie cũ, sai, thiếu hoặc response đảo thứ tự.
Login mới trả ngay token ổn định này, nên lần focus/bootstrap tiếp theo không
xoay cookie trong khi frontend còn giữ header token vừa đăng nhập.

Token có dạng:

```text
deadline_microseconds.s1:<HMAC binding của session>.<HMAC signature của payload>
```

Nonce dùng domain riêng `csrf-session-v1` và seed/hash chỉ có ở server. `s1:`
không trùng nonce token_urlsafe cũ vì nonce cũ không có dấu `:`. Deadline ký
bằng microsecond giữ đúng expiry session, tránh hết hạn CSRF sớm ở phần lẻ giây.
Token không gia hạn session. Signature, matching cookie/header, Origin/Referer,
Sec-Fetch-Site, HttpOnly/SameSite, Secure và phân quyền tiếp tục được kiểm tra.

`csrf_hash` là seed bất biến từ login mới hoặc hash token login cũ. Với session
v4 đã tồn tại, token cũ vẫn phải có signature/TTL hợp lệ và digest khớp đúng
`csrf_hash` của session. Vì thế không cần sửa row, migration, logout cưỡng bức
hoặc thay account/review/approval. Token ổn định của session khác, kể cả cùng
username, bị từ chối. Session đã logout/expired không thể được bootstrap muộn
tạo lại; request ghi tiếp theo vẫn 401.

Không sửa `api.ts`/`AuthGate.tsx`: backend trả cùng cookie và JSON token cho cùng
session nên cách frontend giữ token trong memory vẫn phù hợp. Không thêm retry
POST, không nới origin hoặc bỏ session binding. Database vẫn schema **4**.

## Kiểm thử CPU và Chromium riêng

Default app khi import `main.py` trong pytest cũng được chuyển tới QA state qua
biến môi trường; production database không dùng làm fixture hoặc được migrate.

```bash
cd '/mnt/d/New folder/OCCFusion'
mkdir -p work/csrf-qa-default/state work_dirs/csrf_fix

# Đã chạy nguyên hai regression trên auth cũ trước khi sửa:
VIEWER_STATE_ROOT="$PWD/work/csrf-qa-default/state" \
  .venv-web/bin/python -m pytest web/backend/tests/test_csrf_bootstrap.py -q \
  --junitxml=work_dirs/csrf_fix/regression_before.xml

# Regression mở rộng + auth/review hiện có:
VIEWER_STATE_ROOT="$PWD/work/csrf-qa-default/state" \
  .venv-web/bin/python -m pytest web/backend/tests/test_csrf_bootstrap.py \
  web/backend/tests/test_auth_review.py -q -x \
  --junitxml=work_dirs/csrf_fix/targeted_after.xml -o faulthandler_timeout=60

# Toàn bộ backend (viewer/review/collection/jobs fixture, không GPU):
VIEWER_STATE_ROOT="$PWD/work/csrf-qa-default/state" \
  .venv-web/bin/python -m pytest web/backend/tests -q \
  --junitxml=work_dirs/csrf_fix/all_backend_after.xml -o faulthandler_timeout=60

# Browser QA: không khởi động worker, CLI hoặc GPU:
.venv-web/bin/python tools/check_web_csrf.py
```

Lệnh regression_before chỉ mô tả bằng chứng đã chạy trên mã cũ; với mã đã sửa
test sẽ pass. Không checkout code cũ để chạy lại. Kết quả mở rộng: **45 passed**
(32 test CSRF mới + 13 auth/review), 53.95 s. Các ca gồm:

- Cookie cũ/thiếu/sai/đang hợp lệ; hai thứ tự response, hai snapshot ban đầu.
- Cùng user khác session; sai header/cookie/chữ ký/origin/referer/cross-site;
  token anonymous không thay được authenticated token.
- Legacy v4 row/token giữ nguyên qua bootstrap và restart; vẫn logout được.
- Expiry phần lẻ giây, token legacy đúng TTL, bootstrap muộn sau logout.
- Cookie HttpOnly/SameSite/host-only/Path/Max-Age và Secure cả false/true.

Lượt toàn bộ backend hoàn tất **114 passed**, **0 failures / 0 skipped**,
395.78 s. Bao gồm viewer/geometry, review v1/v2, benchmark collection và jobs
dùng fixture hiện có. Một cảnh báo deprecation TestClient/httpx từ dependency
hiện có; không thay dependency hoặc bỏ kiểm thử.

Browser dùng `work/csrf-qa-*/` riêng, cổng 8003/3003, password ngẫu nhiên, bản
sao artifact thật. Tool xác minh proxy workspace_id trước login. Tái sử dụng
build `.next-qa` của frontend **không đổi**, chỉ chấp nhận rewrite trỏ 8003;
không build hoặc sửa TypeScript. Nếu cache build không có/khác origin, tool
dừng và báo lệnh build cần chạy; không gửi credentials sang workspace khác.

Chromium dùng interception + promise để giữ response A đến sau khi JavaScript
đã nhận B. Cả hai request mang cookie cũ hoặc thiếu cookie; unit regression bên
trên kiểm soát cả snapshot trước server writes. UI dùng nguyên code frontend
để tạo/hủy hai job, mở 3D/LiDAR và đăng ký review draft. Các POST sai bị chặn,
logout về login, API protected trả 401, không page error. Backend QA restart
thật giữ account/session/token/review/jobs; POST logout sau restart đạt.

Lượt QA đạt tại `work/csrf-qa-4kri24dj/`: **2 lịch browser đạt**, **2 job canceled,
launch count 0**, **0 worker**. Health xác nhận không import model. Công cụ QA
ban đầu lỗi lời gọi assertion `expect.setState`; đã sửa công cụ và chạy lại đạt,
không phải một lần browser test đạt giả.

## Bằng chứng và giới hạn

- [Regression trước sửa](../work_dirs/csrf_fix/regression_before.xml),
  [sau sửa](../work_dirs/csrf_fix/regression_after_initial.xml),
  [45 tests](../work_dirs/csrf_fix/targeted_after.xml).
- [Browser/restart evidence](../work_dirs/csrf_fix/browser_evidence.json),
  [ảnh jobs](../work/csrf-qa-4kri24dj/csrf-jobs.png),
  [ảnh viewer/review](../work/csrf-qa-4kri24dj/csrf-viewer-review.png).
- `work_dirs/csrf_fix/all_backend_after.xml`, `all_backend_after.log`,
  `full_test_result.json` ghi lệnh/exit code cho toàn bộ backend.
- [Tổng hợp bằng chứng](../work_dirs/csrf_fix/summary.json) ghi hash mã auth
  trước/sau, số test và đường dẫn evidence; `git diff --check`, `py_compile` đạt.
- `work_dirs/csrf_fix/preservation_before.json` và audit sau ghi hash artifact/
  checkpoint/config/profile cùng digest bảng production đọc **read-only**.
  Snapshot này phục vụ bảo toàn dữ liệu, không dùng làm test fixture.

Audit cuối: **668/668 file không đổi**, cả **12 bảng production** giữ nguyên
digest/số row (gồm users, sessions, review/approval và jobs), schema **v4** không
đổi, SQLite `quick_check=ok`. Xem
[preservation_after.json](../work_dirs/csrf_fix/preservation_after.json).

File thay đổi: `web/backend/app/auth.py`; thêm
`web/backend/tests/test_csrf_bootstrap.py`, `tools/check_web_csrf.py`, tài liệu
này; cập nhật liên kết/diễn giải trong `AUTH_REVIEW.md`, `WEB_INFERENCE_JOBS.md`.
`database.py`, `main.py`, `auth_support.py`, TypeScript ứng dụng và lockfile
không đổi. Vì không sửa TypeScript nên không chạy typecheck/build mới.

Không chạy inference, benchmark GPU hoặc huấn luyện. Browser dùng SwiftShader
CPU; Secure được kiểm tra qua HTTPS TestClient, chưa chạy TLS reverse proxy
thật. Bản sửa chỉ đảm bảo convergence trong **cùng** authenticated session;
response đã chụp trước logout có thể chứa thông tin user cũ nhưng không phục
hồi session hoặc cấp quyền ghi. Chưa restart production để áp dụng bản backend
mới; các kiểm chứng restart trong mốc này đều trên QA riêng.
Khi áp dụng cần nạp bản backend mới đồng nhất cho các ASGI worker; không kiểm
chứng rolling deployment trộn mã cũ còn xoay hash với định dạng token mới.
