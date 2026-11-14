# Benchmark occupancy nuScenes mini_val

Mốc thực thi: **2026-11-14**. Áp dụng [PROJECT_SPEC.md](../PROJECT_SPEC.md),
[INFERENCE_MINI.md](INFERENCE_MINI.md) và giao thức
[OCCUPANCY_EVALUATION.md](OCCUPANCY_EVALUATION.md).

Entry point: [tools/benchmark_occupancy.py](../tools/benchmark_occupancy.py).
Trang phát lại benchmark và review/export collection riêng được hướng dẫn tại
[WEB_BENCHMARK_COLLECTIONS.md](WEB_BENCHMARK_COLLECTIONS.md). Approval của run
một frame không tự bao gồm kết quả pooled này.
Benchmark chỉ ghi trong thư mục mới của nó. Không cập nhật run/evaluation cũ,
manifest review, SQLite hoặc approval. Pooled metric là bằng chứng của **tập
frame**, chưa tích hợp vào review/export của một frame.

## Manifest và phạm vi

Dùng `DatasetCatalog`/nuscenes-devkit đọc v1.0-mini, lấy mini_val chính thức bằng
`create_splits_scenes()`. Dữ liệu hiện tại có **81 frame**: scene-0103 có 40,
scene-0916 có 41. Không dùng scene index cố định. Thứ tự là scene name, rồi
frame index theo sample.next, rồi token; không phụ thuộc kết quả prediction/IoU.

Manifest ghi scene/index, sample token, timestamp sample/LiDAR, LiDAR filename
và sample_data token, GT filename/path/hash/size ban đầu; cùng provenance model,
config và dữ liệu. Toàn tập giữ mọi frame, kể cả frame không thể suy luận.
Không phải benchmark của toàn bộ nuScenes validation.

`--pilot` chọn trước **frame index 1 của mỗi scene mini_val**, độc lập metric.
Scope của manifest là `pilot_two_scenes`, expected=2, dataset_frame_count=81.
Pilot complete không được hiểu là full mini_val complete.

Hai frame index 0 thiếu radar quá khứ theo chính sách đã kiểm chứng:

| Scene | Sample token | Radar thiếu record <= timestamp LiDAR |
| --- | --- | --- |
| scene-0103 | 3e8750f331d7499e9b5123e9eb70f2e2 | RADAR_FRONT, RADAR_BACK_RIGHT |
| scene-0916 | b5989651183643369174912bc5641d3b | Cả năm radar |

Cả hai vẫn có trong manifest. Benchmark ghi lỗi input_validation, không thay
bằng frame khác, không lấy radar tương lai hoặc tạo điểm thay thế. Thông báo
`choose another --frame-index` từ helper một-frame không được dùng để thay
membership của benchmark. Nếu các frame này không đánh giá được, full report
phải là **partial** với danh sách thiếu, dù các frame còn lại thành công.

## Audit radar đầu scene bằng CPU

Audit ngày 2026-11-14 tại [radar_startup_audit.json](../work_dirs/radar_startup_audit.json)
đối chiếu nuScenes API với các bảng JSON có cùng hash như benchmark. JSON ghi
đầy đủ token/timestamp, đường dẫn/size/hash file, toàn bộ chuỗi prev, record
gần nhất thỏa điều kiện và lỗi adapter của từng radar.

Timestamp LiDAR tham chiếu: scene-0103 là **1533151603547590 µs**; scene-0916
là **1538984233547259 µs**. Trong bảng, Δ = radar − LiDAR; dương là tương lai.

| Radar | scene-0103 Δ (ms) | Record <= LiDAR | scene-0916 Δ (ms) | Record <= LiDAR |
| --- | ---: | ---: | ---: | ---: |
| RADAR_FRONT | +8.401 | 0 | +13.575 | 0 |
| RADAR_FRONT_LEFT | -21.242 | 1 | +12.695 | 0 |
| RADAR_FRONT_RIGHT | -34.709 | 1 | +33.719 | 0 |
| RADAR_BACK_LEFT | -25.352 | 1 | +21.259 | 0 |
| RADAR_BACK_RIGHT | +28.833 | 0 | +2.808 | 0 |

**Cả 10 radar keyframe có prev=""**: chuỗi chỉ gồm record được sample trỏ tới,
không có predecessor trong metadata. **Cả 10 file radar và hai file LiDAR đều
tồn tại, không rỗng**; toàn bộ 12 sensor keyframe của mỗi sample cũng qua kiểm
tra tồn tại/không rỗng. Đây không phải lỗi thiếu file vật lý.

Quét độc lập mọi sample_data cùng radar/cùng scene cho kết quả giống chuỗi
prev: bảy kênh có Δ dương không có bất kỳ record <= LiDAR nào. Ba kênh còn lại
của scene-0103 dùng được chính keyframe; thiếu 6 sweep chỉ gây warning, không
tự nó làm frame bị từ chối. Không phát hiện lỗi duyệt/lọc hoặc record hợp lệ
bị adapter bỏ sót trong hai trường hợp này.

`make_frame_info` lọc timestamp tại dòng 192, theo prev ở dòng 196, rồi từ chối
kênh có danh sách rỗng tại dòng 216–220. Gọi riêng hàm dựng metadata trên CPU
tái hiện chính xác cả hai lỗi benchmark. Không nạp model/checkpoint, không
decode ảnh/point cloud, không forward; import Torch/OpenMMLab bị chặn trong
lượt audit. Loader gốc đọc `sweeps[0]` tại `occfusion/loading.py:416`, nên bỏ
guard chưa phải cách xử lý kênh vắng hợp lệ.

Lệnh thực tế: `/home/hatis/.venvs/rav25/bin/python /tmp/occfusion_radar_startup_audit.py`.
Hash, size và mtime của **668 file được bảo vệ** giữ nguyên trước/sau audit,
gồm toàn bộ benchmark, dữ liệu review/SQLite và code liên quan. Chỉ thêm báo
cáo audit và phần tài liệu này; coverage vẫn là partial 79/81. Kết luận giới
hạn trong metadata đang dùng, không khẳng định có/không có dữ liệu bên ngoài
metadata và không sử dụng sweep tương lai hoặc dữ liệu thay thế.

## Lệnh chạy trong WSL

Không thêm dependencies. Dùng Python rav25, không dùng môi trường web:

```bash
cd '/mnt/d/New folder/OCCFusion'

# Các test cache dùng artifact lưu sẵn; model factory bị chặn trong test.
MPLCONFIGDIR=/tmp/occfusion-matplotlib \
  /home/hatis/.venvs/rav25/bin/python -m unittest discover \
  -s tests -p 'test_benchmark_occupancy.py' -v

# Pilot hiện tại: hai frame đại diện, một model/checkpoint cho cả tiến trình.
/home/hatis/.venvs/rav25/bin/python tools/benchmark_occupancy.py \
  --gt-root data/nuscenes/occ_samples/nuscenes_occ/samples \
  --out-dir work_dirs/benchmark_mini_val_pilot_v2 --pilot --verify-reference

# Kiểm chứng resume: tuyệt đối không dựng model hoặc forward.
/home/hatis/.venvs/rav25/bin/python tools/benchmark_occupancy.py \
  --gt-root data/nuscenes/occ_samples/nuscenes_occ/samples \
  --out-dir work_dirs/benchmark_mini_val_pilot_v2 --pilot --verify-reference \
  --resume --evaluate-only

# Sau khi pilot/reference/resume đạt, chạy manifest toàn bộ mini_val.
/home/hatis/.venvs/rav25/bin/python tools/benchmark_occupancy.py \
  --gt-root data/nuscenes/occ_samples/nuscenes_occ/samples \
  --out-dir work_dirs/benchmark_mini_val --verify-reference

# Tiếp tục full benchmark, chỉ chạy mới prediction bị thiếu/không còn hợp lệ.
/home/hatis/.venvs/rav25/bin/python tools/benchmark_occupancy.py \
  --gt-root data/nuscenes/occ_samples/nuscenes_occ/samples \
  --out-dir work_dirs/benchmark_mini_val --verify-reference --resume

# Audit full benchmark bằng cache, không được gọi inference dù cache lỗi.
/home/hatis/.venvs/rav25/bin/python tools/benchmark_occupancy.py \
  --gt-root data/nuscenes/occ_samples/nuscenes_occ/samples \
  --out-dir work_dirs/benchmark_mini_val --verify-reference --resume --evaluate-only
```

Trong Codex, lệnh cần CUDA chạy ngoài sandbox như mốc inference. Resume/evaluate
only không yêu cầu model/GPU forward. Các lệnh thực tế đã để lại output; khi thử
mới phải dùng thư mục mới, hoặc --resume với đúng manifest. Khác --pilot/full,
config/checkpoint/provenance/membership sẽ bị từ chối, không đổi manifest cũ.
GT root tương đối tính từ cwd; chạy từ gốc repo như các ví dụ trên.

`--config`, `--data-root`, `--checkpoint` có thể chọn path, nhưng model, ảnh,
grid, sensors, loaders, pipeline và v1.0-mini phải đúng cấu hình baseline.
Không có tùy chọn giảm ảnh/grid hoặc tự đổi precision. --device mặc định cuda:0;
--warmup-forwards mặc định 1. Exit 0 cho complete của scope manifest, 2 cho
partial, 1 cho lỗi khởi tạo/provenance/orchestration.

## Đường suy luận và vòng đời GPU

Tái sử dụng nguyên các helper đã kiểm chứng từ infer_nuscenes_mini.py:
make_frame_info, prepare_frame/load_points, move_data, strict_load_checkpoint và
export_predictions. Dùng model baseline C+L+R, full-resolution BGR/pad32, grid
200x200x16, voxel 0.5 m, 17 nhãn, FP32, batch 1, CPU preprocessing, past sweeps
và chính sách thiếu sweep như CLI một-frame. Không đưa GT vào model.

Dataset/API được nạp một lần. Model được dựng lazy tại frame đầu thật sự cần
prediction mới, checkpoint nạp strict một lần cho tiến trình. Model eval trước
forward; sau chuyển CUDA gọi eval lại để dựng cache EfficientViT đúng device;
preprocessor giữ CPU. Forward nằm trong inference_mode, đường mode='tensor'.

TF32 của baseline được kiểm tra/ghi: matmul=false, cudnn=true. Autocast=false.
Runtime khác precision/default TF32 bị từ chối thay vì âm thầm đổi baseline.
Sau mỗi frame, xuất prediction CPU rồi xóa prepared/logits GPU và gc.collect;
không giữ logits GPU của cả tập. Ghi allocated_after_release_bytes khi model
vẫn resident. OOM ghi sample/stage/traceback, giải phóng allocator cache để thử
frame sau với cùng baseline; không thay grid/ảnh hoặc nạp model lại mỗi frame.

## Resume và cache

Một file lock ngăn hai tiến trình ghi cùng benchmark. Manifest bất biến, có
identity hash; kiểm tra cả danh sách frame và provenance nhúng khi resume.
Không nhận thư mục run cũ làm benchmark, không import artifact bên ngoài chỉ
bằng tên/token hoặc histogram.

Prediction cache yêu cầu:

- Thư mục attempt đã publish, có prediction_commit.json success.
- Hash/size của occupancy.npz, metadata.json, report.json, resolved_config.py,
  bev.png khớp receipt; token metadata/report/receipt khớp manifest và API.
- Checkpoint SHA-256, resolved/source config, source code model/CLI/adapter,
  phiên bản thư viện preprocessing (gồm OpenCV/Pillow/pyquaternion), TF32/precision
  và hash các bảng dataset khớp provenance.
- Calibration/pose/sweep metadata cùng hash byte của mọi file cảm biến được dùng
  khớp input fingerprint hiện tại. Không chỉ tin mtime/size.

Evaluation cache yêu cầu evaluation_commit.json, hash bản evaluation, prediction
hash/bốn artifact nguồn, GT hash và protocol fingerprint khớp. Hash GT thay đổi
không buộc suy luận lại nếu prediction hợp lệ: chỉ tính lại evaluation CPU và
đánh dấu GT khác manifest ban đầu. Đổi protocol cũng chỉ tính lại evaluation.
Đổi model/config/code/libs làm provenance prediction khác thì dùng benchmark
mới, không ghi đè bộ cũ.

Attempt được ghi trong thư mục .pending rồi rename cùng filesystem sau khi
artifact và receipt đã xong. .pending hoặc file chỉ có tên đúng không được coi
là success. Prediction commit được cập nhật vào progress trước evaluation,
nên có thể phục hồi từ interruption giữa hai bước. Resume kiểm tra cả các
attempt đã commit nếu progress chưa kịp nhận pointer. Attempt cũ không bị xóa.

--evaluate-only cấm model factory: cache prediction không hợp lệ được ghi failed,
không tự chạy inference. Evaluation CPU vẫn có thể tính mới nếu cần.

## Pooled metric

Dùng sparse_to_dense, confusion_statistics, summarize và compare_reference của
evaluator CPU hiện có. GT sparse 0 -> ignore 255; ô không liệt kê -> empty 0.
Bỏ GT ignore, không dùng lidarseg, box rasterization hoặc camera mask thay GT.

Cộng confusion/intersection/GT/prediction counts của các token thành công duy
nhất trong manifest rồi mới tính tỷ lệ. Không trung bình IoU từng frame. Mean
đủ 16 lớp semantic 1..16; union=0 cho IoU 0. Warm-up không bị loại khỏi accuracy.
--verify-reference so pooled với nguyên hàm EvalMetric gốc qua AST/NumPy CPU.

Chỉ complete khi mọi frame manifest đã được đánh giá hợp lệ và quá trình tổng
hợp kết thúc. Partial ghi count kỳ vọng/thành công/thất bại, actual_scene_counts,
danh sách thiếu và phạm vi thực tế. Metric luôn là tỷ lệ [0,1].

## Thời gian và allocator

Bootstrap, setup dataset/config/hash và khởi tạo model/checkpoint/device được
ghi riêng. Frame_wall_s trừ thời gian shared model startup của frame đầu; vẫn
gồm validation/hash, preprocessing, export, evaluation CPU và cleanup của frame.
Timings report mô tả pipeline_total của inference riêng với scope đã nêu.

Forward được synchronize CUDA trước và sau; clock chỉ bao đường tensor forward.
Những forward đầu theo warmup-forwards được đánh dấu, vẫn xuất prediction và
được tính accuracy. Báo median/p95 (NumPy linear percentile), sample count và
scope cho cả tất cả completed forward và loại warm-up, theo invocation hiện
hành và toàn lịch sử benchmark. Có thể có retries trong lịch sử đo.

Cache rows có forward_s trống, không lấy thời gian đọc cache làm latency.
Invocation toàn cache cho số mẫu latency 0 và median/p95 null. CSV cũ giữ các
forward thật đã chạy; không thay bằng thời gian cache của lần resume.

Peak allocated/reserved là **PyTorch allocator**, không phải VRAM vật lý thực
đo. Nominal GPU total memory trong thông tin device không phải mức sử dụng.
Đây là một lần quét sequential trên laptop; không phải nhiều trial độc lập
hoặc cam kết latency ổn định khi nhiệt độ/clock/tải hệ thống thay đổi.

## Cấu trúc đầu ra

```text
benchmark_mini_val/
├── manifest.json
├── progress.json
├── metrics_pooled.json
├── benchmark_report.json
├── timings.csv
├── frames/<sample_token>/<attempt>/
│   ├── occupancy.npz, metadata.json, report.json, bev.png, resolved_config.py
│   └── prediction_commit.json
├── evaluations/<sample_token>/<attempt>/
│   ├── evaluation.json
│   └── evaluation_commit.json
└── failures/<sample_token>/<session_id>.json
```

Progress/report/pooled metric cập nhật sau mỗi frame bằng JSON temp + fsync +
rename. Timings CSV append/flush/fsync. Các trạng thái đang chạy không quảng bá
complete. Không gắn metrics_pooled.json vào run hay approval của một frame.

## Kết quả thực tế ngày 2026-11-14

Đã chạy các lệnh pilot, pilot resume/evaluate-only và full benchmark ở trên
bằng `/home/hatis/.venvs/rav25/bin/python`. Pilot v2 đạt **2/2**, pooled khớp
EvalMetric gốc với sai số tối đa 0. Một model và checkpoint được nạp cho hai
forward; khi resume chỉ có 2 prediction cache và 2 evaluation cache, model/
checkpoint/forward đều bằng 0.

Full benchmark đã thử hết **81/81 frame trong manifest**, tạo mới 79 prediction
và 79 evaluation. Kết quả cuối là **partial**, exit code **2**:

| Phạm vi | Kỳ vọng | GT có sẵn | Đánh giá hợp lệ | Lỗi |
| --- | ---: | ---: | ---: | ---: |
| scene-0103 | 40 | 40 | 39 | 1 |
| scene-0916 | 41 | 41 | 40 | 1 |
| mini_val | 81 | 81 | 79 | 2 |

Hai lỗi là input_validation ở frame 0 đã liệt kê phía trên. Không có OOM/lỗi
forward, không đổi baseline để xử lý thiếu radar. Chưa có metric hợp lệ cho
hai frame này; chưa đạt coverage 81/81. Không coi kết quả 79 frame sau đây là
một lượt mini_val hoàn chỉnh hoặc kết quả toàn nuScenes validation.

| Metric pooled trên **79 frame hợp lệ** | Kết quả |
| --- | ---: |
| Occupancy IoU | 44.1789% |
| Semantic mIoU, đủ 16 lớp | 23.0672% |
| Occupied intersection / union | 2,186,946 / 4,950,209 |
| Số voxel hợp lệ sau bỏ GT ignore | 50,536,501 |
| Sai số tuyệt đối lớn nhất so với EvalMetric gốc | 1.1102230246251565e-16 |

JSON giữ tỷ lệ [0,1]; bảng này chỉ nhân 100 một lần để hiển thị. Không lấy
trung bình IoU của 79 frame. Bằng chứng confusion matrix, counts và token của
tập con nằm ở `work_dirs/benchmark_mini_val/metrics_pooled.json`; từng cặp GT/
prediction có hash riêng trong evaluation và receipt.

| ID | Lớp | IoU | Intersection | Union |
| ---: | --- | ---: | ---: | ---: |
| 1 | barrier | 0.0000% | 0 | 9 |
| 2 | bicycle | 13.5409% | 361 | 2666 |
| 3 | bus | 45.5475% | 17068 | 37473 |
| 4 | car | 39.5037% | 195939 | 496002 |
| 5 | construction_vehicle | 0.0000% | 0 | 17 |
| 6 | motorcycle | 30.4220% | 4563 | 14999 |
| 7 | pedestrian | 31.6294% | 17152 | 54228 |
| 8 | traffic_cone | 14.4258% | 103 | 714 |
| 9 | trailer | 0.0000% | 0 | 0 |
| 10 | truck | 38.0029% | 17937 | 47199 |
| 11 | driveable_surface | 38.8866% | 527365 | 1356161 |
| 12 | other_flat | 0.0000% | 0 | 338 |
| 13 | sidewalk | 25.6637% | 101015 | 393610 |
| 14 | terrain | 18.8354% | 70769 | 375723 |
| 15 | manmade | 29.3217% | 434354 | 1481342 |
| 16 | vegetation | 43.2960% | 450209 | 1039839 |

Lớp trailer có union=0 vẫn nhận IoU 0 và có mặt trong trung bình 16 lớp.

Lượt GPU dùng FP32, autocast=false, batch 1; TF32 matmul=false, cudnn=true.
Dataset/model/checkpoint được nạp **mỗi loại một lần**. Thời gian bootstrap
0.119 s, setup chung 13.283 s, khởi tạo model/checkpoint/device 15.392 s.
`current_invocation.wall_s` của lượt này là 2706.198 s, gồm khởi tạo model,
vòng xử lý, reference và ghi báo cáo; không gồm bootstrap/setup chung. Thời
gian model startup được trừ khỏi frame_wall_s của frame đầu.

| Phép đo forward CUDA đã đồng bộ | Số mẫu | Median | p95 |
| --- | ---: | ---: | ---: |
| Tất cả forward thực sự chạy | 79 | 32.5019 s | 36.3541 s |
| Bỏ một forward đánh dấu warm-up | 78 | 32.5109 s | 36.3683 s |

Warm-up 11.7585 s vẫn có trong accuracy; phạm vi forward quan sát được là
11.7585–39.0334 s. Đây là một lượt sequential trên laptop, chưa phải phép đo
latency ổn định qua nhiều trial.

Peak **PyTorch allocator**: allocated 5,720,832,000 byte (5.32794 GiB), reserved
6,245,318,656 byte (5.81641 GiB). Không diễn giải các số này là VRAM vật lý thực
đo. Allocated sau giải phóng tensor của từng frame đều là 609,836,544 byte,
với model còn resident; không giữ logits GPU của cả tập.

`initial_report.json` giữ báo cáo lượt GPU trước khi resume. File
`benchmark_script_used.py` trong thư mục benchmark giữ đúng source của lượt
đó, SHA-256 `2cd8a4e1476a205844fc22046905454b3ab3ecb671b95ceb997d45566643bc6b`.
Mỗi invocation tự ghi hash script hiện hành; fingerprint adapter suy luận
được ghi riêng để thay đổi xử lý CSV không làm mất cache prediction hợp lệ.

## Kiểm chứng cache và tính toàn vẹn

**15 unittest đạt** trong 3.705 s bằng lệnh test phía trên; py_compile hai file
Python đạt. Test cache dùng bản sao prediction thật và GT nhỏ chỉ phục vụ test,
không dùng các metric test làm số liệu benchmark. Model factory trong các test
artifact là mock sẽ báo lỗi nếu bị gọi.

Các trường hợp đã kiểm tra: thứ tự manifest/pilot cố định; resume thường và
evaluate-only; khôi phục attempt đã commit khi progress chưa có pointer;
prediction thay byte dù giữ nguyên size/mtime; GT/protocol thay đổi chỉ chạy
lại evaluator CPU; evaluation thay hash được tính lại và giữ bản cũ; sai input/
model provenance; .pending không được nhận là success; sửa membership không
biến partial thành complete; không nhận/ghi vào run cũ; cache không trở thành
latency; warm-up tách riêng; timing NaN bị loại; CSV rỗng được ghi lại header.

Full benchmark đã chạy lại bằng **--resume thông thường**, dùng đúng 79
prediction và 79 evaluation cache; tạo mới 0, dataset nạp 1 lần, model/
checkpoint/forward bằng 0. Metric và reference check khớp lượt GPU. Vòng xử lý
cache này mất 93.563 s (không gồm bootstrap/setup), không được coi là latency
inference. `latency_current_invocation` có số mẫu 0 và median/p95 null;
`latency_all_invocations` vẫn chỉ có 79 forward thật từ lượt GPU.
`resume_report.json` giữ riêng bằng chứng này trước lượt audit tiếp theo.

Lượt **--resume --evaluate-only** cũng đã thử hết manifest, dùng 79 prediction
cache và 79 evaluation cache, tạo mới 0, model/checkpoint/forward bằng 0; vòng
xử lý mất 90.214 s. Metric, pooled reference check và 79 mẫu forward lịch sử
giữ nguyên; latency invocation này có 0 mẫu. Báo cáo riêng được lưu tại
`evaluate_only_report.json`. Cả hai lượt cache trả exit code **2** do cùng hai
frame thiếu radar, không phải lỗi cache hoặc thiếu GT.

Audit CPU cuối cùng ghi `work_dirs/benchmark_mini_val/integrity_audit.json`,
`passed=true`, với các kiểm tra:

- Manifest đủ 81 token duy nhất, giữ nguyên hash; 79 success và 2 lỗi đầu scene.
- Hash 81 file GT không đổi so với manifest; hash của 395 file prediction/config
  và 79 evaluation khớp receipt. Không nhận attempt .pending là kết quả hợp lệ.
- Tổng 79 confusion matrix khớp toàn bộ metrics/counts pooled.
- CSV có 243 dòng sau ba invocation: 79 new, 158 cache và 6 failure rows.
  Mọi dòng cache có forward_s trống; chỉ 79 forward thật đóng góp latency.
- Hash của **9 artifact inference/evaluation cũ và 2 manifest review cũ** không
  thay đổi so với snapshot `work_dirs/benchmark_preservation_before.json`.

Chưa có metric cho hai frame đầu scene theo baseline hiện hành. Pooled kết quả
79 frame là bằng chứng riêng của tập con được công bố; không cập nhật run,
revision, approval hoặc gói export của bất kỳ frame nào.
