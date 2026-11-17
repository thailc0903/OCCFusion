# Temporal training readiness v1 — bổ sung GT và kiểm kê input CPU

Tiếp tục checkout WSL 46a891d4 sau mốc persistence, giữ canonical root
/mnt/d/New folder/OCCFusion. Đây là chuẩn bị dữ liệu/kế hoạch, **không huấn
luyện hoặc chạy inference**. Không sửa persistence_v1, protocol v1, viewer,
model/config/checkpoint/profile, queue, database hoặc approval cũ.

## Code và output

- [tools/prepare_temporal_training.py](../tools/prepare_temporal_training.py):
  entry point CPU, dùng import guard cấm Torch/OpenMMLab/OccFusion.
- [gt_archive.py](../tools/forecasting/gt_archive.py): kiểm tra ZIP, staging,
  decode và publish file GT mới độc quyền, không overwrite.
- [training_readiness.py](../tools/forecasting/training_readiness.py): audit GT,
  raw sensor, cache, manifest theo scene và kế hoạch cache.
- [path_snapshot.py](../tools/forecasting/path_snapshot.py): cache metadata
  filesystem theo file, xác minh identity/size/mtime và directory cuối bước.
- [Tests](../tests/test_temporal_training_readiness.py), hồi quy
  [forecasting](../tests/test_forecasting.py).
- Output: work/forecasting/training_readiness_v1; log/fixture QA riêng tại
  work/forecasting/training_readiness_qa_v1. Không nằm trong work_dirs nên
  không bị viewer nhận là run occupancy/detection.

## Archive và GT import

Nguồn được người dùng tải tại data/downloads/surroundocc/train.zip. Công cụ
đọc toàn bộ central directory và tính SHA-256 archive trước khi stage bất
kỳ payload nào. ZIP phải đúng nuscenes_occ/samples/<LiDAR basename>.npy;
từ chối absolute path, traversal, backslash/drive path, tên không canonical,
symlink/special file, member trùng, encryption và compression không hỗ trợ.
Không dùng extractall. Chỉ đọc payload được chọn hoặc cần đối chiếu conflict;
CRC của payload không được chọn chưa được kiểm tra bằng giải nén.

Danh sách được phép bổ sung lấy **đúng missing_gt của mini_train trong
persistence_v1/gt_audit.json**, đối chiếu lại token/scene/frame/split và tên
LiDAR bằng nuscenes-devkit. Không lấy toàn archive, không bổ sung mini_val
vào tập train. GT hiện có không bị thay thế. Trùng basename khác byte hash
là conflict, kể cả member không thuộc danh sách nhập; trùng cùng byte giữ
file cũ nguyên trạng.

Mọi GT chọn được stage trong output, kiểm tra NPY header/payload trước khi
NumPy cấp phát, allow_pickle=False và decoder sparse_to_dense hiện có.
Kiểm tra N×4 numeric, index XYZ trong 200×200×16, nhãn 0..16/255, hữu hạn,
nguyên và duplicate xung đột. Hàng nhãn 0 → ignore 255; ô không liệt kê →
empty 0 khi decode; **không sửa byte NPY nguồn**.

Sau khi cả batch hợp lệ, file được copy riêng vào temp trong GT root, hash
lại rồi publish exclusive hardlink; staging và GT không chia sẻ inode.
Không overwrite file/symlink đã có. Receipt riêng theo token lưu ngay khi
mỗi file được publish. Khi lỗi/gián đoạn, giữ staging/receipt và các file
mới đã kiểm tra; không tự retry hoặc xóa GT để chạy lại.

NPY không có token, bounds hoặc mô tả hệ tọa độ nhúng. Token/split lấy từ
API, gắn bằng basename LiDAR; geometry/class map theo giao thức SurroundOcc
đã công bố trong OCCUPANCY_EVALUATION.md. Local SHA/decoder không phải chữ
ký/checksum của publisher hay xác minh độc lập lịch sử sinh GT. Không có
publisher checksum được cung cấp cùng archive này.

## Bốn tầng kiểm kê, không đánh đồng

| Tầng | Điều kiện |
| --- | --- |
| Có raw sensor | Đủ 6 camera, LiDAR và 5 radar keyframe thật, file tồn tại/không rỗng |
| Adapter metadata hợp lệ | Reuse make_frame_info của CLI, calibration/pose SE(3), prev-chain, sweep timestamp, file được kiểm tra và không radar rỗng |
| Có GT hợp lệ | Tên/token/split đúng API, decode hợp lệ và hash; độc lập với prediction |
| Có cache prediction v1 hợp lệ | Saved OCCFusion labels/max-softmax đúng schema/token/geometry/baseline, commit/artifact hashes và historical raw-input fingerprint/hash đầy đủ |
| Có window ready | Ba history cache hợp lệ, tương lai có GT, cùng scene, timing/cutoff đúng protocol v1 |

Raw audit **không decode toàn bộ ảnh/point payload**, không dựng preprocessor
hoặc model; metadata-valid không bảo đảm forward sau này sẽ thành công.
Keyframe radar tương lai bị adapter loại vẫn được audit nhưng không phải
input. Camera sau timestamp LiDAR của anchor làm window bị chặn; camera
muộn hơn reference của history cũ chỉ được phép nếu vẫn ≤ anchor cutoff.

Saved prediction legacy có thể đúng shape/token và được job publication
hash-pin nhưng thiếu historical raw sensor hash/input fingerprint. Báo
riêng legacy_output_valid_incomplete_input_provenance; không giả là không
có file, không tự nâng thành cache v1. Hash raw sensor **bây giờ** không chứng
minh bytes đã dùng khi inference cũ. Cần khôi phục bằng chứng gốc hoặc được
cho phép tạo cache mới trước khi dùng theo policy v1 hiện có.

**Không dùng GT làm history prediction**, không tạo occupancy.npz giả,
không chạy model để lấp cache trong lượt này. Legacy evaluation.json ghi
missing_gt tiếp tục giữ nguyên dù GT vừa được bổ sung; audit mới ở output
riêng, không mở rộng approval/evaluation cũ.

## Protocol và train/validation

Giữ nguyên protocol/fingerprint từ persistence_v1/protocol.json:
history −1,0 / −0,5 / 0 s, horizon +0,5 / +1,0 / +1,5 s, timestamp LiDAR,
tolerance 50 ms, sample liên tiếp, không vượt scene/padding/skip, cutoff
chính timestamp anchor. Hệ anchor native LIDAR_TOP, XYZ 200×200×16, 0,5 m,
full SE(3), mask history/target tách biệt và semantic cell lookup không đổi.

train_manifest.json chỉ chứa official mini_train scenes. validation_manifest.json
chỉ chứa mini_val, purpose=evaluation_only_not_training. Công cụ kiểm tra
hai scene sets không giao nhau và IDs các validation window ready phải
khớp chính xác 26 window của persistence v1 đã lưu; không chọn theo metric.
Future GT/pose vẫn chỉ là target offline, không phải history input.

Manifest readiness chưa phải tensor dataset đã sẵn sàng huấn luyện. Nó ghi
cả status hiện tại lẫn potential_after_verified_cache_generation. Potential
chỉ có nghĩa metadata/timing/GT cho phép nếu tạo cache hợp lệ sau này, không
được cộng vào số window ready thực tế.

Checkpoint C+L+R giữ identity từ manifest/profile cũ, không đọc weights.
Không có danh sách sample training gốc của checkpoint để chứng minh held-out
độc lập. Report ghi membership mini split và full nuScenes train/val/test
theo devkit; chúng là các scope khác nhau. Mini_val không vào training dù
đã có prediction/GT đầy đủ hơn. Chưa chọn temporal architecture; NPZ hiện
chỉ có labels/max-softmax, không đủ logits/features nếu model tương lai cần.

## Kế hoạch cache, không thực thi

cache_generation_plan.json liệt kê từng token, scene, frame, timestamp,
LiDAR sample_data token/file, raw blockers, legacy output và window cần nó.
Tách tất cả frame thiếu cache, metadata-runnable, sensor-blocked và tập
history tối thiểu của những potential windows. Không tạo job hoặc command
tự chạy GPU; worker và ba job queued giữ nguyên.

Dung lượng ước tính từ median/p95 **file thật** của các cache v1 hiện có:
prediction NPZ + metadata/report/BEV/resolved config/commit. Đây là estimate
payload, không gồm dataset/GT, model, evaluation, feature/logit cache, staging
lỗi hay reserve. Không biến estimate thành dung lượng bảo đảm hoặc latency
model. Code benchmark occupancy hiện chọn mini_val: không dùng nó như lệnh
train cache. Bước GPU tương lai cần đường mini_train được cho phép riêng,
tái sử dụng model/preprocessing đã kiểm chứng và ghi provenance input thật.

## Lệnh WSL

~~~bash
cd '/mnt/d/New folder/OCCFusion'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''

/home/hatis/.venvs/rav25/bin/python -m unittest discover \
  -s tests -p 'test_temporal_training_readiness.py' -v
/home/hatis/.venvs/rav25/bin/python -m unittest discover \
  -s tests -p 'test_forecasting.py' -v
/home/hatis/.venvs/rav25/bin/python -m py_compile \
  tools/prepare_temporal_training.py tools/forecasting/gt_archive.py \
  tools/forecasting/training_readiness.py tests/test_temporal_training_readiness.py

# Lệnh bổ sung GT/audit thực tế; output phải mới, không chạy lại ghi đè:
/home/hatis/.venvs/rav25/bin/python tools/prepare_temporal_training.py \
  --install-missing-gt --out-dir work/forecasting/training_readiness_v1

# Chỉ dùng với audit đã gián đoạn và có resume_state/GT phase đã niêm phong:
/home/hatis/.venvs/rav25/bin/python tools/prepare_temporal_training.py \
  --resume-audit --out-dir work/forecasting/training_readiness_v1
~~~

--inspect-only chỉ kiểm tra archive, không stage/install/audit model.
Không có --install-missing-gt thì chỉ audit dữ liệu hiện tại. Exit 0=ready
hoặc inspect-only thành công; exit 2=audit hoàn tất nhưng training blocked;
exit 1=lỗi input/conflict/IO. Output đã có bị từ chối nếu không phải resume
được yêu cầu rõ ràng với các phase đã niêm phong; không tự retry.
Không source launcher, không bật worker hoặc tải thêm dữ liệu.

## Kết quả thực tế

Ngày 2026-11-17. Mã cuối đạt **31 test readiness + 37 test forecasting hồi
quy = 68 passed**, không fail/skip; py_compile đạt. Trước import đã đạt 22
test readiness; 22 và lượt 30 test trung gian là tập con của 31, không cộng
lặp. Fixture chỉ dùng thư mục
tạm, không dùng production DB. Không chạy backend/TypeScript/build vì không
sửa web.

Archive thật: **3.209.383.603 byte**, SHA-256
0cb17cae2d672dab3493589e3476882c53b5f17ea60d9989441ef78b8733bc77.
Central directory có 28.132 entries, gồm 28.130 NPY; tổng payload khai báo
35.595.865.536 byte. Chỉ stage/install **242 NPY = 321.638.176 byte**, không
giải nén phần còn lại. Có đủ 242 requested filenames, không basename trùng
6.019 GT cũ, **0 conflict**. NPY mới được giữ nguyên byte, kiểm tra CRC/decode
và hash sau publish; mỗi token có receipt riêng.

GT audit sau bổ sung:

| Split / scene | Số frame | GT hợp lệ | Thiếu | Lỗi |
| --- | ---: | ---: | ---: | ---: |
| mini_train / scene-0061 | 39 | 39 | 0 | 0 |
| mini_train / scene-0553 | 41 | 41 | 0 | 0 |
| mini_train / scene-0655 | 41 | 41 | 0 | 0 |
| mini_train / scene-0757 | 41 | 41 | 0 | 0 |
| mini_train / scene-0796 | 40 | 40 | 0 | 0 |
| mini_train / scene-1077 | 41 | 41 | 0 | 0 |
| mini_train / scene-1094 | 40 | 40 | 0 | 0 |
| mini_train / scene-1100 | 40 | 40 | 0 | 0 |
| **mini_train tổng** | **323** | **323** | **0** | **0** |
| mini_val / scene-0103 | 40 | 40 | 0 | 0 |
| mini_val / scene-0916 | 41 | 41 | 0 | 0 |
| **mini_val tổng** | **81** | **81** | **0** | **0** |

### Readiness thực tế

**Kết luận: BLOCKED cho training**, vì chưa có window train đủ ba cache
prediction hợp lệ theo provenance v1. Bước bổ sung GT đã thành công, không
còn thiếu GT trong mini_train/mini_val hiện có.

| Split | Raw keyframe có file | Adapter hợp lệ | GT hợp lệ | Cache v1 unique | Legacy output unique | Window ready | Potential sau tạo cache |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mini_train | 323/323 | 316/323 | 323/323 | **0** | 2 | **0/323 candidate** | **147** |
| mini_val, chỉ evaluation | 81/81 | 79/81 | 81/81 | **79** | 2 | **26/81 candidate** | 26 |

Val window IDs và protocol fingerprint
bb1501449e796e219bb0ec9d7c80ab7100db68fedd16fec4ebfdbed6902f1908
khớp persistence_v1, không đưa scene validation vào training và không chạy
lại metric. Cache inventory đọc 89 bundle: 81 bundle đạt v1 (79 token duy
nhất, gồm copy/cache trùng), 6 legacy bundle thiếu provenance input và 2
bundle bị từ chối vì shared/input fingerprint khác baseline được chọn.
Không sửa, xóa hoặc tự thay thế các bundle đó.

Hai legacy train thật:

- scene-0061 frame **1**, sample 39586f9d59004284a7114a68825e8eec,
  work_dirs/infer_mini_cuda_v2: NPZ/shape/token/geometry đúng, nhưng không có
  historical raw-input fingerprint. CLI source hash a9b0… khác hash 354c…
  của baseline hiện tại; không thể mặc nhiên chứng nhận preprocessing đồng nhất.
- scene-1077 frame **13**, sample 8c8755ebb0bf4c519416323acad6cea1,
  work_dirs/infer_job_e1c62ec025744027a96070e955c333c9: publication hash-pin
  thực và đúng baseline, nhưng vẫn thiếu historical raw sensor hashes như
  benchmark cache v1. Không tự gắn provenance hồi tố hoặc thay approval.

Chúng là prediction đã lưu, không phải GT và không bị gọi là file thiếu.
Tuy nhiên chưa đủ điều kiện cache train theo policy v1 đã giữ nguyên.

| mini_train scene | Raw file | Adapter valid | GT valid | Cache v1 | Window ready | Potential sau cache |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scene-0061 | 39 | 38 | 39 | 0 | 0 | 6 |
| scene-0553 | 41 | 40 | 41 | 0 | 0 | 6 |
| scene-0655 | 41 | 40 | 41 | 0 | 0 | 25 |
| scene-0757 | 41 | 40 | 41 | 0 | 0 | 31 |
| scene-0796 | 40 | 40 | 40 | 0 | 0 | 31 |
| scene-1077 | 41 | 40 | 41 | 0 | 0 | 12 |
| scene-1094 | 40 | 39 | 40 | 0 | 0 | 20 |
| scene-1100 | 40 | 39 | 40 | 0 | 0 | 16 |

### Kế hoạch cache còn thiếu

- Thiếu **323 cache v1**. Trong đó **316 frame metadata-runnable**, **7
  frame bị chặn**. Không có job mới hoặc GPU được chạy.
- Tập history tối thiểu phục vụ 147 potential windows gồm **186 frame**
  chưa có cache. Danh sách token/scene/frame nằm trong cache_generation_plan.json,
  mỗi frame ghi window cần nó; các frame còn lại không mặc nhiên thuộc tập
  tối thiểu này.
- Estimate dựa trên 79 cache thật: per-frame median **1.732.626 byte**,
  p95 **1.881.266,7 byte**. Tổng 316 frame: khoảng **522,15 / 566,94 MiB**
  theo median/p95 mỗi file bundle. Tập tối thiểu 186 frame: **307,34 /
  333,71 MiB**. Đây không phải p95 của tổng dung lượng hay reserve đầy đủ;
  không gồm feature/logit cache, dataset, GT, model, evaluation và lỗi staging.
- Có **64 frame train** có sensor thực dùng sau timestamp LiDAR của chính
  nó. Không đổi cutoff; có thể chỉ dùng cache đó ở history cũ khi timestamp
  vẫn ≤ anchor. Potential windows đã áp dụng đúng quy tắc này.

Bảy frame sensor-blocked đều frame 0, có file radar nhưng không có record
quá khứ hợp lệ cho các kênh sau. Không dùng sweep tương lai hoặc padding:

| Scene | Sample token | Radar không có record ≤ LiDAR |
| --- | --- | --- |
| scene-0061 | ca9a282c9e77460f8360f564131a8af5 | FRONT, FRONT_LEFT, BACK_LEFT |
| scene-0553 | 8687ba92abd3406aa797115b874ebeba | BACK_LEFT |
| scene-0655 | 5991fad3280c4f84b331536c32001a04 | FRONT_LEFT, BACK_LEFT |
| scene-0757 | cd9964f8c3d34383b16e9c2997de1ed0 | FRONT |
| scene-1077 | 5998b71b64c146769bde1d5430741381 | FRONT, FRONT_LEFT, FRONT_RIGHT, BACK_LEFT |
| scene-1094 | e6b0b282aa174a978272dc2d0a89d560 | FRONT_RIGHT, BACK_LEFT |
| scene-1100 | a480496a5988410fbe3d8ed6c84da996 | BACK_LEFT |

Tên kênh trong bảng có prefix RADAR_. Hai frame radar lỗi mini_val cũ vẫn
giữ nguyên trong manifest. Scene-0796 frame 0 qua adapter nhưng không có
nghĩa mọi anchor/window trong scene đều qua timing/cutoff.

Devkit full split cho thấy scene-0553/0796 thuộc full **val** dù nằm trong
official **mini_train**; sáu mini_train scene còn lại thuộc full train.
Hai mini_val scene thuộc full val. Không đánh đồng các split, không có bằng
chứng sample-level training provenance của checkpoint để tuyên bố held-out
độc lập. Không huấn luyện trong lượt này.

### Gián đoạn, tiếp tục và bảo toàn

GT import thực thi **một lần** và hoàn tất đủ 242 receipt. Sau đó audit raw
chỉ đọc gặp >20 phút lặp truy vấn DrvFS (wchan p9_client_rpc). Chỉ tiến trình
CPU audit của lượt này (PID 6661) được dừng bằng SIGTERM/exit 143; không hủy
job trong queue. Code lúc import lưu tại code_at_gt_import, các JSON hoàn
tất được hash-pin trong resume_state.json, log prepare.log giữ nguyên.

Resume rehash archive, 242 GT đã nhập và toàn bộ GT mini đã decode; không
stage/install lại hoặc overwrite JSON GT hoàn tất. Snapshot filesystem cache
fact thật một lần mỗi file, vẫn kiểm tra symlink/containment, rồi kiểm lại
identity/size/mtime tất cả file và directory khi ra khỏi bước. Có **19.027
path / 18 directory** được kiểm tra, stable_at_exit=true. Metadata/warnings/
error của adapter gốc và snapshot khớp chính xác trên scene-0061/1,
scene-1077/1, scene-0103/3 và frame radar lỗi scene-0916/0; bằng chứng tại
training_readiness_qa_v1/path_snapshot_equivalence.json. Không sửa file CLI
inference gốc hoặc thay preprocessing/protocol để tối ưu filesystem.

Lượt resume hoàn tất, exit **2** do training blocked như bảng trên, không
phải import thất bại. Thời gian 879,706 s là CPU data audit/IO của lượt
resume, không phải latency inference. Báo cáo/kiểm chứng bảo toàn cuối ở
training_readiness_v1/preservation_after.json và completion_report.json.

Model load, checkpoint load/đọc weights, model forward, GPU, training, tạo
job và bật worker đều **0**. Production không restart; ba job queued được
giữ nguyên. Manifest/GT mới không được tự gắn vào approval/export cũ.
