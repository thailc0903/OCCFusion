# Báo cáo khả năng chạy baseline OccFusion

Ngày kiểm tra: 2026-11-12. Đặc tả áp dụng: [PROJECT_SPEC.md](../PROJECT_SPEC.md).

**Trạng thái: chưa chạy inference; chưa đủ điều kiện chạy baseline ngay trên môi trường đã kiểm tra.** Checkout có mã mô hình, pipeline huấn luyện/đánh giá và liên kết checkpoint. Các phần đang thiếu là môi trường PyTorch/OpenMMLab, dữ liệu mẫu đầy đủ và checkpoint OccFusion. Máy có GPU NVIDIA RTX 3050 Laptop 4 GB; khả năng đủ VRAM cho cấu hình gốc chưa được kiểm chứng.

Phạm vi lần này là khảo sát và tài liệu hóa. Không cài thư viện, tải dataset/checkpoint, huấn luyện hoặc sửa mã mô hình. Các bước chạy ở mục 8 là **đề xuất chưa thực thi**.

Quy ước bằng chứng:

- **Đã kiểm tra — máy/checkout:** kết quả lệnh đọc trạng thái thực tế.
- **Đã kiểm tra — mã nguồn:** hành vi đọc được từ code/config; chưa đồng nghĩa đã chạy thành công.
- **Suy luận:** hệ quả dự kiến từ mã nguồn, cần xác nhận bằng runtime/dữ liệu thật.
- **Đề xuất:** việc cần làm tiếp, chưa phải tính năng hoặc kết quả hiện có.

## 1. Checkout và thay đổi chưa commit

**Đã kiểm tra — checkout:**

| Hạng mục | Kết quả |
| --- | --- |
| Thư mục | `D:\New folder\OCCFusion` |
| Remote `origin` | `https://github.com/DanielMing123/OCCFusion.git` |
| HEAD | `193a3c3cb38a8ca8978be54c98e436eb96b5e8c3` |
| Commit cuối | `193a3c3 Update dataset.md` |
| Trước nhiệm vụ này | `git status --short` rỗng; không có thay đổi chưa commit |
| Thay đổi của nhiệm vụ này | Thêm `PROJECT_SPEC.md` và báo cáo này; không sửa file đã có |

Bằng chứng: `git rev-parse HEAD`, `git log -1 --format="%h %s"`, `git remote -v`, `git status --short`. Không clone thay thế checkout hiện tại.

## 2. Môi trường thực tế

**Đã kiểm tra — máy/checkout:**

| Thành phần | Kết quả | Giới hạn kết luận |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, tổng VRAM `4096 MiB`, driver `610.62` | Có GPU; chưa chạy model để biết peak VRAM |
| NVIDIA driver | `nvidia-smi` chạy được, báo CUDA UMD Version `13.3` | Đây không phải bằng chứng đã cài CUDA Toolkit hoặc PyTorch CUDA runtime |
| Python tìm bằng `uv` | CPython `3.15.2`, `3.12.16`, `3.12.10` | Không phải Python 3.8 theo hướng dẫn Conda |
| Gói trong ba interpreter trên | Không có `torch`, `torchvision`, `mmengine`, `mmcv`, `mmdet`, `mmdet3d`, `mmsegmentation`, `timm`, `pykitti`, `nuscenes-devkit`, `focal-loss-torch` | Đã đọc metadata gói; không khẳng định mọi virtualenv trên máy đều thiếu |
| Công cụ trên PATH | Có `uv`, `nvidia-smi`, `wsl`; không tìm thấy `python`, `python3`, `py`, `conda`, `pip`, `docker`, `nvcc` | Không tìm thấy trên PATH không có nghĩa chưa cài ở nơi khác; chưa xác minh distro WSL/GPU passthrough |
| Tài nguyên trong repo | Chưa có `data/`, `ckpt/`, `fix_param_small/`, `work_dirs/`; không tìm thấy file trọng số/PKL/nhãn occupancy trong checkout | Không kiểm kê các ổ đĩa ngoài repo |

Bằng chứng môi trường: `Get-Command`, `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader`, `nvidia-smi`, `uv python list --only-installed --offline --no-cache`, và `importlib.metadata.version()` chạy bằng đường dẫn đầy đủ của từng interpreter. Bằng chứng tài nguyên: `Test-Path` và `rg --files --hidden` theo phần mở rộng trọng số/dữ liệu.

**Suy luận:** VRAM 4 GB là rủi ro đối với cấu hình ảnh sáu camera, ResNet-101 và volume 3D. Không có số đo để kết luận chắc chắn chạy được hoặc chắc chắn OOM. Các cấp phát `repeat()` trong view transform cũng cần được đo thực tế: [multi_scale_inverse_matrixVT.py:357](../occfusion/multi_scale_inverse_matrixVT.py#L357). Không ghi thiếu GPU là nguyên nhân chặn; nguyên nhân hiện tại là thiếu môi trường và tài nguyên đầu vào.

## 3. Phiên bản cần dùng: tài liệu chưa khóa một môi trường duy nhất

**Đã kiểm tra — mã nguồn/tài liệu:** hai hướng cài đặt đang khác nhau.

| Thành phần | Hướng Conda | Dockerfile |
| --- | --- | --- |
| Python | `3.8` | Không khai báo phiên bản Python riêng; phụ thuộc base image |
| PyTorch | Không pin phiên bản | `2.1.1` |
| CUDA | `pytorch-cuda=11.8` | `12.1` |
| cuDNN | Không pin | `8` trong tag base image |
| MMEngine | Không pin | Không pin |
| MMCV | `>=2.0.0rc4` | `>=2.0.0rc4`, bật build ops |
| MMDetection | `>=3.0.0` | `>=3.0.0` |
| MMDetection3D | `>=1.1.0` | `>=1.1.0` |
| MMSegmentation | `>=1.0.0` | `>=1.0.0` |
| Gói bổ sung | `focal_loss_torch` | Có thêm `pykitti`, `timm`, `Cython`, `debugpy`; đều không pin |

Nguồn: [install.md:5](install.md#L5), [install.md:11](install.md#L11), [install.md:17](install.md#L17), [Dockerfile:1](../Dockerfile#L1), [Dockerfile:23](../Dockerfile#L23). Không tìm thấy manifest/lockfile khóa toàn bộ môi trường trong checkout.

**Kết luận:** chưa thể đưa ra bộ phiên bản MMCV/MMDetection3D chính xác đã được chứng minh tương thích. Các ràng buộc `>=` trong repo không bảo đảm mọi tổ hợp được resolver chọn đều chạy được. Cũng chưa có bằng chứng Dockerfile build thành công ở thời điểm kiểm tra.

**Đề xuất cho baseline:** chọn một hướng Linux GPU và dùng base `pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel` của repo làm điểm bắt đầu. Sau khi import, CUDA ops và một frame chạy được, lưu phiên bản Python thực tế, danh sách gói chính xác và image digest. Không trộn tùy ý hướng CUDA 11.8 với 12.1. Dockerfile hiện chỉ chuẩn bị môi trường tại `/workdir`, không copy repo hay cấu hình một dịch vụ inference: [Dockerfile:10](../Dockerfile#L10).

Lý do chưa chọn Windows native cho lệnh mẫu: cấu hình runtime dùng `fork`, backend phân tán `nccl`, và hướng dẫn chạy dùng Bash. Đây là căn cứ chọn môi trường thử, không phải khẳng định không thể port sang Windows: [default_runtime.py:11](../configs/_base_/default_runtime.py#L11), [getting_started.md](getting_started.md).

Các import cần kiểm tra ngoài phiên bản là `mmcv.ops` cho DCNv2, `nuscenes.utils.data_classes.RadarPointCloud`, `pykitti`, `timm`, `focal_loss`. Nguồn: [OccFusion.py:56](../configs/OccFusion.py#L56), [loading.py:13](../occfusion/loading.py#L13), [main.py:9](../occfusion/main.py#L9), [Dockerfile:23](../Dockerfile#L23). Import thành công vẫn chưa thay thế phép thử CUDA thực tế.

## 4. Dataset, nhãn và checkpoint phải đi cùng nhau

**Baseline đề xuất:** `configs/OccFusion.py` + camera/LiDAR/radar + nuScenes validation + nhãn SurroundOcc + checkpoint C+L+R từ Model Zoo. Đây là ghép nối dựa trên tài liệu và config; chưa kiểm tra state dict vì chưa có file checkpoint.

| Tài nguyên | Yêu cầu và bằng chứng | Trạng thái local |
| --- | --- | --- |
| Config | `use_lidar=True`, `use_radar=True`, `use_occ3d=False`, `nbr_class=17`: [OccFusion.py:21](../configs/OccFusion.py#L21) | Có |
| Dataset gốc | nuScenes V1.0 full theo [dataset.md:2](dataset.md#L2); các file camera, LiDAR và radar mà record tham chiếu | Chưa có trong repo |
| Info PKL | `data/nuscenes/nuscenes_infos_occfusion_val.pkl`: [OccFusion.py:220](../configs/OccFusion.py#L220); liên kết tác giả ở [dataset.md:6](dataset.md#L6) | Chưa có |
| Occupancy GT | Nhãn SurroundOcc ở `data/nuscenes/occ_samples/`: [dataset.md:13](dataset.md#L13), [loading.py:337](../occfusion/loading.py#L337) | Chưa có |
| LiDAR segmentation GT | Pipeline vẫn đọc `pts_semantic_mask_path`, prefix `lidarseg/v1.0-trainval`: [OccFusion.py:10](../configs/OccFusion.py#L10), [OccFusion.py:179](../configs/OccFusion.py#L179), [nuscenes_dataset.py:126](../occfusion/nuscenes_dataset.py#L126) | Chưa có; thư mục này chưa được vẽ trong cây dữ liệu của tài liệu |
| Checkpoint OccFusion | Chọn liên kết **C+L+R** ở [README.md:36](../README.md#L36) cho config mặc định | Chưa tải; chưa kiểm tra tên file/hash/key/shape |
| Backbone pretrain | `ckpt/r101_dcn_fcos3d_pretrain.pth`: [OccFusion.py:4](../configs/OccFusion.py#L4), [install.md:28](install.md#L28) | Không thay thế được checkpoint toàn bộ OccFusion đã huấn luyện |
| Fixed view-transform params | Chỉ được đọc khi `enable_fix=True`: [multi_scale_inverse_matrixVT.py:238](../occfusion/multi_scale_inverse_matrixVT.py#L238) | Không chặn config mặc định vì [OccFusion.py:78](../configs/OccFusion.py#L78) đặt `False` |

`tools/test.py` nhận checkpoint làm đối số và ghi đè `cfg.load_from`; với checkpoint OccFusion đầy đủ, không cần dùng riêng backbone pretrain để đánh giá: [test.py:122](../tools/test.py#L122). Các chỉ số trong Model Zoo là số tác giả công bố, chưa được tái hiện tại máy này.

**Schema cần xác minh khi lấy PKL:** `images` chứa đường ảnh và `cam2img`, `lidar2cam`, `cam2ego`; `lidar_points` chứa đường LiDAR và `lidar2ego`; có thông tin LiDAR sweeps, `pts_semantic_mask_path`; `radars` chứa danh sách sweeps với `data_path`, `timestamp`, `sensor2lidar_rotation`, `sensor2lidar_translation`. Loader radar đọc cả `lidar2ego` dù cấu hình không dùng Occ3D. Nguồn: [nuscenes_dataset.py:97](../occfusion/nuscenes_dataset.py#L97), [loading.py:159](../occfusion/loading.py#L159), [loading.py:406](../occfusion/loading.py#L406), [loading.py:430](../occfusion/loading.py#L430).

Không mặc định thay custom PKL bằng output của converter nuScenes thông thường trong repo: converter tạo camera/LiDAR sweeps và số đếm `num_radar_pts` của annotation, nhưng không dựng cấu trúc `radars` mà loader này yêu cầu. Nguồn: [nuscenes_converter.py:178](../tools/dataset_converters/nuscenes_converter.py#L178), [nuscenes_converter.py:215](../tools/dataset_converters/nuscenes_converter.py#L215), [nuscenes_converter.py:266](../tools/dataset_converters/nuscenes_converter.py#L266), [loading.py:406](../occfusion/loading.py#L406). Cần kiểm tra schema và đường dẫn thật trước khi chạy.

Với nhãn mặc định, loader đọc `<tên-file-LiDAR>.npy`, chuyển nhãn sparse `0` sang `255`; bước dựng dense GT khởi tạo các ô không được liệt kê bằng `0`. Phải giữ nguyên quy ước này khi đối chiếu GT và prediction: [loading.py:337](../occfusion/loading.py#L337), [main.py:44](../occfusion/main.py#L44). Không trộn bộ nhãn/mask của Occ3D vào baseline này.

## 5. Đường inference hiện phụ thuộc ground truth ở đâu?

**Đã kiểm tra — mã nguồn:**

1. `test_pipeline = val_pipeline`; loader chạy sáu camera, radar tối đa sáu sweeps mỗi radar và LiDAR với cấu hình mười sweeps. Sau lọc vùng, pipeline chạy cả `LoadOccupancy`, `LoadAnnotations3D(with_seg_3d=True)` và `SegLabelMapping`: [OccFusion.py:148](../configs/OccFusion.py#L148).
2. Preprocessor tạo voxel features và chuyển `occ_200` thành `dense_occ_200`: [data_preprocessor.py:40](../occfusion/data_preprocessor.py#L40), [data_preprocessor.py:100](../occfusion/data_preprocessor.py#L100).
3. `_forward()` dùng ảnh, ma trận chiếu và đặc trưng LiDAR/radar; không đọc nhãn GT để tính logits: [main.py:71](../occfusion/main.py#L71).
4. Nhưng `predict()` gọi `_forward()` xong lại đọc `dense_occ_200` để dựng GT và ghi vào `eval_ann_info`. Bỏ nhãn mà giữ nguyên đường này sẽ không hoàn tất prediction: [main.py:163](../occfusion/main.py#L163).
5. `tools/test.py` gọi `runner.test()`, evaluator đọc GT và prediction: [test.py:145](../tools/test.py#L145), [evaluate.py:109](../occfusion/evaluate.py#L109).

**Kết luận:** lõi mạng có thể được bọc thành inference không cần GT, nhưng entry point test/predict hiện tại chưa đáp ứng quy tắc tách inference khỏi evaluation trong đặc tả. Đây là phụ thuộc ở pipeline/wrapper, không phải bằng chứng mạng dùng GT làm đầu vào học khi suy luận.

**Phần cần bổ sung ở mốc sau:** pipeline inference riêng bỏ transform đọc nhãn; `predict()` chỉ trả prediction; chuyển việc dựng GT sang evaluator; giữ metadata định danh/calibration; có phép thử inference trên sample không có GT. Chỉ xóa `LoadOccupancy` khỏi config là chưa đủ.

## 6. Đầu ra model và metadata cho viewer

**Đã kiểm tra — mã nguồn; kích thước dưới đây suy ra từ cấu hình, chưa đo runtime:**

| Đầu ra | Ý nghĩa với config mặc định | Bằng chứng |
| --- | --- | --- |
| `_forward()` ở eval mode | Logits `[B, 200, 200, 16, 17]` | [occhead.py:28](../occfusion/occhead.py#L28), [multi_scale_inverse_matrixVT.py:54](../occfusion/multi_scale_inverse_matrixVT.py#L54), [main.py:174](../occfusion/main.py#L174) |
| `sample.pts_seg_logits.pts_seg_logits` | Logits flatten `[640000, 17]` cho một frame | [main.py:180](../occfusion/main.py#L180), [main.py:189](../occfusion/main.py#L189) |
| `sample.pred_pts_seg.pts_semantic_mask` | Nhãn argmax `[640000]`, giá trị `0..16` | [main.py:187](../occfusion/main.py#L187) |
| GT trong sample | `eval_ann_info['pts_semantic_mask']`, tách biệt với prediction | [main.py:178](../occfusion/main.py#L178) |
| Detection/flow | Đầu ra trên không có detection boxes, velocity/flow hay future occupancy | [main.py:183](../occfusion/main.py#L183), [occhead.py](../occfusion/occhead.py) |

Grid đầu ra là `200×200×16`, khác grid đặc trưng `100×100×8`. Từ bounds `[-50,-50,-5,50,50,3]`, độ phân giải đầu ra suy ra là `0.5 m` mỗi trục. Config mặc định chiếu từ hệ LiDAR (`lidar2img`); nhánh Occ3D dùng `ego2img`: [OccFusion.py:21](../configs/OccFusion.py#L21), [main.py:75](../occfusion/main.py#L75). Cần xác nhận bằng overlay dữ liệu thật trước khi nghiệm thu viewer.

**Đề xuất hợp đồng dữ liệu viewer:**

| Nhóm | Cần giữ/xuất | Hiện trạng và căn cứ |
| --- | --- | --- |
| Định danh/thời gian | dataset/split, scene token, sample token, timestamp keyframe và từng sensor, đơn vị thời gian | Config chỉ pack `lidar2img`: [OccFusion.py:187](../configs/OccFusion.py#L187); chưa có artifact xuất chứa các định danh này |
| Hệ tọa độ | frame LiDAR/ego/global, chiều trục, đơn vị, pose và extrinsic/intrinsic từng camera | Loader tính calibration nhưng nhiều trường không được pack: [loading.py:159](../occfusion/loading.py#L159), [custom_pack.py:199](../occfusion/custom_pack.py#L199) |
| Grid | origin/bounds, shape `[X,Y,Z]`, voxel size, thứ tự flatten, quy ước tâm voxel | Tách metadata này khỏi tensor nhãn; reshape theo `X,Y,Z` của [main.py:180](../occfusion/main.py#L180) |
| Ngữ nghĩa | class map/version, palette, `free=0`, `ignore=255` cho GT, mask đánh giá và nguồn mask | [nuscenes_dataset.py:26](../occfusion/nuscenes_dataset.py#L26), [evaluate.py:58](../occfusion/evaluate.py#L58) |
| Ảnh/point cloud | đường hoặc ID asset, tên/thứ tự camera, kích thước gốc/padding/resize, transform điểm về frame chung | [loading.py:160](../occfusion/loading.py#L160), [data_preprocessor.py:122](../occfusion/data_preprocessor.py#L122) |
| Nguồn kết quả | GT/prediction/replay, commit, config, checkpoint hash, phiên bản môi trường | Chưa được đóng gói thành một output schema bởi [test.py](../tools/test.py) |
| Flow | Chưa có ở baseline; sau này cần vector frame, đơn vị, mốc thời gian/horizon, valid mask | Head hiện chỉ phân loại occupancy: [occhead.py:18](../occfusion/occhead.py#L18) |

Các điểm cần xử lý khi dựng viewer:

- `METAINFO['classes'][0]` đang ghi `IoU` phục vụ bảng metric; không dùng chuỗi này làm tên lớp vật thể. Với occupancy mặc định, `0` là free trong phép tính geometry IoU; `255` là ignore của GT: [nuscenes_dataset.py:28](../occfusion/nuscenes_dataset.py#L28), [evaluate.py:61](../occfusion/evaluate.py#L61).
- Baseline SurroundOcc không xuất một visibility mask riêng. Không tự biến `ignore` hoặc vùng chưa quan sát thành free; ghi rõ mask nào có/không có. Nhánh Occ3D đọc `mask_camera` riêng: [loading.py:310](../occfusion/loading.py#L310).
- Loader đặt `scale_factor=0.2` trong khi đoạn resize 0.2 đang comment. Không dùng giá trị này để dựng overlay mà chưa đối chiếu ảnh/calibration thực tế: [loading.py:189](../occfusion/loading.py#L189), [loading.py:218](../occfusion/loading.py#L218).
- Radar có `time_diff` theo sweep đầu của từng radar, không phải metadata đồng bộ keyframe đầy đủ: [loading.py:416](../occfusion/loading.py#L416). Gộp sweeps không tạo ra head occupancy flow hay recurrent temporal fusion.
- Có thể dựng BEV từ occupancy theo một quy tắc chiếu xuống mặt phẳng được công bố. Repo chưa trả ảnh BEV hoặc artifact viewer hoàn chỉnh; `--show` của test CLI chỉ liệt kê các task detection/point segmentation: [test.py:34](../tools/test.py#L34). Muốn detection mAP phải có nguồn prediction boxes và GT boxes tương ứng; backbone pretrain tên FCOS3D không biến head occupancy thành detector.

## 7. Cảm biến rỗng hoặc thiếu dữ liệu

**Đã kiểm tra — mã nguồn; các lỗi dưới đây là suy luận theo nhánh code, chưa chạy fault injection:**

| Tình huống | Hành vi/rủi ro hiện tại | Bằng chứng |
| --- | --- | --- |
| Thiếu key `radars` | Truy cập key trực tiếp, không có fallback | [loading.py:406](../occfusion/loading.py#L406) |
| Một radar có danh sách sweeps rỗng | Truy cập `sweeps[0]` trước khi đọc điểm | [loading.py:416](../occfusion/loading.py#L416) |
| `radars={}` | `np.concatenate()` trên danh sách rỗng | [loading.py:460](../occfusion/loading.py#L460) |
| Radar không còn điểm sau lọc bounds | Danh sách voxel feature rỗng rồi `torch.stack()` | [loading.py:470](../occfusion/loading.py#L470), [data_preprocessor.py:83](../occfusion/data_preprocessor.py#L83) |
| LiDAR không còn điểm sau lọc | Cùng rủi ro `torch.stack()` trên danh sách rỗng | [OccFusion.py:176](../configs/OccFusion.py#L176), [data_preprocessor.py:55](../occfusion/data_preprocessor.py#L55) |
| Thiếu LiDAR sweeps lịch sử | Config bật `pad_empty_sweeps=True` trong transform MMDetection3D | [OccFusion.py:169](../configs/OccFusion.py#L169); chưa kiểm tra implementation của phiên bản sẽ cài, không suy rộng thành hỗ trợ mất cảm biến |
| Thiếu/hỏng file ảnh, radar hoặc GT | Loader đọc file trực tiếp; không có cơ chế báo lỗi có cấu trúc cho ứng dụng | [loading.py:181](../occfusion/loading.py#L181), [loading.py:340](../occfusion/loading.py#L340), [loading.py:385](../occfusion/loading.py#L385) |
| Thiếu một camera trong metadata | Loader lặp qua dictionary; chưa thấy bước xác thực đủ tên và thứ tự sáu camera | [loading.py:160](../occfusion/loading.py#L160), [OccFusion.py:153](../configs/OccFusion.py#L153) |
| Mất modality trong lúc chạy | `_forward()` vẫn yêu cầu feature của modality đang bật | [main.py:87](../occfusion/main.py#L87) |

Cờ `use_lidar/use_radar` là lựa chọn kiến trúc/cấu hình, chưa phải cơ chế tự suy giảm khi cảm biến hỏng. Pipeline mặc định vẫn có cả loader radar và LiDAR, nên chỉ đổi cờ model không loại bỏ được yêu cầu file đầu vào: [OccFusion.py:148](../configs/OccFusion.py#L148).

**Đề xuất:** ở bước đầu, kiểm tra sample trước inference và báo lỗi rõ sensor/file/record bị thiếu. Nếu sau này hỗ trợ fallback, phải định nghĩa modality mask, hành vi empty tensor và checkpoint phù hợp, rồi đánh giá riêng. Không âm thầm thay cảm biến thiếu bằng dữ liệu giả.

## 8. Phép thử nhỏ nhất: một keyframe thật, có GT

Mục tiêu là chứng minh checkpoint đã huấn luyện chạy hết đường dữ liệu → preprocessing → model → evaluator hiện có. Đây chưa phải inference không GT và chưa phải benchmark chất lượng toàn tập.

### 8.1. Điều kiện trước khi chạy

1. Có môi trường Linux GPU tương thích theo mục 3; ghi lại phiên bản và kiểm tra `torch.cuda.is_available()`, import `occfusion`, import CUDA ops cần thiết.
2. Có checkpoint C+L+R từ Model Zoo, lưu hash và kiểm tra key/shape khi load. `ckpt/occfusion_cam_lidar_radar.pth` trong lệnh dưới là **tên lưu local đề xuất**, chưa phải file đang có.
3. Có custom validation PKL và chọn một record hợp lệ. Nếu dùng index `0` như ví dụ, phải kiểm tra chính record `0`; không giả định frame đầu đầy đủ.
4. Chuẩn bị toàn bộ asset mà record đó tham chiếu: sáu ảnh, LiDAR keyframe/các sweeps được dùng, radar sweeps, occupancy `.npy`, lidarseg GT và calibration. Một keyframe inference vẫn có thể cần nhiều file sweeps lịch sử. Không cần train split để thử đường test này.
5. Kiểm tra đường dẫn PKL sau khi ghép `data_root/data_prefix`, đặc biệt `radars[*].data_path` được đọc trực tiếp. Kiểm tra occupancy indices trong bounds, nhãn hợp lệ và point cloud còn điểm sau lọc. Nguồn: [nuscenes_dataset.py:112](../occfusion/nuscenes_dataset.py#L112), [loading.py:421](../occfusion/loading.py#L421), [main.py:44](../occfusion/main.py#L44).
6. Xác nhận MMEngine được cài hỗ trợ `BaseDataset.__init__(..., indices=...)`. Dataset custom kế thừa `BaseDataset` và truyền `**kwargs`, nhưng thư viện chưa có trên máy để xác minh runtime: [nuscenes_dataset.py:5](../occfusion/nuscenes_dataset.py#L5), [nuscenes_dataset.py:62](../occfusion/nuscenes_dataset.py#L62). Khi build dataset, phải kiểm tra `len(dataset) == 1`.

### 8.2. Lệnh đề xuất sau khi đủ điều kiện

Chạy bằng **Bash trong Linux/WSL/container đã xác minh GPU**, tại thư mục gốc repo. Không dán nguyên cú pháp này vào PowerShell. Chưa thực thi lệnh trong đợt khảo sát này.

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python tools/test.py \
  configs/OccFusion.py \
  ckpt/occfusion_cam_lidar_radar.pth \
  --launcher none \
  --work-dir work_dirs/baseline_one_frame \
  --cfg-options \
  test_dataloader.batch_size=1 \
  test_dataloader.num_workers=0 \
  test_dataloader.persistent_workers=False \
  'test_dataloader.dataset.indices=[0]'
```

Căn cứ lệnh: [test.py:17](../tools/test.py#L17), [test.py:44](../tools/test.py#L44), [test.py:54](../tools/test.py#L54), [OccFusion.py:210](../configs/OccFusion.py#L210). Override loader test thực tế, giảm batch từ 8 xuống 1 và tắt persistent workers khi workers bằng 0. Hướng dẫn tác giả dùng 8 GPU là ví dụ distributed, không phải bằng chứng một frame bắt buộc cần 8 GPU: [getting_started.md:8](getting_started.md#L8).

### 8.3. Tiêu chí xác nhận thành công và bằng chứng cần lưu

- Nạp đúng checkpoint; không bỏ qua warning thiếu key/shape của các nhánh backbone, fusion hoặc occupancy head. Lưu commit, config sau override, hash checkpoint, token/index sample và phiên bản nhãn.
- Dataset thực sự chỉ có một record; log cho thấy test kết thúc bình thường, không lỗi file, shape, CUDA op hoặc OOM.
- Khi kiểm tra tensor bằng debugger/hook: logits hữu hạn, shape `[1,200,200,16,17]`; nhãn reshape được về `[200,200,16]` và thuộc `0..16`. Đó là tiêu chí dự kiến từ mục 6, chưa phải output đã quan sát.
- Có output/log evaluator IoU/mIoU của đúng frame. Chỉ ghi là smoke test một frame; lớp vắng mặt có thể bị `nan_to_num` về 0 nên không dùng kết quả này so sánh Model Zoo: [evaluate.py:74](../occfusion/evaluate.py#L74).
- Muốn lưu tensor cho viewer phải bổ sung exporter/hook ở nhiệm vụ sau; test CLI hiện chưa tự tạo artifact occupancy kèm metadata theo đặc tả: [test.py](../tools/test.py), [main.py:183](../occfusion/main.py#L183).

Nếu OOM ở batch 1, lưu traceback/peak VRAM và đánh dấu chưa đạt trên GPU 4 GB. Bước tái hiện tiếp theo có thể dùng GPU đủ bộ nhớ. Không tự giảm grid, số camera hoặc đổi nhãn rồi vẫn ghi là baseline gốc; shape GT đang được hard-code tại [main.py:174](../occfusion/main.py#L174). Chưa có căn cứ đo để cam kết một mức VRAM tối thiểu cụ thể.

## 9. Metric và latency: cái gì đã có, cái gì chưa được chứng minh?

**Đã kiểm tra — mã nguồn:** `EvalMetric` có geometry IoU occupied/free và mIoU các lớp semantic, bỏ GT `ignore_index`. Không có detection mAP hoặc flow ADE/FDE trong evaluator được config mặc định chọn: [OccFusion.py:226](../configs/OccFusion.py#L226), [evaluate.py:46](../occfusion/evaluate.py#L46).

Script [benchmark.py](../tools/analysis_tools/benchmark.py) chưa nên dùng nguyên trạng làm bằng chứng latency cho đặc tả:

- Đếm iteration làm sample trong công thức FPS, trong khi config test mặc định batch 8: [benchmark.py:75](../tools/analysis_tools/benchmark.py#L75), [OccFusion.py:211](../configs/OccFusion.py#L211).
- Thời gian iteration cuối bị cộng thêm lần nữa khi đạt `args.samples`: [benchmark.py:73](../tools/analysis_tools/benchmark.py#L73), [benchmark.py:80](../tools/analysis_tools/benchmark.py#L80).
- Đo `model.test_step(data)` sau khi dataloader đã yield batch, nên khoảng đo bao gồm preprocessing/predict nhưng không bao trùm đọc dữ liệu trong dataloader; chưa phải tách model-only và toàn pipeline: [benchmark.py:61](../tools/analysis_tools/benchmark.py#L61).

**Đề xuất mốc đo sau:** sửa cách đếm/cộng thời gian; warm-up và CUDA synchronization; đo riêng model và end-to-end; báo batch size, precision, input shape, phần có/không có I/O, số lượt, p50/p95 và peak VRAM. Một lượt startup không đủ làm benchmark latency ổn định. Hiện không có IoU/mIoU/latency/VRAM inference nào đã đo trên checkout này.

## 10. Phần bị chặn và thứ tự công việc tiếp theo

| Mức | Việc cần hoàn tất | Bằng chứng/điều kiện gỡ chặn |
| --- | --- | --- |
| Chặn baseline hiện tại | Chuẩn bị một môi trường PyTorch/OpenMMLab tương thích và khóa phiên bản | Mục 2–3; import/CUDA ops và log môi trường thực tế |
| Chặn baseline hiện tại | Lấy checkpoint C+L+R và kiểm tra tương thích config | Mục 4; file, hash, báo cáo load state dict |
| Chặn baseline hiện tại | Lấy PKL, một record thật và toàn bộ sensor/GT assets cần thiết | Mục 4, 8; kiểm tra schema/path/sample trước chạy |
| Rủi ro chưa đo | GPU 4 GB có thể không đủ cho model gốc | Mục 2, 8; kết quả smoke test và peak VRAM/traceback |
| Sau khi baseline chạy | Tách inference không GT, bổ sung metadata/exporter | Mục 5–6; inference chạy được khi không có file nhãn |
| Sau đó | Xử lý input thiếu/rỗng và đo latency đúng giao thức | Mục 7, 9; các ca kiểm tra thật và số đo có nguồn gốc |

Lệnh đọc/đối chiếu cấu hình đã chạy không thay thế bằng chứng inference. Đợt này chỉ tạo đặc tả và báo cáo; chưa triển khai worker, viewer, detection, temporal flow hoặc các tính năng lab khác trong [PROJECT_SPEC.md](../PROJECT_SPEC.md). Nhiệm vụ lập trình tiếp theo nên giới hạn ở một môi trường đã khóa và smoke test một frame theo các điều kiện trên.
