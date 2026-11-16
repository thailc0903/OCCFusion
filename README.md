# RAV25 Occupancy Lab

Lab offline trên WSL để phát lại dữ liệu nuScenes, xem occupancy/BEV/LiDAR,
box CenterPoint và sáu camera, kiểm tra metric đã lưu, rồi review/export
snapshot theo vai trò Engineer/Reviewer. Dự án không điều khiển xe.
Chưa triển khai temporal fusion hoặc learned 4D flow.

Bản bàn giao, bảng nghiệm thu và giới hạn: [BASIC_ACCEPTANCE.md](BASIC_ACCEPTANCE.md).
Hướng dẫn vận hành đầy đủ: [WSL_OPERATIONS.md](docs/WSL_OPERATIONS.md).

## Quickstart WSL

Dùng đúng đường dẫn của deployment hiện có. Trên máy này alias viết thường
trỏ cùng thư mục nhưng làm đổi identity đường dẫn trong profile/GT provenance;
giữ cách viết đã lưu bên dưới. Không sửa job cũ để đổi fingerprint.

~~~bash
cd '/mnt/d/New folder/OCCFusion'
export PATH="$PWD/.cache/toolchains/node/bin:$PATH"
source web/viewer.env.example

# Chỉ đọc metadata/dependency/build/queue; không tạo user hoặc chạy model.
.venv-web/bin/python -B tools/check_web_readiness.py --mode view

# Chỉ khi cần build/cài môi trường; đừng build đè .next khi Next đang phục vụ.
# python3 -m venv .venv-web
# .venv-web/bin/python -m pip install -r web/backend/requirements.lock
# npm --prefix web/frontend ci
# VIEWER_API_ORIGIN=http://127.0.0.1:8000 VIEWER_NEXT_DIST_DIR=.next \
#   NEXT_TELEMETRY_DISABLED=1 npm --prefix web/frontend run build

# Xem artifact, không nhận job GPU:
VIEWER_START_WORKER=0 bash tools/run_web_viewer.sh
~~~

Mở **http://localhost:3000**; các trang **/benchmarks**, **/detections**, **/jobs**.
Đăng nhập bằng tài khoản hiện có; không có tài khoản/mật khẩu mặc định và
startup không seed account. Worker tắt thì cả job mới và job đang queued tiếp
tục chờ. Bật worker là thao tác chủ động có thể nhận ngay job trong queue:

~~~bash
# Cùng terminal/environment/state/root/profile như web, sau kiểm tra queue:
VIEWER_START_WORKER=1 bash tools/run_web_viewer.sh
~~~

Không chạy cả hai launcher cùng lúc. Nếu tách worker sang terminal khác,
source cùng web/viewer.env.example và giữ cùng GPU lock; xem hướng dẫn WSL.
Readiness/startup không tải model, dựng model, tạo job hoặc approve dữ liệu.

## Dữ liệu, model và môi trường

- Dataset đã cấp quyền tại data/nuscenes: v1.0-mini JSON, samples và sweeps.
- Artifact đã lưu tại work_dirs, gồm benchmark_mini_val và
  benchmark_detection_mini_val. Playback không chạy lại inference/evaluator.
- OCCFusion C+L+R: ckpt/Cam_Lidar_Radar_Atten3D.pth; CenterPoint opt-in:
  ckpt/centerpoint_pillar02_v140. Nguồn/SHA và cặp config/checkpoint được ghi
  trong tài liệu tương ứng. Không có weights/dataset trong Git.
- Web: Python .venv-web với requirements.lock, Node >=24 <27 với package-lock.
  Model dùng /home/hatis/.venvs/rav25/bin/python riêng; không trộn dependencies.
- SurroundOcc GT là dữ liệu riêng cho evaluation, không phải input inference.
  Không có GT thì metric không khả dụng, không thay bằng 0.

## Kết quả dự án đã lưu, không phải số model zoo

| Task / phạm vi | Coverage | Kết quả lưu |
| --- | --- | --- |
| OCCFusion mini_val, pooled 16 lớp | Partial 79/81 | Occupancy IoU 44,1789%; semantic mIoU 23,0672% |
| CenterPoint LiDAR-only mini_val | Complete 81/81 | mAP 41,2801%; NDS 47,7591% |

Hai scene mini_val không đại diện toàn nuScenes validation. Hai occupancy
frame đầu scene thiếu radar quá khứ hợp lệ; không dùng sweep tương lai để bù.
Loader detection có sai khác SE(3) tối đa 0,428603971 m đã công bố. Thời gian
lấy từ lượt GPU gốc, bỏ warm-up đúng scope; allocator không phải VRAM vật lý.
Chỉ sáu ảnh scene-0103/frame 1 đã có kiểm chứng privacy ở QA, chưa phải kiểm
tra PII toàn dataset. Camera source registration trong production không mang
QA approval hoặc tự đánh dấu Engineer đã kiểm tra.

## Tài liệu

- [Nghiệm thu](BASIC_ACCEPTANCE.md), [Vận hành WSL](docs/WSL_OPERATIONS.md).
- [Inference OCCFusion](docs/INFERENCE_MINI.md), [web jobs](docs/WEB_INFERENCE_JOBS.md).
- [Occupancy evaluation](docs/OCCUPANCY_EVALUATION.md), [benchmark](docs/BENCHMARK_MINI_VAL.md).
- [CenterPoint](docs/CENTERPOINT_DETECTION.md), [detection benchmark](docs/DETECTION_BENCHMARK.md).
- [Benchmark/collection viewer](docs/WEB_DETECTION_BENCHMARKS.md).
- [Camera/projection/privacy](docs/CAMERA_VIEWER.md).
- [Auth/review](docs/AUTH_REVIEW.md), [CSRF fix](docs/CSRF_BOOTSTRAP_FIX.md).

## Nguồn upstream OCCFusion

Phần dưới giữ nội dung giới thiệu, kiến trúc, nguồn, acknowledgements và
citation của OCCFusion. Các hình benchmark và bảng Model Zoo là số báo cáo
của tác giả upstream trên phạm vi của họ, không phải kết quả mini_val RAV25
và không dùng để tuyên bố chất lượng của bản bàn giao này.
Dockerfile upstream được giữ để tham khảo; chưa build/test stack web này
bằng Docker. Bản được nghiệm thu ở đây là WSL localhost.

---

# OccFusion
OccFusion: Multi-Sensor Fusion Framework for 3D Semantic Occupancy Prediction [[Paper](https://ieeexplore.ieee.org/document/10663967)]
## Abstract
A comprehensive understanding of 3D scenes is crucial in autonomous vehicles (AVs), and recent models for 3D semantic occupancy prediction have successfully addressed the challenge of describing real-world objects with varied shapes and classes. However, existing methods for 3D semantic occupancy prediction heavily rely on surround-view camera images, making them susceptible to changes in lighting and weather conditions. This paper introduces OccFusion, a novel sensor fusion framework for predicting 3D semantic occupancy. By integrating features from additional sensors, such as lidar and surround view radars, our framework enhances the accuracy and robustness of occupancy prediction, resulting in top-tier performance on the nuScenes benchmark. Furthermore, extensive experiments conducted on the nuScenes and semanticKITTI dataset, including challenging night and rainy scenarios, confirm the superior performance of our sensor fusion strategy across various perception ranges.
## Overall Architecture
<p align='center'>
<img src="./figs/overrall.jpg" width="1200px">
</p>

## Benchmark
### Nuscenes val set
<p align='center'>
<img src="./figs/nuscene_whole.png" width="1200px">
</p>

### Nuscenes val rainy subset
<p align='center'>
<img src="./figs/nuscene_rainy.png" width="1200px">
</p>

### Nuscenes val night subset
<p align='center'>
<img src="./figs/nuscene_night.png" width="1200px">
</p>

### Visualization Result
<p align='center'>
<img src="./figs/vis.png" width="1200px">
</p>

## Model Zoo
| Input Modality | IoU | mIoU | Model Weights |
| :---: | :---: | :---: | :---: | 
| C + R | 32.90 | 20.73 | [ckpt](https://drive.google.com/file/d/1bKGOVez7teuCh91rrMnSYaXVXXm6i18R/view?usp=sharing)
| C + L | 43.53 | 27.55 | [ckpt](https://drive.google.com/file/d/1G4nynOPTzFdNIahBJtPpCTF3AgjmEUHo/view?usp=sharing)
| C + L + R | 43.96 | 28.27 | [ckpt](https://drive.google.com/file/d/1f5KT-w1VZIwvOb-mMGeqplmy7EtZDcvM/view?usp=sharing)
## Getting Started
- [Installation](docs/install.md)
- [Dataset](docs/dataset.md)
- [Run and Eval](docs/getting_started.md)
## Acknowledgement
Many thanks to these excellent projects:
- [Co-Occ](https://github.com/Rorisis/Co-Occ)
- [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy)
- [SurroundOcc](https://github.com/weiyithu/SurroundOcc)
## Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@article{ming2024occfusion,
  title={OccFusion: Multi-Sensor Fusion Framework for 3D Semantic Occupancy Prediction},
  author={Ming, Zhenxing and Berrio, Julie Stephany and Shan, Mao and Worrall, Stewart},
  journal={IEEE Transactions on Intelligent Vehicles},
  year={2024},
  publisher={IEEE}
}

```
