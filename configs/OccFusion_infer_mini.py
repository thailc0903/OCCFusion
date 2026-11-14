"""GT-free, one-frame nuScenes mini inference with the C+L+R baseline.

Run with tools/infer_nuscenes_mini.py, not tools/test.py / Runner.test().
The model, image resolution, normalization and grids are inherited unchanged.
"""

_base_ = ['OccFusion.py']

load_from = 'ckpt/Cam_Lidar_Radar_Atten3D.pth'
dataset_version = 'v1.0-mini'

# Same insertion order as tools/dataset_converters/nuscenes_converter.py.
camera_order = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]
radar_order = [
    'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
    'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT',
]

lidar_loader = _base_.val_pipeline[2]
lidar_sweep_loader = dict(
    sweeps_num=10, load_dim=5, use_dim=5,
    remove_close=True, pad_empty_sweeps=False, test_mode=True,
)
radar_loader = _base_.val_pipeline[1]

# Point loading happens in the CLI before these transforms. Its LiDAR sweep
# adapter applies the full SE(3) transform, independently of the installed
# MMDetection3D version's inverse-translation convention.
inference_pipeline = [
    _base_.val_pipeline[0],
    _base_.val_pipeline[4],
    dict(
        type='Custom3DPack',
        keys=['img', 'points', 'radars'],
        meta_keys=[
            'lidar2img', 'img_path', 'ori_shape', 'img_shape', 'pad_shape',
            'cam2img', 'lidar2cam', 'num_views', 'sample_idx',
        ],
    ),
]

# This entry point creates metadata directly with nuscenes-devkit. It never
# constructs the training/evaluation datasets or reads their PKL/GT files.
train_pipeline = None
val_pipeline = None
test_pipeline = inference_pipeline
train_dataloader = None
val_dataloader = None
test_dataloader = None
val_evaluator = None
test_evaluator = None
train_cfg = None
val_cfg = None
test_cfg = None
