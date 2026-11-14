#!/usr/bin/env python3
"""Infer one real nuScenes mini frame, without occupancy or lidarseg GT."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# nuScenes imports matplotlib even though the exporter below uses Pillow.
os.environ.setdefault('MPLCONFIGDIR', '/tmp/occfusion-matplotlib')

TABLES = (
    'attribute', 'calibrated_sensor', 'category', 'ego_pose', 'instance',
    'log', 'map', 'sample', 'sample_annotation', 'sample_data', 'scene',
    'sensor', 'visibility',
)
ARTIFACTS = ('occupancy.npz', 'bev.png', 'metadata.json', 'report.json',
             'resolved_config.py')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/OccFusion_infer_mini.py')
    parser.add_argument('--data-root', default=None,
                        help='Dataset root; default: data_root in config.')
    parser.add_argument('--checkpoint', default=None,
                        help='Full OccFusion checkpoint; default: config load_from.')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--scene-index', type=int, default=None,
                           help='Zero-based index in nuScenes.scene JSON order; default 0.')
    selection.add_argument('--scene-name', help='Exact nuScenes scene name, e.g. scene-0103.')
    parser.add_argument('--frame-index', type=int, default=0,
                        help='Zero-based sample index along scene.next links.')
    parser.add_argument('--out-dir', required=True,
                        help='Use a fresh output directory for each attempt.')
    parser.add_argument('--device', default='cuda:0', help='cuda:N or cpu.')
    parser.add_argument('--check-only', action='store_true',
                        help='Validate/load/preprocess on CPU; never run forward.')
    return parser.parse_args(argv)


def repo_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def code_version():
    def git(*args):
        result = subprocess.run(['git', '-C', str(ROOT), *args],
                                capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return dict(commit=git('rev-parse', 'HEAD'),
                dirty=bool(git('status', '--porcelain')),
                inference_script_sha256=file_hash(Path(__file__)))


def validate_paths(checkpoint, data_root, version):
    paths = [checkpoint] + [data_root / version / f'{name}.json'
                            for name in TABLES]
    missing = [str(path) for path in paths if not path.is_file()]
    empty = [str(path) for path in paths
             if path.is_file() and path.stat().st_size == 0]
    if missing or empty:
        raise FileNotFoundError(f'Missing input files: {missing}; empty files: {empty}')


def resolve_scene_index(nusc, scene_index=None, scene_name=None):
    if scene_name is not None:
        if scene_index is not None:
            raise ValueError('Use either --scene-name or --scene-index')
        matches = [i for i, scene in enumerate(nusc.scene) if scene['name'] == scene_name]
        if len(matches) != 1:
            raise ValueError(f'Expected one scene named {scene_name!r}, found {len(matches)}')
        return matches[0]
    return 0 if scene_index is None else scene_index


def select_sample(nusc, scene_index, frame_index):
    if not 0 <= scene_index < len(nusc.scene):
        raise IndexError(f'--scene-index {scene_index}: valid range 0..{len(nusc.scene)-1}')
    scene = nusc.scene[scene_index]
    if not 0 <= frame_index < scene['nbr_samples']:
        raise IndexError(f'--frame-index {frame_index}: scene {scene["name"]} '
                         f'has {scene["nbr_samples"]} frames (zero-based)')
    sample = nusc.get('sample', scene['first_sample_token'])
    for _ in range(frame_index):
        sample = nusc.get('sample', sample['next'])
    if sample['scene_token'] != scene['token']:
        raise ValueError('Sample.next crossed into a different scene')
    return scene, sample


def make_frame_info(nusc, sample, cfg, warnings):
    """Construct sensor metadata, using each record's own calibration/pose.

    All matrices here act on COLUMN vectors. Points in the loaders use
    xyz @ rotation.T + translation. No annotation/box API is called.
    """
    import numpy as np
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion

    channels = list(cfg.camera_order) + ['LIDAR_TOP'] + list(cfg.radar_order)
    missing_channels = [name for name in channels if name not in sample['data']]
    if missing_channels:
        raise ValueError(f'Sample {sample["token"]} missing sensors: {missing_channels}')

    def pose(record):
        return transform_matrix(record['translation'], Quaternion(record['rotation']))

    ref_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ref_cal = nusc.get('calibrated_sensor', ref_sd['calibrated_sensor_token'])
    ref_pose = nusc.get('ego_pose', ref_sd['ego_pose_token'])
    lidar_to_ego = pose(ref_cal)
    lidar_to_global = pose(ref_pose) @ lidar_to_ego
    global_to_lidar = np.linalg.inv(lidar_to_global)
    reference_us = ref_sd['timestamp']
    checked_files = {}

    def sensor_info(token, channel):
        sd = nusc.get('sample_data', token)
        cal = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        sensor = nusc.get('sensor', cal['sensor_token'])
        if sensor['channel'] != channel:
            raise ValueError(f'{token}: expected {channel}, got {sensor["channel"]}')
        if nusc.get('sample', sd['sample_token'])['scene_token'] != sample['scene_token']:
            raise ValueError(f'{channel}: sweep {token} crosses a scene boundary')
        ego = nusc.get('ego_pose', sd['ego_pose_token'])
        sensor_to_ego = pose(cal)
        ego_to_global = pose(ego)
        sensor_to_lidar = global_to_lidar @ ego_to_global @ sensor_to_ego
        path = Path(nusc.get_sample_data_path(token)).resolve()
        info = dict(
            channel=channel, sample_data_token=token, path=str(path),
            timestamp_us=int(sd['timestamp']),
            age_to_reference_s=(reference_us - sd['timestamp']) * 1e-6,
            calibrated_sensor_token=cal['token'], ego_pose_token=ego['token'],
            sensor_to_ego=sensor_to_ego.tolist(),
            ego_to_global=ego_to_global.tolist(),
            sensor_to_reference_lidar=sensor_to_lidar.tolist(),
        )
        if sensor['modality'] == 'camera':
            info.update(intrinsic=cal['camera_intrinsic'],
                        width=sd['width'], height=sd['height'])
        checked_files[token] = info
        return info

    keyframes = {name: sensor_info(sample['data'][name], name) for name in channels}

    def collect_past(start, count, channel):
        records, skipped, visited = [], [], set()
        token = start
        previous_time = None
        while token and len(records) < count:
            if token in visited:
                raise ValueError(f'{channel}: cycle in sample_data.prev at {token}')
            visited.add(token)
            sd = nusc.get('sample_data', token)
            if previous_time is not None and sd['timestamp'] >= previous_time:
                raise ValueError(f'{channel}: prev timestamps are not strictly decreasing')
            previous_time = sd['timestamp']
            if sd['timestamp'] <= reference_us:
                records.append(sensor_info(token, channel))
            else:
                skipped.append(dict(sample_data_token=token, timestamp_us=sd['timestamp']))
            token = sd['prev']
        if len(records) < count:
            warnings.append(f'{channel}: only {len(records)}/{count} past sweeps; '
                            'using available records without padding/repetition')
        return records, skipped

    lidar_sweeps, _ = collect_past(ref_sd['prev'], cfg.lidar_sweep_loader.sweeps_num,
                                 'LIDAR_TOP')
    radars, skipped_future = {}, {}
    for name in cfg.radar_order:
        radars[name], skipped_future[name] = collect_past(
            sample['data'][name], cfg.radar_loader.sweeps_num, name)

    missing_files = [f'{r["channel"]} {r["sample_data_token"]}: {r["path"]}'
                     for r in checked_files.values() if not Path(r['path']).is_file()]
    empty_files = [f'{r["channel"]}: {r["path"]}' for r in checked_files.values()
                   if Path(r['path']).is_file() and Path(r['path']).stat().st_size == 0]
    if missing_files or empty_files:
        raise FileNotFoundError('Missing sensor files:\n' + '\n'.join(missing_files)
                                + '\nEmpty sensor files:\n' + '\n'.join(empty_files))
    empty_channels = [name for name, records in radars.items() if not records]
    if empty_channels:
        raise ValueError(f'No radar sweep at or before LiDAR timestamp {reference_us} '
                         f'for {empty_channels}; choose another --frame-index. '
                         'Future sweeps and replacement data are not allowed.')

    images = {}
    for name in cfg.camera_order:
        record = keyframes[name]
        images[name] = dict(
            img_path=record['path'], cam2img=record['intrinsic'],
            cam2ego=record['sensor_to_ego'],
            lidar2cam=np.linalg.inv(record['sensor_to_reference_lidar']).tolist(),
        )
    radar_input = {}
    for name, records in radars.items():
        radar_input[name] = [dict(
            data_path=r['path'], timestamp=r['timestamp_us'],
            sensor2lidar_rotation=np.asarray(r['sensor_to_reference_lidar'])[:3, :3],
            sensor2lidar_translation=np.asarray(r['sensor_to_reference_lidar'])[:3, 3],
        ) for r in records]
    results = dict(
        token=sample['token'], sample_idx=sample['token'],
        timestamp=reference_us * 1e-6, images=images, radars=radar_input,
        lidar_points=dict(lidar_path=keyframes['LIDAR_TOP']['path'],
                          lidar2ego=lidar_to_ego.tolist()),
    )
    metadata = dict(
        sample_token=sample['token'], timestamp_us=int(sample['timestamp']),
        reference_timestamp_us=int(reference_us), timestamp_unit='microseconds',
        coordinate_system=dict(
            name='LIDAR_TOP at reference_timestamp_us', unit='metres',
            axes='native right-handed LIDAR_TOP xyz; calibration is authoritative',
            ego_axes='x forward, y left, z up',
            matrix_convention='column vectors; target_from_source @ [x,y,z,1]',
            lidar_to_ego=lidar_to_ego.tolist(),
            lidar_to_global=lidar_to_global.tolist(),
        ),
        sensors=dict(
            camera_order=list(cfg.camera_order), radar_order=list(cfg.radar_order),
            keyframes=keyframes, lidar_past_sweeps=lidar_sweeps,
            radar_sweeps=radars, excluded_future_radar_records=skipped_future,
            checked_file_count=len(checked_files),
            requested_lidar_past_sweeps=cfg.lidar_sweep_loader.sweeps_num,
            requested_radar_sweeps_per_sensor=cfg.radar_loader.sweeps_num,
            sweep_policy='prev links only, timestamp <= reference, no repetition',
            lidar_features=['x', 'y', 'z', 'intensity', 'reference_age_s'],
            radar_features=['x', 'y', 'z', 'vx_comp', 'vy_comp', 'channel_age_s'],
            radar_time_feature='age relative to latest USED sweep of each radar, '
                               'as in baseline LoadRadarPointsMultiSweeps; '
                               'age_to_reference_s is recorded separately',
            radar_velocity='ego-compensated velocity, rotated into reference LiDAR; '
                           'no object-motion extrapolation',
        ),
    )
    return results, lidar_sweeps, metadata


def load_points(results, lidar_sweeps, cfg, warnings):
    """Reuse file readers and the repo radar loader; correct LiDAR sweep SE(3)."""
    import numpy as np
    from mmdet3d.registry import TRANSFORMS
    from occfusion.loading import LoadRadarPointsMultiSweeps

    lidar_reader = TRANSFORMS.build(cfg.lidar_loader)
    counts = dict(lidar=[], radar=[])

    def read_lidar(path):
        record = lidar_reader(dict(lidar_points=dict(lidar_path=path)))
        points = record['points']
        if not np.isfinite(points.tensor.numpy()).all():
            raise ValueError(f'Non-finite LiDAR points: {path}')
        counts['lidar'].append(dict(path=path, points=len(points)))
        if len(points) == 0:
            warnings.append(f'Empty LiDAR point cloud: {path}')
        return points

    key_points = read_lidar(results['lidar_points']['lidar_path'])
    key_points.tensor[:, 4] = 0  # time lag replaces nuScenes ring index.
    all_points = [key_points]
    for sweep in lidar_sweeps:
        pts = read_lidar(sweep['path']).tensor.numpy().copy()
        # Baseline removes the 1 m square near the sensor in sweep coordinates.
        if cfg.lidar_sweep_loader.remove_close:
            pts = pts[~((np.abs(pts[:, 0]) < 1) & (np.abs(pts[:, 1]) < 1))]
        transform = np.asarray(sweep['sensor_to_reference_lidar'])
        pts[:, :3] = pts[:, :3] @ transform[:3, :3].T + transform[:3, 3]
        pts[:, 4] = sweep['age_to_reference_s']
        all_points.append(key_points.new_point(pts))
    results['points'] = key_points.cat(all_points)

    class CheckedRadarLoader(LoadRadarPointsMultiSweeps):
        def _load_points(self, filename):
            pts = super()._load_points(filename)
            if not np.isfinite(pts).all():
                raise ValueError(f'Non-finite radar points: {filename}')
            counts['radar'].append(dict(path=filename, points_after_devkit_filters=len(pts)))
            if len(pts) == 0:
                warnings.append(f'Empty radar sweep after devkit filters: {filename}')
            return pts

    radar_cfg = dict(cfg.radar_loader)
    radar_cfg.pop('type')
    results = CheckedRadarLoader(**radar_cfg)(results)
    return results, counts


def strict_load_checkpoint(model, path, report):
    """Compare every persistent tensor BEFORE loading or running any forward."""
    import torch

    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state, dict) or not state:
        raise TypeError('Checkpoint must contain a nonempty state_dict')
    prefix = ''
    if all(key.startswith('module.') for key in state):
        prefix = 'module.'
        state = {key[len(prefix):]: value for key, value in state.items()}
    expected = model.state_dict()
    missing = sorted(expected.keys() - state.keys())
    unexpected = sorted(state.keys() - expected.keys())
    mismatched, invalid = [], []
    for key in sorted(expected.keys() & state.keys()):
        if not isinstance(state[key], torch.Tensor):
            invalid.append(key)
        elif expected[key].shape != state[key].shape:
            mismatched.append(dict(key=key, model_shape=list(expected[key].shape),
                                   checkpoint_shape=list(state[key].shape)))
    comparison = dict(
        missing_keys=missing, unexpected_keys=unexpected,
        shape_mismatches=mismatched, non_tensor_values=invalid,
        removed_prefix=prefix, model_state_tensor_count=len(expected),
        checkpoint_tensor_count=len(state),
        model_parameter_count=sum(p.numel() for p in model.parameters()),
        checkpoint_dtypes=dict(Counter(str(v.dtype) for v in state.values()
                                      if isinstance(v, torch.Tensor))),
        compatible=not (missing or unexpected or mismatched or invalid),
    )
    report['checkpoint_comparison'] = comparison
    print('Checkpoint comparison:', json.dumps(comparison, ensure_ascii=False), flush=True)
    if not comparison['compatible']:
        raise RuntimeError('Checkpoint incompatible; refused to run with missing, '
                           'unexpected or incorrectly sized model weights')
    model.load_state_dict(state, strict=True)
    report['checkpoint_comparison']['strict_load_succeeded'] = True


def prepare_frame(results, lidar_sweeps, cfg, model, warnings):
    import numpy as np
    import torch
    from mmengine.dataset import Compose, pseudo_collate

    results, counts = load_points(results, lidar_sweeps, cfg, warnings)
    # Validate before packing/preprocessor (torch.stack fails on empty voxels).
    # The two first transforms load full-resolution images and filter LiDAR.
    transforms = Compose(cfg.inference_pipeline)
    for transform in transforms.transforms[:-1]:
        results = transform(results)
    if len(results['points']) == 0:
        raise ValueError('LiDAR is empty after sweep loading / point-cloud range filter')
    if len(results['radars']) == 0:
        raise ValueError('Radar aggregate is empty after devkit / point-cloud range filters')
    for name, image in zip(cfg.camera_order, results['img']):
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f'{name}: expected a readable three-channel BGR image')
    counts.update(lidar_after_range_filter=len(results['points']),
                  radar_after_range_filter=len(results['radars']))
    packed = transforms.transforms[-1](results)
    if set(packed['inputs']) != {'img', 'points', 'radars'}:
        raise ValueError(f'Unexpected inputs, GT is forbidden: {packed["inputs"].keys()}')
    # CPU preprocessing preserves the repo's feature ordering/truncation and
    # avoids its per-voxel torch.zeros allocations on CPU with points on CUDA.
    with torch.inference_mode():
        prepared = model.data_preprocessor(pseudo_collate([packed]), training=False)
    inputs = prepared['inputs']
    details = dict(
        image_shape=list(inputs['imgs'].shape), image_dtype=str(inputs['imgs'].dtype),
        lidar_voxels=int(inputs['lidar_voxel_feats'][0].shape[0]),
        radar_voxels=int(inputs['radar_voxel_feats'][0].shape[0]),
        point_counts=counts, preprocessor_device='cpu', batch_size=1,
    )
    if not torch.isfinite(inputs['imgs']).all():
        raise ValueError('Non-finite normalized images')
    for modality in ('lidar', 'radar'):
        if not torch.isfinite(inputs[f'{modality}_voxel_feats'][0]).all():
            raise ValueError(f'Non-finite {modality} voxel features')
    if not np.isfinite(prepared['data_samples'][0].lidar2img).all():
        raise ValueError('Non-finite camera projection matrices')
    return prepared, details


def move_data(value, device):
    """Transfer tensors/samples; keep NumPy projection metadata on the CPU."""
    import torch
    from mmengine.structures import BaseDataElement

    if isinstance(value, (torch.Tensor, BaseDataElement)):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_data(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_data(item, device) for item in value)
    return value


def export_predictions(logits, metadata, out_dir):
    import numpy as np
    import torch
    from PIL import Image

    expected = [1, *metadata['grid_shape'], len(metadata['class_names'])]
    if not isinstance(logits, torch.Tensor) or list(logits.shape) != expected:
        raise ValueError(f'Expected logits {expected}, got {getattr(logits, "shape", None)}')
    # Compute softmax on CPU in FP32 without another GPU-sized probability volume.
    logits = logits[0].detach().float().cpu()
    if not torch.isfinite(logits).all():
        raise ValueError('Non-finite logits; no prediction artifacts will be written')
    probability, labels = logits.softmax(dim=-1).max(dim=-1)
    labels = labels.numpy().astype(np.uint8)
    probability = probability.numpy().astype(np.float32)

    # Highest non-free voxel at each (x,y). Free-only columns stay class 0.
    occupied = labels != 0
    has_occupied = occupied.any(axis=2)
    highest_z = labels.shape[2] - 1 - occupied[:, :, ::-1].argmax(axis=2)
    bev_labels = np.take_along_axis(labels, highest_z[:, :, None], axis=2)[:, :, 0]
    bev_labels[~has_occupied] = 0
    palette = np.asarray(metadata['palette'], dtype=np.uint8)
    # PNG row 0 = largest y; column 0 = smallest x. No interpolation.
    rgb = palette[bev_labels].transpose(1, 0, 2)[::-1]
    npz_temp = out_dir / 'occupancy.tmp.npz'
    png_temp = out_dir / 'bev.tmp.png'
    np.savez_compressed(npz_temp, labels=labels, max_softmax_probability=probability)
    Image.fromarray(rgb).save(png_temp)
    npz_temp.replace(out_dir / 'occupancy.npz')
    png_temp.replace(out_dir / 'bev.png')
    return dict(logits_shape=expected, labels_dtype=str(labels.dtype),
                probability_dtype=str(probability.dtype),
                predicted_voxel_counts={str(i): int(n) for i, n in
                                        zip(*np.unique(labels, return_counts=True))})


def main():
    args = parse_args()
    out_dir = repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [str(out_dir / name) for name in ARTIFACTS if (out_dir / name).exists()]
    if existing:
        print(f'Refusing to overwrite an earlier attempt; use a fresh --out-dir: {existing}',
              file=sys.stderr)
        return 1

    started = time.perf_counter()
    report = dict(
        status='started', command=[sys.executable, *sys.argv],
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        check_only=args.check_only, precision='float32', autocast=False,
        timing_note='Single frame, no warm-up; not a stable benchmark. '
                    'Pipeline includes sensor I/O, CPU preprocessing, transfer, '
                    'forward, CPU softmax and NPZ/BEV export; excludes startup '
                    '(imports, JSON/index construction, checkpoint/model/device setup) '
                    'and final metadata/report writes.',
        confidence_note='Max softmax probability is not calibrated uncertainty.',
        metrics=dict(occupancy_iou=None, occupancy_miou=None,
                     reason='Unavailable: no occupancy GT loaded.'),
        timings_s={}, warnings=[],
        vram=dict(peak_allocated_bytes=None, peak_reserved_bytes=None,
                  scope='CUDA not used'),
    )
    metadata = None
    stage = 'imports'
    stage_started = started
    pipeline_started = None
    cuda_device = None
    torch = None
    try:
        import numpy as np
        import torch
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmdet3d.registry import MODELS
        from mmdet3d.utils import register_all_modules
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.splits import create_splits_scenes
        from occfusion.nuscenes_dataset import NuScenesSegDataset
        from mmdet3d.models import Base3DSegmentor

        config_path = repo_path(args.config)
        cfg = Config.fromfile(str(config_path))
        if cfg.dataset_version != 'v1.0-mini':
            raise ValueError('This CLI requires dataset_version=v1.0-mini')
        baseline = Config.fromfile(str(ROOT / 'configs/OccFusion.py'))
        if cfg.model != baseline.model:
            raise ValueError('Inference model differs from configs/OccFusion.py baseline')
        if len(cfg.camera_order) != 6 or len(set(cfg.camera_order)) != 6:
            raise ValueError('Expected exactly six distinct cameras')
        if len(cfg.radar_order) != 5 or len(set(cfg.radar_order)) != 5:
            raise ValueError('Expected exactly five distinct radars')
        # Validate the installed framework API instead of relying on test_step.
        signature = inspect.signature(Base3DSegmentor.forward)
        if 'mode' not in signature.parameters:
            raise RuntimeError(f'Framework has no tensor forward API: {signature}')
        report['framework_forward_api'] = str(signature)
        report['environment'] = dict(
            python=sys.version, executable=sys.executable,
            packages={name: importlib.metadata.version(name) for name in
                      ('torch', 'torchvision', 'mmengine', 'mmcv', 'mmdet',
                       'mmdet3d', 'nuscenes-devkit', 'numpy')},
            cuda_build=torch.version.cuda,
        )
        report['code'] = code_version()
        cfg.dump(str(out_dir / 'resolved_config.py'))

        stage = 'input_validation'
        stage_started = time.perf_counter()
        checkpoint = repo_path(args.checkpoint or cfg.load_from)
        data_root = repo_path(args.data_root or cfg.data_root)
        report.update(config=str(config_path), checkpoint=str(checkpoint),
                      data_root=str(data_root))
        validate_paths(checkpoint, data_root, cfg.dataset_version)
        nusc = NuScenes(version=cfg.dataset_version, dataroot=str(data_root), verbose=False)
        scene_index = resolve_scene_index(nusc, args.scene_index, args.scene_name)
        scene, sample = select_sample(nusc, scene_index, args.frame_index)
        official_splits = create_splits_scenes()
        memberships = [name for name in ('mini_train', 'mini_val')
                       if scene['name'] in official_splits[name]]
        if len(memberships) != 1:
            raise ValueError('Scene must belong to exactly one official nuScenes mini split')
        split = memberships[0]
        report.update(sample_token=sample['token'], scene_name=scene['name'],
                      scene_index=scene_index, frame_index=args.frame_index, split=split,
                      split_source='nuscenes.utils.splits.create_splits_scenes')
        results, lidar_sweeps, metadata = make_frame_info(nusc, sample, cfg, report['warnings'])
        bounds = np.asarray(cfg.point_cloud_range, dtype=float)
        grid = np.asarray(cfg.model.view_transformer.grid_size[0], dtype=int) * 2
        class_names = ['free', *NuScenesSegDataset.METAINFO['classes'][1:]]
        metadata.update(
            result_source='prediction', dataset=dict(root=str(data_root), version=cfg.dataset_version,
                                                   split=split, split_source=report['split_source']),
            scene=dict(name=scene['name'], token=scene['token'], index=scene_index),
            frame_index=args.frame_index, bounds=bounds.tolist(), grid_shape=grid.tolist(),
            voxel_size=((bounds[3:] - bounds[:3]) / grid).tolist(),
            grid_axis_order=['x', 'y', 'z'], flatten_order='C (z varies fastest)',
            voxel_center='bounds_min + (index + 0.5) * voxel_size',
            class_names=class_names, palette=NuScenesSegDataset.METAINFO['palette'],
            free_label=0, gt_ignore_label=255, visibility_mask=None,
            visibility_note='No visibility mask is produced or inferred from predictions.',
            config=dict(path=str(config_path), sha256=file_hash(config_path),
                        resolved_file='resolved_config.py',
                        resolved_sha256=file_hash(out_dir / 'resolved_config.py'),
                        baseline_path=str(ROOT / 'configs/OccFusion.py'),
                        baseline_sha256=file_hash(ROOT / 'configs/OccFusion.py')),
            checkpoint=dict(path=str(checkpoint), bytes=checkpoint.stat().st_size,
                            sha256=file_hash(checkpoint)),
            code=report['code'], environment=report['environment'],
            image_preprocessing=dict(
                color_order='BGR', resize=False, scale_factor_actual=1.0,
                mean=list(cfg.model.data_preprocessor.mean),
                std=list(cfg.model.data_preprocessor.std),
                pad_size_divisor=cfg.model.data_preprocessor.pad_size_divisor,
                note='Repo image loader advertises scale_factor=0.2 despite no resize; '
                     'this field is not packed or used by tensor forward.'),
            sensor_configuration=dict(use_camera=True, use_lidar=True, use_radar=True,
                                      use_occ3d=False, points_per_voxel=35),
            bev=dict(projection='highest non-free voxel along z; free if none',
                     rows='y decreasing', columns='x increasing',
                     width=int(grid[0]), height=int(grid[1])),
            confidence_note=report['confidence_note'], metrics=report['metrics'],
        )
        report['timings_s']['input_validation'] = time.perf_counter() - stage_started
        print(f'Sample {sample["token"]}, {scene["name"]}, frame {args.frame_index}; '
              f'validated {metadata["sensors"]["checked_file_count"]} sensor files', flush=True)

        stage = 'model_checkpoint_cpu'
        stage_started = time.perf_counter()
        register_all_modules(init_default_scope=False)
        init_default_scope('mmdet3d')
        model = MODELS.build(cfg.model).cpu()
        strict_load_checkpoint(model, checkpoint, report)
        model.eval()
        report['timings_s']['model_checkpoint_cpu'] = time.perf_counter() - stage_started

        stage = 'device_setup'
        stage_started = time.perf_counter()
        device = torch.device('cpu' if args.check_only else args.device)
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('--device must be cpu or cuda:N')
        report['device'] = str(device)
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable in this process; no CPU fallback was used')
            cuda_device = device
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
            properties = torch.cuda.get_device_properties(device)
            report['gpu'] = dict(name=properties.name, total_memory_bytes=properties.total_memory)
            model.to(device)
            # EfficientViT's train(False) caches attention biases in a plain
            # tensor (ab), which Module.to() does not move. Rebuild the eval
            # caches AFTER moving the weights, using the loaded checkpoint.
            model.eval()
            # Repo preprocessor allocates voxel storage on CPU. Keep it there
            # and transfer the finished batch explicitly after normalization.
            model.data_preprocessor.cpu()
            torch.cuda.synchronize(device)
        report['tf32'] = dict(matmul=torch.backends.cuda.matmul.allow_tf32,
                              cudnn=torch.backends.cudnn.allow_tf32)
        report['timings_s']['device_setup'] = time.perf_counter() - stage_started

        stage = 'preprocessing'
        stage_started = pipeline_started = time.perf_counter()
        prepared, details = prepare_frame(results, lidar_sweeps, cfg, model, report['warnings'])
        if cuda_device is not None:
            prepared = move_data(prepared, device)
            torch.cuda.synchronize(device)
        report['preprocessing'] = details
        metadata['image_preprocessing']['batch_shape'] = details['image_shape']
        metadata['sensors']['point_counts'] = details['point_counts']
        report['timings_s']['preprocessing'] = time.perf_counter() - stage_started
        print('Preprocessing:', json.dumps({key: value for key, value in details.items()
                                           if key != 'point_counts'}), flush=True)

        if args.check_only:
            report['status'] = 'check_only_passed'
            report['timings_s']['forward'] = None
            print('CHECK ONLY PASSED on CPU; no forward or occupancy predictions.', flush=True)
        else:
            stage = 'forward'
            stage_started = time.perf_counter()
            if cuda_device is not None:
                torch.cuda.synchronize(device)
            with torch.inference_mode():
                logits = model(inputs=prepared['inputs'],
                               data_samples=prepared['data_samples'], mode='tensor')
            if cuda_device is not None:
                torch.cuda.synchronize(device)
            report['timings_s']['forward'] = time.perf_counter() - stage_started
            stage = 'postprocessing_export'
            stage_started = time.perf_counter()
            report['prediction'] = export_predictions(logits, metadata, out_dir)
            report['timings_s']['postprocessing_export'] = time.perf_counter() - stage_started
            report['status'] = 'success'
        report['timings_s']['pipeline_total'] = time.perf_counter() - pipeline_started
    except Exception as error:
        is_oom = torch is not None and isinstance(error, torch.cuda.OutOfMemoryError)
        report['status'] = 'cuda_out_of_memory' if is_oom else 'failed'
        report['error'] = dict(stage=stage, type=type(error).__name__, message=str(error),
                               traceback=traceback.format_exc(), cuda_out_of_memory=is_oom)
        if cuda_device is not None:
            try:
                torch.cuda.synchronize(cuda_device)
            except Exception as sync_error:
                report['error']['cuda_synchronize_error'] = str(sync_error)
        report['timings_s'][f'{stage}_failed_elapsed'] = time.perf_counter() - stage_started
        if pipeline_started is not None:
            report['timings_s']['pipeline_until_failure'] = time.perf_counter() - pipeline_started
        print(f'FAILED at {stage}: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        if cuda_device is not None:
            report['vram'] = dict(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(cuda_device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(cuda_device),
                scope='device setup and one-frame pipeline, no warm-up',
            )
        report['timings_s']['process_until_report'] = time.perf_counter() - started
        report['warnings'] = list(dict.fromkeys(report['warnings']))
        for warning in report['warnings']:
            print(f'Warning: {warning}', file=sys.stderr)
        if metadata is not None:
            metadata.update(status=report['status'], precision=report['precision'],
                            autocast=report['autocast'], warnings=report['warnings'],
                            timings_s=report['timings_s'], vram=report['vram'],
                            timing_note=report['timing_note'])
            write_json(out_dir / 'metadata.json', metadata)
        write_json(out_dir / 'report.json', report)
        print(f'Report: {out_dir / "report.json"} ({report["status"]})', flush=True)
    return 0 if report['status'] in ('success', 'check_only_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
