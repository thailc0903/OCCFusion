"""Frozen persistence forecasting v1 protocol and provenance helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools import evaluate_occupancy as occupancy

HISTORY_OFFSETS_US = (-1_000_000, -500_000, 0)
HORIZONS_US = (500_000, 1_000_000, 1_500_000)
TOLERANCE_US = 50_000


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return dict(sha256=digest.hexdigest(), bytes=path.stat().st_size)


def write_json(path, value):
    """Never overwrite an existing result or publish a partially written JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    if path.exists():
        raise FileExistsError(path)
    try:
        with pending.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        # Exclusive hardlink publishes atomically without clobbering another writer.
        path.hardlink_to(pending)
    finally:
        pending.unlink(missing_ok=True)


def protocol():
    return dict(
        name='rav25-occupancy-forecasting-persistence-v1', version=1,
        baseline='persistence; no learned temporal model',
        history_frame_count=3, history_offsets_us=list(HISTORY_OFFSETS_US),
        nominal_horizons_us=list(HORIZONS_US), timestamp_source='sample_data LIDAR_TOP timestamp, microseconds',
        tolerance_us=TOLERANCE_US,
        selection='Within the same scene, nearest LiDAR timestamp to each nominal offset; ties choose earlier timestamp then token. Require tolerance, distinct consecutive sample links and every adjacent interval within 0.5s +/- tolerance. Never pad/skip a sample or cross a scene.',
        input_cutoff='anchor LIDAR_TOP timestamp; every actually used history camera/LiDAR/radar record must be <= cutoff. Unused future radar keyframes are audited separately, never treated as input.',
        availability_limit='Sensor acquisition timestamps only; transport/inference processing latency is not measured. Not a real-time deadline benchmark.',
        future_data='GT and future ego poses are used only by the offline target/evaluator, never by the persistence predictor.',
        reference='native LIDAR_TOP at anchor t; metres, physical full SE(3)',
        transform='anchor_from_source = inv(global_from_anchor_lidar) @ global_from_source_lidar; own calibration and ego pose at each timestamp',
        resampling='Pull anchor voxel centres into source; cell lookup floor((p_source - bounds_min) / voxel_size). Nearest centre, ties choose higher index, half-open source bounds. No linear interpolation.',
        prediction='O(t+h) = saved anchor prediction O(t), in the unchanged anchor grid, all horizons; older history is supplied/aligned but unused by this baseline.',
        bounds=list(occupancy.BOUNDS), grid_shape=list(occupancy.GRID), voxel_size=list(occupancy.VOXEL_SIZE),
        grid_axis_order=['x', 'y', 'z'], flatten_order='C (z varies fastest)',
        voxel_center='bounds_min + (index + 0.5) * voxel_size',
        class_names=list(occupancy.CLASS_NAMES), free_label=0, ignore_label=255,
        gt_protocol=occupancy.protocol(),
        masks=dict(history='separate per-history source in-bounds / known / valid masks; unknown output label 255',
                   target='future source in-bounds AND decoded GT != 255; separate from all history masks',
                   evaluation='target-valid only, independent of confidence, predicted class or correctness; anchor prediction is required to cover the complete grid'),
        semantic_mean_class_ids=list(range(1, 17)), semantic_mean_class_count=16,
        union_zero_iou=0.0, aggregation='sum integer 17x17 GT-row/prediction-column confusion statistics over windows separately per horizon, then ratios; no mean window IoU',
        metric_unit='fraction [0,1]', visibility='No visibility/occlusion mask inferred; SurroundOcc unlisted cells remain protocol empty, not independently observed free space.',
        unsupported=dict(flow=None, EPE=None, ADE=None, FDE=None, status='not_implemented'),
        limitations=['Overlapping windows are correlated, not independent samples.',
                     'Quantized semantic lookup is not object-motion compensation or volume-conservative splatting.',
                     'mini_val is not full validation; checkpoint training overlap has not been independently excluded.'],
    )
