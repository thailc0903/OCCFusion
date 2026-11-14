#!/usr/bin/env python3
"""CPU evaluation of saved OccFusion predictions against SurroundOcc sparse GT.

No model, checkpoint, lidarseg, boxes or web/review database is loaded. Outputs
are separate from the four inference artifacts and existing approval bundles.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ('occupancy.npz', 'metadata.json', 'report.json', 'bev.png')
GRID = (200, 200, 16)
BOUNDS = (-50, -50, -5, 50, 50, 3)
VOXEL_SIZE = (.5, .5, .5)
CLASS_NAMES = (
    'free', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation',
)
GT_SOURCE = 'https://github.com/weiyithu/SurroundOcc/blob/main/docs/data.md'


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def json_object(content):
    def reject(value):
        raise ValueError(f'Non-finite JSON constant: {value}')
    value = json.loads(content, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    return value


def exact_vector(value, expected, name):
    array = np.asarray(value)
    if array.dtype.kind not in 'iuf' or array.shape != np.shape(expected) or not np.isfinite(array).all() or not np.array_equal(array, expected):
        raise ValueError(f'{name} must be {list(expected)} for SurroundOcc, got {value!r}')


def validate_geometry(metadata):
    exact_vector(metadata['bounds'], BOUNDS, 'bounds XYZ')
    exact_vector(metadata['voxel_size'], VOXEL_SIZE, 'voxel_size XYZ')
    exact_vector(metadata['grid_shape'], GRID, 'grid_shape')
    if metadata['grid_axis_order'] != ['x', 'y', 'z']:
        raise ValueError('SurroundOcc requires array axis order [x,y,z]; no implicit transpose')
    if metadata['voxel_center'] != 'bounds_min + (index + 0.5) * voxel_size':
        raise ValueError('Unsupported voxel center convention')
    coordinate = metadata['coordinate_system']
    if coordinate['name'] != 'LIDAR_TOP at reference_timestamp_us' or coordinate['unit'] != 'metres':
        raise ValueError('Prediction must use native reference LiDAR coordinates in metres')
    if metadata['class_names'] != list(CLASS_NAMES) or metadata['free_label'] != 0 or metadata['gt_ignore_label'] != 255:
        raise ValueError('Prediction class map, empty or ignore label differs from SurroundOcc protocol')
    if metadata.get('sensor_configuration', {}).get('use_occ3d', False):
        raise ValueError('Occ3D predictions cannot use this SurroundOcc protocol')


def sparse_to_dense(rows, grid=GRID, num_classes=len(CLASS_NAMES)):
    """Implement LoadOccupancy + ratio-one multiscale_supervision, with validation."""
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 4 or rows.dtype.kind not in 'iuf':
        raise ValueError('SurroundOcc GT must be a numeric N x 4 array of x,y,z,label')
    if not np.isfinite(rows).all() or not np.equal(rows, np.floor(rows)).all():
        raise ValueError('GT coordinates and labels must be finite integers, not metric points')
    coordinates, labels = rows[:, :3], rows[:, 3]
    bad = np.any((coordinates < 0) | (coordinates >= np.asarray(grid)), axis=1)
    if bad.any():
        raise ValueError(f'GT indices outside grid {grid}: {coordinates[bad][:5].tolist()}')
    valid_labels = (labels >= 0) & (labels < num_classes) | (labels == 255)
    if not valid_labels.all():
        raise ValueError(f'Invalid GT labels: {np.unique(labels[~valid_labels]).tolist()}')
    coordinates = coordinates.astype(np.int64)
    normalized = labels.astype(np.int64)
    normalized[normalized == 0] = 255
    flat = np.ravel_multi_index(coordinates.T, grid)
    order = np.argsort(flat, kind='stable')
    ordered, values = flat[order], normalized[order]
    conflicts = (ordered[1:] == ordered[:-1]) & (values[1:] != values[:-1])
    if conflicts.any():
        indices = np.flatnonzero(conflicts)[:5]
        locations = coordinates[order[indices]].tolist()
        raise ValueError(f'Conflicting GT labels at duplicate voxel coordinates: {locations}')
    unique_count = int(np.unique(flat).size)
    dense = np.zeros(grid, dtype=np.uint8)
    dense.reshape(-1)[flat] = normalized.astype(np.uint8)
    return dense, dict(
        sparse_rows=len(rows), unique_listed_voxels=unique_count,
        duplicate_identical_rows=len(rows)-unique_count,
        rows_label_zero_remapped_to_ignore=int((labels == 0).sum()),
        ignored_voxels=int((dense == 255).sum()),
        unlisted_empty_voxels=int(dense.size-unique_count),
    )


def confusion_statistics(prediction, ground_truth, num_classes=len(CLASS_NAMES)):
    prediction, ground_truth = np.asarray(prediction), np.asarray(ground_truth)
    if prediction.shape != ground_truth.shape or prediction.dtype.kind not in 'iu' or ground_truth.dtype.kind not in 'iu':
        raise ValueError('Prediction and dense GT must be integer arrays of equal shape')
    if np.any(prediction < 0) or np.any(prediction >= num_classes):
        raise ValueError('Prediction labels are outside the semantic class map')
    valid = ground_truth != 255
    gt = ground_truth[valid].astype(np.int64)
    pred = prediction[valid].astype(np.int64)
    if np.any(gt < 0) or np.any(gt >= num_classes):
        raise ValueError('Dense GT labels are outside the semantic class map')
    return np.bincount(gt*num_classes + pred, minlength=num_classes**2).reshape(num_classes, num_classes)


def summarize(confusion, class_names=CLASS_NAMES):
    """Ratios of summed counts. Union zero is 0, including in the semantic mean."""
    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.shape != (len(class_names), len(class_names)) or np.any(confusion < 0):
        raise ValueError('Invalid confusion matrix')
    gt, pred = confusion.sum(axis=1), confusion.sum(axis=0)
    intersection = int(confusion[1:, 1:].sum())
    gt_occupied, pred_occupied = int(gt[1:].sum()), int(pred[1:].sum())
    union = gt_occupied + pred_occupied - intersection
    semantic = []
    for label in range(1, len(class_names)):
        tp = int(confusion[label, label])
        total = int(gt[label] + pred[label] - tp)
        semantic.append(dict(label=label, name=class_names[label], intersection=tp,
                             union=total, gt_count=int(gt[label]), prediction_count=int(pred[label]),
                             iou=tp/total if total else 0.0))
    return dict(
        occupancy_iou=intersection/union if union else 0.0,
        semantic_miou=float(np.mean([entry['iou'] for entry in semantic])),
        semantic_mean_class_count=len(semantic), per_class_semantic=semantic,
        occupancy=dict(intersection=intersection, union=union,
                       gt_occupied=gt_occupied, prediction_occupied=pred_occupied),
        valid_voxel_count=int(confusion.sum()), confusion_matrix=confusion.tolist(),
    )


def reference_method():
    """Execute the exact NumPy method from this checkout, without OpenMMLab imports."""
    path = ROOT / 'occfusion/evaluate.py'
    source = path.read_bytes()
    tree = ast.parse(source, filename=str(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'EvalMetric')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'evaluation_semantic')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[method.name], sha256(source)


def compare_reference(ground_truths, predictions, metrics, class_names=CLASS_NAMES):
    function, source_hash = reference_method()
    names = {0: 'IoU', **{i: name for i, name in enumerate(class_names) if i}}
    with np.errstate(divide='ignore', invalid='ignore'):
        reference = function(None, ground_truths, predictions, names, 255)
    ours = {'IoU': metrics['occupancy_iou'], 'mIoU': metrics['semantic_miou']}
    ours.update({entry['name']: entry['iou'] for entry in metrics['per_class_semantic']})
    error = max(abs(float(reference[name])-value) for name, value in ours.items())
    if error > 1e-12:
        raise ValueError(f'Reference EvalMetric disagreement: maximum absolute difference {error}')
    return dict(passed=True, max_abs_difference=error, source_sha256=source_hash,
                method='occfusion.evaluate.EvalMetric.evaluation_semantic',
                execution='Exact method extracted by AST; NumPy CPU, no model/framework initialization')


class GroundTruthIndex:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.files = defaultdict(list)
        if self.root.is_dir():
            for path in self.root.rglob('*.npy'):
                resolved = path.resolve()
                if not resolved.is_relative_to(self.root):
                    raise ValueError(f'GT symlink escapes --gt-root: {path}')
                self.files[path.name].append(resolved)

    def lookup(self, lidar_filename):
        key = Path(lidar_filename).name + '.npy'
        matches = self.files.get(key, [])
        if len(matches) > 1:
            raise ValueError(f'Ambiguous GT filename {key}: {[str(path) for path in matches]}')
        return key, matches[0] if matches else None


class DatasetCatalog:
    def __init__(self, root, version='v1.0-mini'):
        os.environ.setdefault('MPLCONFIGDIR', '/tmp/occfusion-matplotlib')
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.splits import create_splits_scenes
        if version != 'v1.0-mini':
            raise ValueError('This milestone uses official mini_train/mini_val on v1.0-mini')
        self.root = Path(root).expanduser().resolve()
        self.version = version
        self.nusc = NuScenes(version=version, dataroot=str(self.root), verbose=False)
        self.splits = create_splits_scenes()
        self.samples = {}
        for scene_index, scene in enumerate(self.nusc.scene):
            token, index, visited = scene['first_sample_token'], 0, set()
            memberships = [name for name in ('mini_train', 'mini_val') if scene['name'] in self.splits[name]]
            if len(memberships) != 1:
                raise ValueError(f'Invalid official mini split for {scene["name"]}')
            while token:
                if token in visited:
                    raise ValueError(f'Cycle in sample links for {scene["name"]}')
                visited.add(token)
                sample = self.nusc.get('sample', token)
                if sample['scene_token'] != scene['token']:
                    raise ValueError('Sample chain crosses scene boundary')
                sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                self.samples[token] = dict(sample=sample, scene=scene, scene_index=scene_index,
                                           frame_index=index, lidar=sd, split=memberships[0])
                token, index = sample['next'], index+1
            if index != scene['nbr_samples']:
                raise ValueError('Scene frame count disagrees with sample chain')

    def inventory(self, split, gt):
        frames = []
        for token, item in self.samples.items():
            if item['split'] != split:
                continue
            key, path = gt.lookup(item['lidar']['filename'])
            frames.append(dict(sample_token=token, scene=item['scene']['name'],
                               frame_index=item['frame_index'], lidar_filename=item['lidar']['filename'],
                               expected_gt_filename=key, gt_path=str(path) if path else None,
                               status='has_gt' if path else 'missing_gt'))
        counts = Counter(frame['status'] for frame in frames)
        return dict(frame_count=len(frames), has_gt=counts['has_gt'], missing_gt=counts['missing_gt'], frames=frames)

    def validate_sample(self, metadata, report):
        from nuscenes.utils.geometry_utils import transform_matrix
        from pyquaternion import Quaternion
        token = metadata['sample_token']
        if token != report['sample_token'] or token not in self.samples:
            raise ValueError('metadata/report sample token mismatch or unknown to nuScenes')
        item = self.samples[token]
        sample, scene, sd = item['sample'], item['scene'], item['lidar']
        if metadata['dataset']['version'] != self.version:
            raise ValueError('Prediction dataset version differs from nuScenes API version')
        if metadata['scene'] != dict(name=scene['name'], token=scene['token'], index=item['scene_index']):
            raise ValueError('Prediction scene metadata differs from nuScenes API')
        if report['scene_name'] != scene['name'] or report['scene_index'] != item['scene_index']:
            raise ValueError('Report scene differs from nuScenes API')
        if metadata['frame_index'] != item['frame_index'] or report['frame_index'] != item['frame_index']:
            raise ValueError('Frame index differs from nuScenes sample chain')
        for declared in (metadata['dataset'].get('split'), report.get('split')):
            if declared is not None and declared != item['split']:
                raise ValueError('Declared split differs from nuscenes-devkit')
        record = metadata['sensors']['keyframes']['LIDAR_TOP']
        cal = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        ego = self.nusc.get('ego_pose', sd['ego_pose_token'])
        sensor = self.nusc.get('sensor', cal['sensor_token'])
        if sd['sample_token'] != token or not sd['is_key_frame'] or sensor['channel'] != 'LIDAR_TOP':
            raise ValueError('LiDAR is not the sample keyframe in the reference channel')
        if record['sample_data_token'] != sd['token'] or Path(record['path']).name != Path(sd['filename']).name:
            raise ValueError('Metadata LiDAR token/filename differs from nuScenes API')
        if record['channel'] != 'LIDAR_TOP' or record['calibrated_sensor_token'] != sd['calibrated_sensor_token'] or record['ego_pose_token'] != sd['ego_pose_token']:
            raise ValueError('LiDAR calibration/pose tokens differ from nuScenes API')
        if metadata['timestamp_us'] != sample['timestamp'] or metadata['reference_timestamp_us'] != sd['timestamp'] or record['timestamp_us'] != sd['timestamp']:
            raise ValueError('Prediction/reference LiDAR timestamps differ from nuScenes API')
        if metadata['timestamp_unit'] != 'microseconds':
            raise ValueError('Prediction timestamps must use microseconds')
        if not (self.root / sd['filename']).is_file():
            raise ValueError(f'Missing sample LiDAR file: {self.root / sd["filename"]}')
        lidar2ego = transform_matrix(cal['translation'], Quaternion(cal['rotation']))
        lidar2global = transform_matrix(ego['translation'], Quaternion(ego['rotation'])) @ lidar2ego
        matrices = [(record['sensor_to_reference_lidar'], np.eye(4)),
                    (metadata['coordinate_system']['lidar_to_ego'], lidar2ego),
                    (metadata['coordinate_system']['lidar_to_global'], lidar2global)]
        for actual, expected in matrices:
            actual = np.asarray(actual, dtype=float)
            if actual.shape != (4, 4) or not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=0, atol=1e-8):
                raise ValueError('Prediction reference transform differs from native sample LiDAR frame')
        return item


def load_prediction(folder):
    folder = Path(folder).expanduser().resolve()
    files = {name: (folder/name).read_bytes() for name in ARTIFACTS}
    metadata, report = json_object(files['metadata.json']), json_object(files['report.json'])
    if metadata['status'] != 'success' or report['status'] != 'success' or report.get('check_only') or metadata['result_source'] != 'prediction':
        raise ValueError('Evaluation requires a successful saved prediction, not check-only/failed runs')
    validate_geometry(metadata)
    with np.load(BytesIO(files['occupancy.npz']), allow_pickle=False) as archive:
        if set(archive.files) != {'labels', 'max_softmax_probability'}:
            raise ValueError('Unexpected prediction NPZ schema; no NPZ sample token is assumed')
        labels, probability = archive['labels'], archive['max_softmax_probability']
    if labels.shape != GRID or labels.dtype.kind not in 'iu' or np.any(labels < 0) or np.any(labels >= len(CLASS_NAMES)):
        raise ValueError('Invalid prediction label shape/range/dtype')
    if probability.shape != GRID or probability.dtype.kind != 'f' or not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
        raise ValueError('Invalid max_softmax_probability shape/range/non-finite values')
    histogram = {str(int(label)): int(count) for label, count in zip(*np.unique(labels, return_counts=True))}
    if histogram != report['prediction']['predicted_voxel_counts'] or report['prediction']['logits_shape'] != [1, *GRID, len(CLASS_NAMES)]:
        raise ValueError('Prediction histogram/logits shape differs from report')
    if metadata['precision'] != report['precision']:
        raise ValueError('metadata/report precision mismatch')
    hashes = {name: dict(sha256=sha256(value), bytes=len(value)) for name, value in files.items()}
    return metadata, report, labels, hashes


def protocol():
    source_paths = ['occfusion/loading.py', 'occfusion/main.py', 'occfusion/evaluate.py', 'tools/evaluate_occupancy.py']
    return dict(
        name='surroundocc-200x200x16-occfusion-v1', ground_truth_source=GT_SOURCE,
        geometry_source='https://github.com/weiyithu/SurroundOcc/blob/main/tools/generate_occupancy_nuscenes/config.yaml',
        bounds=list(BOUNDS), voxel_size=list(VOXEL_SIZE), grid_shape=list(GRID),
        grid_axis_order=['x', 'y', 'z'], coordinate_system='native sample LIDAR_TOP, metres',
        voxel_center='bounds_min + (index + 0.5) * voxel_size',
        sparse_format='NPY N x 4 [index_x,index_y,index_z,semantic_label]',
        sparse_zero_label_becomes=255, ignore_label=255, unlisted_voxel_label=0,
        accepted_raw_gt_labels=[*range(len(CLASS_NAMES)), 255], class_names=list(CLASS_NAMES),
        occupancy='GT != 0 versus prediction != 0, excluding GT == 255',
        semantic_mean_class_ids=list(range(1, len(CLASS_NAMES))), semantic_mean_class_count=16,
        union_zero_iou=0.0, absent_classes_included_in_mean=True,
        aggregation='sum confusion/intersection/GT/prediction counts across evaluated frames, then divide',
        original_evaluator_equivalence='Original averages counts before ratios; equivalent to summed counts, not mean frame IoU',
        visibility_mask='None; only GT ignore voxels are excluded',
        gt_metadata_limitation='Sparse NPY has no embedded token/bounds/coordinate schema; fixed protocol declared here; matching by nuScenes LiDAR basename',
        npz_token_source='Prediction token comes from metadata.json/report.json, not from NPZ',
        source_hashes={name: sha256((ROOT/name).read_bytes()) for name in source_paths},
    )


def evaluate_runs(run_dirs, catalog, gt, split, verify_reference=False):
    pooled = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    frames, seen, reference_gt, reference_pred = [], set(), [], []
    for folder in run_dirs:
        frame = dict(run_dir=str(Path(folder).expanduser().resolve()), status='invalid_input', metrics=None)
        frames.append(frame)
        try:
            metadata, report, prediction, hashes = load_prediction(folder)
            frame.update(sample_token=metadata['sample_token'], artifact_hashes=hashes,
                         prediction_sha256=hashes['occupancy.npz']['sha256'], gt_sha256=None)
            item = catalog.validate_sample(metadata, report)
            frame.update(scene=item['scene']['name'], frame_index=item['frame_index'], split=item['split'],
                         timestamp_us=metadata['timestamp_us'], reference_timestamp_us=metadata['reference_timestamp_us'],
                         lidar_sample_data_token=item['lidar']['token'], lidar_filename=item['lidar']['filename'])
            if metadata['sample_token'] in seen:
                raise ValueError('Duplicate sample token; compare different predictions in separate evaluations')
            seen.add(metadata['sample_token'])
            key, path = gt.lookup(item['lidar']['filename'])
            frame.update(expected_gt_filename=key, gt_path=str(path) if path else None,
                         gt_status='has_gt' if path else 'missing_gt')
            if path:
                content = path.read_bytes()
                frame.update(gt_sha256=sha256(content), gt_bytes=len(content))
            if item['split'] != split:
                frame.update(status='split_mismatch', reason=f'Sample belongs to {item["split"]}, requested {split}')
                continue
            if path is None:
                frame.update(status='missing_gt', reason='No exact LiDAR-basename .npy under --gt-root; metrics unavailable')
                continue
            rows = np.load(BytesIO(content), allow_pickle=False)
            ground_truth, details = sparse_to_dense(rows)
            confusion = confusion_statistics(prediction, ground_truth)
            metrics = summarize(confusion)
            frame.update(gt_statistics=details)
            if verify_reference:
                frame['reference_check'] = compare_reference([ground_truth], [prediction], metrics)
            pooled += confusion
            if verify_reference:
                reference_gt.append(ground_truth)
                reference_pred.append(prediction)
            frame.update(status='evaluated', metrics=metrics)
        except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile, EOFError) as error:
            frame.update(status='invalid_input', error=f'{type(error).__name__}: {error}', metrics=None)
    counts = Counter(frame['status'] for frame in frames)
    total = counts['evaluated']
    metrics = summarize(pooled) if total else None
    status = ('success' if total == len(frames) and total else
              'partial' if total else 'missing_gt' if counts['missing_gt'] == len(frames) else 'unavailable')
    result = dict(status=status, frame_count_requested=len(frames), frame_count_evaluated=total,
                  frame_status_counts=dict(counts), all_requested_frames_evaluated=bool(frames) and total == len(frames),
                  metrics_scope='Pooled over evaluated frames only; not a full-split benchmark',
                  metrics=metrics, frames=frames, sample_tokens=[frame.get('sample_token') for frame in frames])
    if verify_reference and total:
        result['pooled_reference_check'] = compare_reference(reference_gt, reference_pred, metrics)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', action='append', default=[], help='Saved run directory; repeat for pooled evaluation.')
    parser.add_argument('--gt-root', required=True, help='Extracted SurroundOcc directory; recursively match exact LiDAR .bin.npy basenames.')
    parser.add_argument('--data-root', default=str(ROOT/'data/nuscenes'))
    parser.add_argument('--split', choices=('mini_val', 'mini_train'), default='mini_val')
    parser.add_argument('--inventory-only', action='store_true', help='List GT availability for every sample in the official split; no prediction needed.')
    parser.add_argument('--verify-reference', action='store_true', help='Compare exact original EvalMetric on CPU, per frame and pooled.')
    parser.add_argument('--out', required=True, help='New evaluation.json path; existing files are never overwritten.')
    args = parser.parse_args(argv)
    if args.inventory_only and args.run_dir or not args.inventory_only and not args.run_dir:
        parser.error('Use --inventory-only OR at least one --run-dir')
    output = Path(args.out).expanduser().resolve()
    if output.name != 'evaluation.json':
        parser.error('--out must name a separate evaluation.json file')
    if output.exists():
        parser.error('Output already exists; use a fresh evaluation directory')
    return args


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.out).expanduser().resolve()
    result = dict(
        schema_version=1, status='failed', generated_at_utc=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable, *sys.argv], protocol=protocol(), split=args.split,
        split_source='nuscenes.utils.splits.create_splits_scenes',
        dataset=dict(root=str(Path(args.data_root).expanduser().resolve()), version='v1.0-mini'),
        gt_root=str(Path(args.gt_root).expanduser().resolve()),
        approval_scope=dict(inference_artifacts_modified=False, included_in_approved_export=False),
        execution=dict(device='cpu', model_loaded=False, numpy_version=np.__version__),
        metrics=None,
    )
    exit_code = 1
    try:
        from importlib.metadata import version
        catalog = DatasetCatalog(args.data_root)
        gt = GroundTruthIndex(args.gt_root)
        result['execution']['nuscenes_devkit_version'] = version('nuscenes-devkit')
        result['official_split_scene_names'] = catalog.splits[args.split]
        result['gt_root_exists'] = gt.root.is_dir()
        result['gt_npy_file_count'] = sum(map(len, gt.files.values()))
        result['coverage'] = catalog.inventory(args.split, gt)
        if args.inventory_only:
            result.update(status='inventory', frame_count_evaluated=0)
            exit_code = 0 if result['coverage']['missing_gt'] == 0 else 2
        else:
            result.update(evaluate_runs(args.run_dir, catalog, gt, args.split, args.verify_reference))
            exit_code = 0 if result['status'] == 'success' else 1 if result['frame_status_counts'].get('invalid_input') else 2
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with output.open('x', encoding='utf-8') as stream:
        stream.write(content)
    print(f'Evaluation: {output} ({result["status"]})')
    if 'coverage' in result:
        coverage = result['coverage']
        print(f'{args.split} GT coverage: {coverage["has_gt"]}/{coverage["frame_count"]}; missing_gt={coverage["missing_gt"]}')
    for frame in result.get('frames', []):
        print(f'{frame.get("scene", "?")} frame {frame.get("frame_index", "?")} '
              f'sample={frame.get("sample_token", "?")}: {frame["status"]}; GT={frame.get("gt_path")}')
    if result.get('metrics'):
        print(f'Occupancy IoU={result["metrics"]["occupancy_iou"]:.8f}; '
              f'semantic mIoU={result["metrics"]["semantic_miou"]:.8f} (16 classes, union-zero=0)')
    else:
        print('Metrics unavailable: no evaluated prediction/GT pairs.')
    if 'error' in result:
        print(result['error'], file=sys.stderr)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
