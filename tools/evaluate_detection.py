#!/usr/bin/env python3
"""CPU nuScenes detection_cvpr_2019 evaluation of committed benchmark predictions."""
from __future__ import annotations
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.detection_benchmark_common import (
    atomic_json, canonical, file_record, fingerprint, read_json, utc,
    valid_predictions, verify_files, verify_manifest,
)

DEFAULT_ATTRIBUTES = {
    'car': 'vehicle.parked', 'truck': 'vehicle.parked', 'trailer': 'vehicle.parked',
    'construction_vehicle': 'vehicle.parked', 'bus': 'vehicle.moving',
    'pedestrian': 'pedestrian.moving', 'motorcycle': 'cycle.without_rider',
    'bicycle': 'cycle.without_rider', 'barrier': '', 'traffic_cone': '',
}


def predicted_attribute(name, velocity):
    """MMDetection3D v1.4 policy: prediction class and global XY speed only."""
    moving = math.hypot(*velocity) > 0.2
    if moving and name in {'car', 'truck', 'bus', 'trailer', 'construction_vehicle'}:
        return 'vehicle.moving'
    if moving and name in {'bicycle', 'motorcycle'}:
        return 'cycle.with_rider'
    if not moving and name == 'pedestrian':
        return 'pedestrian.standing'
    if not moving and name == 'bus':
        return 'vehicle.stopped'
    return DEFAULT_ATTRIBUTES[name]


def global_boxes(document, metadata):
    """Geometric LiDAR centers/dx,dy,dz -> global centers/w,l,h; no GT access."""
    import numpy as np
    from pyquaternion import Quaternion
    transform = np.asarray(metadata['coordinate_system']['lidar_to_global'], dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('Invalid LiDAR-to-global matrix')
    rotation = Quaternion(matrix=transform[:3, :3])
    result = []
    for box in document['boxes']:  # ALL saved baseline boxes; no viewer score filter.
        center = transform[:3, :3] @ np.asarray(box['center']) + transform[:3, 3]
        velocity = transform[:3, :3] @ np.array([*box['velocity'], 0.0])
        orientation = rotation * Quaternion(axis=[0, 0, 1], radians=box['yaw'])
        name = document['class_names'][box['label']]
        result.append(dict(sample_token=document['sample_token'], translation=center.tolist(),
            size=[box['dimensions'][1], box['dimensions'][0], box['dimensions'][2]],
            rotation=orientation.normalised.elements.tolist(), velocity=velocity[:2].tolist(),
            detection_name=name, detection_score=box['score'],
            attribute_name=predicted_attribute(name, velocity[:2])))
    return result


def protocol():
    import nuscenes
    from nuscenes.eval.detection.config import config_factory
    package = Path(nuscenes.__file__).resolve().parent
    files = ['eval/detection/evaluate.py', 'eval/detection/algo.py',
             'eval/detection/data_classes.py', 'eval/detection/utils.py',
             'eval/detection/configs/detection_cvpr_2019.json',
             'eval/common/loaders.py', 'eval/common/utils.py', 'utils/splits.py']
    return dict(name='detection_cvpr_2019', evaluator='nuscenes.eval.detection.evaluate.DetectionEval',
        devkit_version=version('nuscenes-devkit'), configuration=config_factory('detection_cvpr_2019').serialize(),
        installed_source_files={name: file_record(package / name) for name in files},
        evaluator_script=file_record(Path(__file__)),
        attribute_policy=dict(name='mmdet3d_v1.4_global_xy_speed_gt_0.2_mps',
            threshold_mps=0.2, defaults=DEFAULT_ATTRIBUTES, uses_gt_attributes=False),
        box_conversion=dict(input_center='geometric', input_dimensions=['dx', 'dy', 'dz'],
            output_size=['dy', 'dx', 'dz'], rotation='global_R_lidar * yaw_about_lidar_Z; quaternion wxyz',
            velocity='global_R_lidar @ [vx,vy,0]; global XY m/s; no translation',
            extra_display_score_filter=False, extra_formatting_range_filter=False),
        filtering='Official DetectionEval class range, point-count and bicycle-rack filtering only',
        metric_units='mAP, NDS and per-class AP are fractions in [0,1], not percent',
        averaging='Official 10 detection classes and 4 center-distance AP thresholds; no per-frame averaging',
        undefined_tp_values='Normalized JSON uses null for official undefined NaN TP errors; raw devkit outputs preserved',
        scope='Complete official mini_val token set; not the full nuScenes val split')


def coverage(expected, predicted, ground_truth):
    expected = list(expected)
    if len(expected) != len(set(expected)):
        raise ValueError('Duplicate manifest sample token')
    expected, predicted, ground_truth = set(expected), set(predicted), set(ground_truth)
    if expected != ground_truth:
        raise ValueError('Manifest token set differs from official mini_val GT')
    unexpected = sorted(predicted - expected)
    if unexpected:
        raise ValueError('Unexpected prediction tokens: ' + ','.join(unexpected))
    return dict(expected=len(expected), predicted=len(predicted), ground_truth=len(ground_truth),
        missing=sorted(expected - predicted), complete=predicted == expected)


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluate(out, manifest, progress, nusc=None, evaluator_class=None):
    """Only publish mAP/NDS with exact complete GT/prediction coverage."""
    from nuscenes import NuScenes
    from nuscenes.eval.common.loaders import load_gt
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.eval.detection.evaluate import DetectionEval
    out = Path(out)
    entries = verify_manifest(manifest)
    verify_files(manifest['prediction_provenance'])
    current_protocol = protocol()
    if current_protocol != manifest['evaluation_protocol']:
        raise ValueError('Evaluator/protocol changed from benchmark manifest')
    nusc = nusc or NuScenes(version='v1.0-mini', dataroot=manifest['prediction_provenance']['data_root'], verbose=False)
    gt = load_gt(nusc, 'mini_val', DetectionBox, verbose=False)
    predictions, missing = valid_predictions(out, manifest, progress)
    counts = coverage([e['sample_token'] for e in entries], predictions, gt.sample_tokens)
    # Unknown progress keys cannot silently extend the scope.
    if set(progress['frames']) - {e['sample_token'] for e in entries}:
        raise ValueError('Progress contains tokens outside the manifest')
    results = {entry['sample_token']: global_boxes(predictions[entry['sample_token']]['document'],
        predictions[entry['sample_token']]['metadata']) for entry in entries if entry['sample_token'] in predictions}
    prediction_json = dict(meta=dict(use_camera=False, use_lidar=True, use_radar=False,
        use_map=False, use_external=False), results=results)
    # This JSON intentionally has NO placeholder entry for any missing prediction.
    destination = out / ('predictions_nuscenes.json' if counts['complete'] else 'predictions_nuscenes_partial.json')
    atomic_json(destination, prediction_json)
    identity = dict(manifest_identity_sha256=manifest['identity_sha256'],
        prediction=file_record(destination), protocol_sha256=fingerprint(current_protocol),
        frame_artifacts={token: value['commit']['artifacts'] for token, value in predictions.items()},
        gt_table_hashes=manifest['dataset_table_hashes'], split='mini_val')
    base = dict(schema_version=1, generated_at_utc=utc(), task='detection', split='mini_val',
        status='complete' if counts['complete'] else 'partial', coverage=counts,
        sample_tokens=list(results), missing_frames=missing, protocol=current_protocol,
        prediction_json=str(destination.relative_to(out)), identity=identity,
        review_scope='Set-level result; not attached to any single-frame review or approval',
        torch_imported='torch' in sys.modules,
        model_modules_loaded=any(name.startswith('mmdet3d.models') for name in sys.modules),
        execution='CPU devkit only; this evaluator never constructs a model or calls forward')
    if not counts['complete']:
        value = dict(base, metrics=None, reason='Missing/invalid predictions; full mini_val mAP/NDS unavailable',
            evaluation_source='not_run')
        atomic_json(out / 'evaluation.json', value)
        return value
    # A cached evaluation needs current prediction bytes, GT bytes, protocol and all official outputs.
    pointer = out / 'evaluation.json'
    if pointer.is_file():
        previous = read_json(pointer)
        try:
            directory = (out / previous['evaluation_directory']).resolve()
            if not directory.is_relative_to(out.resolve()):
                raise ValueError('Evaluation escaped benchmark root')
            commit = read_json(directory / 'evaluation_commit.json')
            core = {k: v for k, v in commit.items() if k != 'fingerprint_sha256'}
            if (commit['identity'] == identity and commit['status'] == 'complete'
                    and fingerprint(core) == commit['fingerprint_sha256']
                    and all(file_record(directory / name) == record for name, record in commit['files'].items())):
                saved = read_json(directory / 'evaluation.json')
                if saved['identity'] != identity:
                    raise ValueError('Evaluation snapshot identity mismatch')
                value = dict(saved, evaluation_source='cache', torch_imported=base['torch_imported'],
                    model_modules_loaded=base['model_modules_loaded'])
                atomic_json(pointer, value)
                return value
        except (OSError, KeyError, ValueError, TypeError):
            pass  # Re-evaluation is CPU only; corrupt snapshots are retained, never reused.
    identifier = uuid.uuid4().hex
    pending = out / 'evaluations' / ('.pending-' + identifier)
    pending.mkdir(parents=True)
    official = (evaluator_class or DetectionEval)(nusc, config_factory('detection_cvpr_2019'),
        str(destination), 'mini_val', output_dir=str(pending / 'official'), verbose=False)
    raw = official.main(plot_examples=0, render_curves=False)
    normalized = finite_json(raw)
    if not all(isinstance(raw[k], (int, float)) and math.isfinite(raw[k]) and -1e-12 <= raw[k] <= 1 + 1e-12
               for k in ('mean_ap', 'nd_score')):
        raise ValueError('Official evaluator returned invalid mAP/NDS')
    metrics = dict(mAP=raw['mean_ap'], NDS=raw['nd_score'],
        per_class_ap=normalized['mean_dist_aps'], per_class_ap_by_distance=normalized['label_aps'],
        tp_errors=normalized['tp_errors'], per_class_tp_errors=normalized['label_tp_errors'])
    def class_counts(boxes):
        return {name: sum(box.detection_name == name for token in boxes.sample_tokens for box in boxes[token])
            for name in current_protocol['configuration']['class_range']}
    support = dict(gt_before_official_filters=class_counts(gt),
        gt_after_official_filters=class_counts(official.gt_boxes),
        predictions_after_official_filters=class_counts(official.pred_boxes),
        saved_prediction_box_count=sum(len(boxes) for boxes in results.values()),
        display_score_filter_applied=False)
    final = pending.with_name(identifier)
    value = dict(base, metrics=metrics, support=support, evaluation_source='new',
        evaluation_directory=str(final.relative_to(out)),
        official_output_files={name: file_record(pending / 'official' / name)
            for name in ('metrics_summary.json', 'metrics_details.json')})
    atomic_json(pending / 'evaluation.json', value)
    commit = dict(status='complete', identity=identity, files={
        name: file_record(pending / name) for name in
        ('evaluation.json', 'official/metrics_summary.json', 'official/metrics_details.json')})
    commit['fingerprint_sha256'] = fingerprint(commit)
    atomic_json(pending / 'evaluation_commit.json', commit)
    pending.rename(final)
    atomic_json(pointer, value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-dir', required=True)
    args = parser.parse_args(argv)
    out = Path(args.benchmark_dir).resolve()
    from tools.detection_benchmark_common import exclusive_lock
    with exclusive_lock(out / '.benchmark.lock'):
        result = evaluate(out, read_json(out / 'manifest.json'), read_json(out / 'progress.json'))
    print(json.dumps(dict(status=result['status'], coverage=result['coverage'], metrics=result['metrics'],
        model_modules_loaded=result['model_modules_loaded']), ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
