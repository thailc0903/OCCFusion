"""CPU-safe artifact identities shared by detection benchmark and evaluation."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_FILES = ('detections.json', 'metadata.json', 'report.json', 'resolved_config.py')
KIND = 'centerpoint_detection_benchmark'


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError('Non-finite JSON: ' + value)
    return json.loads(Path(path).read_bytes(), parse_constant=reject)


def file_record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return dict(sha256=digest.hexdigest(), bytes=path.stat().st_size)


def file_records(folder, names=BUNDLE_FILES):
    return {name: file_record(Path(folder) / name) for name in names}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with pending.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


@contextmanager
def exclusive_lock(path):
    """Same Linux flock/inode as the web worker, held across the entire model lifetime."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Lock is busy: ' + str(path)) from error
        try:
            yield stream.fileno()
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def gpu_lock_path():
    source = Path(os.environ.get('VIEWER_CONFIG', ROOT / 'web/viewer.config.json')).resolve()
    config = read_json(source)
    value = Path(os.environ.get('VIEWER_GPU_LOCK', config.get('gpu_lock_path', '../work/inference-worker/gpu.lock')))
    return (source.parent / value).resolve() if not value.is_absolute() else value.resolve()


def ordered_frames(nusc):
    from nuscenes.utils.splits import create_splits_scenes
    names = sorted(create_splits_scenes()['mini_val'])
    scenes = {scene['name']: scene for scene in nusc.scene if scene['name'] in names}
    if set(scenes) != set(names):
        raise ValueError('Dataset lacks an official mini_val scene')
    result, seen = [], set()
    for name in names:
        scene = scenes[name]
        token, previous, last = scene['first_sample_token'], '', -1
        frame = 0
        while token:
            sample = nusc.get('sample', token)
            if token in seen or sample['scene_token'] != scene['token'] or sample['prev'] != previous or sample['timestamp'] <= last:
                raise ValueError('Invalid sample chain: ' + token)
            seen.add(token)
            lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            result.append(dict(scene=name, scene_token=scene['token'], frame_index=frame,
                sample_token=token, timestamp_us=sample['timestamp'],
                reference_timestamp_us=lidar['timestamp'], lidar_sample_data_token=lidar['token'],
                lidar_filename=lidar['filename']))
            previous, last, token = token, sample['timestamp'], sample['next']
            frame += 1
        if frame != scene['nbr_samples'] or previous != scene['last_sample_token']:
            raise ValueError('Scene count/end token disagrees with sample chain')
    official = {s['token'] for s in nusc.sample if nusc.get('scene', s['scene_token'])['name'] in names}
    if seen != official:
        raise ValueError('Sample chain does not cover the complete official mini_val metadata')
    return names, result


def audit_frame(nusc, entry, pipeline_fingerprint):
    import numpy as np
    from tools.infer_centerpoint_mini import frame_info
    inputs, metadata = frame_info(nusc, entry['scene'], entry['frame_index'])
    if metadata['sample_token'] != entry['sample_token']:
        raise ValueError('Adapter selected a different sample')
    records = [metadata['sensors']['keyframes']['LIDAR_TOP'], *metadata['sensors']['lidar_sweeps']]
    for record in records:
        points = np.fromfile(record['path'], dtype='<f4').reshape(-1, 5)
        if not len(points) or not np.isfinite(points).all():
            raise ValueError('Empty/non-finite input: ' + record['path'])
    sweeps = metadata['sensors']['lidar_sweeps']
    deviations = [float(np.linalg.norm(s['official_loader_translation_difference_m'])) for s in sweeps]
    # The unchanged loader pads only completely absent history; it does not pad 1..8 sweeps.
    audit = dict(status='passed', past_sweeps=len(sweeps), requested_past_sweeps=9,
        padded_keyframe_copies=9 if not sweeps else 0, remove_close=True, future_sweeps_used=0,
        sweep_lag_ms=[(metadata['reference_timestamp_us'] - s['timestamp_us']) / 1000 for s in sweeps],
        loader_translation_difference_m=deviations,
        max_loader_vs_se3_translation_m=max(deviations, default=0.0),
        source='Unchanged tools.infer_centerpoint_mini.frame_info and official cfg.test_pipeline')
    identity = dict(metadata=metadata, pipeline_fingerprint=pipeline_fingerprint)
    return inputs, metadata, audit, fingerprint(identity)


def verify_files(provenance):
    """Re-hash source/config/checkpoint/data bytes, including evaluate-only invocations."""
    for path, expected in provenance['files'].items():
        if file_record(path) != expected:
            raise ValueError('Provenance file changed: ' + path)


def validate_bundle(folder, entry, provenance, origin):
    """Reuse the viewer's strict detection/sensor validator; never import a model."""
    from web.backend.app.config import Settings
    from web.backend.app.detections import validate_detection
    folder = Path(folder)
    settings = Settings(folder.parent, (Path(provenance['data_root']),))
    metadata, report, doc = validate_detection(folder, settings)
    if report.get('fixture') or metadata.get('fixture') or report.get('check_only'):
        raise ValueError('Fixture/check-only output cannot be benchmark prediction')
    if report.get('forward_calls') != 1 or report.get('forward_completed') is not True:
        raise ValueError('Prediction needs a completed real forward')
    for key, value in entry['input_metadata'].items():
        if metadata.get(key) != value:
            raise ValueError('Input metadata mismatch: ' + key)
    sensors = metadata['sensors']
    for sensor in [sensors['keyframes']['LIDAR_TOP'], *sensors['lidar_sweeps']]:
        path = Path(sensor['path']).resolve()
        if not path.is_relative_to(Path(provenance['data_root']).resolve()):
            raise ValueError('Input escaped dataset root')
        if file_record(path) != dict(sha256=sensor['sha256'], bytes=sensor['bytes']):
            raise ValueError('LiDAR/sweep input bytes changed: ' + str(path))
    for key, expected in [('checkpoint', provenance['checkpoint']), ('config', provenance['config'])]:
        if metadata[key] != expected:
            raise ValueError('Model/config mismatch: ' + key)
    if report['tf32'] != provenance['tf32'] or metadata['test_pipeline'] != provenance['test_pipeline']:
        raise ValueError('Preprocessing/precision flags mismatch')
    if any(metadata['framework'][name] != provenance['packages'][name] for name in ('mmdet3d', 'torch')):
        raise ValueError('Framework versions differ from the cached prediction')
    expected_code = provenance['reference_cli_sha256'] if origin == 'external_smoke' else provenance['runtime_sha256']
    if metadata['code']['inference_script_sha256'] != expected_code:
        raise ValueError('Prediction code provenance mismatch')
    if origin == 'benchmark' and metadata['benchmark_provenance'] != dict(
            prediction_fingerprint=fingerprint(provenance), input_fingerprint=entry['input_fingerprint']):
        raise ValueError('Benchmark prediction fingerprint mismatch')
    if origin == 'external_smoke':
        receipt = read_json(folder / 'job_result.json')
        core = {k: v for k, v in receipt.items() if k != 'fingerprint_sha256'}
        if receipt['status'] != 'complete' or receipt['task'] != 'detection' or receipt['execution_mode'] != 'real':
            raise ValueError('Smoke receipt is not a complete real detection')
        if receipt['sample_token'] != entry['sample_token'] or receipt['artifacts'] != file_records(folder) or fingerprint(core) != receipt['fingerprint_sha256']:
            raise ValueError('Smoke publication hash/token mismatch')
    return metadata, report, doc


def committed_prediction(folder, entry, provenance):
    folder = Path(folder)
    commit = read_json(folder / 'prediction_commit.json')
    core = {k: v for k, v in commit.items() if k != 'fingerprint_sha256'}
    if commit['status'] != 'complete' or fingerprint(core) != commit['fingerprint_sha256']:
        raise ValueError('Uncommitted or corrupt prediction')
    if (commit['sample_token'] != entry['sample_token'] or commit['input_fingerprint'] != entry['input_fingerprint']
            or commit['prediction_fingerprint'] != fingerprint(provenance)):
        raise ValueError('Cached prediction identity mismatch')
    names = (*BUNDLE_FILES, 'job_result.json') if commit['origin'] == 'external_smoke' else BUNDLE_FILES
    if commit['artifacts'] != file_records(folder, names):
        raise ValueError('Cached prediction hash mismatch')
    metadata, report, doc = validate_bundle(folder, entry, provenance, commit['origin'])
    return dict(folder=folder, commit=commit, metadata=metadata, report=report, document=doc)


def verify_manifest(manifest):
    if manifest.get('kind') != KIND or manifest['identity_sha256'] != fingerprint(manifest['identity']):
        raise ValueError('Invalid detection benchmark manifest')
    frames = manifest['identity']['frames']
    tokens = [e['sample_token'] for e in frames]
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError('Manifest has duplicate/empty sample tokens')
    if (manifest['identity']['prediction_fingerprint'] != fingerprint(manifest['prediction_provenance'])
            or manifest['identity']['evaluation_protocol_fingerprint'] != fingerprint(manifest['evaluation_protocol'])):
        raise ValueError('Manifest provenance/protocol fingerprint mismatch')
    return frames


def valid_predictions(out, manifest, progress):
    """Missing/failed tokens stay absent. Never synthesize empty result entries."""
    entries = verify_manifest(manifest)
    predictions, missing = {}, []
    provenance = manifest['prediction_provenance']
    for entry in entries:
        state = progress['frames'].get(entry['sample_token'], {})
        try:
            if state.get('status') != 'complete':
                raise ValueError('Frame status: ' + state.get('status', 'pending'))
            folder = (Path(out) / state['prediction_directory']).resolve()
            if not folder.is_relative_to(Path(out).resolve()):
                raise ValueError('Prediction directory escaped benchmark root')
            predictions[entry['sample_token']] = committed_prediction(folder, entry, provenance)
        except (OSError, ValueError, KeyError, TypeError) as error:
            missing.append(dict(sample_token=entry['sample_token'], scene=entry['scene'],
                frame_index=entry['frame_index'], status=state.get('status', 'pending'), error=str(error)))
    return predictions, missing
