"""Verified saved occupancy sequences; input and offline target paths stay separate."""
from __future__ import annotations

from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
import zipfile
import numpy as np

from tools import evaluate_occupancy as occupancy
from tools.forecasting.geometry import anchor_from_source, pull_labels, se3
from tools.forecasting.persistence import HistoryInputs
from tools.forecasting.protocol import (
    HISTORY_OFFSETS_US, HORIZONS_US, TOLERANCE_US, file_record, fingerprint,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return occupancy.json_object(Path(path).read_bytes())


def select_window(scene_frames, anchor, tolerance_us=TOLERANCE_US):
    """Time-first matching; ordinals/links only reject skipped samples afterward."""
    require(bool(scene_frames), 'Empty scene')
    require(all(f['scene'] == anchor['scene'] for f in scene_frames), 'Scene boundary crossed')
    times = [f['reference_timestamp_us'] for f in scene_frames]
    require(times == sorted(set(times)), 'Scene LiDAR times must be strictly increasing')
    require(anchor in scene_frames, 'Unknown anchor')
    cutoff = anchor['reference_timestamp_us']
    reasons, history, future, timing = [], [], [], []
    for role, offsets, selected in [('history', HISTORY_OFFSETS_US, history), ('future', HORIZONS_US, future)]:
        for offset in offsets:
            desired = cutoff + offset
            nearest = min(scene_frames, key=lambda f: (abs(f['reference_timestamp_us'] - desired), f['reference_timestamp_us'], f['sample_token']))
            error = nearest['reference_timestamp_us'] - desired
            boundary = desired < times[0] - tolerance_us or desired > times[-1] + tolerance_us
            matched = not boundary and abs(error) <= tolerance_us
            if boundary:
                reasons.append(role + '_scene_boundary')
            elif not matched:
                reasons.append(role + '_timestamp_tolerance')
            selected.append(nearest if matched else None)
            timing.append(dict(role=role, nominal_offset_us=offset, desired_timestamp_us=desired,
                               nearest_token=nearest['sample_token'], actual_timestamp_us=nearest['reference_timestamp_us'],
                               error_us=error, matched=matched))
    chosen = history + future
    if all(f is not None for f in chosen):
        indices = [f['frame_index'] for f in chosen]
        if any(b - a != 1 for a, b in zip(indices, indices[1:])):
            reasons.append('nonconsecutive_samples')
        intervals = np.diff([f['reference_timestamp_us'] for f in chosen])
        if any(abs(int(dt) - 500_000) > tolerance_us for dt in intervals):
            reasons.append('timestamp_gap')
    return dict(anchor=anchor, history=history, future=future, timing=timing,
                input_cutoff_us=cutoff, reasons=list(dict.fromkeys(reasons)))


def window_asset_reasons(history, future, predictions, gt, cutoff):
    """No metrics are read when deciding window membership."""
    reasons = []
    for frame in history:
        if frame is None:
            continue
        item = predictions.get(frame['sample_token'])
        if item is None or item['status'] == 'missing_prediction':
            reasons.append('missing_prediction')
        elif item['status'] != 'valid_prediction':
            reasons.append('invalid_prediction')
        elif item['latest_used_timestamp_us'] > cutoff:
            reasons.append('input_after_cutoff')
    for frame in future:
        if frame is None:
            continue
        item = gt.get(frame['sample_token'])
        if item is None or item['status'] == 'missing_gt':
            reasons.append('missing_gt')
        elif item['status'] != 'valid_gt':
            reasons.append('invalid_gt')
    return list(dict.fromkeys(reasons))


class SequenceDataset:
    """Load NuScenes once. Never imports inference/model or reads checkpoint bytes."""
    def __init__(self, data_root, benchmark_root, gt_root):
        self.catalog = occupancy.DatasetCatalog(data_root)
        self.root = Path(benchmark_root).resolve()
        self.gt_index = occupancy.GroundTruthIndex(gt_root)
        self.manifest = read_json(self.root / 'manifest.json')
        self.progress = read_json(self.root / 'progress.json')
        self.source_hashes = {n: file_record(self.root / n) for n in ('manifest.json', 'progress.json')}
        self.file_hash_cache = {}
        self.predictions, self.gt, self.dense_gt = {}, {}, {}
        self.labels, self.metadata = {}, {}
        self.geometry_errors = []
        self.frames = sorted((self.frame_record(t, x) for t, x in self.catalog.samples.items() if x['split'] == 'mini_val'),
                             key=lambda f: (f['scene'], f['reference_timestamp_us'], f['sample_token']))
        self._validate_source_manifest()

    def frame_record(self, token, item):
        return dict(sample_token=token, scene=item['scene']['name'], scene_token=item['scene']['token'],
                    frame_index=item['frame_index'], reference_timestamp_us=int(item['lidar']['timestamp']),
                    sample_timestamp_us=int(item['sample']['timestamp']), lidar_sample_data_token=item['lidar']['token'],
                    lidar_filename=item['lidar']['filename'])

    def _validate_source_manifest(self):
        m = self.manifest
        require(m['kind'] == 'occfusion_occupancy_benchmark', 'Not an occupancy benchmark')
        require(fingerprint(m['identity']) == m['identity_sha256'], 'Benchmark manifest identity hash mismatch')
        require(self.progress['manifest_sha256'] == self.source_hashes['manifest.json']['sha256'], 'Progress manifest hash mismatch')
        self.shared = m['prediction_provenance']
        self.shared_fingerprint = fingerprint(self.shared)
        require(self.shared_fingerprint == m['identity']['prediction_provenance'], 'Benchmark prediction provenance fingerprint mismatch')
        require(self.shared['data_root'] == str(self.catalog.root), 'Dataset root differs from original canonical provenance')
        require(m['identity']['split'] == 'mini_val' and m['identity']['scope'] == 'mini_val', 'Need complete mini_val source membership, including failed frames')
        require(sorted(m['identity']['official_scene_names']) == sorted(self.catalog.splits['mini_val']), 'Official split mismatch')
        require(len(m['frames']) == len(self.frames) == m['identity']['expected_frame_count'], 'Source token coverage/count mismatch')
        by_token = {f['sample_token']: f for f in m['frames']}
        require(len(by_token) == len(m['frames']), 'Duplicate source manifest token')
        require(set(by_token) == {f['sample_token'] for f in self.frames}, 'Source manifest token set differs from official split')
        for frame in self.frames:
            declared = by_token[frame['sample_token']]
            require(all(declared[k] == frame[k] for k in ('scene', 'frame_index', 'reference_timestamp_us', 'lidar_sample_data_token', 'lidar_filename')), 'Source frame metadata mismatch')
        self.source_frames = by_token
        self.dataset_tables = {}
        for table, expected in self.shared['dataset_table_hashes'].items():
            record = file_record(self.catalog.root / self.catalog.version / (table + '.json'))
            require(record['sha256'] == expected, 'Dataset metadata hash mismatch: ' + table)
            self.dataset_tables[table] = record
        # Declared checkpoint hash is retained as provenance; checkpoint bytes are never opened.
        require(m['initial_evaluation_protocol'] == occupancy.protocol(), 'Saved GT decoding/evaluation protocol differs from current verified evaluator')

    def global_from_lidar(self, token):
        """Only called for history/anchor by load_history; futures only by load_targets."""
        from nuscenes.utils.geometry_utils import transform_matrix
        from pyquaternion import Quaternion
        sd = self.catalog.samples[token]['lidar']
        cal = self.catalog.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        ego = self.catalog.nusc.get('ego_pose', sd['ego_pose_token'])
        return se3(transform_matrix(ego['translation'], Quaternion(ego['rotation'])) @
                   transform_matrix(cal['translation'], Quaternion(cal['rotation'])))

    def _past_tokens(self, token, count, reference_us, scene):
        found, excluded, visited = [], [], set()
        previous = None
        while token and len(found) < count:
            require(token not in visited, 'Sensor prev cycle')
            visited.add(token)
            sd = self.catalog.nusc.get('sample_data', token)
            require(previous is None or sd['timestamp'] < previous, 'Sensor prev timestamps not decreasing')
            previous = sd['timestamp']
            sample = self.catalog.nusc.get('sample', sd['sample_token'])
            require(sample['scene_token'] == scene, 'Sensor sweep crosses scene')
            if sd['timestamp'] <= reference_us:
                found.append(token)
            else:
                excluded.append(dict(sample_data_token=token, timestamp_us=sd['timestamp']))
            token = sd['prev']
        return found, excluded

    def _sensor_audit(self, metadata):
        from nuscenes.utils.geometry_utils import transform_matrix
        from pyquaternion import Quaternion
        nusc, sensors = self.catalog.nusc, metadata['sensors']
        item = self.catalog.samples[metadata['sample_token']]
        reference = item['lidar']['timestamp']
        require(len(set(sensors['camera_order'])) == 6 and len(set(sensors['radar_order'])) == 5, 'Wrong baseline sensor orders')
        expected_channels = set(sensors['camera_order']) | set(sensors['radar_order']) | {'LIDAR_TOP'}
        require(set(sensors['keyframes']) == expected_channels, 'Keyframe channel set mismatch')
        lidar_tokens, _ = self._past_tokens(item['lidar']['prev'], sensors['requested_lidar_past_sweeps'], reference, item['scene']['token'])
        require([r['sample_data_token'] for r in sensors['lidar_past_sweeps']] == lidar_tokens, 'Saved LiDAR sweep selection differs from timestamp/prev protocol')
        for channel in sensors['radar_order']:
            tokens, excluded = self._past_tokens(item['sample']['data'][channel], sensors['requested_radar_sweeps_per_sensor'], reference, item['scene']['token'])
            require(bool(tokens), 'No valid radar history')
            require(tokens == [r['sample_data_token'] for r in sensors['radar_sweeps'][channel]], 'Radar sweep selection mismatch')
            require(excluded == sensors['excluded_future_radar_records'][channel], 'Excluded radar provenance mismatch')
        used = {sensors['keyframes'][channel]['sample_data_token'] for channel in [*sensors['camera_order'], 'LIDAR_TOP']}
        used.update(lidar_tokens)
        used.update(r['sample_data_token'] for arr in sensors['radar_sweeps'].values() for r in arr)
        records = [*sensors['keyframes'].values(), *sensors['lidar_past_sweeps'],
                   *(r for arr in sensors['radar_sweeps'].values() for r in arr)]
        unique = {r['sample_data_token']: r for r in records}
        require(set(unique) == set(metadata['benchmark_provenance']['sensor_files']), 'Sensor file provenance token set mismatch')
        audit, hashes = [], {}
        inverse_reference = np.linalg.inv(self.global_from_lidar(metadata['sample_token']))
        for token, record in sorted(unique.items()):
            sd = nusc.get('sample_data', token)
            cal = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            ego = nusc.get('ego_pose', sd['ego_pose_token'])
            channel = nusc.get('sensor', cal['sensor_token'])['channel']
            require(record['channel'] == channel, 'Sensor channel mismatch')
            require(nusc.get('sample', sd['sample_token'])['scene_token'] == item['scene']['token'], 'Sensor belongs to a different scene')
            require(record['timestamp_us'] == sd['timestamp'] and record['calibrated_sensor_token'] == cal['token'] and record['ego_pose_token'] == ego['token'], 'Sensor timestamp/calibration/ego token mismatch')
            if token in {r['sample_data_token'] for r in sensors['keyframes'].values()}:
                require(item['sample']['data'][channel] == token, 'Keyframe sample_data token mismatch')
            path = (self.catalog.root / sd['filename']).resolve()
            require(path.is_relative_to(self.catalog.root) and str(path) == record['path'], 'Sensor path/root mismatch')
            if str(path) not in self.file_hash_cache:
                self.file_hash_cache[str(path)] = file_record(path)
            actual = self.file_hash_cache[str(path)]
            require(actual['bytes'] > 0, 'Empty sensor file')
            hashes[token] = dict(path=str(path), **actual)
            require(hashes[token] == metadata['benchmark_provenance']['sensor_files'][token], 'Sensor file hash mismatch: ' + channel)
            sensor_to_ego = transform_matrix(cal['translation'], Quaternion(cal['rotation']))
            ego_to_global = transform_matrix(ego['translation'], Quaternion(ego['rotation']))
            error = max(float(np.max(np.abs(np.asarray(record[key]) - expected))) for key, expected in
                        [('sensor_to_ego', sensor_to_ego), ('ego_to_global', ego_to_global),
                         ('sensor_to_reference_lidar', inverse_reference @ ego_to_global @ sensor_to_ego)])
            require(error <= 1e-8, 'Sensor calibration/pose matrix mismatch')
            if channel.startswith('CAM_'):
                require(record['intrinsic'] == cal['camera_intrinsic'] and record['width'] == sd['width'] and record['height'] == sd['height'], 'Camera intrinsic/dimensions mismatch')
            audit.append(dict(channel=channel, sample_data_token=token, timestamp_us=int(sd['timestamp']),
                              ego_pose_timestamp_us=int(ego['timestamp']), delta_to_frame_lidar_us=int(sd['timestamp'] - reference),
                              used_in_prediction=token in used, path=str(path), file=actual,
                              calibrated_sensor_token=cal['token'], ego_pose_token=ego['token'],
                              max_matrix_abs_difference=error))
        base = {k: metadata[k] for k in ('sample_token', 'timestamp_us', 'reference_timestamp_us', 'timestamp_unit', 'coordinate_system', 'sensors')}
        base['sensors'] = dict(sensors)
        base['sensors'].pop('point_counts', None)
        actual_input = fingerprint(dict(shared_fingerprint=self.shared_fingerprint, metadata=base, sensor_files=hashes))
        require(actual_input == metadata['benchmark_provenance']['input_fingerprint'], 'Preprocessing input fingerprint mismatch')
        return audit

    def audit_predictions(self):
        for frame in self.frames:
            token = frame['sample_token']
            state = self.progress['frames'].get(token, {})
            result = dict(sample_token=token, scene=frame['scene'], frame_index=frame['frame_index'])
            self.predictions[token] = result
            if state.get('status') != 'success':
                result.update(status='missing_prediction', source_failure=state.get('error'), sensors=[])
                continue
            try:
                folder = Path(state['prediction_dir']).resolve()
                require(folder.is_relative_to(self.root / 'frames') and not folder.name.endswith('.pending'), 'Unpublished or external source prediction')
                receipt = read_json(folder / 'prediction_commit.json')
                metadata, report, labels, hashes = occupancy.load_prediction(folder)
                require(receipt['status'] == 'success' and receipt['sample_token'] == token == metadata['sample_token'], 'Prediction token/status mismatch')
                self.catalog.validate_sample(metadata, report)
                hashes['resolved_config.py'] = file_record(folder / 'resolved_config.py')
                require(receipt['artifact_hashes'] == hashes, 'Prediction artifact hash mismatch')
                require(state['prediction_sha256'] == hashes['occupancy.npz']['sha256'], 'Progress prediction hash mismatch')
                require(receipt['shared_fingerprint'] == metadata['benchmark_provenance']['shared_fingerprint'] == self.shared_fingerprint, 'Prediction shared provenance mismatch')
                require(metadata['checkpoint'] == self.shared['checkpoint'] and metadata['config']['resolved_sha256'] == self.shared['resolved_config_sha256'], 'Checkpoint/config identity mismatch')
                require(metadata['config']['sha256'] == self.shared['config_source_sha256'] and hashes['resolved_config.py']['sha256'] == self.shared['resolved_config_sha256'], 'Config hash mismatch')
                require(report['tf32'] == self.shared['tf32'] and metadata['precision'] == self.shared['precision'] and metadata['environment'] == self.shared['environment'], 'Saved baseline precision/environment mismatch')
                require(receipt['input_fingerprint'] == metadata['benchmark_provenance']['input_fingerprint'], 'Receipt input identity mismatch')
                sensors = self._sensor_audit(metadata)
                times = [max(r['timestamp_us'], r['ego_pose_timestamp_us']) for r in sensors if r['used_in_prediction']]
                result.update(status='valid_prediction', source_dir=str(folder), artifact_hashes=hashes,
                              prediction_commit=file_record(folder / 'prediction_commit.json'), input_fingerprint=receipt['input_fingerprint'],
                              latest_used_timestamp_us=max(times), earliest_used_timestamp_us=min(times), sensors=sensors,
                              npz_keys=['labels', 'max_softmax_probability'], logits_or_features_available=False,
                              used_after_local_lidar=[r['sample_data_token'] for r in sensors if r['used_in_prediction'] and max(r['timestamp_us'], r['ego_pose_timestamp_us']) > frame['reference_timestamp_us']])
                self.labels[token], self.metadata[token] = labels, metadata
            except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as error:
                result.update(status='invalid_prediction', error=f'{type(error).__name__}: {error}', sensors=[])
        return dict(frame_count=len(self.frames), status_counts=dict(Counter(p['status'] for p in self.predictions.values())),
                    checkpoint_bytes_read=False, frames=list(self.predictions.values()))

    def audit_gt(self):
        inventories = {}
        for split in ('mini_train', 'mini_val'):
            frames = []
            for token, item in sorted(self.catalog.samples.items(), key=lambda pair: (pair[1]['scene']['name'], pair[1]['frame_index'])):
                if item['split'] != split:
                    continue
                frame = self.frame_record(token, item)
                key, path = self.gt_index.lookup(item['lidar']['filename'])
                frame.update(expected_gt_filename=key, gt_path=str(path) if path else None)
                if path is None:
                    frame['status'] = 'missing_gt'
                else:
                    try:
                        content = path.read_bytes()
                        record = dict(sha256=occupancy.sha256(content), bytes=len(content))
                        if split == 'mini_val':
                            require(record == self.source_frames[token]['gt_initial'], 'GT hash differs from benchmark manifest')
                        dense, stats = occupancy.sparse_to_dense(np.load(BytesIO(content), allow_pickle=False))
                        frame.update(status='valid_gt', file=record, sparse=stats)
                        if split == 'mini_val':
                            self.dense_gt[token] = dense
                    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as error:
                        frame.update(status='invalid_gt', error=f'{type(error).__name__}: {error}')
                self.gt[token] = frame
                frames.append(frame)
            scenes = {}
            for name in self.catalog.splits[split]:
                group = [f for f in frames if f['scene'] == name]
                scenes[name] = dict(frame_count=len(group), status_counts=dict(Counter(f['status'] for f in group)),
                                    missing=[dict(sample_token=f['sample_token'], frame_index=f['frame_index'], expected_gt_filename=f['expected_gt_filename']) for f in group if f['status'] == 'missing_gt'])
            inventories[split] = dict(frame_count=len(frames), status_counts=dict(Counter(f['status'] for f in frames)), scenes=scenes, frames=frames)
        return dict(source=occupancy.GT_SOURCE, root=str(self.gt_index.root), sparse_protocol=occupancy.protocol(),
                    gt_has_no_embedded_token_or_geometry=True, matching='nuScenes API LiDAR basename + .npy',
                    training_performed=False, mini_val_used_as_independent_training_test=False,
                    checkpoint_training_overlap_not_independently_excluded=True, splits=inventories)

    def _reference(self, frame, role):
        if frame is None:
            return None
        token = frame['sample_token']
        result = dict(frame)
        if role == 'history':
            evidence = self.predictions[token]
            result['prediction'] = {k: v for k, v in evidence.items() if k not in ('sensors', 'source_failure')}
        else:
            result['target'] = {k: self.gt[token].get(k) for k in ('status', 'gt_path', 'file', 'expected_gt_filename')}
        return result

    def sequence_manifest(self, protocol_fingerprint):
        by_scene = defaultdict(list)
        for frame in self.frames:
            by_scene[frame['scene']].append(frame)
        windows = []
        for anchor in self.frames:
            window = select_window(by_scene[anchor['scene']], anchor)
            reasons = window['reasons'] + window_asset_reasons(window['history'], window['future'], self.predictions, self.gt, window['input_cutoff_us'])
            reasons = list(dict.fromkeys(reasons))
            windows.append(dict(window_id=anchor['sample_token'], scene=anchor['scene'], anchor=dict(anchor),
                                input_cutoff_us=window['input_cutoff_us'], history=[self._reference(f, 'history') for f in window['history']],
                                future=[self._reference(f, 'future') for f in window['future']], timing=window['timing'],
                                status='excluded' if reasons else 'valid', exclusion_reasons=reasons))
        valid = [w for w in windows if w['status'] == 'valid']
        return dict(kind='rav25_forecasting_sequence_manifest', schema_version=1, protocol_fingerprint=protocol_fingerprint,
                    split='mini_val', official_scenes=sorted(self.catalog.splits['mini_val']),
                    candidate_window_count=len(windows), valid_window_count=len(valid), excluded_window_count=len(windows) - len(valid),
                    primary_exclusion_counts=dict(Counter(w['exclusion_reasons'][0] for w in windows if w['exclusion_reasons'])),
                    any_exclusion_counts=dict(Counter(r for w in windows for r in w['exclusion_reasons'])),
                    reason_count_note='Primary counts sum to excluded windows; any-reason counts overlap.',
                    scene_counts={s:dict(candidate=len(group), valid=sum(w['status']=='valid' for w in windows if w['scene']==s)) for s,group in by_scene.items()},
                    source_benchmark=dict(root=str(self.root), hashes=self.source_hashes, shared_fingerprint=self.shared_fingerprint),
                    windows=windows)

    def load_history(self, window):
        require(window['status'] == 'valid', 'Cannot load excluded window')
        anchor = window['anchor']['sample_token']
        global_anchor = self.global_from_lidar(anchor)
        labels, valid, bounds_masks, known_masks, times, used_times, transforms = [], [], [], [], [], [], []
        for frame in window['history']:
            token = frame['sample_token']
            evidence = self.predictions[token]
            require(evidence['status'] == 'valid_prediction', 'History prediction unavailable')
            transform = np.eye(4) if token == anchor else anchor_from_source(global_anchor, self.global_from_lidar(token))
            warped = pull_labels(self.labels[token], transform)
            labels.append(warped.labels); valid.append(warped.valid)
            bounds_masks.append(warped.in_bounds); known_masks.append(warped.known)
            times.append(frame['reference_timestamp_us'])
            used_times.append(tuple(max(r['timestamp_us'], r['ego_pose_timestamp_us']) for r in evidence['sensors'] if r['used_in_prediction']))
            transforms.append(transform)
        inputs = HistoryInputs(np.stack(labels), np.stack(valid), tuple(times), tuple(used_times),
                               window['anchor']['reference_timestamp_us'], window['input_cutoff_us'])
        inputs.validate()
        return inputs, dict(in_bounds=np.stack(bounds_masks), known=np.stack(known_masks), anchor_from_history=np.stack(transforms))

    def load_targets(self, window):
        """Offline only: future GT/ego poses cannot flow into HistoryInputs/predict."""
        global_anchor = self.global_from_lidar(window['anchor']['sample_token'])
        targets, matrices = [], []
        for frame in window['future']:
            token = frame['sample_token']
            require(self.gt[token]['status'] == 'valid_gt', 'Future GT unavailable')
            transform = anchor_from_source(global_anchor, self.global_from_lidar(token))
            targets.append(pull_labels(self.dense_gt[token], transform))
            matrices.append(transform)
        return targets, np.stack(matrices)
