"""Read saved evaluation evidence from configured roots; never run evaluation."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import re
from threading import RLock

import numpy as np

from .store import contained, InvalidArtifact, OutsideRoot


ARTIFACTS = ('occupancy.npz', 'metadata.json', 'report.json', 'bev.png')
MAX_BYTES = 8 * 1024 * 1024


def digest(content):
    return hashlib.sha256(content).hexdigest()


def decode(content):
    def reject(value):
        raise InvalidArtifact(f'Non-finite evaluation JSON: {value}')
    value = json.loads(content, parse_constant=reject)
    if not isinstance(value, dict):
        raise InvalidArtifact('Evaluation must be a JSON object')
    return value


def ratio(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or not 0 <= value <= 1:
        raise InvalidArtifact('Evaluation metrics must be finite ratios in [0,1]')
    return float(value)


def validate_metrics(metrics, classes, max_voxels=16_000_000):
    """Check the saved metrics against saved count evidence, without loading GT."""
    count = len(classes)
    matrix = np.asarray(metrics['confusion_matrix'])
    if matrix.shape != (count, count) or matrix.dtype.kind not in 'iu' or np.any(matrix < 0) or np.any(matrix > max_voxels):
        raise InvalidArtifact('Invalid evaluation confusion matrix')
    matrix = matrix.astype(np.int64)
    if matrix.sum() > max_voxels:
        raise InvalidArtifact('Evaluation voxel counts exceed the declared frame scope')
    gt, pred = matrix.sum(1), matrix.sum(0)
    intersection = int(matrix[1:, 1:].sum())
    union = int(gt[1:].sum() + pred[1:].sum() - intersection)
    expected = intersection / union if union else 0.0
    if not np.isclose(ratio(metrics['occupancy_iou']), expected, rtol=0, atol=1e-12):
        raise InvalidArtifact('Occupancy IoU disagrees with count evidence')
    if metrics['occupancy'] != dict(intersection=intersection, union=union, gt_occupied=int(gt[1:].sum()), prediction_occupied=int(pred[1:].sum())):
        raise InvalidArtifact('Occupancy intersection/union disagree with confusion matrix')
    if metrics['valid_voxel_count'] != int(matrix.sum()) or metrics['semantic_mean_class_count'] != count-1:
        raise InvalidArtifact('Evaluation voxel/class count mismatch')
    entries = metrics['per_class_semantic']
    if len(entries) != count-1:
        raise InvalidArtifact('Evaluation must contain every semantic class')
    values = []
    for label, entry in enumerate(entries, 1):
        tp = int(matrix[label, label])
        total = int(gt[label] + pred[label] - tp)
        value = tp/total if total else 0.0
        if entry['label'] != label or entry['name'] != classes[label] or entry['intersection'] != tp or entry['union'] != total or entry['gt_count'] != int(gt[label]) or entry['prediction_count'] != int(pred[label]):
            raise InvalidArtifact('Semantic class/count evidence mismatch')
        if not np.isclose(ratio(entry['iou']), value, rtol=0, atol=1e-12):
            raise InvalidArtifact('Semantic IoU disagrees with count evidence')
        values.append(value)
    if not np.isclose(ratio(metrics['semantic_miou']), np.mean(values), rtol=0, atol=1e-12):
        raise InvalidArtifact('Semantic mean disagrees with per-class evidence')


class EvaluationStore:
    def __init__(self, settings):
        self.roots = tuple(Path(root).resolve() for root in settings.evaluation_directories)
        self.cache = OrderedDict()
        self.lock = RLock()

    def paths(self):
        result = {}
        for root in self.roots:
            if not root.is_dir():
                continue
            for path in sorted(root.rglob('evaluation.json')):
                if any(part.startswith('.') or part.endswith('.pending') for part in path.relative_to(root).parts[:-1]):
                    continue
                # ID names a configured logical location, independent of its content.
                identifier = digest(str(root).encode() + b'\0' + str(path.relative_to(root)).encode())
                result[identifier] = (path, root)
        return result

    def read(self, identifier, paths=None):
        if not re.fullmatch('[0-9a-f]{64}', identifier):
            raise InvalidArtifact('Invalid evaluation_id')
        paths = self.paths() if paths is None else paths
        if identifier not in paths:
            raise FileNotFoundError('Evaluation is absent from configured evaluation roots')
        path, root = paths[identifier]
        path = contained(path, root)
        if path.stat().st_size > MAX_BYTES:
            raise InvalidArtifact('Evaluation exceeds 8 MiB limit')
        content = path.read_bytes()
        if len(content) > MAX_BYTES:
            raise InvalidArtifact('Evaluation exceeds 8 MiB limit')
        fingerprint = digest(content)
        with self.lock:
            if fingerprint not in self.cache:
                self.cache[fingerprint] = decode(content)
                if len(self.cache) > 16:
                    self.cache.popitem(last=False)
            document = self.cache[fingerprint]
        return dict(id=identifier, label=path.parent.name, sha256=fingerprint, bytes=len(content),
                    content=content, document=document)

    @staticmethod
    def relevant(document, metadata, prediction_hash, run_id):
        frames = document.get('frames', [])
        if not isinstance(frames, list):
            return False
        return any(frame.get('sample_token') == metadata['sample_token'] or
                   frame.get('prediction_sha256') == prediction_hash or
                   Path(frame.get('run_dir', '')).name == run_id
                   for frame in frames if isinstance(frame, dict))

    def match(self, item, metadata, files):
        document = item['document']
        frames = document.get('frames', [])
        if not isinstance(frames, list) or not all(isinstance(frame, dict) for frame in frames):
            raise InvalidArtifact('Evaluation frames must be a list of frame objects')
        frame_matches = [f for f in frames if f.get('sample_token') == metadata['sample_token']]
        if len(frame_matches) != 1:
            return dict(status='token_mismatch', reason='Sample token trong evaluation không khớp metadata/report.')
        frame = frame_matches[0]
        hashes = {name: digest(content) for name, content in files.items()}
        if frame.get('prediction_sha256') != hashes['occupancy.npz']:
            return dict(status='hash_mismatch', reason='SHA-256 prediction trong evaluation không khớp occupancy.npz hiện tại.')
        if any(frame.get('artifact_hashes', {}).get(name, {}).get('sha256') != hashes[name] for name in ARTIFACTS):
            return dict(status='hash_mismatch', reason='Hash bộ artifact nguồn của evaluation không khớp bộ inference hiện tại.')
        if frame.get('status') == 'missing_gt':
            return dict(status='missing_gt', reason='Chưa có GT cho sample này; evaluation không có metric.')
        if document.get('status') != 'success' or frame.get('status') != 'evaluated':
            return dict(status='unavailable', reason='Evaluation chưa đánh giá thành công cặp prediction/GT này.')
        if document.get('frame_count_evaluated') != 1 or document.get('frame_count_requested') != 1 or len(frames) != 1:
            return dict(status='unsupported_scope', reason='Viewer hiện ghép evaluation một frame; không dùng pooled metric làm metric của frame.')
        if document.get('schema_version') != 1 or document.get('all_requested_frames_evaluated') is not True or document.get('sample_tokens') != [metadata['sample_token']]:
            raise InvalidArtifact('Invalid evaluation schema/sample list/completeness')
        if document['dataset']['version'] != metadata['dataset']['version']:
            raise InvalidArtifact('Evaluation dataset version mismatch')
        if frame['scene'] != metadata['scene']['name'] or frame['frame_index'] != metadata['frame_index'] or frame['timestamp_us'] != metadata['timestamp_us'] or frame['reference_timestamp_us'] != metadata['reference_timestamp_us']:
            raise InvalidArtifact('Evaluation scene/frame/timestamp mismatch')
        record = metadata['sensors']['keyframes']['LIDAR_TOP']
        if frame['lidar_sample_data_token'] != record['sample_data_token'] or Path(frame['lidar_filename']).name != Path(record['path']).name:
            raise InvalidArtifact('Evaluation LiDAR sample mismatch')
        split = document['split']
        if frame['split'] != split or metadata['dataset'].get('split', split) != split or metadata['scene']['name'] not in document['official_split_scene_names']:
            raise InvalidArtifact('Evaluation split mismatch')
        protocol = document['protocol']
        if protocol['name'] != 'surroundocc-200x200x16-occfusion-v1' or protocol['semantic_mean_class_ids'] != list(range(1, 17)) or protocol['semantic_mean_class_count'] != 16 or protocol['union_zero_iou'] != 0 or protocol['absent_classes_included_in_mean'] is not True:
            raise InvalidArtifact('Unsupported evaluation protocol/mean/union-zero convention')
        if protocol['grid_shape'] != [200, 200, 16] or protocol['bounds'] != [-50, -50, -5, 50, 50, 3] or protocol['voxel_size'] != [.5, .5, .5] or protocol['grid_axis_order'] != ['x', 'y', 'z'] or len(protocol['class_names']) != 17:
            raise InvalidArtifact('Unsupported SurroundOcc geometry/classes')
        for key in ('bounds', 'voxel_size', 'grid_shape', 'grid_axis_order', 'voxel_center', 'class_names'):
            if protocol[key] != metadata[key]:
                raise InvalidArtifact(f'Evaluation {key} mismatch')
        if protocol['ignore_label'] != 255 or protocol['unlisted_voxel_label'] != 0 or protocol['sparse_zero_label_becomes'] != 255 or frame['gt_status'] != 'has_gt' or not re.fullmatch('[0-9a-f]{64}', frame['gt_sha256']):
            raise InvalidArtifact('Invalid GT protocol/hash evidence')
        validate_metrics(frame['metrics'], metadata['class_names'])
        if frame['metrics'] != document['metrics']:
            raise InvalidArtifact('Single-frame metrics disagree with report aggregate')
        summary = dict(
            id=item['id'], label=item['label'], sha256=item['sha256'], bytes=item['bytes'],
            metric_unit='ratio_0_1', split=split, scene=frame['scene'], frame_index=frame['frame_index'],
            sample_token=metadata['sample_token'], prediction_sha256=hashes['occupancy.npz'],
            frame_count_requested=document['frame_count_requested'], frame_count_evaluated=document['frame_count_evaluated'],
            metrics_scope=document['metrics_scope'], protocol_name=protocol['name'],
            semantic_mean_class_count=protocol['semantic_mean_class_count'], union_zero_iou=protocol['union_zero_iou'],
            absent_classes_included_in_mean=protocol['absent_classes_included_in_mean'],
            ground_truth_sha256=frame['gt_sha256'], metrics=frame['metrics'],
            reference_check=frame.get('reference_check'),
        )
        return dict(status='matched', reason='Token và SHA-256 của prediction/bộ artifact đã khớp.', matched=summary)

    def describe(self, run_id, files):
        metadata = decode(files['metadata.json'])
        report = decode(files['report.json'])
        if metadata.get('sample_token') != report.get('sample_token'):
            return dict(status='token_mismatch', reason='Metadata/report sample token mismatch', matched=None, candidates=[])
        candidates, matches = [], []
        paths = self.paths()
        for identifier in paths:
            try:
                item = self.read(identifier, paths)
                if not self.relevant(item['document'], metadata, digest(files['occupancy.npz']), run_id):
                    continue
                try:
                    status = self.match(item, metadata, files)
                except (KeyError, ValueError, TypeError, IndexError) as error:
                    status = dict(status='invalid', reason=str(error))
                candidates.append(dict(id=identifier, label=item['label'], sha256=item['sha256'],
                                       status=status['status'], reason=status['reason']))
                if status['status'] == 'matched':
                    matches.append(status['matched'])
            except (OSError, ValueError, TypeError, OutsideRoot):
                # Malformed/unassociated reports have no authority to name a run.
                continue
        unique = {item['sha256']: item for item in matches}
        if len(unique) == 1:
            return dict(status='matched', reason='Evaluation đã ghép đúng sample và nội dung prediction.',
                        matched=next(iter(unique.values())), candidates=candidates)
        if unique:
            return dict(status='ambiguous', reason='Nhiều evaluation khác nội dung cùng khớp; chọn một nguồn trong cấu hình.', matched=None, candidates=candidates)
        status = candidates[0]['status'] if len(candidates) == 1 else 'unmatched'
        reason = candidates[0]['reason'] if len(candidates) == 1 else 'Chưa có evaluation ghép đúng token và SHA-256 của prediction.'
        return dict(status=status, reason=reason, matched=None, candidates=candidates)

    def require_match(self, identifier, metadata, files):
        item = self.read(identifier)
        status = self.match(item, metadata, files)
        if status['status'] != 'matched':
            raise InvalidArtifact(status['reason'])
        return item, status['matched']
