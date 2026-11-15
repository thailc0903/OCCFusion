"""Validate/cache the schema actually exported by infer_nuscenes_mini.py.

No model import, subprocess, checkpoint load, or inference entry point exists.
The browser only selects run IDs. All filenames are fixed or from allowlisted
nuScenes tables, never from a browser path or an unchecked metadata path.
"""

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
from threading import RLock

import numpy as np

from .config import Settings


class InvalidArtifact(ValueError):
    pass


class OutsideRoot(PermissionError):
    pass


def contained(path: Path, root: Path) -> Path:
    path, root = path.resolve(), root.resolve()
    if not path.is_relative_to(root):
        raise OutsideRoot('File resolves outside its configured data directory')
    return path


def read_json(path):
    def reject_constant(value):
        raise InvalidArtifact(f'{path.name}: non-finite JSON constant {value}')
    try:
        value = json.loads(path.read_text(encoding='utf-8'), parse_constant=reject_constant)
    except (ValueError, OSError) as error:
        raise InvalidArtifact(f'Cannot read {path.name}: {error}') from error
    if not isinstance(value, dict):
        raise InvalidArtifact(f'{path.name}: expected a JSON object')
    return value


def geometry(metadata):
    """Return array-axis order, shape, and world XYZ bounds/voxel size."""
    try:
        axes = tuple(metadata['grid_axis_order'])
        raw_shape = np.asarray(metadata['grid_shape'])
        bounds = np.asarray(metadata['bounds'], dtype=np.float64)
        size = np.asarray(metadata['voxel_size'], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidArtifact(f'Incomplete voxel geometry: {error}') from error
    if len(axes) != 3 or set(axes) != {'x', 'y', 'z'}:
        raise InvalidArtifact('grid_axis_order must be a permutation of x,y,z')
    if raw_shape.shape != (3,) or raw_shape.dtype.kind not in 'iu' or np.any(raw_shape <= 0):
        raise InvalidArtifact('grid_shape must contain three positive integers')
    if bounds.shape != (6,) or size.shape != (3,) or not np.isfinite(bounds).all() or not np.isfinite(size).all():
        raise InvalidArtifact('bounds/voxel_size must be finite XYZ vectors')
    if np.any(size <= 0) or np.any(bounds[3:] <= bounds[:3]):
        raise InvalidArtifact('Invalid spatial bounds or voxel size')
    xyz_shape = raw_shape[[axes.index(axis) for axis in ('x', 'y', 'z')]]
    if not np.allclose(xyz_shape * size, bounds[3:] - bounds[:3], rtol=0, atol=1e-6):
        raise InvalidArtifact('grid_shape, grid_axis_order, bounds and voxel_size disagree')
    if int(np.prod(raw_shape)) > 16_000_000:
        raise InvalidArtifact('Grid exceeds the local viewer limit of 16 million cells')
    if metadata.get('voxel_center') != 'bounds_min + (index + 0.5) * voxel_size':
        raise InvalidArtifact('Unsupported or absent voxel-center convention')
    return axes, tuple(int(n) for n in raw_shape), bounds, size


def voxel_centers(indices, metadata):
    axes, shape, bounds, size = geometry(metadata)
    indices = np.asarray(indices)
    if indices.ndim != 2 or indices.shape[1] != 3 or indices.dtype.kind not in 'iu':
        raise InvalidArtifact('Voxel indices must be an integer [N,3] array')
    if np.any(indices < 0) or np.any(indices >= np.asarray(shape)):
        raise InvalidArtifact('Voxel index outside grid_shape')
    xyz = indices[:, [axes.index(axis) for axis in ('x', 'y', 'z')]]
    return (bounds[:3] + (xyz + 0.5) * size).astype('<f4')


def quaternion_pose(rotation, translation):
    q = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if q.shape != (4,) or translation.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(translation).all() or np.linalg.norm(q) < 1e-12:
        raise InvalidArtifact('Invalid nuScenes calibration/ego pose')
    w, x, y, z = q / np.linalg.norm(q)
    result = np.eye(4)
    result[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]
    result[:3, 3] = translation
    return result


class SensorCatalog:
    def __init__(self, settings):
        self.settings = settings
        self.indexes = {}

    def camera_frame(self, metadata):
        from .camera_sensors import camera_frame
        return camera_frame(self, metadata)

    def tables(self, root, version):
        if not re.fullmatch(r'v[0-9]+\.[0-9]+-[a-z]+', version):
            raise InvalidArtifact('Unsupported dataset version identifier')
        names = ('sample', 'sample_data', 'scene', 'calibrated_sensor', 'sensor', 'ego_pose')
        paths = [contained(root / version / f'{name}.json', root) for name in names]
        signature = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
        key = (root, version)
        if key not in self.indexes or self.indexes[key][0] != signature:
            tables = {}
            for name, path in zip(names, paths):
                rows = json.loads(path.read_text(encoding='utf-8'))
                tables[name] = {row['token']: row for row in rows}
            self.indexes[key] = (signature, tables)
        return self.indexes[key][1]

    def lidar(self, metadata, bounds=None):
        detector_bounds = bounds is not None
        token = metadata['sample_token']
        record = metadata['sensors']['keyframes']['LIDAR_TOP']
        version = metadata['dataset']['version']
        candidates = []
        for root in self.settings.dataset_roots:
            if not (root / version / 'sample.json').is_file():
                continue
            tables = self.tables(root, version)
            if token in tables['sample']:
                candidates.append((root, tables))
        if not candidates:
            raise InvalidArtifact('Sample token not found in any configured nuScenes dataset')
        declared_root = Path(metadata['dataset']['root']).resolve()
        chosen = [pair for pair in candidates if pair[0].resolve() == declared_root]
        if len(candidates) > 1 and len(chosen) != 1:
            raise InvalidArtifact('Sample is ambiguous across configured dataset roots')
        root, tables = chosen[0] if chosen else candidates[0]
        sample = tables['sample'][token]
        sd = tables['sample_data'].get(record['sample_data_token'])
        if sd is None or sd['sample_token'] != token or not sd['is_key_frame']:
            raise InvalidArtifact('LiDAR sample_data token does not belong to this keyframe sample')
        cal = tables['calibrated_sensor'][sd['calibrated_sensor_token']]
        ego = tables['ego_pose'][sd['ego_pose_token']]
        if tables['sensor'][cal['sensor_token']]['channel'] != 'LIDAR_TOP' or record['channel'] != 'LIDAR_TOP':
            raise InvalidArtifact('Sensor record is not LIDAR_TOP')
        scene = tables['scene'][sample['scene_token']]
        if scene['token'] != metadata['scene']['token'] or scene['name'] != metadata['scene']['name']:
            raise InvalidArtifact('Scene differs between metadata and nuScenes sample')
        cursor = tables['sample'][scene['first_sample_token']]
        frame = metadata['frame_index']
        if not isinstance(frame, int) or not 0 <= frame < scene['nbr_samples']:
            raise InvalidArtifact('frame_index outside the declared nuScenes scene')
        for _ in range(frame):
            cursor = tables['sample'][cursor['next']]
        if cursor['token'] != token or metadata['timestamp_us'] != sample['timestamp']:
            raise InvalidArtifact('Frame index/timestamp does not match the nuScenes sample')
        if sd['timestamp'] != metadata['reference_timestamp_us'] or sd['timestamp'] != record['timestamp_us']:
            raise InvalidArtifact('LiDAR reference timestamp mismatch')
        if record['calibrated_sensor_token'] != cal['token'] or record['ego_pose_token'] != ego['token']:
            raise InvalidArtifact('LiDAR calibration/ego pose token mismatch')
        sensor_to_ego = quaternion_pose(cal['rotation'], cal['translation'])
        ego_to_global = quaternion_pose(ego['rotation'], ego['translation'])
        for actual, expected in (
            (record['sensor_to_ego'], sensor_to_ego),
            (record['ego_to_global'], ego_to_global),
            (record['sensor_to_reference_lidar'], np.eye(4)),
            (metadata['coordinate_system']['lidar_to_ego'], sensor_to_ego),
            (metadata['coordinate_system']['lidar_to_global'], ego_to_global @ sensor_to_ego),
        ):
            matrix = np.asarray(actual)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix, expected, rtol=0, atol=1e-6):
                raise InvalidArtifact('LiDAR transform disagrees with nuScenes calibration/ego pose')
        try:
            declared_relative = PurePosixPath(record['path']).relative_to(PurePosixPath(metadata['dataset']['root']))
        except ValueError as error:
            raise InvalidArtifact('Metadata LiDAR path is outside its declared dataset root') from error
        if declared_relative.as_posix() != sd['filename']:
            raise InvalidArtifact('Metadata LiDAR filename differs from sample_data')
        # Open ONLY the catalog filename under an allowlisted root. The absolute
        # metadata path is compared above but is never opened.
        path = contained(root / sd['filename'], root)
        if not path.is_file() or path.stat().st_size == 0 or path.stat().st_size % 20:
            raise InvalidArtifact('LiDAR keyframe must be a nonempty float32 [N,5] binary')
        points = np.fromfile(path, dtype='<f4').reshape(-1, 5)
        if not np.isfinite(points).all():
            raise InvalidArtifact('LiDAR keyframe contains non-finite values')
        if bounds is None:
            _, _, bounds, _ = geometry(metadata)
        else:
            bounds = np.asarray(bounds,dtype=np.float64)
            if bounds.shape!=(6,) or not np.isfinite(bounds).all() or np.any(bounds[3:]<=bounds[:3]):raise InvalidArtifact('Invalid detector bounds')
        inside = ((points[:, :3] > bounds[:3]) & (points[:, :3] < bounds[3:])).all(axis=1)
        visible = np.ascontiguousarray(points[inside, :4], dtype='<f4')
        return visible, dict(
            sample_token=token, sample_data_token=sd['token'],
            timestamp_us=sd['timestamp'], source='LiDAR keyframe only',
            original_point_count=len(points), point_count=len(visible),
            range_filter='strictly inside detector XYZ bounds' if detector_bounds else 'strictly inside occupancy XYZ bounds',
            coordinate_system=metadata['coordinate_system']['name'],
            verified_against='nuScenes sample, sample_data, scene, calibration and ego pose JSON',
            fields=['x', 'y', 'z', 'intensity'], stride=4, dtype='float32-le',
        )


@dataclass
class LoadedRun:
    detail: dict
    occupancy: bytes | None = None
    lidar: bytes | None = None
    bev: bytes | None = None


class RunStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.catalog = SensorCatalog(settings)
        self.cache = OrderedDict()
        self.lock = RLock()
        self.hits = 0
        self.misses = 0

    def folder(self, run_id):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', run_id) or run_id in ('.', '..'):
            raise OutsideRoot('Invalid run_id')
        folder = contained(self.settings.runs_root / run_id, self.settings.runs_root)
        if not folder.is_dir():
            raise FileNotFoundError('Run not found')
        if run_id.startswith('infer_job_'):
            from .job_results import verify_publication
            verify_publication(folder)
        return folder

    def artifact(self, folder, name):
        return contained(folder / name, folder)

    def list_runs(self):
        items = []
        if not self.settings.runs_root.is_dir():
            return items
        for folder in sorted(self.settings.runs_root.iterdir()):
            if not folder.is_dir() or folder.name.startswith('.'):
                continue
            try:
                folder = self.folder(folder.name)
                if not self.artifact(folder, 'report.json').is_file():
                    continue
                report = read_json(self.artifact(folder, 'report.json'))
                if report.get('task')=='detection':continue
                status = report.get('status', 'unknown')
                check = report.get('check_only') is True or status == 'check_only_passed'
                prediction = status == 'success' and not check and all(
                    self.artifact(folder, name).is_file() for name in ('occupancy.npz', 'metadata.json', 'bev.png'))
                items.append(dict(
                    run_id=folder.name, status=status,
                    kind='check_only' if check else 'prediction' if prediction else 'failed',
                    prediction_available=prediction, sample_token=report.get('sample_token'),
                    scene_name=report.get('scene_name'), frame_index=report.get('frame_index'),
                ))
            except (ValueError, OSError, OutsideRoot):
                # Never follow an escaped symlink or let one broken run hide others.
                continue
        return items

    def get(self, run_id):
        with self.lock:
            folder = self.folder(run_id)
            paths = [self.artifact(folder, name) for name in ('report.json', 'metadata.json', 'occupancy.npz', 'bev.png')]
            signature = tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.is_file() else None for p in paths)
            if run_id in self.cache and self.cache[run_id][0] == signature:
                self.hits += 1
                self.cache.move_to_end(run_id)
                return self.cache[run_id][1]
            self.misses += 1
            loaded = self.load(run_id, folder)
            self.cache[run_id] = (signature, loaded)
            self.cache.move_to_end(run_id)
            while len(self.cache) > self.settings.cache_entries:
                self.cache.popitem(last=False)
            return loaded

    def load(self, run_id, folder):
        report = read_json(self.artifact(folder, 'report.json'))
        metadata_path = self.artifact(folder, 'metadata.json')
        meta = read_json(metadata_path) if metadata_path.is_file() else None
        status = report.get('status', 'unknown')
        check = report.get('check_only') is True or status == 'check_only_passed'
        if meta:
            for key in ('sample_token', 'status', 'precision', 'frame_index'):
                if meta.get(key) != report.get(key):
                    raise InvalidArtifact(f'{key} differs between metadata.json and report.json')
            if meta['scene']['name'] != report['scene_name'] or meta['scene']['index'] != report['scene_index']:
                raise InvalidArtifact('Scene differs between metadata.json and report.json')
        times = report.get('timings_s', {})
        total, pipeline = times.get('process_until_report'), times.get('pipeline_total')
        startup = max(0, total - pipeline) if total is not None and pipeline is not None else None
        detail = dict(
            run_id=run_id, status=status,
            kind='check_only' if check else 'prediction' if status == 'success' else 'failed',
            prediction_available=False, replay=True, inference_enabled=False,
            scene=meta.get('scene') if meta else dict(name=report.get('scene_name')),
            frame_index=report.get('frame_index'), sample_token=report.get('sample_token'),
            precision=report.get('precision'), checkpoint=meta.get('checkpoint') if meta else None,
            timings_s=times, startup_derived_s=startup,
            timing_note=report.get('timing_note'), allocator=report.get('vram', {}),
            allocator_note='PyTorch allocator statistics; not independently measured physical VRAM residency.',
            metrics=report.get('metrics', {}), error=report.get('error'),
            warnings=report.get('warnings', []), started_at_utc=report.get('started_at_utc'),
        )
        if check or status != 'success':
            return LoadedRun(detail)
        if meta is None:
            raise InvalidArtifact('Successful run has no metadata.json')
        if meta.get('result_source') != 'prediction':
            raise InvalidArtifact('Expected saved prediction artifacts')
        if meta['coordinate_system'].get('name') != 'LIDAR_TOP at reference_timestamp_us' or meta['coordinate_system'].get('unit') != 'metres':
            raise InvalidArtifact('This viewer supports native reference LIDAR_TOP coordinates in metres only')
        axes, shape, bounds, size = geometry(meta)
        classes, palette = meta['class_names'], np.asarray(meta['palette'])
        free = meta['free_label']
        if not isinstance(free, int) or not 0 <= free < len(classes) or classes[free] != 'free':
            raise InvalidArtifact('Background label does not match the metadata class map')
        if palette.shape != (len(classes), 3) or not np.isfinite(palette).all() or np.any(palette < 0) or np.any(palette > 255):
            raise InvalidArtifact('Invalid semantic palette')
        with np.load(self.artifact(folder, 'occupancy.npz'), allow_pickle=False) as archive:
            if set(archive.files) != {'labels', 'max_softmax_probability'}:
                raise InvalidArtifact('Expected NPZ keys labels and max_softmax_probability')
            labels, probability = archive['labels'], archive['max_softmax_probability']
        if labels.shape != shape or probability.shape != shape:
            raise InvalidArtifact('NPZ array shape differs from metadata grid_shape')
        if labels.dtype.kind not in 'iu' or labels.min() < 0 or labels.max() >= len(classes):
            raise InvalidArtifact('Predicted labels outside the metadata class map')
        if probability.dtype.kind != 'f' or not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
            raise InvalidArtifact('Probability must be finite and in [0,1]')
        prediction = report.get('prediction', {})
        if prediction.get('logits_shape') != [1, *shape, len(classes)]:
            raise InvalidArtifact('report logits_shape differs from the NPZ grid/class map')
        histogram = {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
        if histogram != prediction.get('predicted_voxel_counts'):
            raise InvalidArtifact('NPZ label counts differ from report.json')
        indices = np.argwhere(labels != free)
        chosen = tuple(indices.T)
        positions = voxel_centers(indices, meta)
        sparse = np.empty((len(indices), 5), dtype='<f4')
        sparse[:, :3], sparse[:, 3], sparse[:, 4] = positions, labels[chosen], probability[chosen]
        lidar, lidar_info = self.catalog.lidar(meta)
        bev_path = self.artifact(folder, 'bev.png')
        bev = bev_path.read_bytes()
        if not bev.startswith(b'\x89PNG\r\n\x1a\n'):
            raise InvalidArtifact('bev.png is not a PNG image')
        detail.update(
            prediction_available=True, bounds=bounds.tolist(), grid_shape=list(shape),
            grid_axis_order=list(axes), voxel_size=size.tolist(),
            coordinate_system=meta['coordinate_system'], free_label=free,
            classes=[dict(id=i, name=name, color=[int(c) for c in palette[i]],
                          count=histogram.get(str(i), 0), background=i == free)
                     for i, name in enumerate(classes)],
            occupancy=dict(url=f'/api/runs/{run_id}/occupancy', point_count=len(indices),
                           fields=['x', 'y', 'z', 'label', 'max_softmax_probability'],
                           stride=5, dtype='float32-le', background_omitted=True),
            lidar=dict(**lidar_info, url=f'/api/runs/{run_id}/lidar'),
            bev_url=f'/api/runs/{run_id}/bev', bev=meta['bev'],
            confidence=dict(min=float(probability.min()), max=float(probability.max()),
                            note='Max softmax probability; confidence chưa hiệu chuẩn.'),
            validation=dict(
                shape=True, label_range=True, finite_probability=True,
                metadata_report_sample_token=True, sensor_sample_token=True,
                sensor_calibration=True, report_label_counts=True,
                npz_token='NPZ has no embedded sample token; shape and class counts are '
                          'cross-checked against report. This is not cryptographic provenance.',
            ),
        )
        return LoadedRun(detail, sparse.tobytes(), lidar.tobytes(), bev)
