"""Unit fixtures are synthetic and isolated in pytest tmp_path, never run data.

The real saved run is additionally checked by HTTP/browser acceptance tests.
"""
import base64
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from web.backend.tests.auth_support import engineer_client

from web.backend.app.config import Settings
from web.backend.app.main import create_app
from web.backend.app.store import InvalidArtifact, OutsideRoot, RunStore, voxel_centers


REPO = Path(__file__).resolve().parents[3]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def fixture(tmp_path):
    root, data = tmp_path / 'runs', tmp_path / 'dataset'
    folder = root / 'prediction'
    folder.mkdir(parents=True)
    identity = np.eye(4).tolist()
    filename = 'samples/LIDAR_TOP/test.pcd.bin'
    binary = data / filename
    binary.parent.mkdir(parents=True)
    # Unequal coordinates and a negative Z detect axis swaps and sign flips.
    np.array([[0.5, 1.5, -0.5, 12, 0], [20, 20, 0, 2, 0]], dtype='<f4').tofile(binary)
    tables = {
        'sample': [dict(token='test-sample', scene_token='test-scene', timestamp=123456, next='')],
        'sample_data': [dict(token='test-lidar', sample_token='test-sample', is_key_frame=True,
                             calibrated_sensor_token='test-cal', ego_pose_token='test-ego',
                             timestamp=123456, filename=filename)],
        'scene': [dict(token='test-scene', name='fixture-scene', first_sample_token='test-sample', nbr_samples=1)],
        'calibrated_sensor': [dict(token='test-cal', sensor_token='test-sensor', rotation=[1, 0, 0, 0], translation=[0, 0, 0])],
        'sensor': [dict(token='test-sensor', channel='LIDAR_TOP')],
        'ego_pose': [dict(token='test-ego', rotation=[1, 0, 0, 0], translation=[0, 0, 0])],
    }
    for name, rows in tables.items():
        write(data / 'v1.0-mini' / f'{name}.json', rows)
    record = dict(channel='LIDAR_TOP', sample_data_token='test-lidar', path=str(binary),
                  calibrated_sensor_token='test-cal', ego_pose_token='test-ego', timestamp_us=123456,
                  sensor_to_ego=identity, ego_to_global=identity, sensor_to_reference_lidar=identity)
    metadata = dict(
        status='success', result_source='prediction', precision='float32',
        sample_token='test-sample', timestamp_us=123456, reference_timestamp_us=123456,
        scene=dict(token='test-scene', name='fixture-scene', index=0), frame_index=0,
        grid_shape=[2, 3, 2], grid_axis_order=['x', 'y', 'z'],
        bounds=[-1, -1, -1, 1, 2, 1], voxel_size=[1, 1, 1],
        voxel_center='bounds_min + (index + 0.5) * voxel_size',
        class_names=['free', 'car'], palette=[[0, 0, 0], [0, 150, 245]], free_label=0,
        coordinate_system=dict(name='LIDAR_TOP at reference_timestamp_us', unit='metres', lidar_to_ego=identity, lidar_to_global=identity),
        sensors=dict(keyframes=dict(LIDAR_TOP=record)), dataset=dict(root=str(data), version='v1.0-mini'),
        checkpoint=dict(path='test-only.pth', sha256='test-only'), bev=dict(projection='test-only'),
    )
    report = dict(
        status='success', check_only=False, precision='float32', sample_token='test-sample',
        scene_name='fixture-scene', scene_index=0, frame_index=0,
        prediction=dict(logits_shape=[1, 2, 3, 2, 2], predicted_voxel_counts={'0': 11, '1': 1}),
        timings_s=dict(forward=1.0, pipeline_total=2.0, process_until_report=5.0),
        metrics=dict(occupancy_iou=None, occupancy_miou=None),
    )
    write(folder / 'metadata.json', metadata)
    write(folder / 'report.json', report)
    labels = np.zeros((2, 3, 2), dtype=np.uint8)
    labels[1, 2, 0] = 1
    np.savez(folder / 'occupancy.npz', labels=labels, max_softmax_probability=np.full(labels.shape, .8, dtype=np.float32))
    (folder / 'bev.png').write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXS8AAAAASUVORK5CYII='))
    settings = Settings(root, (data,), 2)
    return settings, folder, data


def test_real_axis_formula_and_permutation():
    metadata = dict(grid_shape=[200, 200, 16], grid_axis_order=['x', 'y', 'z'],
                    bounds=[-50, -50, -5, 50, 50, 3], voxel_size=[.5, .5, .5],
                    voxel_center='bounds_min + (index + 0.5) * voxel_size')
    actual = voxel_centers(np.array([[0, 0, 0], [199, 199, 15], [100, 100, 10], [10, 30, 4]]), metadata)
    np.testing.assert_array_equal(actual, [
        [-49.75, -49.75, -4.75], [49.75, 49.75, 2.75], [.25, .25, .25],
        [-44.75, -34.75, -2.75],
    ])
    permuted = dict(metadata, grid_shape=[4, 2, 3], grid_axis_order=['z', 'x', 'y'],
                    bounds=[10, 20, 30, 14, 29, 32], voxel_size=[2, 3, .5])
    np.testing.assert_allclose(voxel_centers(np.array([[0, 0, 0], [3, 1, 2]]), permuted),
                               [[11, 21.5, 30.25], [13, 27.5, 31.75]])
    with pytest.raises(InvalidArtifact):
        voxel_centers(np.array([[200, 0, 0]]), metadata)


def test_payload_alignment_and_background(fixture):
    settings, _, _ = fixture
    app = create_app(settings)
    with engineer_client(app) as client:
        detail = client.get('/api/runs/prediction').json()
        assert detail['prediction_available'] and detail['validation']['sensor_sample_token']
        assert detail['startup_derived_s'] == 3
        assert detail['metrics']['occupancy_iou'] is None
        voxels = np.frombuffer(client.get(detail['occupancy']['url']).content, dtype='<f4').reshape(-1, 5)
        lidar = np.frombuffer(client.get(detail['lidar']['url']).content, dtype='<f4').reshape(-1, 4)
        assert voxels.shape == (1, 5) and voxels[0, 3] == 1
        np.testing.assert_array_equal(voxels[0, :3], [.5, 1.5, -.5])
        np.testing.assert_array_equal(voxels[0, :3], lidar[0, :3])
        assert detail['lidar']['original_point_count'] == 2 and detail['lidar']['point_count'] == 1
        assert client.get('/api/health').json()['inference_enabled'] is False
        assert 'torch' not in sys.modules and 'occfusion' not in sys.modules
        assert client.post('/api/runs/prediction/infer').status_code == 404
        assert client.get('/api/files?path=/etc/passwd').status_code == 404


@pytest.mark.skipif(
    not (REPO / 'work_dirs/infer_mini_cuda_v2/metadata.json').is_file()
    or not (REPO / 'data/nuscenes/v1.0-mini/sample_data.json').is_file(),
    reason='Optional integration test requires the supplied real inference run and nuScenes mini',
)
def test_real_run_voxel_centers_and_lidar_native_xyz():
    folder = REPO / 'work_dirs/infer_mini_cuda_v2'
    dataset = (REPO / 'data/nuscenes').resolve()
    metadata = json.loads((folder / 'metadata.json').read_text())
    np.testing.assert_array_equal(
        voxel_centers(np.array([[10, 30, 4]]), metadata),
        [[-44.75, -34.75, -2.75]],
    )
    loaded = RunStore(Settings(folder.parent, (dataset,))).get(folder.name)
    with np.load(folder / 'occupancy.npz', allow_pickle=False) as archive:
        labels = archive['labels']
    indices = np.argwhere(labels != metadata['free_label'])
    xyz = indices[:, [metadata['grid_axis_order'].index(axis) for axis in ('x', 'y', 'z')]]
    expected = (np.asarray(metadata['bounds'][:3])
                + (xyz + .5) * np.asarray(metadata['voxel_size'])).astype('<f4')
    packet = np.frombuffer(loaded.occupancy, dtype='<f4').reshape(-1, 5)
    np.testing.assert_array_equal(packet[:, :3], expected)
    np.testing.assert_array_equal(packet[:, 3], labels[tuple(indices.T)])

    token = metadata['sensors']['keyframes']['LIDAR_TOP']['sample_data_token']
    rows = json.loads((dataset / metadata['dataset']['version'] / 'sample_data.json').read_text())
    record = next(row for row in rows if row['token'] == token)
    assert record['sample_token'] == metadata['sample_token']
    source_path = (dataset / record['filename']).resolve()
    assert source_path.is_relative_to(dataset)
    raw = np.fromfile(source_path, dtype='<f4').reshape(-1, 5)
    bounds = np.asarray(metadata['bounds'])
    inside = ((raw[:, :3] > bounds[:3]) & (raw[:, :3] < bounds[3:])).all(axis=1)
    lidar = np.frombuffer(loaded.lidar, dtype='<f4').reshape(-1, 4)
    np.testing.assert_array_equal(lidar, raw[inside, :4])
    np.testing.assert_allclose(
        metadata['sensors']['keyframes']['LIDAR_TOP']['sensor_to_reference_lidar'],
        np.eye(4), atol=1e-10, rtol=0,
    )


def test_cache_and_invalidation(fixture):
    settings, folder, _ = fixture
    store = RunStore(settings)
    first = store.get('prediction')
    assert store.get('prediction') is first
    assert store.hits == 1 and store.misses == 1
    report = json.loads((folder / 'report.json').read_text())
    report['warnings'] = ['changed artifact']
    write(folder / 'report.json', report)
    assert store.get('prediction').detail['warnings'] == ['changed artifact']
    assert store.misses == 2


def test_check_only_has_no_prediction(fixture):
    settings, folder, _ = fixture
    for filename in ['report.json', 'metadata.json']:
        value = json.loads((folder / filename).read_text())
        value['status'] = 'check_only_passed'
        if filename == 'report.json':
            value['check_only'] = True
        write(folder / filename, value)
    with engineer_client(create_app(settings)) as client:
        detail = client.get('/api/runs/prediction').json()
        assert detail['kind'] == 'check_only' and not detail['prediction_available']
        assert client.get('/api/runs').json()['runs'][0]['kind'] == 'check_only'
        for suffix in ['occupancy', 'lidar', 'bev']:
            assert client.get(f'/api/runs/prediction/{suffix}').status_code == 409


@pytest.mark.parametrize('case', ['shape', 'label', 'nan', 'probability_range', 'histogram'])
def test_invalid_prediction_rejected(fixture, case):
    settings, folder, _ = fixture
    with np.load(folder / 'occupancy.npz') as source:
        labels, probability = source['labels'], source['max_softmax_probability']
    if case == 'shape':
        labels = labels[:1]
    elif case == 'label':
        labels[0, 0, 0] = 255
    elif case == 'nan':
        probability[0, 0, 0] = np.nan
    elif case == 'probability_range':
        probability[0, 0, 0] = 1.1
    else:
        labels[0, 0, 0] = 1
    np.savez(folder / 'occupancy.npz', labels=labels, max_softmax_probability=probability)
    with engineer_client(create_app(settings)) as client:
        assert client.get('/api/runs/prediction').status_code == 422


@pytest.mark.parametrize('case', ['report_token', 'sensor_token', 'timestamp', 'calibration', 'coordinate_frame'])
def test_sample_and_calibration_mismatch_rejected(fixture, case):
    settings, folder, data = fixture
    if case == 'report_token':
        path = folder / 'report.json'
        value = json.loads(path.read_text())
        value['sample_token'] = 'another-sample'
    elif case in ('calibration', 'coordinate_frame'):
        path = folder / 'metadata.json'
        value = json.loads(path.read_text())
        if case == 'calibration':
            value['sensors']['keyframes']['LIDAR_TOP']['sensor_to_reference_lidar'][0][3] = 1
        else:
            value['coordinate_system']['name'] = 'global'
    else:
        path = data / 'v1.0-mini/sample_data.json'
        value = json.loads(path.read_text())
        value[0]['sample_token' if case == 'sensor_token' else 'timestamp'] = 'another-sample' if case == 'sensor_token' else 1
    write(path, value)
    with engineer_client(create_app(settings)) as client:
        assert client.get('/api/runs/prediction').status_code == 422


def test_run_and_sensor_symlink_escape(fixture, tmp_path):
    settings, folder, data = fixture
    outside = tmp_path / 'outside'
    outside.mkdir()
    write(outside / 'report.json', {'status': 'success'})
    (settings.runs_root / 'escaped-run').symlink_to(outside, target_is_directory=True)
    with engineer_client(create_app(settings)) as client:
        assert client.get('/api/runs/escaped-run').status_code == 403
        assert all(item['run_id'] != 'escaped-run' for item in client.get('/api/runs').json()['runs'])
    with pytest.raises(OutsideRoot):
        RunStore(settings).folder('../outside')
    sensor = data / 'samples/LIDAR_TOP/test.pcd.bin'
    target = outside / 'sensor.bin'
    sensor.rename(target)
    sensor.symlink_to(target)
    with engineer_client(create_app(settings)) as client:
        assert client.get('/api/runs/prediction').status_code == 403


@pytest.mark.parametrize('name', ['report.json', 'metadata.json', 'occupancy.npz', 'bev.png'])
def test_artifact_symlink_escape(fixture, tmp_path, name):
    settings, folder, _ = fixture
    artifact = folder / name
    outside = tmp_path / name
    artifact.rename(outside)
    artifact.symlink_to(outside)
    with engineer_client(create_app(settings)) as client:
        assert client.get('/api/runs/prediction').status_code == 403
