"""Independent quaternion-chain checks of real full-SE(3) voxel-centre geometry."""
import numpy as np

from tools.forecasting.geometry import anchor_from_source, centres, transform_points


def future_gt_independence(dataset, window, inputs, expected_forecast):
    """Perturb future GT in RAM only, restore it, compare the actual forecast bytes."""
    import hashlib
    from tools.forecasting.persistence import predict
    tokens = [f['sample_token'] for f in window['future']]
    original = {t: dataset.dense_gt[t] for t in tokens}
    before = hashlib.sha256(expected_forecast.tobytes(order='C')).hexdigest()
    try:
        for token, values in original.items():
            dataset.dense_gt[token] = np.full_like(values, 7)
        after = hashlib.sha256(predict(inputs).tobytes(order='C')).hexdigest()
    finally:
        dataset.dense_gt.update(original)
    if before != after:
        raise ValueError('Future GT leaked into persistence forecast')
    return dict(passed=True, original_forecast_labels_sha256=before, perturbed_future_gt_forecast_labels_sha256=after,
                future_tokens=tokens, perturbation='All future dense GT cells set to class 7 in RAM, then restored; no source file modified')


def check_window(dataset, window):
    from pyquaternion import Quaternion
    anchor = window['anchor']['sample_token']
    nusc = dataset.catalog.nusc

    def records(token):
        sd = dataset.catalog.samples[token]['lidar']
        return (nusc.get('calibrated_sensor', sd['calibrated_sensor_token']),
                nusc.get('ego_pose', sd['ego_pose_token']))

    anchor_cal, anchor_ego = records(anchor)
    indices = np.array([[0, 0, 0], [199, 199, 15], [10, 30, 4], [100, 100, 8],
                        [0, 199, 15], [199, 0, 0], [53, 71, 5]], dtype=np.int64)
    points = centres(indices)
    errors, roundtrips, checks = [], [], []
    for frame in window['history'] + window['future']:
        token = frame['sample_token']
        cal, ego = records(token)
        matrix = anchor_from_source(dataset.global_from_lidar(anchor), dataset.global_from_lidar(token))
        transformed = transform_points(points, matrix)
        expected = []
        for point in points:
            p = Quaternion(cal['rotation']).rotate(point) + np.asarray(cal['translation'])
            p = Quaternion(ego['rotation']).rotate(p) + np.asarray(ego['translation'])
            p = Quaternion(anchor_ego['rotation']).inverse.rotate(p - anchor_ego['translation'])
            p = Quaternion(anchor_cal['rotation']).inverse.rotate(p - anchor_cal['translation'])
            expected.append(p)
        error = np.linalg.norm(transformed - expected, axis=1)
        roundtrip = np.linalg.norm(transform_points(transformed, np.linalg.inv(matrix)) - points, axis=1)
        errors.extend(error.tolist()); roundtrips.extend(roundtrip.tolist())
        checks.append(dict(sample_token=token, anchor_from_source=matrix.tolist(),
                           quaternion_chain_max_error_m=float(error.max()), roundtrip_max_error_m=float(roundtrip.max())))
    maximum = max(errors)
    if maximum > 1e-8 or max(roundtrips) > 1e-8:
        raise ValueError('Independent real geometry verification exceeds 1e-8 metres')
    return dict(window_id=anchor, precision='float64', test_voxel_indices=indices.tolist(),
                test_centres_m=points.tolist(), method='SDK transform_matrix chains versus independent pyquaternion.rotate/inverse pose chain',
                max_error_m=maximum, p95_error_m=float(np.percentile(errors, 95)),
                roundtrip_max_error_m=max(roundtrips), point_comparisons=len(errors), checks=checks)
