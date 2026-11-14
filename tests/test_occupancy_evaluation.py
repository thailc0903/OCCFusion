"""Hand-computed protocol tests. Fixtures are synthetic and stay in temp dirs."""
import ast
from contextlib import redirect_stderr
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from tools import evaluate_occupancy as evaluation
from tools import infer_nuscenes_mini as inference


def source_method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_bytes(), filename=str(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[method_name]


class MetricTests(unittest.TestCase):
    def metrics(self, gt, pred):
        return evaluation.summarize(evaluation.confusion_statistics(np.asarray(pred), np.asarray(gt)))

    def check_reference(self, gt, pred, metrics):
        result = evaluation.compare_reference([np.asarray(gt)], [np.asarray(pred)], metrics)
        self.assertTrue(result['passed'])
        self.assertLessEqual(result['max_abs_difference'], 1e-12)

    def test_perfect_prediction_and_all_sixteen_classes_in_mean(self):
        gt, pred = [0, 1, 1, 2], [0, 1, 1, 2]
        metrics = self.metrics(gt, pred)
        self.assertEqual(metrics['occupancy_iou'], 1)
        self.assertEqual(metrics['semantic_miou'], 2/16)
        self.assertEqual(metrics['semantic_mean_class_count'], 16)
        self.assertEqual(metrics['per_class_semantic'][2]['iou'], 0)
        self.assertEqual(metrics['per_class_semantic'][2]['union'], 0)
        self.check_reference(gt, pred, metrics)

    def test_false_positive(self):
        gt, pred = [1, 0], [1, 1]
        metrics = self.metrics(gt, pred)
        self.assertEqual(metrics['occupancy_iou'], .5)
        self.assertEqual(metrics['per_class_semantic'][0]['iou'], .5)
        self.assertEqual(metrics['semantic_miou'], .5/16)
        self.check_reference(gt, pred, metrics)

    def test_false_negative(self):
        gt, pred = [1, 1], [1, 0]
        metrics = self.metrics(gt, pred)
        self.assertEqual(metrics['occupancy_iou'], .5)
        self.assertEqual(metrics['semantic_miou'], .5/16)
        self.check_reference(gt, pred, metrics)

    def test_wrong_semantic_class_is_still_occupied(self):
        gt, pred = [1], [2]
        metrics = self.metrics(gt, pred)
        self.assertEqual(metrics['occupancy_iou'], 1)
        self.assertEqual(metrics['semantic_miou'], 0)
        self.check_reference(gt, pred, metrics)

    def test_ignore_excludes_both_false_positives_and_false_negatives(self):
        gt, pred = [255, 255, 1, 0], [16, 0, 1, 0]
        metrics = self.metrics(gt, pred)
        self.assertEqual(metrics['valid_voxel_count'], 2)
        self.assertEqual(metrics['occupancy_iou'], 1)
        self.assertEqual(metrics['semantic_miou'], 1/16)
        self.check_reference(gt, pred, metrics)

    def test_union_zero_and_all_ignore_match_original_zero_convention(self):
        for gt, pred in [([0, 0], [0, 0]), ([255, 255], [1, 2])]:
            with self.subTest(gt=gt):
                metrics = self.metrics(gt, pred)
                self.assertEqual(metrics['occupancy_iou'], 0)
                self.assertEqual(metrics['semantic_miou'], 0)
                self.check_reference(gt, pred, metrics)

    def test_pooled_iou_is_not_mean_frame_iou(self):
        ground_truth = [np.ones(9, dtype=int), np.ones(1, dtype=int)]
        predictions = [np.ones(9, dtype=int), np.zeros(1, dtype=int)]
        matrices = [evaluation.confusion_statistics(pred, gt) for gt, pred in zip(ground_truth, predictions)]
        pooled = evaluation.summarize(sum(matrices))
        self.assertEqual(pooled['occupancy_iou'], .9)
        self.assertEqual(pooled['semantic_miou'], .9/16)
        self.assertEqual(pooled['occupancy']['intersection'], 9)
        self.assertEqual(pooled['occupancy']['union'], 10)
        self.assertEqual(np.mean([evaluation.summarize(matrix)['occupancy_iou'] for matrix in matrices]), .5)
        self.assertTrue(evaluation.compare_reference(ground_truth, predictions, pooled)['passed'])

    def test_invalid_prediction_labels_are_rejected_even_at_ignored_gt(self):
        for pred in [[255], [-1], [17]]:
            with self.subTest(pred=pred), self.assertRaises(ValueError):
                self.metrics([255], pred)


class SparseTests(unittest.TestCase):
    def test_sparse_zero_is_ignore_and_unlisted_is_empty(self):
        gt, stats = evaluation.sparse_to_dense(np.array([[0, 0, 0, 0], [1, 2, 1, 4]]), (2, 3, 2))
        self.assertEqual(gt[0, 0, 0], 255)
        self.assertEqual(gt[1, 2, 1], 4)
        self.assertEqual(gt[0, 0, 1], 0)
        self.assertEqual(stats['unlisted_empty_voxels'], 10)
        self.assertEqual(stats['ignored_voxels'], 1)

    def test_duplicate_identical_or_equivalent_ignore_rows_are_deduplicated(self):
        rows = np.array([[0, 0, 0, 0], [0, 0, 0, 255], [1, 0, 0, 4], [1, 0, 0, 4]])
        gt, stats = evaluation.sparse_to_dense(rows, (2, 1, 1))
        np.testing.assert_array_equal(gt[:, 0, 0], [255, 4])
        self.assertEqual(stats['duplicate_identical_rows'], 2)

    def test_conflicting_duplicates_are_rejected(self):
        for labels in [(1, 2), (0, 1)]:
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, 'Conflicting'):
                evaluation.sparse_to_dense(np.array([[0, 0, 0, labels[0]], [0, 0, 0, labels[1]]]))

    def test_bad_coordinates_labels_and_nonfinite_values_are_rejected(self):
        examples = [[[-1, 0, 0, 1]], [[200, 0, 0, 1]], [[0, 200, 0, 1]], [[0, 0, 16, 1]],
                    [[.5, 0, 0, 1]], [[0, 0, 0, 17]], [[0, 0, 0, -1]], [[0, 0, 0, 1.5]],
                    [[0, 0, 0, np.nan]], [[np.inf, 0, 0, 1]], [[0, 0, 0]], [['x', 'y', 'z', 'car']]]
        for rows in examples:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                evaluation.sparse_to_dense(np.asarray(rows))

    def test_empty_sparse_array_and_integral_float_rows(self):
        gt, stats = evaluation.sparse_to_dense(np.empty((0, 4)), (2, 2, 2))
        self.assertEqual(int(gt.sum()), 0)
        self.assertEqual(stats['unlisted_empty_voxels'], 8)
        gt, _ = evaluation.sparse_to_dense(np.array([[1., 0., 0., 2.]]), (2, 1, 1))
        self.assertEqual(gt[1, 0, 0], 2)

    def test_exact_original_loader_and_multiscale_supervision_cpu(self):
        import os
        import torch
        load = source_method(evaluation.ROOT/'occfusion/loading.py', 'LoadOccupancy', 'transform',
                             {'np': np, 'torch': torch, 'os': os})
        dense = source_method(evaluation.ROOT/'occfusion/main.py', 'OccFusion', 'multiscale_supervision',
                              {'np': np, 'torch': torch})
        rows = np.array([[0, 0, 0, 0], [1, 2, 1, 4], [0, 1, 1, 2]])
        with TemporaryDirectory(prefix='occupancy-unit-') as tmp:
            root = Path(tmp)
            (root/'occ_samples').mkdir()
            np.save(root/'occ_samples/frame.pcd.bin.npy', rows)
            result = load(SimpleNamespace(use_occ3d=False), {'lidar_points': {'lidar_path': str(root/'samples/LIDAR_TOP/frame.pcd.bin')}})
            reference = dense(None, [result['occ_200']], [1, 1, 1], [1, 2, 3, 2]).numpy()[0]
            actual, _ = evaluation.sparse_to_dense(rows, (2, 3, 2))
            np.testing.assert_array_equal(actual, reference)


class SavedRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix='occupancy-unit-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root/'run'
        self.run.mkdir()
        self.gt_root = self.root/'gt'
        self.gt_root.mkdir()
        self.token, self.sd_token = 'sample-token', 'lidar-token'
        self.scene = dict(name='scene-fixture', token='scene-token')
        self.filename = 'samples/LIDAR_TOP/frame.pcd.bin'
        (self.root/'samples/LIDAR_TOP').mkdir(parents=True)
        (self.root/self.filename).write_bytes(b'unit test fixture only')
        self.sd = dict(token=self.sd_token, sample_token=self.token, filename=self.filename,
                       timestamp=10, is_key_frame=True, calibrated_sensor_token='cal', ego_pose_token='ego')
        tables = {('calibrated_sensor', 'cal'): dict(translation=[0, 0, 0], rotation=[1, 0, 0, 0], sensor_token='sensor'),
                  ('ego_pose', 'ego'): dict(translation=[0, 0, 0], rotation=[1, 0, 0, 0]),
                  ('sensor', 'sensor'): dict(channel='LIDAR_TOP')}
        self.catalog = evaluation.DatasetCatalog.__new__(evaluation.DatasetCatalog)
        self.catalog.root, self.catalog.version = self.root, 'v1.0-mini'
        self.catalog.nusc = SimpleNamespace(get=lambda table, token: tables[(table, token)])
        self.catalog.samples = {self.token: dict(sample=dict(timestamp=10), scene=self.scene,
            scene_index=0, frame_index=1, lidar=self.sd, split='mini_val')}
        self.metadata = dict(status='success', result_source='prediction', sample_token=self.token,
            dataset=dict(version='v1.0-mini'), scene=dict(self.scene, index=0), frame_index=1,
            timestamp_us=10, reference_timestamp_us=10, timestamp_unit='microseconds',
            bounds=list(evaluation.BOUNDS), voxel_size=list(evaluation.VOXEL_SIZE), grid_shape=list(evaluation.GRID),
            grid_axis_order=['x', 'y', 'z'], voxel_center='bounds_min + (index + 0.5) * voxel_size',
            class_names=list(evaluation.CLASS_NAMES), free_label=0, gt_ignore_label=255, precision='float32',
            coordinate_system=dict(name='LIDAR_TOP at reference_timestamp_us', unit='metres',
                                   lidar_to_ego=np.eye(4).tolist(), lidar_to_global=np.eye(4).tolist()),
            sensors=dict(keyframes=dict(LIDAR_TOP=dict(channel='LIDAR_TOP', sample_data_token=self.sd_token,
                path=str(self.root/self.filename), calibrated_sensor_token='cal', ego_pose_token='ego',
                timestamp_us=10, sensor_to_reference_lidar=np.eye(4).tolist()))))
        self.labels = np.zeros(evaluation.GRID, dtype=np.uint8)
        self.labels[0, 0, 0] = 1
        self.report = dict(status='success', sample_token=self.token, scene_name=self.scene['name'],
            scene_index=0, frame_index=1, precision='float32', prediction=dict(
                logits_shape=[1, *evaluation.GRID, 17], predicted_voxel_counts={'0': 639999, '1': 1}))
        self.gt_path = self.gt_root/'frame.pcd.bin.npy'
        np.save(self.gt_path, np.array([[0, 0, 0, 1]]))
        self.save()

    def save(self):
        (self.run/'metadata.json').write_text(json.dumps(self.metadata))
        (self.run/'report.json').write_text(json.dumps(self.report))
        (self.run/'bev.png').write_bytes(b'unit fixture; metrics do not decode BEV')
        np.savez_compressed(self.run/'occupancy.npz', labels=self.labels,
                            max_softmax_probability=np.ones(evaluation.GRID, dtype=np.float32))

    def evaluate(self, split='mini_val', runs=None):
        return evaluation.evaluate_runs(runs or [self.run], self.catalog,
                                       evaluation.GroundTruthIndex(self.gt_root), split, True)

    def test_success_hashes_reference_and_original_files_preserved(self):
        before = {name: (self.run/name).read_bytes() for name in evaluation.ARTIFACTS}
        result = self.evaluate()
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['metrics']['occupancy_iou'], 1)
        self.assertTrue(result['pooled_reference_check']['passed'])
        frame = result['frames'][0]
        self.assertEqual(frame['sample_token'], self.token)
        self.assertEqual(frame['gt_sha256'], evaluation.sha256(self.gt_path.read_bytes()))
        self.assertEqual(frame['prediction_sha256'], evaluation.sha256(before['occupancy.npz']))
        self.assertEqual(before, {name: (self.run/name).read_bytes() for name in evaluation.ARTIFACTS})

    def test_missing_gt_metrics_are_unavailable(self):
        self.gt_path.unlink()
        result = self.evaluate()
        self.assertEqual(result['status'], 'missing_gt')
        self.assertIsNone(result['metrics'])
        self.assertIsNone(result['frames'][0]['metrics'])
        self.assertIsNone(result['frames'][0]['gt_sha256'])

    def test_wrong_split_is_not_evaluated(self):
        result = self.evaluate('mini_train')
        self.assertEqual(result['frames'][0]['status'], 'split_mismatch')
        self.assertEqual(result['frame_count_evaluated'], 0)
        self.assertIsNone(result['metrics'])

    def test_token_filename_calibration_timestamp_and_frame_mismatches(self):
        original = deepcopy(self.metadata)
        mutations = [lambda m: m.update(sample_token='unknown'),
                     lambda m: m.update(frame_index=2),
                     lambda m: m.update(reference_timestamp_us=11),
                     lambda m: m['dataset'].update(split='mini_train'),
                     lambda m: m['sensors']['keyframes']['LIDAR_TOP'].update(path='different.pcd.bin'),
                     lambda m: m['sensors']['keyframes']['LIDAR_TOP'].update(sample_data_token='wrong'),
                     lambda m: m['coordinate_system']['lidar_to_ego'][0].__setitem__(3, 1)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.metadata = deepcopy(original)
                mutate(self.metadata)
                self.save()
                result = self.evaluate()
                self.assertEqual(result['frames'][0]['status'], 'invalid_input')
                self.assertIsNone(result['metrics'])

    def test_geometry_axis_and_class_map_mismatches(self):
        original = deepcopy(self.metadata)
        for key, bad in [('bounds', [-40, -40, -1, 40, 40, 5.4]), ('voxel_size', [.4]*3),
                         ('grid_axis_order', ['y', 'x', 'z']), ('class_names', ['free', 'car'])]:
            with self.subTest(key=key):
                self.metadata = deepcopy(original)
                self.metadata[key] = bad
                self.save()
                self.assertIsNone(self.evaluate()['metrics'])

    def test_nonfinite_probability_and_label_range_are_rejected(self):
        probability = np.ones(evaluation.GRID, dtype=np.float32)
        probability[0, 0, 0] = np.nan
        np.savez_compressed(self.run/'occupancy.npz', labels=self.labels, max_softmax_probability=probability)
        self.assertIsNone(self.evaluate()['metrics'])
        self.labels[0, 0, 0] = 17
        self.save()
        self.assertIsNone(self.evaluate()['metrics'])

    def test_duplicate_gt_filename_is_ambiguous(self):
        (self.gt_root/'other').mkdir()
        (self.gt_root/'other'/self.gt_path.name).write_bytes(self.gt_path.read_bytes())
        result = self.evaluate()
        self.assertIn('Ambiguous', result['frames'][0]['error'])
        self.assertIsNone(result['metrics'])

    def test_duplicate_sample_does_not_double_count(self):
        result = self.evaluate(runs=[self.run, self.run])
        self.assertEqual(result['frame_count_evaluated'], 1)
        self.assertEqual(result['status'], 'partial')
        self.assertIn('Duplicate sample', result['frames'][1]['error'])

    def test_output_cannot_overwrite_inference_or_existing_evaluation(self):
        for name in (*evaluation.ARTIFACTS, 'evaluation.json'):
            path = self.run/name
            if not path.exists():
                path.write_text('existing evidence')
            before = path.read_bytes()
            with self.subTest(name=name), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                evaluation.parse_args(['--run-dir', str(self.run), '--gt-root', str(self.gt_root), '--out', str(path)])
            self.assertEqual(path.read_bytes(), before)


class SceneSelectionTests(unittest.TestCase):
    def test_scene_name_uses_api_order_and_old_index_default_is_preserved(self):
        scenes = [dict(name='scene-0916'), dict(name='scene-0061'), dict(name='scene-0103')]
        fake = SimpleNamespace(scene=scenes)
        self.assertEqual(inference.resolve_scene_index(fake, scene_name='scene-0103'), 2)
        self.assertEqual(inference.resolve_scene_index(fake, scene_name='scene-0916'), 0)
        self.assertEqual(inference.resolve_scene_index(fake), 0)
        self.assertEqual(inference.resolve_scene_index(fake, scene_index=1), 1)
        with self.assertRaises(ValueError):
            inference.resolve_scene_index(fake, scene_name='unknown')
        with self.assertRaises(ValueError):
            inference.resolve_scene_index(fake, scene_index=0, scene_name='scene-0103')
        old = inference.parse_args(['--scene-index', '0', '--frame-index', '1', '--out-dir', 'test'])
        self.assertEqual(old.scene_index, 0)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            inference.parse_args(['--scene-index', '0', '--scene-name', 'scene-0103', '--out-dir', 'test'])

    def test_class_map_matches_repository_dataset(self):
        tree = ast.parse((evaluation.ROOT/'occfusion/nuscenes_dataset.py').read_bytes())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'NuScenesSegDataset')
        value = next(node.value for node in cls.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'METAINFO' for t in node.targets))
        names = next(v for k, v in zip(value.keys, value.values) if isinstance(k, ast.Constant) and k.value == 'classes')
        self.assertEqual(ast.literal_eval(names), ('IoU', *evaluation.CLASS_NAMES[1:]))


if __name__ == '__main__':
    unittest.main()
