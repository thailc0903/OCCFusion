"""CPU-only tests: real saved smoke copies and explicitly synthetic metric fixtures."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from pyquaternion import Quaternion
from tools import benchmark_detection as benchmark
from tools import evaluate_detection as evaluator
from tools import detection_benchmark_common as common

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / 'work/centerpoint-smoke-wyp1nl__/runs/infer_job_65e1b30833e343c3b434da8dc71423f9'
DRAFT = Path(os.environ.get('DETECTION_TEST_AUDIT', ROOT / 'work_dirs/benchmark_detection_mini_val'))
if not DRAFT.is_dir():
    DRAFT = ROOT / 'work_dirs/detection_benchmark_preparation/cpu_audit_draft'


def manifest_for(entries, provenance, protocol=None):
    protocol = protocol or evaluator.protocol()
    identity = dict(split='mini_val', scene_names=['scene-0103', 'scene-0916'], frames=entries,
        prediction_fingerprint=common.fingerprint(provenance),
        evaluation_protocol_fingerprint=common.fingerprint(protocol))
    return dict(kind=common.KIND, identity=identity, identity_sha256=common.fingerprint(identity),
        prediction_provenance=provenance, evaluation_protocol=protocol,
        dataset_table_hashes={'sample_annotation': {'source': 'explicit synthetic test fixture'}})


class TransformTests(unittest.TestCase):
    def test_geometric_center_wlh_quaternion_velocity_known_rotations(self):
        lidar_to_ego = np.eye(4)
        lidar_to_ego[:3, :3] = Quaternion(axis=[0, 0, 1], radians=np.pi / 2).rotation_matrix
        lidar_to_ego[:3, 3] = [1, 2, 3]
        ego_to_global = np.eye(4)
        ego_to_global[:3, :3] = Quaternion(axis=[0, 0, 1], radians=np.pi / 2).rotation_matrix
        ego_to_global[:3, 3] = [100, 200, 3]
        meta = {'coordinate_system': {'lidar_to_global': (ego_to_global @ lidar_to_ego).tolist()}}
        doc = dict(sample_token='synthetic', class_names=['car'], boxes=[dict(center=[2, 0, 1],
            dimensions=[4, 2, 6], yaw=np.pi / 2, velocity=[3, 4], score=.11, label=0)])
        box = evaluator.global_boxes(doc, meta)[0]
        np.testing.assert_allclose(box['translation'], [96, 201, 7], atol=1e-12)
        np.testing.assert_allclose(box['velocity'], [-3, -4], atol=1e-12)
        self.assertEqual(box['size'], [2, 4, 6])
        np.testing.assert_allclose(Quaternion(box['rotation']).rotation_matrix,
            Quaternion(axis=[0, 0, 1], radians=3 * np.pi / 2).rotation_matrix, atol=1e-12)
        self.assertEqual(box['attribute_name'], 'vehicle.moving')
        self.assertEqual(box['detection_score'], .11)  # Below viewer .35 must survive.

    def test_real_smoke_all_boxes_match_two_step_nuscenes_box_transform(self):
        from nuscenes.utils.data_classes import Box
        metadata, doc = (common.read_json(SMOKE / n) for n in ('metadata.json', 'detections.json'))
        actual = evaluator.global_boxes(doc, metadata)
        self.assertEqual(len(actual), 151)
        self.assertGreater(sum(b['detection_score'] < .35 for b in actual), 0)
        key = metadata['sensors']['keyframes']['LIDAR_TOP']
        for source, result in zip(doc['boxes'], actual):
            reference = Box(source['center'], [source['dimensions'][1], source['dimensions'][0], source['dimensions'][2]],
                Quaternion(axis=[0, 0, 1], radians=source['yaw']), velocity=(*source['velocity'], 0))
            for value in (key['sensor_to_ego'], key['ego_to_global']):
                matrix = np.asarray(value)
                reference.rotate(Quaternion(matrix=matrix[:3, :3]))
                reference.translate(matrix[:3, 3])
            np.testing.assert_allclose(result['translation'], reference.center, atol=1e-10, rtol=0)
            np.testing.assert_allclose(result['size'], reference.wlh, atol=1e-12, rtol=0)
            np.testing.assert_allclose(result['velocity'], reference.velocity[:2], atol=1e-10, rtol=0)
            np.testing.assert_allclose(Quaternion(result['rotation']).rotation_matrix,
                reference.orientation.rotation_matrix, atol=1e-10, rtol=0)

    def test_attribute_policy_prediction_only_and_exact_threshold(self):
        cases = [('car', 0, 'vehicle.parked'), ('car', .2, 'vehicle.parked'),
            ('truck', .21, 'vehicle.moving'), ('bus', 0, 'vehicle.stopped'),
            ('bus', .21, 'vehicle.moving'), ('pedestrian', 0, 'pedestrian.standing'),
            ('pedestrian', 1, 'pedestrian.moving'), ('bicycle', 0, 'cycle.without_rider'),
            ('motorcycle', 1, 'cycle.with_rider'), ('barrier', 1, ''), ('traffic_cone', 0, '')]
        for name, speed, expected in cases:
            with self.subTest(name=name, speed=speed):
                self.assertEqual(evaluator.predicted_attribute(name, [speed, 0]), expected)

    def test_coverage_missing_unknown_duplicate_and_real_empty(self):
        self.assertFalse(evaluator.coverage(['a', 'b'], ['a'], ['a', 'b'])['complete'])
        with self.assertRaises(ValueError):
            evaluator.coverage(['a', 'b'], ['a', 'c'], ['a', 'b'])
        with self.assertRaises(ValueError):
            evaluator.coverage(['a', 'a'], ['a'], ['a'])
        with self.assertRaises(ValueError):
            evaluator.coverage(['a', 'b'], ['a', 'b'], ['a'])
        self.assertEqual(evaluator.global_boxes(dict(boxes=[]), dict(coordinate_system={'lidar_to_global': np.eye(4)})), [])


class SyntheticNuScenes:
    """Small, labeled synthetic fixture solely for running official CPU evaluator tests."""
    version = 'v1.0-mini'

    def __init__(self):
        self.attribute = [dict(token='parked', name='vehicle.parked')]
        self.sample = []
        self.sample_annotation = []
        self.tables = {name: {} for name in ('scene', 'sample', 'sample_data', 'ego_pose', 'sample_annotation')}
        for i, name in enumerate(('scene-0103', 'scene-0916')):
            token = 'synthetic-' + str(i)
            sample = dict(token=token, scene_token=name, anns=['ann-' + token], data={'LIDAR_TOP': token})
            annotation = dict(token='ann-' + token, translation=[5., 0., 0.], size=[2., 4., 2.],
                rotation=[1., 0., 0., 0.], category_name='vehicle.car', attribute_tokens=['parked'],
                num_lidar_pts=10, num_radar_pts=0)
            self.sample.append(sample)
            self.sample_annotation.append(annotation)
            for table, key, value in [('scene', name, dict(name=name)), ('sample', token, sample),
                ('sample_data', token, dict(ego_pose_token=token)), ('ego_pose', token, dict(translation=[0, 0, 0])),
                ('sample_annotation', annotation['token'], annotation)]:
                self.tables[table][key] = value

    def get(self, table, token):
        return self.tables[table][token]

    def box_velocity(self, token):
        return np.zeros(3)


class OfficialMetricTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='detection-official-cpu-')
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.nusc = SyntheticNuScenes()
        self.entries = [dict(sample_token=s['token'], scene=s['scene_token'], frame_index=0) for s in self.nusc.sample]
        self.provenance = dict(files={}, data_root=str(self.out))
        self.manifest = manifest_for(self.entries, self.provenance)
        self.predictions = {e['sample_token']: dict(document=dict(sample_token=e['sample_token'],
            class_names=['car'], boxes=[dict(center=[5., 0, 0], dimensions=[4., 2., 2.],
                yaw=0., velocity=[0., 0.], score=.9, label=0)]),
            metadata=dict(coordinate_system={'lidar_to_global': np.eye(4).tolist()}),
            commit=dict(artifacts={})) for e in self.entries}
        self.progress = dict(frames={e['sample_token']: {'status': 'complete'} for e in self.entries})

    def run_evaluation(self, predictions=None, missing=None, factory=None):
        with patch.object(evaluator, 'valid_predictions', return_value=(
                self.predictions if predictions is None else predictions, missing or [])), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return evaluator.evaluate(self.out, self.manifest, self.progress, nusc=self.nusc,
                evaluator_class=factory)

    def test_official_metrics_known_perfect_car_absent_classes_and_reference(self):
        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import DetectionEval
        result = self.run_evaluation()
        self.assertEqual(result['status'], 'complete')
        self.assertAlmostEqual(result['metrics']['per_class_ap']['car'], 1)
        self.assertAlmostEqual(result['metrics']['mAP'], .1)  # 1 present class / all 10 classes.
        self.assertEqual(result['metrics']['per_class_ap']['truck'], 0)
        self.assertIsNone(result['metrics']['per_class_tp_errors']['traffic_cone']['attr_err'])
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            reference = DetectionEval(self.nusc, config_factory('detection_cvpr_2019'),
                str(self.out / 'predictions_nuscenes.json'), 'mini_val', str(self.out / 'reference'), False)
            raw, _ = reference.evaluate()
        self.assertEqual(result['metrics']['mAP'], raw.mean_ap)
        self.assertEqual(result['metrics']['NDS'], raw.nd_score)

    def test_all_false_positives_missed_gt_official_zero_ap(self):
        for value in self.predictions.values():
            value['document']['boxes'][0]['center'] = [20., 0., 0.]
        result = self.run_evaluation()
        self.assertEqual(result['metrics']['mAP'], 0)

    def test_missing_prediction_partial_no_empty_placeholder_and_no_official_call(self):
        token = self.entries[0]['sample_token']
        factory = Mock(side_effect=AssertionError('Incomplete split must not publish full-split metrics'))
        result = self.run_evaluation({token: self.predictions[token]}, [dict(sample_token=self.entries[1]['sample_token'])], factory)
        self.assertEqual(result['status'], 'partial')
        self.assertIsNone(result['metrics'])
        factory.assert_not_called()
        self.assertEqual(set(common.read_json(self.out / 'predictions_nuscenes_partial.json')['results']), {token})

    def test_evaluation_cache_and_corrupt_official_summary(self):
        first = self.run_evaluation()
        forbidden = Mock(side_effect=AssertionError('Valid CPU evaluation cache should be used'))
        second = self.run_evaluation(factory=forbidden)
        forbidden.assert_not_called()
        self.assertEqual(second['evaluation_source'], 'cache')
        self.assertEqual(first['metrics'], second['metrics'])
        path = self.out / first['evaluation_directory'] / 'official/metrics_summary.json'
        path.write_bytes(path.read_bytes() + b' ')
        third = self.run_evaluation()
        self.assertEqual(third['evaluation_source'], 'new')
        self.assertEqual(first['metrics'], third['metrics'])
        self.assertNotEqual(first['evaluation_directory'], third['evaluation_directory'])

    def test_gt_hash_or_protocol_change_cannot_reuse_evaluation(self):
        first = self.run_evaluation()
        self.manifest['dataset_table_hashes']['sample_annotation'] = {'sha256': 'changed-synthetic-GT-hash'}
        second = self.run_evaluation()
        self.assertEqual(second['evaluation_source'], 'new')
        self.assertNotEqual(first['evaluation_directory'], second['evaluation_directory'])
        wrong = deepcopy(self.manifest)
        wrong['evaluation_protocol']['attribute_policy']['threshold_mps'] = .35
        wrong['identity']['evaluation_protocol_fingerprint'] = common.fingerprint(wrong['evaluation_protocol'])
        wrong['identity_sha256'] = common.fingerprint(wrong['identity'])
        self.manifest = wrong
        with self.assertRaisesRegex(ValueError, 'Evaluator/protocol changed'):
            self.run_evaluation()


class CacheResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='detection-cache-cpu-')
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / 'benchmark'
        original = common.read_json(DRAFT / 'manifest.json')
        self.entry = deepcopy(next(e for e in original['identity']['frames']
            if e['sample_token'] == common.read_json(SMOKE / 'report.json')['sample_token']))
        self.provenance = original['prediction_provenance']
        self.context = dict(manifest=manifest_for([self.entry], self.provenance), nusc=None, cfg=None,
            resolved_config=(DRAFT / 'resolved_config.py').read_text(), setup_wall_s=0.,
            input_audit=dict(status='passed', frames=[]))
        self.args = SimpleNamespace(out_dir=str(self.out), resume=False, prepare_only=True,
            evaluate_only=False, smoke_run=str(SMOKE))
        self.gpu_patch = patch.object(benchmark, 'gpu_lock_path', return_value=Path(self.temp.name) / 'qa-gpu.lock')
        self.gpu_patch.start()
        self.addCleanup(self.gpu_patch.stop)
        self.factory = Mock(side_effect=AssertionError('Artifact-only tests must never construct a model'))
        self.evaluation = lambda *a, **k: dict(status='partial', metrics=None, evaluation_source='not_run')
        self.execute()

    def execute(self, factory=None):
        return benchmark.execute(self.args, self.context, self.factory if factory is None else factory, self.evaluation)

    def import_smoke(self):
        self.args.resume, self.args.prepare_only, self.args.evaluate_only = True, False, True
        return self.execute()

    def test_real_smoke_import_resume_evaluate_only_no_model_no_source_mutation(self):
        before = common.file_records(SMOKE, (*common.BUNDLE_FILES, 'job_result.json'))
        first = self.import_smoke()
        self.assertEqual(first['successful_frames'], 1)
        self.assertEqual(first['current_invocation']['predictions_imported'], 1)
        second = self.execute()
        self.factory.assert_not_called()
        self.assertEqual(second['current_invocation']['model_load_count'], 0)
        self.assertEqual(second['current_invocation']['forward_calls'], 0)
        self.assertEqual(second['current_invocation']['predictions_cached'], 1)
        self.assertEqual(second['latency_all_invocations']['all_completed_forwards']['sample_count'], 0)
        self.assertEqual(before, common.file_records(SMOKE, (*common.BUNDLE_FILES, 'job_result.json')))

    def test_corrupt_prediction_cache_is_rejected_without_inference(self):
        self.import_smoke()
        progress = common.read_json(self.out / 'progress.json')
        path = self.out / progress['frames'][self.entry['sample_token']]['prediction_directory'] / 'detections.json'
        path.write_bytes(path.read_bytes() + b' ')
        self.args.evaluate_only = False
        result = self.execute()
        self.factory.assert_not_called()
        self.assertEqual(result['successful_frames'], 0)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(common.read_json(self.out / 'progress.json')['frames'][self.entry['sample_token']]['status'], 'invalid_cache')

    def test_missing_commit_or_pending_directory_is_not_success(self):
        self.import_smoke()
        progress = common.read_json(self.out / 'progress.json')
        folder = self.out / progress['frames'][self.entry['sample_token']]['prediction_directory']
        (folder / 'prediction_commit.json').unlink()
        predictions, missing = common.valid_predictions(self.out, self.context['manifest'], progress)
        self.assertEqual(predictions, {})
        self.assertEqual(len(missing), 1)
        self.execute()
        self.factory.assert_not_called()

    def test_wrong_token_input_or_fixture_rejected(self):
        with tempfile.TemporaryDirectory(prefix='detection-bad-smoke-') as temp:
            folder = Path(temp)
            for name in (*common.BUNDLE_FILES, 'job_result.json'):
                shutil.copyfile(SMOKE / name, folder / name)
            report = common.read_json(folder / 'report.json')
            report['fixture'] = True
            common.atomic_json(folder / 'report.json', report)
            with self.assertRaises(ValueError):
                common.validate_bundle(folder, self.entry, self.provenance, 'external_smoke')
        for change in ('sample_token', 'input_metadata'):
            with self.subTest(change=change):
                wrong = deepcopy(self.entry)
                if change == 'sample_token':
                    wrong['input_metadata']['sample_token'] = 'wrong'
                else:
                    wrong['input_metadata']['sensors']['lidar_sweeps'][0]['timestamp_us'] += 1
                with self.assertRaises(ValueError):
                    common.validate_bundle(SMOKE, wrong, self.provenance, 'external_smoke')

    def test_changed_model_code_pipeline_or_sweep_hash_rejected(self):
        for field in ('checkpoint', 'config', 'reference_cli_sha256', 'test_pipeline', 'packages'):
            with self.subTest(field=field):
                wrong = deepcopy(self.provenance)
                if field in ('checkpoint', 'config'):
                    wrong[field]['sha256'] = '0' * 64
                elif field == 'test_pipeline':
                    wrong[field] = []
                elif field == 'packages':
                    wrong[field]['mmdet3d'] = 'wrong-version'
                else:
                    wrong[field] = '0' * 64
                with self.assertRaises(ValueError):
                    common.validate_bundle(SMOKE, self.entry, wrong, 'external_smoke')
        original = common.file_record
        sweep = Path(self.entry['input_metadata']['sensors']['lidar_sweeps'][0]['path'])
        def changed(path):
            value = original(path)
            return dict(value, sha256='0' * 64) if Path(path) == sweep else value
        with patch.object(common, 'file_record', side_effect=changed), self.assertRaisesRegex(ValueError, 'input bytes changed'):
            common.validate_bundle(SMOKE, self.entry, self.provenance, 'external_smoke')

    def test_resume_changed_manifest_refused_before_model(self):
        self.args.resume = True
        self.context['manifest'] = deepcopy(self.context['manifest'])
        self.context['manifest']['identity']['frames'][0]['input_fingerprint'] = 'changed'
        self.context['manifest']['identity_sha256'] = common.fingerprint(self.context['manifest']['identity'])
        with self.assertRaises(ValueError):
            self.execute()
        self.factory.assert_not_called()

    def test_truncated_timing_journal_blocks_before_model(self):
        (self.out / 'timings.csv').write_text('invocation_id,sample_token\npartial')
        self.args.resume, self.args.prepare_only = True, False
        with self.assertRaisesRegex(ValueError, 'Truncated timing journal'):
            self.execute()
        self.factory.assert_not_called()

    def test_ambiguous_restart_is_interrupted_not_retried(self):
        progress = common.read_json(self.out / 'progress.json')
        progress['frames'][self.entry['sample_token']] = dict(status='running', pending_directory='frames/.pending')
        common.atomic_json(self.out / 'progress.json', progress)
        self.args.resume, self.args.prepare_only = True, False
        self.execute()
        self.factory.assert_not_called()
        self.assertEqual(common.read_json(self.out / 'progress.json')['frames'][self.entry['sample_token']]['status'], 'interrupted')

    def test_crash_after_publication_recovers_commit_without_forward(self):
        self.import_smoke()
        progress = common.read_json(self.out / 'progress.json')
        state = progress['frames'][self.entry['sample_token']]
        final = self.out / state['prediction_directory']
        state.update(status='running', pending_directory=str(final.with_name('.pending-' + final.name).relative_to(self.out)))
        del state['prediction_directory']
        common.atomic_json(self.out / 'progress.json', progress)
        self.args.evaluate_only = False
        result = self.execute()
        self.factory.assert_not_called()
        self.assertEqual(result['current_invocation']['predictions_recovered'], 1)
        self.assertEqual(result['successful_frames'], 1)

    def test_fake_forward_failure_persisted_and_never_automatically_retried(self):
        class FakeSession:
            setup = {'fixture': True}
            forward_calls = 0
            def __init__(self, context):
                pass
            def run(self, entry, folder):
                self.forward_calls += 1  # Simulated call count, NOT a GPU/model call.
                common.atomic_json(folder / 'report.json', dict(status='failed', forward_completed=False,
                    error=dict(stage='forward', message='explicit synthetic failure fixture'), timings_s={}))
                raise RuntimeError('explicit synthetic failure fixture')
            def close(self):
                pass
        self.args.resume, self.args.prepare_only, self.args.smoke_run = True, False, None
        first = self.execute(FakeSession)
        self.assertEqual(first['status'], 'partial')
        state = common.read_json(self.out / 'progress.json')['frames'][self.entry['sample_token']]
        self.assertEqual(state['error']['stage'], 'forward')
        self.execute()
        self.factory.assert_not_called()

    def test_common_model_initialization_excluded_from_first_frame_end_to_end(self):
        clock = [0.]
        class FakeSession:
            setup = {'fixture': True, 'model_checkpoint_cpu_s': 1000.}
            forward_calls = 0
            def __init__(self, context):
                clock[0] += 1000.
            def run(self, entry, folder):
                clock[0] += 2.
                self.forward_calls = 1
                common.atomic_json(folder / 'report.json', dict(fixture=True, forward_completed=True,
                    latency_warmup=True, timings_s={'forward': 1.}))
            def close(self):
                pass
        def fake_commit(pending, *args):
            clock[0] += 1.
            final = pending.with_name(pending.name.removeprefix('.pending-'))
            pending.rename(final)
            return final
        self.args.resume, self.args.prepare_only, self.args.smoke_run = True, False, None
        with patch.object(benchmark.time, 'perf_counter', side_effect=lambda: clock[0]), \
                patch.object(benchmark, 'commit_prediction', side_effect=fake_commit), \
                patch.object(benchmark, 'valid_predictions', return_value=({}, [dict(status='synthetic')])):
            result = self.execute(FakeSession)
        self.assertEqual(result['current_invocation']['model_load_count'], 1)
        self.assertEqual(result['latency_all_invocations']['all_completed_forwards']['sample_count'], 1)
        import csv
        with (self.out / 'timings.csv').open() as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(float(row['end_to_end_s']), 3.)


class AuditAndLockTests(unittest.TestCase):
    def test_official_cpu_preprocessing_smoke_count_and_empty_history_padding(self):
        import torch
        from mmengine.config import Config
        from mmengine.dataset import Compose
        from mmdet3d.utils import register_all_modules
        from mmdet3d.registry import MODELS
        from nuscenes import NuScenes
        from tools.infer_centerpoint_mini import frame_info
        profile = next(p for p in common.read_json(ROOT / 'web/centerpoint_profiles.json')['profiles'] if p.get('task') == 'detection')
        cfg = Config.fromfile(profile['config'])
        nusc = NuScenes(version='v1.0-mini', dataroot=str(ROOT / 'data/nuscenes'), verbose=False)
        register_all_modules(init_default_scope=True)
        forbidden = Mock(side_effect=AssertionError('CPU pipeline test must not construct a model'))
        with patch.object(MODELS, 'build', forbidden):
            inputs, _ = frame_info(nusc, 'scene-0103', 1)
            packed = Compose(cfg.test_pipeline)(inputs)
            self.assertEqual(len(packed['inputs']['points']), common.read_json(SMOKE / 'report.json')['point_count'])
            self.assertEqual(packed['inputs']['points'].dtype, torch.float32)
            inputs, meta = frame_info(nusc, 'scene-0103', 0)
            raw = np.fromfile(meta['sensors']['keyframes']['LIDAR_TOP']['path'], dtype='<f4').reshape(-1, 5)
            outside_close = ~((np.abs(raw[:, 0]) < 1) & (np.abs(raw[:, 1]) < 1))
            packed = Compose(cfg.test_pipeline)(inputs)
            self.assertEqual(len(packed['inputs']['points']), len(raw) + 9 * int(outside_close.sum()))
        forbidden.assert_not_called()
        self.assertFalse(torch.cuda.is_initialized())

    def test_full_manifest_includes_both_radar_startup_failure_tokens_and_padding(self):
        manifest = common.read_json(DRAFT / 'manifest.json')
        entries = common.verify_manifest(manifest)
        self.assertEqual(len(entries), 81)
        self.assertEqual(len({e['sample_token'] for e in entries}), 81)
        self.assertEqual([sum(e['scene'] == s for e in entries) for s in ('scene-0103', 'scene-0916')], [40, 41])
        for token in ('3e8750f331d7499e9b5123e9eb70f2e2', 'b5989651183643369174912bc5641d3b'):
            entry = next(e for e in entries if e['sample_token'] == token)
            self.assertEqual(entry['input_audit']['past_sweeps'], 0)
            self.assertEqual(entry['input_audit']['padded_keyframe_copies'], 9)
        audit = common.read_json(DRAFT / 'input_audit.json')
        self.assertGreater(audit['loader_vs_se3_translation_m']['max'], .4)
        self.assertEqual(audit['sweep_count'], 711)

    def test_actual_adapter_rejects_future_sweep_missing_file_and_wrong_scene(self):
        from nuscenes import NuScenes
        from tools.infer_centerpoint_mini import frame_info
        nusc = NuScenes(version='v1.0-mini', dataroot=str(ROOT / 'data/nuscenes'), verbose=False)
        sample = nusc.get('sample', common.read_json(SMOKE / 'report.json')['sample_token'])
        ref = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        original_get = nusc.get
        def future(table, token):
            value = original_get(table, token)
            return dict(value, timestamp=ref['timestamp'] + 1) if table == 'sample_data' and token == ref['prev'] else value
        with patch.object(nusc, 'get', side_effect=future), self.assertRaisesRegex(ValueError, 'future'):
            frame_info(nusc, 'scene-0103', 1)
        with patch.object(nusc, 'get_sample_data_path', return_value='/tmp/nonexistent-rav25-test-lidar.bin'), self.assertRaises(FileNotFoundError):
            frame_info(nusc, 'scene-0103', 1)
        sweep_sample = original_get('sample_data', ref['prev'])['sample_token']
        def boundary(table, token):
            value = original_get(table, token)
            return dict(value, scene_token='wrong-scene') if table == 'sample' and token == sweep_sample else value
        with patch.object(nusc, 'get', side_effect=boundary), self.assertRaises(ValueError):
            frame_info(nusc, 'scene-0103', 1)

    def test_content_hash_change_same_size_and_worker_shares_flock(self):
        with tempfile.TemporaryDirectory(prefix='detection-lane-cpu-') as temp:
            path = Path(temp) / 'input.bin'
            path.write_bytes(b'abcd')
            provenance = {'files': {str(path): common.file_record(path)}}
            path.write_bytes(b'abce')
            with self.assertRaises(ValueError):
                common.verify_files(provenance)
            lock = Path(temp) / 'gpu.lock'
            with common.exclusive_lock(lock):
                script = ('from pathlib import Path;import sys;'
                    'from web.backend.app.job_worker import gpu_lock\n'
                    'with gpu_lock(Path(sys.argv[1])) as fd:sys.exit(0 if fd is None else 1)')
                result = subprocess.run([str(ROOT / '.venv-web/bin/python'), '-c', script, str(lock)],
                    cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_latency_excludes_cache_and_first_manifest_forward_but_not_accuracy(self):
        rows = [dict(source='new', forward_completed=True, warmup=True, forward_s=10., end_to_end_s=12.),
            dict(source='new', forward_completed=True, warmup=False, forward_s=1., end_to_end_s=2.),
            dict(source='new', forward_completed=True, warmup=False, forward_s=3., end_to_end_s=4.),
            dict(source='cache', forward_completed=True, warmup=False, forward_s=.001, end_to_end_s=.001)]
        result = benchmark.timing_statistics(rows)
        self.assertEqual(result['all_completed_forwards']['sample_count'], 3)
        self.assertEqual(result['excluding_marked_warmup']['sample_count'], 2)
        self.assertEqual(result['excluding_marked_warmup']['median_s'], 2)
        self.assertAlmostEqual(result['excluding_marked_warmup']['p95_s'], 2.9)
        self.assertEqual(result['end_to_end_excluding_marked_warmup']['median_s'], 3)


if __name__ == '__main__':
    unittest.main()
