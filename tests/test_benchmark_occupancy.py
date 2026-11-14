"""Resume tests use saved artifact copies and a model factory that must never run."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tools import benchmark_occupancy as benchmark
from tools import evaluate_occupancy as evaluator


ROOT = Path(__file__).resolve().parents[1]


class TimingTests(unittest.TestCase):
    def test_cache_io_is_excluded_and_warmup_has_separate_distribution(self):
        rows = [dict(prediction_source='new',forward_s=10,warmup=True),
                dict(prediction_source='new',forward_s=1,warmup=False),
                dict(prediction_source='new',forward_s=3,warmup=False),
                dict(prediction_source='cache',forward_s=.001,warmup=False),
                dict(prediction_source='new',forward_s=None,warmup=False)]
        value = benchmark.timing_statistics(rows)
        self.assertEqual(value['all_completed_forwards']['sample_count'],3)
        self.assertEqual(value['all_completed_forwards']['median_s'],3)
        self.assertEqual(value['excluding_marked_warmup']['sample_count'],2)
        self.assertEqual(value['excluding_marked_warmup']['median_s'],2)
        self.assertAlmostEqual(value['excluding_marked_warmup']['p95_s'],2.9)
        self.assertEqual(value['warmup_completed_count'],1)
        self.assertIsNone(benchmark.timing_statistics([])['all_completed_forwards']['median_s'])

    def test_malformed_timing_is_not_a_latency_measurement(self):
        value = benchmark.timing_statistics([dict(prediction_source='new',forward_s='NaN',warmup=False)])
        self.assertEqual(value['malformed_timing_rows_excluded'],1)
        self.assertEqual(value['all_completed_forwards']['sample_count'],0)

    def test_resume_writes_header_if_timing_file_was_created_but_left_empty(self):
        with TemporaryDirectory(prefix='benchmark-csv-') as directory:
            path = Path(directory)/'timings.csv'
            path.touch()
            benchmark.append_timing(path,dict(session_id='resumed',sample_token='sample',
                prediction_source='new',forward_s=2.5,warmup=False))
            rows = benchmark.read_timings(path)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['session_id'],'resumed')
        self.assertEqual(benchmark.timing_statistics(rows)['all_completed_forwards']['sample_count'],1)

    def test_order_is_official_scene_then_frame_without_metric_selection(self):
        items = [dict(scene=dict(name=scene),frame_index=index,sample=dict(token=f'{scene}-{index}'),split='mini_val')
                 for scene,index in [('b',1),('a',1),('b',0),('a',0)]]
        catalog = SimpleNamespace(samples={str(i):item for i,item in enumerate(items)},splits={'mini_val':['b','a']})
        self.assertEqual([x['sample']['token'] for x in benchmark.ordered_samples(catalog)],['a-0','a-1','b-0','b-1'])
        self.assertEqual([x['sample']['token'] for x in benchmark.ordered_samples(catalog,True)],['a-1','b-1'])


class CacheTests(unittest.TestCase):
    def setUp(self):
        source = ROOT/'work_dirs/infer_mini_val_scene0103_f1'
        if not source.is_dir():
            self.skipTest('Saved real prediction fixture unavailable')
        self.temp = TemporaryDirectory(prefix='benchmark-cache-')
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.metadata = evaluator.json_object((source/'metadata.json').read_bytes())
        token = self.metadata['sample_token']
        self.token = token
        self.item = dict(scene=dict(name=self.metadata['scene']['name'],token=self.metadata['scene']['token']),
            scene_index=self.metadata['scene']['index'],frame_index=self.metadata['frame_index'],split='mini_val',
            sample=dict(token=token,timestamp=self.metadata['timestamp_us']),
            lidar=dict(token=self.metadata['sensors']['keyframes']['LIDAR_TOP']['sample_data_token'],
                       timestamp=self.metadata['reference_timestamp_us'],filename='samples/LIDAR_TOP/test.pcd.bin'))
        gt = self.out/'fixture_gt.npy'
        np.save(gt,np.array([[0,0,0,1]],dtype=np.int64))
        self.gt = gt
        shared = dict(checkpoint=self.metadata['checkpoint'],resolved_config_sha256=self.metadata['config']['resolved_sha256'],
                      tf32=dict(benchmark.TF32))
        self.context = dict(shared=shared,shared_fingerprint=benchmark.fingerprint(shared),
            protocol=evaluator.protocol(),protocol_fingerprint=benchmark.fingerprint(evaluator.protocol()),
            catalog=SimpleNamespace(samples={token:self.item},splits={'mini_val':[self.item['scene']['name']]},nusc=None,
                validate_sample=lambda metadata,report:self.assertEqual(metadata['sample_token'],token)),
            gt_index=SimpleNamespace(lookup=lambda filename:('fixture_gt.npy',gt)),cfg=None,setup_wall_s=0.)
        self.args = SimpleNamespace(out_dir=str(self.out),resume=True,evaluate_only=True,pilot=False,
                                    verify_reference=True,warmup_forwards=1)
        self.input_fingerprint = 'test-input-fingerprint'
        self.folder = self.out/'frames'/token/'completed'
        shutil.copytree(source,self.folder)
        self.metadata['benchmark_provenance'] = dict(shared_fingerprint=self.context['shared_fingerprint'],input_fingerprint=self.input_fingerprint)
        benchmark.atomic_json(self.folder/'metadata.json',self.metadata)
        benchmark.atomic_json(self.folder/'prediction_commit.json',dict(status='success',sample_token=token,
            input_fingerprint=self.input_fingerprint,shared_fingerprint=self.context['shared_fingerprint'],
            artifact_hashes=benchmark.file_records(self.folder,[*benchmark.ARTIFACTS,'resolved_config.py'])))
        manifest = benchmark.build_manifest(self.context,False)
        self.entry = manifest['frames'][0]
        benchmark.atomic_json(self.out/'manifest.json',manifest)
        benchmark.atomic_json(self.out/'progress.json',dict(frames={},sessions=[]))

    def execute(self):
        factory = unittest.mock.Mock(side_effect=AssertionError('Saved artifact test must never load a model'))
        with patch.object(benchmark.inference,'make_frame_info',return_value=({},[],{})),\
             patch.object(benchmark,'sensor_identity',return_value=(self.input_fingerprint,{})),redirect_stdout(StringIO()):
            result = benchmark.execute(self.args,self.context,factory)
        factory.assert_not_called()
        return result

    def test_cache_only_creates_evaluation_then_reuses_both_without_forward(self):
        first = self.execute()
        self.assertEqual(first['status'],'complete')
        self.assertEqual(first['current_invocation']['predictions_cached'],1)
        self.assertEqual(first['current_invocation']['evaluations_new'],1)
        self.assertTrue(first['reference_check']['passed'])
        second = self.execute()
        self.assertEqual(second['current_invocation']['predictions_new'],0)
        self.assertEqual(second['current_invocation']['evaluations_cached'],1)
        self.assertEqual(second['current_invocation']['model_load_count'],0)
        self.assertEqual(second['latency_current_invocation']['all_completed_forwards']['sample_count'],0)
        self.assertEqual(first['metrics'],second['metrics'])

    def test_gt_change_recomputes_only_evaluation(self):
        self.execute()
        np.save(self.gt,np.array([[1,1,1,2]],dtype=np.int64))
        value = self.execute()
        self.assertEqual(value['status'],'complete')
        self.assertEqual(value['current_invocation']['predictions_cached'],1)
        self.assertEqual(value['current_invocation']['evaluations_new'],1)
        state = benchmark.read_json(self.out/'progress.json')['frames'][self.token]
        self.assertTrue(state['gt_changed_since_manifest'])

    def test_normal_resume_uses_saved_prediction_without_initializing_model(self):
        self.execute()
        self.args.evaluate_only = False
        value = self.execute()
        self.assertEqual(value['status'],'complete')
        self.assertEqual(value['current_invocation']['predictions_cached'],1)
        self.assertEqual(value['current_invocation']['evaluations_cached'],1)
        self.assertEqual(value['current_invocation']['model_load_count'],0)
        self.assertEqual(value['current_invocation']['completed_forwards'],0)

    def test_committed_attempts_are_recovered_without_progress_pointers(self):
        self.execute()
        progress = benchmark.read_json(self.out/'progress.json')
        progress['frames'][self.token].pop('prediction_dir')
        progress['frames'][self.token].pop('evaluation_dir')
        benchmark.atomic_json(self.out/'progress.json',progress)
        value = self.execute()
        self.assertEqual(value['status'],'complete')
        self.assertEqual(value['current_invocation']['predictions_cached'],1)
        self.assertEqual(value['current_invocation']['evaluations_cached'],1)

    def test_modified_evaluation_is_recomputed_without_inference(self):
        self.execute()
        progress = benchmark.read_json(self.out/'progress.json')
        previous = Path(progress['frames'][self.token]['evaluation_dir'])
        with (previous/'evaluation.json').open('a') as stream:
            stream.write(' ')
        value = self.execute()
        self.assertEqual(value['status'],'complete')
        self.assertEqual(value['current_invocation']['predictions_cached'],1)
        self.assertEqual(value['current_invocation']['evaluations_new'],1)
        self.assertEqual(value['current_invocation']['evaluations_cached'],0)
        current = benchmark.read_json(self.out/'progress.json')['frames'][self.token]
        self.assertNotEqual(current['evaluation_dir'],str(previous))
        self.assertTrue((previous/'evaluation.json').is_file())

    def test_protocol_change_recomputes_only_evaluation(self):
        self.execute()
        self.context['protocol'] = dict(self.context['protocol'],test_version='changed')
        self.context['protocol_fingerprint'] = benchmark.fingerprint(self.context['protocol'])
        value = self.execute()
        self.assertEqual(value['current_invocation']['evaluations_new'],1)
        self.assertEqual(value['current_invocation']['predictions_cached'],1)

    def test_corrupt_prediction_same_stat_is_not_reused_or_inferred_in_cache_only(self):
        target = self.folder/'occupancy.npz'
        stat = target.stat()
        data = bytearray(target.read_bytes());data[-1] ^= 1;target.write_bytes(data)
        os.utime(target,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        result = self.execute()
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['successful_frame_count'],0)
        self.assertEqual(result['current_invocation']['predictions_new'],0)
        self.assertEqual(result['failed_frame_count'],1)

    def test_pending_directory_with_files_is_never_a_success(self):
        staged = self.folder.with_name(self.folder.name+'.pending')
        self.folder.rename(staged)
        value = self.execute()
        self.assertEqual(value['status'],'partial')
        self.assertEqual(value['current_invocation']['predictions_cached'],0)

    def test_input_and_shared_provenance_must_match(self):
        value,reason = benchmark.prediction_cache(self.folder,self.entry,self.context,'changed-input')
        self.assertIsNone(value);self.assertIn('provenance',reason)
        other = dict(self.context,shared_fingerprint='changed-code-or-checkpoint')
        value,reason = benchmark.prediction_cache(self.folder,self.entry,other,self.input_fingerprint)
        self.assertIsNone(value);self.assertIn('provenance',reason)

    def test_tampered_membership_cannot_turn_partial_split_into_complete(self):
        manifest = benchmark.read_json(self.out/'manifest.json')
        manifest['frames'] = []
        benchmark.atomic_json(self.out/'manifest.json',manifest)
        with self.assertRaisesRegex(ValueError,'frame list'):
            self.execute()

    def test_nonbenchmark_runs_are_not_adopted_or_modified(self):
        old = self.out/'old_run';old.mkdir()
        (old/'metadata.json').write_text('original source')
        for resume in (False,True):
            with self.subTest(resume=resume),self.assertRaises(ValueError):
                with benchmark.output_lock(old,resume):
                    self.fail('Cannot acquire existing nonbenchmark directory')
        self.assertEqual([p.name for p in old.iterdir()],['metadata.json'])


if __name__ == '__main__':
    unittest.main()
