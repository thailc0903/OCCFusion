"""CPU forecasting protocol tests; synthetic fixtures never become real outputs."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from tools import evaluate_occupancy as occupancy
from tools.forecasting.dataset import SequenceDataset, select_window, window_asset_reasons
from tools.forecasting.evaluation import statistics, pooled
from tools.forecasting.geometry import anchor_from_source, centres, pull_labels, transform_points, se3
from tools.forecasting.persistence import HistoryInputs, predict
from tools.forecasting.protocol import HORIZONS_US, file_record, fingerprint, protocol, write_json
from tools.forecast_persistence import main, save_npz
from tools.forecasting.validation import future_gt_independence


def frames(count=10, scene='scene-test'):
    return [dict(scene=scene, sample_token=f'{scene}-{i}', frame_index=i, reference_timestamp_us=i * 500_000) for i in range(count)]


def history():
    labels = np.zeros((3, 2, 3, 2), dtype=np.uint8)
    labels[-1, 1, 2, 1] = 4
    return HistoryInputs(labels, np.ones_like(labels, dtype=bool), (0, 500_000, 1_000_000),
                         ((0,), (500_000,), (1_000_000,)), 1_000_000, 1_000_000)


class GeometryTests(unittest.TestCase):
    def test_exact_baseline_xyz_centre(self):
        np.testing.assert_array_equal(centres([[10, 30, 4]]), [[-44.75, -34.75, -2.75]])

    def test_identity_non_cubic_grid_preserves_xyz_and_ignore(self):
        source = (np.arange(24).reshape(4, 3, 2) % 17).astype(np.uint8)
        source[1, 2, 1] = 255
        result = pull_labels(source, np.eye(4), bounds=(0, 0, 0, 4, 3, 2), voxel_size=(1, 1, 1), grid=source.shape, chunk_size=7)
        np.testing.assert_array_equal(result.labels, source)
        self.assertTrue(result.in_bounds.all())
        self.assertFalse(result.known[1, 2, 1])
        self.assertEqual(int(result.valid.sum()), source.size - 1)

    def test_known_translation_and_oob_are_not_free(self):
        source = np.zeros((4, 3, 2), dtype=np.uint8); source[1, 2, 1] = 4
        matrix = np.eye(4); matrix[0, 3] = 1
        result = pull_labels(source, matrix, bounds=(0, 0, 0, 4, 3, 2), voxel_size=(1, 1, 1), grid=source.shape)
        self.assertEqual(result.labels[2, 2, 1], 4)
        self.assertTrue((result.labels[0] == 255).all())
        self.assertFalse(result.in_bounds[0].any())
        self.assertFalse(result.valid[0].any())

    def test_known_ninety_degree_yaw_cell_lookup(self):
        source = np.zeros((4, 4, 2), dtype=np.uint8); source[1, 0, 0] = 4
        matrix = np.array([[0, -1, 0, 4], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        result = pull_labels(source, matrix, bounds=(0, 0, 0, 4, 4, 2), voxel_size=(1, 1, 1), grid=source.shape)
        self.assertEqual(result.labels[3, 1, 0], 4)
        self.assertEqual(int((result.labels == 4).sum()), 1)

    def test_full_se3_roll_changes_height_without_axis_swap(self):
        source = np.zeros((4, 4, 4), dtype=np.uint8); source[1, 1, 0] = 7
        matrix = np.array([[1, 0, 0, 0], [0, 0, -1, 4], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
        result = pull_labels(source, matrix, bounds=(0, 0, 0, 4, 4, 4), voxel_size=(1, 1, 1), grid=source.shape)
        self.assertEqual(result.labels[1, 3, 1], 7)

    def test_cell_boundary_tie_and_half_open_maximum(self):
        source = np.array([1, 4, 7, 9], dtype=np.uint8).reshape(4, 1, 1)
        matrix = np.eye(4); matrix[0, 3] = -.5
        result = pull_labels(source, matrix, bounds=(0, 0, 0, 4, 1, 1), voxel_size=(1, 1, 1), grid=source.shape)
        np.testing.assert_array_equal(result.labels[:, 0, 0], [4, 7, 9, 255])

    def test_column_vector_relative_transform_and_roundtrip(self):
        anchor = np.array([[0, -1, 0, 10], [1, 0, 0, -4], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=float)
        source = np.eye(4); source[:3, 3] = [12, -1, 5]
        relative = anchor_from_source(anchor, source)
        np.testing.assert_array_equal(transform_points([[0, 0, 0]], relative), [[3, -2, 2]])
        points = np.array([[1, 2, 3], [-44.75, -34.75, -2.75]])
        np.testing.assert_allclose(transform_points(transform_points(points, relative), np.linalg.inv(relative)), points, atol=1e-12, rtol=0)

    def test_invalid_nonrigid_and_nonfinite_transforms_rejected(self):
        for bad in (np.diag([2, 1, 1, 1]), np.full((4, 4), np.nan), np.eye(3)):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                se3(bad)


class WindowTests(unittest.TestCase):
    def test_time_selected_offsets_and_actual_deviation(self):
        data = frames(); data[6]['reference_timestamp_us'] += 12_345
        window = select_window(data, data[4])
        self.assertFalse(window['reasons'])
        self.assertEqual([f['frame_index'] for f in window['history']], [2, 3, 4])
        self.assertEqual([f['frame_index'] for f in window['future']], [5, 6, 7])
        self.assertEqual(window['timing'][4]['error_us'], 12_345)

    def test_scene_begin_and_end_never_padded(self):
        data = frames()
        first, last = select_window(data, data[0]), select_window(data, data[-1])
        self.assertIn('history_scene_boundary', first['reasons'])
        self.assertIsNone(first['history'][0])
        self.assertIn('future_scene_boundary', last['reasons'])
        self.assertTrue(all(f is None for f in last['future']))

    def test_cross_scene_rejected(self):
        data = frames(); data[0]['scene'] = 'different'
        with self.assertRaisesRegex(ValueError, 'Scene boundary'):
            select_window(data, data[4])

    def test_gap_rejected_even_if_frame_indices_look_contiguous(self):
        data = frames()
        for f in data[5:]: f['reference_timestamp_us'] += 100_000
        window = select_window(data, data[4])
        self.assertIn('future_timestamp_tolerance', window['reasons'])

    def test_no_skipping_extra_sample_inside_timestamp_targets(self):
        data = frames()
        extra = dict(scene='scene-test', sample_token='extra', frame_index=5, reference_timestamp_us=2_750_000)
        for f in data[5:]: f['frame_index'] += 1
        data.insert(6, extra); data.sort(key=lambda f: f['reference_timestamp_us'])
        self.assertIn('nonconsecutive_samples', select_window(data, data[4])['reasons'])

    def test_interval_gap_rejected_even_with_each_offset_in_tolerance(self):
        data = frames(); data[3]['reference_timestamp_us'] += 40_000; data[4]['reference_timestamp_us'] -= 40_000
        window = select_window(data, data[5])
        self.assertIn('timestamp_gap', window['reasons'])

    def test_missing_prediction_gt_and_cutoff_have_distinct_reasons(self):
        data = frames(); w = select_window(data, data[4])
        pred = {f['sample_token']:dict(status='valid_prediction',latest_used_timestamp_us=f['reference_timestamp_us']) for f in data}
        gt = {f['sample_token']:dict(status='valid_gt') for f in data}
        del pred[data[2]['sample_token']]; gt[data[5]['sample_token']]['status'] = 'missing_gt'
        pred[data[4]['sample_token']]['latest_used_timestamp_us'] += 1
        reasons = window_asset_reasons(w['history'],w['future'],pred,gt,w['input_cutoff_us'])
        self.assertEqual(set(reasons), {'missing_prediction','missing_gt','input_after_cutoff'})

    def test_older_frame_later_camera_allowed_only_before_anchor_cutoff(self):
        data = frames();w = select_window(data,data[4])
        pred={f['sample_token']:dict(status='valid_prediction',latest_used_timestamp_us=f['reference_timestamp_us']) for f in data}
        gt={f['sample_token']:dict(status='valid_gt') for f in data}
        pred[data[2]['sample_token']]['latest_used_timestamp_us'] += 1_000
        self.assertFalse(window_asset_reasons(w['history'],w['future'],pred,gt,w['input_cutoff_us']))


class PersistenceAndMetricTests(unittest.TestCase):
    def test_persistence_copies_anchor_for_all_horizons_not_oldest_history(self):
        inputs=history();inputs.labels[0].fill(7)
        result=predict(inputs)
        self.assertEqual(result.shape[0],len(HORIZONS_US))
        for forecast in result:np.testing.assert_array_equal(forecast,inputs.labels[-1])
        result[0].fill(1)
        self.assertEqual(inputs.labels[-1,0,0,0],0)

    def test_future_sensor_rejected_and_unknown_anchor_not_filled(self):
        inputs=history()
        with self.assertRaisesRegex(ValueError,'Noncausal'):
            predict(HistoryInputs(inputs.labels,inputs.valid,inputs.timestamps_us,((0,),(500_000,),(1_000_001,)),1_000_000,1_000_000))
        inputs.labels[-1,0,0,0]=255;inputs.valid[-1,0,0,0]=False
        with self.assertRaisesRegex(ValueError,'complete saved anchor'):
            predict(inputs)

    def test_modifying_future_gt_file_does_not_change_forecast(self):
        inputs=history()
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'future_gt.npy'
            np.save(path,inputs.labels[-1])
            a=predict(inputs);gt=np.load(path,allow_pickle=False)
            first=occupancy.confusion_statistics(a[0],gt)
            np.save(path,np.full_like(gt,7))
            b=predict(inputs);gt=np.load(path,allow_pickle=False)
            second=occupancy.confusion_statistics(b[0],gt)
            np.testing.assert_array_equal(a,b)
            self.assertNotEqual(fingerprint(first.tolist()),fingerprint(second.tolist()))

    def test_history_loader_does_not_even_require_future_field_or_pose(self):
        ds=SequenceDataset.__new__(SequenceDataset)
        tokens=['past2','past1','anchor'];times=[0,500_000,1_000_000]
        ds.labels={t:np.full(occupancy.GRID,i,dtype=np.uint8) for i,t in enumerate(tokens)}
        ds.predictions={t:dict(status='valid_prediction',sensors=[dict(timestamp_us=ts,ego_pose_timestamp_us=ts,used_in_prediction=True)]) for t,ts in zip(tokens,times)}
        ds.global_from_lidar=Mock(side_effect=lambda t:np.eye(4) if t in tokens else (_ for _ in ()).throw(AssertionError('Future pose read')))
        window=dict(status='valid',anchor=dict(sample_token='anchor',reference_timestamp_us=1_000_000),input_cutoff_us=1_000_000,
                    history=[dict(sample_token=t,reference_timestamp_us=ts) for t,ts in zip(tokens,times)])
        inputs,_=ds.load_history(window)
        self.assertTrue((predict(inputs)==2).all())
        self.assertTrue(all(call.args[0] in tokens for call in ds.global_from_lidar.call_args_list))

    def test_future_gt_perturbation_check_restores_original_targets(self):
        inputs=history();original=np.full((2,3,2),4,dtype=np.uint8)
        dataset=SimpleNamespace(dense_gt={'future':original})
        result=future_gt_independence(dataset,dict(future=[dict(sample_token='future')]),inputs,predict(inputs))
        self.assertTrue(result['passed'])
        self.assertIs(dataset.dense_gt['future'],original)
        self.assertTrue((original==4).all())

    def test_ignore_and_oob_mask_do_not_become_free(self):
        target=np.array([1,255,255,0],dtype=np.uint8);pred=np.array([1,7,4,1],dtype=np.uint8)
        inside=np.array([True,True,False,True]);known=np.array([True,False,False,True])
        result=occupancy.summarize(statistics(pred,target,inside,known))
        self.assertEqual(result['valid_voxel_count'],2)
        self.assertEqual(result['occupancy_iou'],.5)
        self.assertEqual(result['semantic_miou'],.5/16)
        self.assertTrue(occupancy.compare_reference([target],[pred],result)['passed'])

    def test_target_mask_cannot_hide_a_valid_label(self):
        with self.assertRaisesRegex(ValueError,'mask disagree'):
            statistics(np.array([0]),np.array([1]),np.array([False]),np.array([False]))

    def test_pooled_hand_computed_counts_not_average_window_iou(self):
        gt=[np.ones(9,dtype=int),np.ones(1,dtype=int)]
        pred=[np.ones(9,dtype=int),np.zeros(1,dtype=int)]
        matrices=[occupancy.confusion_statistics(p,g) for p,g in zip(pred,gt)]
        result=pooled(matrices)
        self.assertEqual(result['occupancy_iou'],.9)
        self.assertEqual(result['semantic_miou'],.9/16)
        self.assertEqual(result['valid_voxel_count'],10)
        self.assertTrue(occupancy.compare_reference(gt,pred,result)['passed'])
        self.assertNotEqual(result['occupancy_iou'],np.mean([occupancy.summarize(m)['occupancy_iou'] for m in matrices]))

    def test_union_zero_and_all_sixteen_classes_absent_or_present(self):
        result=occupancy.summarize(occupancy.confusion_statistics(np.array([0]),np.array([0])))
        self.assertEqual(result['occupancy_iou'],0)
        self.assertEqual(result['semantic_miou'],0)
        self.assertEqual(len(result['per_class_semantic']),16)

    def test_original_surroundocc_sparse_decoding_is_reused(self):
        gt,stats=occupancy.sparse_to_dense(np.array([[0,0,0,0],[1,1,1,4]]),grid=(2,2,2))
        self.assertEqual(gt[0,0,0],255);self.assertEqual(gt[0,0,1],0);self.assertEqual(gt[1,1,1],4)
        self.assertEqual(stats['ignored_voxels'],1)
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            occupancy.sparse_to_dense(np.array([[0,0,0,1],[0,0,0,2]]))


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);root=Path(self.temp.name)
        self.folder=root/'frames/sample/done';self.folder.mkdir(parents=True)
        ds=SequenceDataset.__new__(SequenceDataset);self.ds=ds
        ds.root=root;ds.frames=[dict(sample_token='sample',scene='fixture',frame_index=4,reference_timestamp_us=1)]
        ds.shared=dict(checkpoint={'sha256':'declared-no-checkpoint-file'},resolved_config_sha256=fingerprint('config'),config_source_sha256='config-source',tf32={},precision='float32',environment={})
        ds.shared_fingerprint='shared';ds.predictions={};ds.labels={};ds.metadata={}
        ds.catalog=SimpleNamespace(validate_sample=lambda m,r:None)
        ds._sensor_audit=Mock(return_value=[dict(timestamp_us=1,ego_pose_timestamp_us=1,used_in_prediction=True)])
        config=b'fixture resolved config';(self.folder/'resolved_config.py').write_bytes(config)
        ds.shared['resolved_config_sha256']=file_record(self.folder/'resolved_config.py')['sha256']
        self.metadata=dict(status='success',result_source='prediction',sample_token='sample',precision='float32',
            bounds=list(occupancy.BOUNDS),voxel_size=list(occupancy.VOXEL_SIZE),grid_shape=list(occupancy.GRID),grid_axis_order=['x','y','z'],
            voxel_center='bounds_min + (index + 0.5) * voxel_size',coordinate_system=dict(name='LIDAR_TOP at reference_timestamp_us',unit='metres'),
            class_names=list(occupancy.CLASS_NAMES),free_label=0,gt_ignore_label=255,checkpoint=ds.shared['checkpoint'],
            config=dict(sha256='config-source',resolved_sha256=ds.shared['resolved_config_sha256']),environment={},
            benchmark_provenance=dict(shared_fingerprint='shared',input_fingerprint='input'))
        self.report=dict(status='success',sample_token='sample',precision='float32',tf32={},
            prediction=dict(predicted_voxel_counts={'0':int(np.prod(occupancy.GRID))},logits_shape=[1,*occupancy.GRID,17]))
        (self.folder/'metadata.json').write_text(json.dumps(self.metadata));(self.folder/'report.json').write_text(json.dumps(self.report))
        (self.folder/'bev.png').write_bytes(b'fixture-bev')
        np.savez_compressed(self.folder/'occupancy.npz',labels=np.zeros(occupancy.GRID,dtype=np.uint8),max_softmax_probability=np.ones(occupancy.GRID,dtype=np.float32))
        self.commit()

    def commit(self):
        hashes={n:file_record(self.folder/n) for n in [*occupancy.ARTIFACTS,'resolved_config.py']}
        receipt=dict(status='success',sample_token='sample',shared_fingerprint='shared',input_fingerprint='input',artifact_hashes=hashes)
        (self.folder/'prediction_commit.json').write_text(json.dumps(receipt))
        self.ds.progress=dict(frames={'sample':dict(status='success',prediction_dir=str(self.folder),prediction_sha256=hashes['occupancy.npz']['sha256'])})

    def test_complete_fixture_uses_real_two_key_npz_schema(self):
        result=self.ds.audit_predictions()
        self.assertEqual(result['status_counts'],{'valid_prediction':1})
        self.assertFalse(result['frames'][0]['logits_or_features_available'])

    def test_corrupt_artifact_hash_is_rejected(self):
        (self.folder/'bev.png').write_bytes(b'tampered')
        result=self.ds.audit_predictions()['frames'][0]
        self.assertEqual(result['status'],'invalid_prediction');self.assertIn('hash mismatch',result['error'])

    def test_wrong_token_is_rejected(self):
        self.metadata['sample_token']='other';(self.folder/'metadata.json').write_text(json.dumps(self.metadata));self.commit()
        result=self.ds.audit_predictions()['frames'][0]
        self.assertEqual(result['status'],'invalid_prediction');self.assertIn('token',result['error'])

    def test_incomplete_prediction_not_backfilled(self):
        self.ds.progress['frames']['sample']=dict(status='failed',error={'message':'missing past radar'})
        result=self.ds.audit_predictions()['frames'][0]
        self.assertEqual(result['status'],'missing_prediction')
        self.ds._sensor_audit.assert_not_called()

    def test_unpublished_pending_is_not_a_prediction(self):
        pending=self.folder.with_name('attempt.pending');self.folder.rename(pending)
        self.ds.progress['frames']['sample']['prediction_dir']=str(pending)
        result=self.ds.audit_predictions()['frames'][0]
        self.assertEqual(result['status'],'invalid_prediction');self.assertIn('Unpublished',result['error'])

    def configure_gt(self, path):
        self.ds.catalog=SimpleNamespace(samples={'sample':dict(split='mini_val',frame_index=4,
            scene=dict(name='fixture',token='fixture-scene'),sample=dict(timestamp=1),
            lidar=dict(timestamp=1,token='lidar-token',filename='frame.bin'))},splits={'mini_train':[],'mini_val':['fixture']})
        self.ds.gt_index=SimpleNamespace(root=Path(self.temp.name),lookup=lambda _:('frame.bin.npy',path))
        self.ds.source_frames={'sample':dict(gt_initial={'sha256':'stale-hash','bytes':0})}
        self.ds.gt={};self.ds.dense_gt={}

    def test_gt_hash_mismatch_is_not_evaluated(self):
        path=Path(self.temp.name)/'frame.bin.npy';np.save(path,np.array([[1,1,1,4]]))
        self.configure_gt(path)
        result=self.ds.audit_gt()['splits']['mini_val']['frames'][0]
        self.assertEqual(result['status'],'invalid_gt');self.assertIn('GT hash',result['error'])
        self.assertNotIn('sample',self.ds.dense_gt)

    def test_missing_gt_is_reported_without_replacement(self):
        self.configure_gt(None)
        result=self.ds.audit_gt()['splits']['mini_val']
        self.assertEqual(result['status_counts'],{'missing_gt':1})
        self.assertEqual(result['scenes']['fixture']['missing'][0]['sample_token'],'sample')


class SafetyTests(unittest.TestCase):
    def test_outputs_cannot_overwrite_or_escape_forecasting(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'result.json';write_json(path,{'old':True})
            with self.assertRaises(FileExistsError):write_json(path,{'new':True})
            self.assertEqual(json.loads(path.read_text()),{'old':True})
            with self.assertRaises(SystemExit):main(['--out-dir',temp])

    def test_npz_publish_is_exclusive(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'forecast.npz';save_npz(path,labels=np.array([4]))
            before=path.read_bytes()
            with self.assertRaises(FileExistsError):save_npz(path,labels=np.array([7]))
            self.assertEqual(path.read_bytes(),before)

    def test_model_framework_import_guard_in_fresh_cpu_process(self):
        code='import sys; from tools.forecast_persistence import CPUOnlyImports; sys.meta_path.insert(0,CPUOnlyImports()); import torch'
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('CPU contract forbids',result.stderr)

    def test_protocol_records_null_unimplemented_flow_and_sixteen_class_mean(self):
        p=protocol();self.assertEqual(p['semantic_mean_class_count'],16)
        for name in ['flow','EPE','ADE','FDE']:self.assertIsNone(p['unsupported'][name])


if __name__=='__main__':unittest.main()
