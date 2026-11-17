"""Selective GT import and temporal readiness; all mutations use temporary QA data."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import zipfile

import numpy as np

from tools.forecasting.gt_archive import (
    inspect_archives,stage_missing,publish_staged,validate_member,decode_content,
)
from tools.forecasting.protocol import file_record,fingerprint,protocol
from tools.forecasting.training_readiness import (
    sensor_config,requested_train_gt,build_manifests,cache_inventory,cache_plan,NameIndex,
)
from tools.forecasting.path_snapshot import DatasetPaths


def array_bytes(rows=None):
    buf=BytesIO();np.save(buf,np.asarray([[1,2,3,4]] if rows is None else rows),allow_pickle=False);return buf.getvalue()


def basename(i=1):return f'n000-unit__LIDAR_TOP__{i}.pcd.bin.npy'


def request(i=1):
    return dict(sample_token=f'token-{i}',scene='scene-test',frame_index=i,
                lidar_filename='samples/LIDAR_TOP/'+basename(i)[:-4],expected_gt_filename=basename(i))


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.gt=self.root/'gt';self.gt.mkdir();self.staging=self.root/'staging'

    def zip(self,name='train.zip',members=None):
        path=self.root/name
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for filename,value in (members or {'nuscenes_occ/samples/'+basename():array_bytes()}).items():z.writestr(filename,value)
        return path

    def test_inspection_does_not_extract_and_only_requested_file_is_installed(self):
        path=self.zip(members={'nuscenes_occ/samples/'+basename(1):array_bytes(),
                              'nuscenes_occ/samples/'+basename(2):array_bytes()})
        index,reports=inspect_archives([path]);self.assertEqual(reports[0]['npy_count'],2)
        self.assertFalse(self.staging.exists());self.assertFalse(list(self.gt.iterdir()))
        staged=stage_missing(index,[request(1)],self.staging,self.gt)
        self.assertEqual(staged['selected_count'],1)
        self.assertFalse(list(self.gt.iterdir()))
        result=publish_staged(staged,self.gt,self.root/'receipts')
        self.assertEqual(result['installed'],1)
        self.assertTrue((self.gt/basename(1)).is_file());self.assertFalse((self.gt/basename(2)).exists())
        self.assertEqual(file_record(self.gt/basename(1)),staged['files'][0]['file'])
        self.assertNotEqual((self.gt/basename(1)).stat().st_ino,(self.staging/basename(1)).stat().st_ino)

    def test_existing_different_content_is_conflict_before_any_publish(self):
        existing=self.gt/basename(2);existing.write_bytes(array_bytes([[1,2,3,7]]));before=existing.read_bytes()
        path=self.zip(members={'nuscenes_occ/samples/'+basename(1):array_bytes(),
                              'nuscenes_occ/samples/'+basename(2):array_bytes()})
        index,_=inspect_archives([path])
        with self.assertRaisesRegex(ValueError,'conflict'):
            stage_missing(index,[request(1)],self.staging,self.gt)
        self.assertEqual(existing.read_bytes(),before);self.assertFalse((self.gt/basename(1)).exists())

    def test_identical_existing_kept_byte_for_byte(self):
        existing=self.gt/basename();existing.write_bytes(array_bytes());before=existing.stat().st_mtime_ns
        index,_=inspect_archives([self.zip()]);staged=stage_missing(index,[request()],self.staging,self.gt)
        result=publish_staged(staged,self.gt,self.root/'receipts')
        self.assertEqual(result['installed'],0);self.assertEqual(result['identical_existing_kept'],1)
        self.assertEqual(existing.stat().st_mtime_ns,before)

    def test_missing_requested_member_does_not_create_gt(self):
        index,_=inspect_archives([self.zip()])
        with self.assertRaises(FileNotFoundError):stage_missing(index,[request(2)],self.staging,self.gt)
        self.assertFalse(list(self.gt.iterdir()))

    def test_path_traversal_absolute_backslash_drive_and_noncanonical_rejected(self):
        bad=['../'+basename(),'/nuscenes_occ/samples/'+basename(),'C:/'+basename(),
             'nuscenes_occ\\samples\\'+basename(),'nuscenes_occ/../samples/'+basename(),
             'nuscenes_occ//samples/'+basename()]
        for i,name in enumerate(bad):
            with self.subTest(name=name),self.assertRaises(ValueError):inspect_archives([self.zip(f'bad-{i}.zip',{name:array_bytes()})])

    def test_symlink_special_encrypted_and_oversize_members_rejected(self):
        for mode,flags,size in [(stat.S_IFLNK|0o777,0,10),(stat.S_IFIFO|0o600,0,10),
                                (stat.S_IFREG|0o600,1,10),(stat.S_IFREG|0o600,0,65*1024*1024)]:
            info=zipfile.ZipInfo('nuscenes_occ/samples/'+basename());info.external_attr=mode<<16;info.flag_bits=flags;info.file_size=size
            with self.subTest(mode=mode,flags=flags,size=size),self.assertRaises(ValueError):validate_member(info)

    def test_null_name_not_silently_truncated(self):
        info=zipfile.ZipInfo('nuscenes_occ/samples/'+basename()+'\x00extra');info.file_size=1
        with self.assertRaises(ValueError):validate_member(info)

    def test_duplicate_archive_member_rejected(self):
        path=self.root/'duplicate.zip'
        with zipfile.ZipFile(path,'w') as z:
            z.writestr('nuscenes_occ/samples/'+basename(),array_bytes())
            with self.assertWarns(UserWarning):z.writestr('nuscenes_occ/samples/'+basename(),array_bytes())
        with self.assertRaisesRegex(ValueError,'Duplicate'):inspect_archives([path])

    def test_same_name_different_content_across_archives_rejected(self):
        a=self.zip('a.zip');b=self.zip('b.zip',{'nuscenes_occ/samples/'+basename():array_bytes([[1,2,3,7]])})
        index,_=inspect_archives([a,b])
        with self.assertRaisesRegex(ValueError,'Conflicting'):stage_missing(index,[request()],self.staging,self.gt)

    def test_corrupt_zip_crc_rejected_before_install(self):
        path=self.root/'crc.zip'
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:z.writestr('nuscenes_occ/samples/'+basename(),array_bytes())
        index,_=inspect_archives([path]);content=bytearray(path.read_bytes());needle=array_bytes();at=content.index(needle)
        content[at+len(needle)-1]^=1;path.write_bytes(content)
        with self.assertRaises(zipfile.BadZipFile):stage_missing(index,[request()],self.staging,self.gt)
        self.assertFalse(list(self.gt.iterdir()))

    def test_staged_hash_change_blocks_publication(self):
        index,_=inspect_archives([self.zip()]);staged=stage_missing(index,[request()],self.staging,self.gt)
        (self.staging/basename()).write_bytes(array_bytes([[1,2,3,7]]))
        with self.assertRaisesRegex(ValueError,'hash changed'):publish_staged(staged,self.gt,self.root/'receipts')
        self.assertFalse(list(self.gt.iterdir()))

    def test_symlink_destination_escape_rejected_without_touching_outside(self):
        index,_=inspect_archives([self.zip()]);staged=stage_missing(index,[request()],self.staging,self.gt)
        outside=self.root/'outside.npy';outside.write_bytes(b'unchanged')
        (self.gt/basename()).symlink_to(outside)
        with self.assertRaisesRegex(ValueError,'Symlink'):publish_staged(staged,self.gt,self.root/'receipts')
        self.assertEqual(outside.read_bytes(),b'unchanged')

    def test_invalid_gt_shape_label_oob_conflict_nan_rejected_in_staging(self):
        examples=[[[0,0,0]],[[200,0,0,1]],[[0,0,16,1]],[[0,0,0,17]],
                  [[0,0,0,np.nan]],[[0,0,0,1],[0,0,0,2]]]
        for i,rows in enumerate(examples):
            index,_=inspect_archives([self.zip(f'invalid-{i}.zip',{'nuscenes_occ/samples/'+basename():array_bytes(rows)})])
            with self.subTest(rows=rows),self.assertRaises(ValueError):stage_missing(index,[request()],self.root/f'stage-{i}',self.gt)
        self.assertFalse(list(self.gt.iterdir()))

    def test_sparse_zero_ignore_and_unlisted_empty_use_original_decoder(self):
        _,dense,stats=decode_content(array_bytes([[0,0,0,0],[1,2,3,4]]))
        self.assertEqual(dense[0,0,0],255);self.assertEqual(dense[1,2,3],4);self.assertEqual(dense[0,0,1],0)
        self.assertEqual(stats['ignored_voxels'],1)

    def test_invalid_header_payload_and_object_arrays_rejected(self):
        with self.assertRaisesRegex(ValueError,'length'):decode_content(array_bytes()[:-8])
        stream=BytesIO();np.save(stream,np.array([[object(),object(),object(),object()]],dtype=object),allow_pickle=True)
        with self.assertRaisesRegex(ValueError,'numeric'):decode_content(stream.getvalue())
        stream=BytesIO();np.lib.format.write_array_header_1_0(stream,dict(descr='<i8',fortran_order=False,shape=(10**10,4)))
        with self.assertRaisesRegex(ValueError,'length'):decode_content(stream.getvalue())


class ConfigAndSplitTests(unittest.TestCase):
    def test_literal_sensor_config_never_executes_config_code(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'config.py'
            path.write_text("camera_order=['a','b','c','d','e','f']\nradar_order=['a','b','c','d','e']\nlidar_sweep_loader=dict(sweeps_num=10)\nradar_loader=dict(sweeps_num=6)\n")
            cfg=sensor_config(path,file_record(path)['sha256']);self.assertEqual(cfg.radar_loader.sweeps_num,6)
            with self.assertRaisesRegex(ValueError,'hash'):sensor_config(path,'stale')
            path.write_text("camera_order=__import__('os').system('false')\n")
            with self.assertRaises(ValueError):sensor_config(path,file_record(path)['sha256'])

    def test_prior_missing_list_is_bound_to_api_train_token_and_filename(self):
        req=request();token=req['sample_token']
        item=dict(split='mini_train',scene=dict(name='scene-test'),frame_index=1,
            sample=dict(data={'LIDAR_TOP':'lidar'}),lidar=dict(token='lidar',sample_token=token,filename=req['lidar_filename'],timestamp=1))
        cat=SimpleNamespace(samples={token:item})
        old=dict(splits={'mini_train':dict(frames=[dict(**req,status='missing_gt')])})
        self.assertEqual(len(requested_train_gt(old,cat)),1)
        item['split']='mini_val'
        with self.assertRaisesRegex(ValueError,'split'):requested_train_gt(old,cat)


def synthetic_inventory():
    frames=[];raw={};gt={};selected={}
    for split,scene in [('mini_train','train-scene'),('mini_val','val-scene')]:
        for i in range(10):
            token=f'{scene}-{i}';t=i*500_000
            frame=dict(sample_token=token,scene=scene,split=split,frame_index=i,reference_timestamp_us=t,
                lidar_sample_data_token='lidar-'+token,lidar_filename=token+'.bin',valid_v1_cache=split=='mini_val',legacy_valid_output_folders=[])
            frames.append(frame);raw[token]=dict(adapter_valid=True,latest_used_timestamp_us=t,causal_at_own_lidar=True,errors=[])
            gt[token]=dict(status='valid_gt',gt_path='target-only')
            if split=='mini_val':selected[token]=dict(latest_used_timestamp_us=t,artifact_hashes={'occupancy.npz':{'bytes':100},'metadata.json':{'bytes':200}},prediction_commit={'bytes':10})
    cat=SimpleNamespace(splits={'mini_train':['train-scene'],'mini_val':['val-scene']})
    return cat,frames,raw,gt,selected


class ManifestTests(unittest.TestCase):
    def test_gt_cannot_replace_missing_history_prediction(self):
        cat,frames,raw,gt,selected=synthetic_inventory()
        manifests=build_manifests(cat,frames,raw,gt,selected,fingerprint(protocol()))
        self.assertEqual(manifests['mini_train']['ready_windows'],0)
        self.assertGreater(manifests['mini_train']['potential_windows_after_cache'],0)
        for w in manifests['mini_train']['windows']:
            for history in w['history']:
                if history:
                    self.assertIsNone(history['history_input']['cache'])
                    self.assertNotIn('gt_path',history);self.assertNotIn('target_gt',history)

    def test_train_and_validation_scenes_are_disjoint_and_eval_only(self):
        cat,frames,raw,gt,selected=synthetic_inventory();m=build_manifests(cat,frames,raw,gt,selected,'v1')
        self.assertEqual(m['mini_val']['purpose'],'evaluation_only_not_training')
        self.assertTrue(set(m['mini_train']['official_scenes']).isdisjoint(m['mini_val']['official_scenes']))
        for split in m:
            self.assertTrue(all(w['scene'] in m[split]['official_scenes'] for w in m[split]['windows']))
        cat.splits['mini_train'].append('val-scene')
        with self.assertRaisesRegex(ValueError,'overlap'):build_manifests(cat,frames,raw,gt,selected,'v1')

    def test_camera_after_anchor_cutoff_blocks_potential_window(self):
        cat,frames,raw,gt,selected=synthetic_inventory()
        raw['train-scene-4']['latest_used_timestamp_us']+=1
        m=build_manifests(cat,frames,raw,gt,selected,'v1')
        w=next(w for w in m['mini_train']['windows'] if w['window_id']=='train-scene-4')
        self.assertFalse(w['potential_after_verified_cache_generation']);self.assertIn('input_after_cutoff',w['potential_blockers'])

    def test_cache_plan_is_nonexecuting_and_storage_uses_real_reference_sizes(self):
        cat,frames,raw,gt,selected=synthetic_inventory();m=build_manifests(cat,frames,raw,gt,selected,'v1')
        raw['train-scene-0']['adapter_valid']=False;raw['train-scene-0']['errors']=[dict(code='no_past_radar')]
        plan=cache_plan(frames,raw,m,dict(selected=selected))
        self.assertTrue(plan['plan_only']);self.assertFalse(plan['jobs_created']);self.assertFalse(plan['inference_launched'])
        self.assertEqual(plan['sensor_blocked_frames'],1)
        self.assertEqual(plan['sizing']['per_frame_median'],310)
        self.assertEqual(plan['sizing']['all_metadata_runnable_estimated_bytes_median'],310*plan['metadata_runnable_frames'])


class CacheTierTests(unittest.TestCase):
    def test_legacy_output_not_promoted_without_historical_input_provenance(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'legacy';folder.mkdir();(folder/'occupancy.npz').write_bytes(b'unit-fixture')
            (folder/'resolved_config.py').write_bytes(b'config')
            resolved=file_record(folder/'resolved_config.py')['sha256']
            shared=dict(checkpoint={'sha256':'declared-only'},resolved_config_sha256=resolved,config_source_sha256='cfg',tf32={},precision='float32')
            md=dict(sample_token='sample',checkpoint=shared['checkpoint'],config=dict(resolved_sha256=resolved,sha256='cfg'),precision='float32',code=dict(inference_script_sha256='old'))
            catalog=SimpleNamespace(validate_sample=lambda m,r:dict(scene=dict(name='train'),split='mini_train',frame_index=1))
            source=SimpleNamespace(catalog=catalog,shared=shared)
            with patch('tools.forecasting.training_readiness.occupancy.load_prediction',return_value=(md,dict(tf32={}),None,{})):
                result=cache_inventory(source,root,dict(frames=[]))
            self.assertEqual(result['selected_unique_count'],0)
            self.assertEqual(result['candidates'][0]['status'],'legacy_output_valid_incomplete_input_provenance')


class PathSnapshotTests(unittest.TestCase):
    def test_absolute_sensor_path_outside_root_is_rejected(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)/'data';root.mkdir()
            with self.assertRaisesRegex(ValueError,'escapes'):DatasetPaths(root).resolve(Path(temp)/'outside')

    def test_canonical_file_and_stat_equal_original_pathlib(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);path=root/'sensor.bin';path.write_bytes(b'abc')
            snapshot=DatasetPaths(root)
            self.assertEqual(snapshot.resolve(path),path.resolve())
            self.assertEqual(snapshot.stat(path).st_size,path.stat().st_size)
            self.assertEqual(snapshot.resolve(path),path.resolve())
            self.assertEqual(len(snapshot.files),1);snapshot.verify_unchanged()

    def test_inside_symlink_resolves_exactly_like_original(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);path=root/'sensor.bin';path.write_bytes(b'abc');link=root/'link';link.symlink_to(path)
            snapshot=DatasetPaths(root)
            self.assertEqual(snapshot.resolve(link),link.resolve())
            self.assertEqual(snapshot.stat(link).st_size,3);snapshot.verify_unchanged()

    def test_outside_symlink_still_rejected(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)/'data';root.mkdir();outside=Path(temp)/'outside';outside.write_bytes(b'not input')
            link=root/'link';link.symlink_to(outside)
            with self.assertRaisesRegex(ValueError,'escapes'):DatasetPaths(root).resolve(link)

    def test_changed_file_is_detected_after_cached_stat(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);path=root/'sensor.bin';path.write_bytes(b'abc');snapshot=DatasetPaths(root);snapshot.stat(path)
            path.write_bytes(b'different length')
            with self.assertRaisesRegex(ValueError,'file changed'):snapshot.verify_unchanged()

    def test_changed_directory_is_detected(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);path=root/'sensor.bin';path.write_bytes(b'abc');snapshot=DatasetPaths(root);snapshot.stat(path)
            (root/'new-file').write_bytes(b'changed directory')
            with self.assertRaisesRegex(ValueError,'directory changed'):snapshot.verify_unchanged()

    def test_adapter_path_binding_and_catalog_are_restored(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);path=root/'sensor.bin';path.write_bytes(b'abc');module=SimpleNamespace(Path=Path)
            catalog=SimpleNamespace(root=root);snapshot=DatasetPaths(root)
            with snapshot.use(catalog,module):
                self.assertEqual(module.Path(path).resolve(),path.resolve())
                self.assertTrue(module.Path(path).is_file())
                self.assertEqual((catalog.root/'sensor.bin').stat().st_size,3)
            self.assertIs(module.Path,Path);self.assertIs(catalog.root,root)

    def test_name_index_checks_containment_on_lookup(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)/'gt';root.mkdir();outside=Path(temp)/basename();outside.write_bytes(array_bytes())
            (root/basename()).symlink_to(outside)
            index=NameIndex(root)
            with self.assertRaisesRegex(ValueError,'escapes'):index.lookup(basename()[:-4])

    def test_name_index_rejects_ambiguous_basename(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ('a','b'):
                p=root/name;p.mkdir();(p/basename()).write_bytes(array_bytes())
            with self.assertRaisesRegex(ValueError,'Ambiguous'):NameIndex(root).lookup(basename()[:-4])


if __name__=='__main__':unittest.main()
