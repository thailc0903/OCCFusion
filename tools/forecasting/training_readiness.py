"""CPU inventories and plans only; no predictor, queue or model execution."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import numpy as np

from tools import evaluate_occupancy as occupancy
from tools import infer_nuscenes_mini as adapter
from tools.forecasting.dataset import SequenceDataset, select_window
from tools.forecasting.gt_archive import decode_content, contained
from tools.forecasting.geometry import se3
from tools.forecasting.protocol import file_record, fingerprint, protocol


def safe_literal(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict' and not node.args:
        if any(k.arg is None for k in node.keywords):
            raise ValueError('Config dict expansion is not allowed')
        return {k.arg:safe_literal(k.value) for k in node.keywords}
    if isinstance(node, ast.List):
        return [safe_literal(n) for n in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(safe_literal(n) for n in node.elts)
    if isinstance(node, ast.Dict):
        return {safe_literal(k):safe_literal(v) for k,v in zip(node.keys,node.values)}
    return ast.literal_eval(node)


def sensor_config(resolved_path, expected_hash):
    if file_record(resolved_path)['sha256'] != expected_hash:
        raise ValueError('Resolved baseline config hash mismatch')
    wanted = {'camera_order','radar_order','lidar_sweep_loader','radar_loader'}
    values = {}
    for node in ast.parse(Path(resolved_path).read_bytes()).body:
        if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in wanted:
            values[node.targets[0].id] = safe_literal(node.value)
    if set(values) != wanted or len(values['camera_order'])!=6 or len(values['radar_order'])!=5:
        raise ValueError('Incomplete pinned sensor config')
    return SimpleNamespace(camera_order=values['camera_order'], radar_order=values['radar_order'],
        lidar_sweep_loader=SimpleNamespace(**values['lidar_sweep_loader']), radar_loader=SimpleNamespace(**values['radar_loader']))


def requested_train_gt(previous_audit, catalog):
    """Authorize only the missing official mini_train tokens in the previous audit."""
    requested = []
    missing = [f for f in previous_audit['splits']['mini_train']['frames'] if f['status']=='missing_gt']
    seen = set()
    for frame in missing:
        token = frame['sample_token']
        item = catalog.samples[token]
        if token in seen or item['split']!='mini_train' or frame['scene']!=item['scene']['name'] or frame['frame_index']!=item['frame_index']:
            raise ValueError('Prior missing-GT token/split/scene mismatch')
        seen.add(token)
        expected = Path(item['lidar']['filename']).name+'.npy'
        if frame['expected_gt_filename']!=expected or frame['lidar_filename']!=item['lidar']['filename']:
            raise ValueError('Prior GT filename does not match nuScenes LiDAR')
        if item['sample']['data']['LIDAR_TOP']!=item['lidar']['token'] or item['lidar']['sample_token']!=token:
            raise ValueError('LiDAR sample_data token mismatch')
        requested.append(dict(sample_token=token,scene=item['scene']['name'],frame_index=item['frame_index'],
                              split='mini_train',lidar_filename=item['lidar']['filename'],
                              lidar_sample_data_token=item['lidar']['token'],timestamp_us=item['lidar']['timestamp'],
                              expected_gt_filename=expected))
    return requested


class NameIndex(occupancy.GroundTruthIndex):
    """Index names once; resolve/confine only GT actually selected for use."""
    def __init__(self,root):
        self.root=Path(root).resolve();self.files=defaultdict(list)
        if self.root.is_dir():
            for path in self.root.rglob('*.npy'):
                if path.is_file():self.files[path.name].append(path)

    def lookup(self,lidar_filename):
        key,path=super().lookup(lidar_filename)
        if path is not None:
            path=path.resolve()
            if not path.is_relative_to(self.root):raise ValueError('GT escapes configured root')
        return key,path


def make_source(data_root,benchmark_root,gt_root):
    """Use the unchanged SequenceDataset manifest validator without eager GT resolve."""
    source=SequenceDataset.__new__(SequenceDataset)
    source.catalog=occupancy.DatasetCatalog(data_root);source.root=Path(benchmark_root).resolve()
    source.gt_index=NameIndex(gt_root)
    source.manifest=occupancy.json_object((source.root/'manifest.json').read_bytes())
    source.progress=occupancy.json_object((source.root/'progress.json').read_bytes())
    source.source_hashes={n:file_record(source.root/n) for n in ('manifest.json','progress.json')}
    source.file_hash_cache={};source.predictions={};source.gt={};source.dense_gt={};source.labels={};source.metadata={};source.geometry_errors=[]
    source.frames=sorted((source.frame_record(t,x) for t,x in source.catalog.samples.items() if x['split']=='mini_val'),
        key=lambda f:(f['scene'],f['reference_timestamp_us'],f['sample_token']))
    source._validate_source_manifest()
    return source


def audit_all_gt(catalog, gt_root, previous_audit):
    index = NameIndex(gt_root)
    old = {f['sample_token']:f for s in previous_audit['splits'].values() for f in s['frames']}
    frames = {}
    for token,item in sorted(catalog.samples.items(),key=lambda pair:(pair[1]['scene']['name'],pair[1]['frame_index'])):
        key,path = index.lookup(item['lidar']['filename'])
        value = dict(sample_token=token,scene=item['scene']['name'],split=item['split'],frame_index=item['frame_index'],
                     timestamp_us=item['lidar']['timestamp'],lidar_sample_data_token=item['lidar']['token'],
                     lidar_filename=item['lidar']['filename'],expected_gt_filename=key,gt_path=str(path) if path else None)
        if path is None:
            value['status']='missing_gt'
        else:
            try:
                content = path.read_bytes();record=dict(sha256=occupancy.sha256(content),bytes=len(content))
                if old[token]['status']=='valid_gt' and old[token]['file']!=record:
                    raise ValueError('Previously valid GT hash changed')
                rows,_,stats=decode_content(content)
                value.update(status='valid_gt',file=record,shape=list(rows.shape),dtype=str(rows.dtype),decoder=stats,
                             geometry='Declared SurroundOcc XYZ 200x200x16 / 0.5m protocol; sparse NPY has no embedded geometry/token')
            except (OSError,ValueError,TypeError,EOFError) as error:
                value.update(status='invalid_gt',error=f'{type(error).__name__}: {error}')
        frames[token]=value
    return frames


def raw_inventory(catalog, cfg, path_snapshot=None):
    """Metadata/file audit; does not decode images/points or invoke preprocessing."""
    helper=SequenceDataset.__new__(SequenceDataset);helper.catalog=catalog
    output={}
    for token,item in sorted(catalog.samples.items(),key=lambda pair:(pair[1]['scene']['name'],pair[1]['frame_index'])):
        reference=item['lidar']['timestamp'];records={};errors=[];keyframes={};used=set();selection={};excluded={}
        def record(sd_token,channel,role):
            sd=catalog.nusc.get('sample_data',sd_token)
            cal=catalog.nusc.get('calibrated_sensor',sd['calibrated_sensor_token'])
            ego=catalog.nusc.get('ego_pose',sd['ego_pose_token'])
            actual_channel=catalog.nusc.get('sensor',cal['sensor_token'])['channel']
            if actual_channel!=channel or catalog.nusc.get('sample',sd['sample_token'])['scene_token']!=item['scene']['token']:
                raise ValueError('Sensor channel/scene mismatch')
            if path_snapshot is None:
                path=contained(catalog.root/sd['filename'],catalog.root)
                size=path.stat().st_size if path.is_file() else None
            else:
                path=path_snapshot.resolve(catalog.root/sd['filename'])
                try:
                    info=path_snapshot.stat(path);size=info.st_size if stat.S_ISREG(info.st_mode) else None
                except FileNotFoundError:size=None
            r=records.setdefault(sd_token,dict(channel=channel,sample_data_token=sd_token,timestamp_us=int(sd['timestamp']),
                ego_pose_timestamp_us=int(ego['timestamp']),calibrated_sensor_token=cal['token'],ego_pose_token=ego['token'],
                delta_to_frame_lidar_us=int(sd['timestamp']-reference),path=str(path),file_exists=size is not None,
                bytes=size,roles=[],used_in_prediction=False))
            r['roles'].append(role)
            return r
        channels=[*cfg.camera_order,'LIDAR_TOP',*cfg.radar_order]
        try:
            for channel in channels:
                r=record(item['sample']['data'][channel],channel,'keyframe');keyframes[channel]=r
                if channel in cfg.camera_order or channel=='LIDAR_TOP':used.add(r['sample_data_token'])
            tokens,_=helper._past_tokens(item['lidar']['prev'],cfg.lidar_sweep_loader.sweeps_num,reference,item['scene']['token'])
            selection['LIDAR_TOP']=tokens
            for sd_token in tokens:record(sd_token,'LIDAR_TOP','past_sweep');used.add(sd_token)
            for channel in cfg.radar_order:
                tokens,skipped=helper._past_tokens(item['sample']['data'][channel],cfg.radar_loader.sweeps_num,reference,item['scene']['token'])
                selection[channel]=tokens;excluded[channel]=skipped
                if not tokens:errors.append(dict(code='no_past_radar',channel=channel,reference_timestamp_us=reference))
                for sd_token in tokens:record(sd_token,channel,'radar_sweep');used.add(sd_token)
        except (OSError,ValueError,KeyError,TypeError) as error:
            errors.append(dict(code='sensor_metadata_error',message=f'{type(error).__name__}: {error}'))
        for sd_token,r in records.items():
            r['used_in_prediction']=sd_token in used
            if not r['file_exists'] or not r['bytes']:
                errors.append(dict(code='missing_or_empty_sensor_file',channel=r['channel'],sample_data_token=sd_token,path=r['path']))
        # Reuse the actual verified CLI adapter, with literal config, no model/framework imports.
        warnings=[]
        try:
            _,_,base=adapter.make_frame_info(catalog.nusc,item['sample'],cfg,warnings)
            all_records=[*base['sensors']['keyframes'].values(),*base['sensors']['lidar_past_sweeps'],
                         *(r for arr in base['sensors']['radar_sweeps'].values() for r in arr)]
            for r in all_records:
                for name in ('sensor_to_ego','ego_to_global','sensor_to_reference_lidar'):se3(r[name])
            expected={k:[r['sample_data_token'] for r in v] for k,v in base['sensors']['radar_sweeps'].items()}
            expected['LIDAR_TOP']=[r['sample_data_token'] for r in base['sensors']['lidar_past_sweeps']]
            if expected!=selection or errors:
                raise ValueError('Readiness metadata selection disagrees with original adapter')
            adapter_valid=True;adapter_error=None
        except (OSError,ValueError,KeyError,TypeError) as error:
            adapter_valid=False;adapter_error=f'{type(error).__name__}: {error}'
            if not errors:errors.append(dict(code='adapter_rejected',message=adapter_error))
        times=[max(r['timestamp_us'],r['ego_pose_timestamp_us']) for r in records.values() if r['used_in_prediction']]
        raw_keyframes_present=len(keyframes)==12 and all(r['file_exists'] and r['bytes'] for r in keyframes.values())
        output[token]=dict(sample_token=token,scene=item['scene']['name'],split=item['split'],frame_index=item['frame_index'],
            reference_timestamp_us=int(reference),raw_keyframes_present=bool(raw_keyframes_present),adapter_valid=adapter_valid,
            adapter_error=adapter_error,errors=errors,warnings=warnings,earliest_used_timestamp_us=min(times) if times else None,
            latest_used_timestamp_us=max(times) if times else None,
            causal_at_own_lidar=bool(times) and max(times)<=reference,
            requested_sweeps=dict(lidar=cfg.lidar_sweep_loader.sweeps_num,radar_each=cfg.radar_loader.sweeps_num),
            selected_sweeps=selection,excluded_future_radar=excluded,records=list(records.values()),
            audit_scope='metadata/calibration adapter + existence/nonempty file; no image/point decode, no model, not a forward-success guarantee')
    return output


def publication_pin(folder, hashes):
    marker=json.loads((folder/'job_result.json').read_bytes())
    core={k:v for k,v in marker.items() if k!='fingerprint_sha256'}
    digest=hashlib.sha256(json.dumps(core,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    if marker['status']!='complete' or marker['run_id']!=folder.name or marker.get('execution_mode')!='real' or digest!=marker['fingerprint_sha256'] or marker['artifacts']!=hashes:
        raise ValueError('Job publication hash/status/real-execution mismatch')
    return dict(kind='real_job_publication',file=file_record(folder/'job_result.json'),sample_token=marker['sample_token'])


def cache_inventory(source, runs_root, previous_inputs):
    """Keep strict v1 caches separate from legacy valid-output/incomplete-input bundles."""
    prior={f['sample_token']:f.get('source_dir') for f in previous_inputs['frames'] if f['status']=='valid_prediction'}
    candidates=[];selected={}
    for npz in sorted(Path(runs_root).rglob('occupancy.npz')):
        folder=npz.parent.resolve()
        relative=folder.relative_to(Path(runs_root).resolve())
        if any(p.startswith('.') or p.endswith('.pending') for p in relative.parts):continue
        result=dict(folder=str(folder),status='invalid_cache')
        try:
            md,rr,_,hashes=occupancy.load_prediction(folder)
            token=md['sample_token'];item=source.catalog.validate_sample(md,rr)
            result.update(sample_token=token,scene=item['scene']['name'],split=item['split'],frame_index=item['frame_index'])
            hashes['resolved_config.py']=file_record(folder/'resolved_config.py')
            result['artifact_hashes']=hashes
            if md['checkpoint']!=source.shared['checkpoint'] or md['config']['resolved_sha256']!=source.shared['resolved_config_sha256'] or md['config']['sha256']!=source.shared['config_source_sha256'] or hashes['resolved_config.py']['sha256']!=source.shared['resolved_config_sha256'] or rr.get('tf32')!=source.shared['tf32'] or md['precision']!=source.shared['precision']:
                raise ValueError('Saved baseline checkpoint/config/precision mismatch')
            marker=folder/'prediction_commit.json'
            if not marker.exists():
                pin=publication_pin(folder,hashes) if (folder/'job_result.json').exists() else None
                if pin and pin['sample_token']!=token:raise ValueError('Publication sample token mismatch')
                result.update(status='legacy_output_valid_incomplete_input_provenance',bundle_hash_pin=pin,
                    reason='Successful shape/token/geometry/baseline artifact; no historical raw-sensor hashes/input fingerprint required by v1 benchmark cache. Current hashes cannot retroactively prove original inputs.',
                    inference_script_sha256=md['code'].get('inference_script_sha256'),strict_training_cache_usable=False)
            else:
                receipt=json.loads(marker.read_bytes())
                if receipt['status']!='success' or receipt['sample_token']!=token or receipt['artifact_hashes']!=hashes:
                    raise ValueError('Prediction commit hash/token mismatch')
                if receipt['shared_fingerprint']!=source.shared_fingerprint or md['benchmark_provenance']['shared_fingerprint']!=source.shared_fingerprint or receipt['input_fingerprint']!=md['benchmark_provenance']['input_fingerprint']:
                    raise ValueError('Saved input/shared fingerprint mismatch')
                if md['environment']!=source.shared['environment']:raise ValueError('Preprocessing environment mismatch')
                sensors=source._sensor_audit(md)
                times=[max(r['timestamp_us'],r['ego_pose_timestamp_us']) for r in sensors if r['used_in_prediction']]
                result.update(status='valid_v1_prediction_cache',strict_training_cache_usable=True,
                    latest_used_timestamp_us=max(times),earliest_used_timestamp_us=min(times),input_fingerprint=receipt['input_fingerprint'],
                    prediction_commit=file_record(marker),historical_raw_sensor_hashes_verified=True)
                old=selected.get(token)
                if old is None or str(folder)==prior.get(token):selected[token]=dict(result)
        except (OSError,ValueError,KeyError,TypeError,EOFError) as error:
            result.update(status='invalid_cache',error=f'{type(error).__name__}: {error}',strict_training_cache_usable=False)
        candidates.append(result)
    return dict(candidate_count=len(candidates),status_counts=dict(Counter(c['status'] for c in candidates)),
                selected_unique_count=len(selected),selected=selected,candidates=candidates,
                policy='Preserve forecasting v1 benchmark cache requirements. Legacy valid output is a separate tier, not GT or a fabricated prediction. No new historical provenance is invented.')


def frame_summary(catalog, raw, gt, caches):
    frames=[]
    legacy=defaultdict(list)
    for item in caches['candidates']:
        if item['status']=='legacy_output_valid_incomplete_input_provenance':legacy[item['sample_token']].append(item['folder'])
    helper=SequenceDataset.__new__(SequenceDataset);helper.catalog=catalog
    for token,item in sorted(catalog.samples.items(),key=lambda pair:(pair[1]['scene']['name'],pair[1]['frame_index'])):
        frames.append(dict(**helper.frame_record(token,item),split=item['split'],
            raw_keyframes_present=raw[token]['raw_keyframes_present'],adapter_valid=raw[token]['adapter_valid'],
            gt_status=gt[token]['status'],gt=gt[token],
            valid_v1_cache=token in caches['selected'],cache=caches['selected'].get(token),
            legacy_valid_output_folders=legacy[token]))
    return frames


def build_manifests(catalog, frames, raw, gt, selected, protocol_fingerprint):
    groups=defaultdict(list)
    for frame in frames:groups[frame['scene']].append(frame)
    scene_sets={s:set(catalog.splits[s]) for s in ('mini_train','mini_val')}
    if scene_sets['mini_train'] & scene_sets['mini_val']:raise ValueError('Train/validation scenes overlap')
    manifests={}
    for split in ('mini_train','mini_val'):
        windows=[]
        for anchor in [f for f in frames if f['split']==split]:
            picked=select_window(groups[anchor['scene']],anchor)
            reasons=list(picked['reasons']);potential=list(reasons)
            history=[];future=[]
            for f in picked['history']:
                if f is None:history.append(None);continue
                token=f['sample_token'];r=raw[token];cache=selected.get(token)
                if not r['adapter_valid']:potential.append('invalid_raw_sensor_input')
                if r['latest_used_timestamp_us'] is None or r['latest_used_timestamp_us']>picked['input_cutoff_us']:potential.append('input_after_cutoff')
                if cache is None:reasons.append('missing_valid_prediction_cache')
                elif cache['latest_used_timestamp_us']>picked['input_cutoff_us']:reasons.append('input_after_cutoff')
                history.append(dict(sample_token=token,scene=f['scene'],frame_index=f['frame_index'],reference_timestamp_us=f['reference_timestamp_us'],
                    history_input=dict(source_kind='saved_OCCFusion_prediction',cache=cache),raw_adapter_valid=r['adapter_valid']))
            for f in picked['future']:
                if f is None:future.append(None);continue
                token=f['sample_token']
                if gt[token]['status']!='valid_gt':reasons.append('missing_or_invalid_target_gt');potential.append('missing_or_invalid_target_gt')
                future.append(dict(sample_token=token,scene=f['scene'],frame_index=f['frame_index'],reference_timestamp_us=f['reference_timestamp_us'],target_gt=gt[token]))
            # A cache cannot make a missing/corrupt current raw source pass provenance checks.
            reasons.extend(r for r in potential if r in ('invalid_raw_sensor_input','input_after_cutoff'))
            reasons=list(dict.fromkeys(reasons));potential=list(dict.fromkeys(potential))
            windows.append(dict(window_id=anchor['sample_token'],scene=anchor['scene'],anchor_token=anchor['sample_token'],
                anchor_timestamp_us=anchor['reference_timestamp_us'],input_cutoff_us=picked['input_cutoff_us'],history=history,future=future,timing=picked['timing'],
                status='ready' if not reasons else 'blocked',reasons=reasons,
                potential_after_verified_cache_generation=not potential,potential_blockers=potential))
        manifests[split]=dict(kind='temporal_training_readiness_manifest',schema_version=1,split=split,
            purpose='training' if split=='mini_train' else 'evaluation_only_not_training',protocol_fingerprint=protocol_fingerprint,
            official_scenes=sorted(scene_sets[split]),candidate_windows=len(windows),ready_windows=sum(w['status']=='ready' for w in windows),
            potential_windows_after_cache=sum(w['potential_after_verified_cache_generation'] for w in windows),
            primary_blocker_counts=dict(Counter(w['reasons'][0] for w in windows if w['reasons'])),
            any_blocker_counts=dict(Counter(r for w in windows for r in w['reasons'])),windows=windows)
    return manifests


def cache_plan(frames, raw, manifests, source):
    strict=source['selected']
    samples=[sum(v['bytes'] for v in c['artifact_hashes'].values())+c.get('prediction_commit',{}).get('bytes',0)
             for c in strict.values()]
    median=float(np.median(samples)) if samples else None
    p95=float(np.percentile(samples,95)) if samples else None
    required=defaultdict(list)
    for w in manifests['mini_train']['windows']:
        if w['potential_after_verified_cache_generation']:
            for f in w['history']:required[f['sample_token']].append(w['window_id'])
    entries=[]
    for frame in frames:
        if frame['split']!='mini_train' or frame['valid_v1_cache']:continue
        token=frame['sample_token'];r=raw[token]
        entries.append(dict(sample_token=token,scene=frame['scene'],frame_index=frame['frame_index'],timestamp_us=frame['reference_timestamp_us'],
            lidar_sample_data_token=frame['lidar_sample_data_token'],lidar_filename=frame['lidar_filename'],
            action='blocked_sensor_input' if not r['adapter_valid'] else 'plan_new_verified_cache',
            required_by_potential_windows=required[token],legacy_output_folders=frame['legacy_valid_output_folders'],
            current_raw_errors=r['errors'],latest_used_timestamp_us=r['latest_used_timestamp_us'],causal_at_own_lidar=r['causal_at_own_lidar'],
            estimated_payload_bytes_median=median,estimated_payload_bytes_p95=p95))
    runnable=[e for e in entries if e['action']=='plan_new_verified_cache']
    minimum=[e for e in runnable if e['required_by_potential_windows']]
    return dict(plan_only=True,inference_launched=False,jobs_created=False,worker_started=False,split='mini_train',
        missing_strict_cache_frames=len(entries),metadata_runnable_frames=len(runnable),sensor_blocked_frames=len(entries)-len(runnable),
        minimum_missing_frames_for_potential_windows=len(minimum),
        sizing=dict(unit='bytes',reference_unique_real_caches=len(samples),per_frame_median=median,per_frame_p95=p95,
            all_metadata_runnable_estimated_bytes_median=median*len(runnable) if median else None,
            all_metadata_runnable_estimated_bytes_p95=p95*len(runnable) if p95 else None,
            minimum_window_set_estimated_bytes_median=median*len(minimum) if median else None,
            minimum_window_set_estimated_bytes_p95=p95*len(minimum) if p95 else None,
            limitation='Estimated saved labels/max-softmax + metadata/report/BEV/config/receipt only. Excludes GPU model, features/logits, failure staging, dataset, GT, evaluation and reserve. Not a storage guarantee.'),
        generator_limit='Current benchmark_occupancy.py selects mini_val. A future explicitly authorized mini_train cache path must reuse its verified session/preprocessing and capture real historical raw-sensor hashes. Do not run the mini_val benchmark as a train cache generator.',
        provenance_limit='Legacy outputs are not silently promoted to v1. Recover authentic original input evidence or explicitly authorize regenerating their cache; hashing inputs now is not proof of historical input bytes.',
        frames=entries)
