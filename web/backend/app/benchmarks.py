"""Confined, hash-verified playback of saved set-level benchmark evidence.

Only JSON/NPZ/PNG and sensor bytes are read. No model, checkpoint, GT evaluator,
subprocess or GPU library is imported. Saved absolute path hints never select
files: attempts are anchored below the configured benchmark directory.
"""
from collections import Counter, OrderedDict
import copy
import csv
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path, PurePosixPath
import re
from threading import RLock

import numpy as np

from .evaluations import ARTIFACTS, EvaluationStore, decode, digest, validate_metrics
from .review import canonical
from .store import contained, geometry, InvalidArtifact, OutsideRoot, read_json


TOKEN = re.compile(r'^[0-9a-f]{32}$')
IDENTIFIER = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')
ROOT_FILES = ('manifest.json', 'progress.json', 'metrics_pooled.json',
              'initial_report.json', 'benchmark_report.json', 'timings.csv')
PRED_FILES = (*ARTIFACTS, 'resolved_config.py')


def fingerprint(value):
    return digest(canonical(value).encode())


def benchmark_fingerprint(value):
    # The CLI uses json.dumps' default ensure_ascii=True for provenance.
    import json
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def record(content):
    return dict(sha256=digest(content), bytes=len(content))


@dataclass
class CollectionSnapshot:
    detail: dict
    manifest: dict
    paths: dict

    @property
    def fingerprint(self):
        return self.manifest['fingerprint_sha256']


class BenchmarkStore:
    def __init__(self, settings, runs):
        self.settings, self.runs = settings, runs
        self.roots = tuple(Path(p).resolve() for p in settings.benchmark_directories)
        self.validated = OrderedDict()
        self.rendered = OrderedDict()
        self.lock = RLock()
        self.evaluations = EvaluationStore(settings)

    def folder(self, identifier):
        if not IDENTIFIER.fullmatch(identifier) or identifier in ('.', '..'):
            raise OutsideRoot('Invalid benchmark_id')
        matches = []
        for root in self.roots:
            folder = contained(root / identifier, root)
            if (folder/'manifest.json').is_file():
                value = read_json(contained(folder/'manifest.json', folder))
                if value.get('kind') == 'occfusion_occupancy_benchmark':
                    matches.append(folder)
        if len(set(matches)) != 1:
            raise InvalidArtifact('Benchmark is absent or ambiguous across configured roots')
        return matches[0]

    def list_benchmarks(self):
        items, seen = [], set()
        for root in self.roots:
            if not root.is_dir():
                continue
            for path in sorted(root.iterdir()):
                if not path.is_dir() or path.name in seen or not IDENTIFIER.fullmatch(path.name):
                    continue
                try:
                    folder = self.folder(path.name)
                    manifest = read_json(contained(folder/'manifest.json', folder))
                    if manifest['identity']['scope'] != 'mini_val':
                        continue
                    report = read_json(contained(folder/'metrics_pooled.json', folder))
                    items.append(dict(benchmark_id=path.name, status=report['status'],
                        split=report['split'], expected_frame_count=report['expected_frame_count'],
                        successful_frame_count=report['successful_frame_count'],
                        validation='Full hashes/protocol checked when opening collection'))
                    seen.add(path.name)
                except (OSError, ValueError, KeyError, OutsideRoot):
                    continue
        return items

    @staticmethod
    def attempt(folder, state, kind, token):
        hint = PurePosixPath(state['prediction_dir' if kind == 'frames' else 'evaluation_dir'])
        name = hint.name
        if tuple(hint.parts[-3:-1]) != (kind, token) or not re.fullmatch(r'[A-Za-z0-9_.+\-]{1,120}', name) or name.endswith('.pending') or name in ('.', '..'):
            raise InvalidArtifact('Unpublished or invalid benchmark attempt')
        return contained(folder/kind/token/name, folder)

    def dataset(self, provenance):
        if provenance['dataset_version'] != 'v1.0-mini':
            raise InvalidArtifact('Only v1.0-mini benchmark playback is supported')
        matches = []
        for root in self.settings.dataset_roots:
            try:
                if all(digest(contained(root/'v1.0-mini'/(name+'.json'),root).read_bytes()) == value
                       for name,value in provenance['dataset_table_hashes'].items()):
                    matches.append(root)
            except FileNotFoundError:
                continue
        if len(matches) != 1:
            raise InvalidArtifact('Dataset metadata hashes do not match exactly one configured root')
        return matches[0], self.runs.catalog.tables(matches[0], 'v1.0-mini')

    @staticmethod
    def validate_prediction(files, meta, report):
        _, shape, _, _ = geometry(meta)
        if meta['coordinate_system']['name'] != 'LIDAR_TOP at reference_timestamp_us' or meta['coordinate_system']['unit'] != 'metres':
            raise InvalidArtifact('Prediction is outside native reference LiDAR coordinates')
        with np.load(BytesIO(files['occupancy.npz']), allow_pickle=False) as archive:
            if set(archive.files) != {'labels','max_softmax_probability'}:
                raise InvalidArtifact('Unexpected NPZ schema')
            labels, probability = archive['labels'], archive['max_softmax_probability']
        if labels.shape != shape or probability.shape != shape or labels.dtype.kind not in 'iu' or np.any(labels<0) or np.any(labels>=len(meta['class_names'])):
            raise InvalidArtifact('Invalid prediction labels/grid')
        if probability.dtype.kind != 'f' or not np.isfinite(probability).all() or np.any(probability<0) or np.any(probability>1):
            raise InvalidArtifact('Invalid saved probability')
        histogram = {str(int(k)):int(n) for k,n in zip(*np.unique(labels,return_counts=True))}
        if histogram != report['prediction']['predicted_voxel_counts'] or report['prediction']['logits_shape'] != [1,*shape,len(meta['class_names'])]:
            raise InvalidArtifact('Prediction counts/shape differ from report')
        if meta['free_label'] != 0 or meta['class_names'][0] != 'free' or not files['bev.png'].startswith(b'\x89PNG\r\n\x1a\n'):
            raise InvalidArtifact('Invalid background class or BEV')
        palette=np.asarray(meta['palette'])
        if palette.shape!=(len(meta['class_names']),3) or not np.isfinite(palette).all() or np.any(palette<0) or np.any(palette>255):
            raise InvalidArtifact('Invalid semantic palette')

    def frame_evidence(self, folder, entry, state, protocol, provenance, dataset, paths, hashes):
        token = entry['sample_token']
        pred = self.attempt(folder,state,'frames',token)
        ev = self.attempt(folder,state,'evaluations',token)
        def load(path, arc):
            content = contained(path,folder).read_bytes()
            paths[arc] = path; hashes[arc] = record(content)
            return content
        files = {name:load(pred/name,f'frames/{token}/{name}') for name in PRED_FILES}
        receipt = decode(load(pred/'prediction_commit.json',f'frames/{token}/prediction_commit.json'))
        actual = {name:record(value) for name,value in files.items()}
        if receipt['status'] != 'success' or receipt['sample_token'] != token or receipt['artifact_hashes'] != actual:
            raise InvalidArtifact(f'{token}: prediction receipt/token/hash mismatch')
        if receipt['shared_fingerprint'] != benchmark_fingerprint(provenance) or receipt['input_fingerprint'] != state['input_fingerprint']:
            raise InvalidArtifact(f'{token}: prediction provenance mismatch')
        meta, report = decode(files['metadata.json']), decode(files['report.json'])
        for key in ('sample_token','frame_index','precision','status'):
            if meta[key] != report[key]:
                raise InvalidArtifact(f'{token}: metadata/report {key} mismatch')
        if meta['sample_token'] != token or state['sample_token'] != token or meta['frame_index'] != entry['frame_index'] or meta['scene']['name'] != entry['scene'] or meta['timestamp_us'] != entry['timestamp_us'] or meta['reference_timestamp_us'] != entry['reference_timestamp_us']:
            raise InvalidArtifact(f'{token}: manifest sample/scene/frame/timestamp mismatch')
        if meta['status'] != 'success' or report.get('check_only') or meta['result_source'] != 'prediction':
            raise InvalidArtifact(f'{token}: frame is not a successful saved prediction')
        if meta['checkpoint'] != provenance['checkpoint'] or meta['config']['resolved_sha256'] != provenance['resolved_config_sha256'] or meta['precision'] != provenance['precision'] or report['tf32'] != provenance['tf32']:
            raise InvalidArtifact(f'{token}: model/config/precision mismatch')
        if meta['benchmark_provenance']['input_fingerprint'] != receipt['input_fingerprint'] or meta['benchmark_provenance']['shared_fingerprint'] != receipt['shared_fingerprint']:
            raise InvalidArtifact(f'{token}: preprocessing fingerprint mismatch')
        if actual['occupancy.npz']['sha256'] != state['prediction_sha256']:
            raise InvalidArtifact(f'{token}: progress prediction hash mismatch')
        for key in ('grid_shape','bounds','voxel_size','grid_axis_order','class_names','voxel_center'):
            if meta[key] != protocol[key]:
                raise InvalidArtifact(f'{token}: prediction/protocol {key} mismatch')
        content = load(ev/'evaluation.json',f'frames/{token}/evaluation.json')
        commit = decode(load(ev/'evaluation_commit.json',f'frames/{token}/evaluation_commit.json'))
        four = {name:actual[name] for name in ARTIFACTS}
        if commit['status'] != 'success' or commit['sample_token'] != token or commit['evaluation'] != record(content) or commit['artifact_hashes'] != four or commit['gt'] != state['gt'] or commit['protocol_fingerprint'] != benchmark_fingerprint(protocol) or state['evaluation_sha256'] != digest(content):
            raise InvalidArtifact(f'{token}: evaluation receipt/hash/protocol mismatch')
        value = decode(content)
        if value['protocol'] != protocol or value['frames'][0]['gt_sha256'] != state['gt']['sha256']:
            raise InvalidArtifact(f'{token}: evaluation protocol/GT hash mismatch')
        item = dict(id=digest(f'benchmark:{folder.name}:{token}'.encode()),label=f'{entry["scene"]} / {entry["frame_index"]}',
                    sha256=digest(content),bytes=len(content),document=value)
        matched = self.evaluations.match(item,meta,{name:files[name] for name in ARTIFACTS})
        if matched['status'] != 'matched':
            raise InvalidArtifact(f'{token}: {matched["reason"]}')
        matched['candidates']=[dict(id=item['id'],label=item['label'],sha256=item['sha256'],
                                    status='matched',reason=matched['reason'])]
        root,tables = dataset
        sd = tables['sample_data'][entry['lidar_sample_data_token']]
        lidar_path = contained(root/sd['filename'],root)
        lidar_hash = record(lidar_path.read_bytes())
        declared = meta['benchmark_provenance']['sensor_files'][sd['token']]
        if any(lidar_hash[key] != declared[key] for key in ('sha256','bytes')):
            raise InvalidArtifact(f'{token}: LiDAR bytes changed from prediction provenance')
        key = fingerprint(dict(artifacts=actual,lidar=lidar_hash,metadata_tables=provenance['dataset_table_hashes']))
        with self.lock:
            if key not in self.validated:
                self.validate_prediction(files,meta,report)
                self.runs.catalog.lidar(meta)  # Token/frame/calibration/pose and finite point validation.
                self.validated[key] = True
                if len(self.validated)>256:
                    self.validated.popitem(last=False)
        return dict(sample_token=token,scene=entry['scene'],frame_index=entry['frame_index'],
            timestamp_us=entry['timestamp_us'],status='success',prediction_available=True,
            prediction_sha256=actual['occupancy.npz']['sha256'],evaluation_sha256=digest(content),
            metrics=value['metrics'],evaluation=matched,cache_key=key,prediction_folder=pred,
            url=f'/api/benchmarks/{folder.name}/frames/{token}')

    @staticmethod
    def latency(initial, content):
        session = initial['current_invocation']
        if session['evaluate_only'] or not session['model_load_count'] or session['completed_forwards']<=0:
            raise InvalidArtifact('initial_report.json is not an original inference invocation')
        try:
            saved_rows=list(csv.DictReader(StringIO(content.decode(),newline='')))
            if any(None in row or any(value is None for value in row.values()) for row in saved_rows):
                raise InvalidArtifact('Malformed timing CSV row')
            rows = [row for row in saved_rows if row['session_id']==session['id'] and row['prediction_source']=='new' and row['forward_s']]
        except (csv.Error,UnicodeError,KeyError) as error:
            raise InvalidArtifact(f'Invalid saved timing CSV: {error}') from error
        values = [float(row['forward_s']) for row in rows]
        steady = [float(row['forward_s']) for row in rows if row['warmup'].lower() not in ('true','1')]
        if not values or not np.isfinite(values).all() or min(values)<0 or len(values)!=session['completed_forwards']:
            raise InvalidArtifact('Original forward timing rows are absent/invalid')
        expected = dict(sample_count=len(steady),median_s=float(np.median(steady)) if steady else None,
                        p95_s=float(np.percentile(steady,95)) if steady else None)
        if expected != initial['latency_current_invocation']['excluding_marked_warmup'] or len(rows)-len(steady) != initial['latency_current_invocation']['warmup_completed_count']:
            raise InvalidArtifact('Original report latency disagrees with saved forward CSV evidence')
        return dict(**expected,source_report='initial_report.json',source_session_id=session['id'],
            warmup_excluded=True,warmup_count=len(rows)-len(steady),all_forward_sample_count=len(rows),
            min_forward_s=min(values),max_forward_s=max(values),units='seconds',
            scope='Original synchronized GPU tensor forwards, excluding marked warm-up; cache rows excluded',
            startup=dict(bootstrap_s=session['bootstrap_wall_s'],shared_setup_s=session['shared_setup_wall_s'],
                         model_initialization_s=session['model_initialization_wall_s'],invocation_wall_s=session['wall_s']))

    def radar_audit(self, content, missing, dataset, provenance):
        value = decode(content)
        if value['kind'] != 'occfusion_radar_startup_metadata_audit' or {k:v['sha256'] for k,v in value['data']['metadata_tables'].items()} != provenance['dataset_table_hashes'] or value['source']['inference_script_sha256'] != provenance['source_hashes']['tools/infer_nuscenes_mini.py']:
            raise InvalidArtifact('Radar audit metadata/source provenance mismatch')
        root,tables = dataset
        linked=[]
        for failed in missing:
            matches=[f for f in value['frames'] if f['sample_token']==failed['sample_token']]
            if len(matches)!=1:
                raise InvalidArtifact('Radar audit does not identify the failed sample exactly')
            frame=matches[0]; sample=tables['sample'][failed['sample_token']]
            if frame['scene']!=failed['scene'] or frame['frame_index']!=failed['frame_index'] or sample['timestamp']!=frame['sample_timestamp_us']:
                raise InvalidArtifact('Radar audit scene/frame/timestamp mismatch')
            reference=frame['lidar_reference'];sd=tables['sample_data'][reference['sample_data_token']]
            if sd['sample_token']!=sample['token'] or sd['timestamp']!=reference['timestamp_us']:
                raise InvalidArtifact('Radar audit reference LiDAR mismatch')
            for radar in frame['radars']:
                saved=radar['sample_radar'];actual=tables['sample_data'][saved['sample_data_token']]
                sensor=tables['sensor'][tables['calibrated_sensor'][actual['calibrated_sensor_token']]['sensor_token']]
                if actual['sample_token']!=sample['token'] or sensor['channel']!=radar['channel'] or actual['timestamp']!=saved['timestamp_us'] or actual['prev']!=saved['prev_token'] or (actual['timestamp']-sd['timestamp'])/1000!=saved['radar_minus_lidar_ms']:
                    raise InvalidArtifact('Radar audit token/time/channel mismatch')
                file=contained(root/actual['filename'],root)
                if record(file.read_bytes())!={'sha256':saved['file_sha256'],'bytes':saved['file_bytes']}:
                    raise InvalidArtifact('Radar audit physical file hash mismatch')
                if actual['prev'] or not actual['is_key_frame'] or not saved['file_exists'] or actual['filename']!=saved['filename']:
                    raise InvalidArtifact('Radar startup audit history/file declaration mismatch')
                same_channel=[r for r in tables['sample_data'].values()
                    if tables['sample'][r['sample_token']]['scene_token']==sample['scene_token'] and
                    tables['sensor'][tables['calibrated_sensor'][r['calibrated_sensor_token']]['sensor_token']]['channel']==radar['channel']]
                eligible=[r for r in same_channel if r['timestamp']<=sd['timestamp']]
                scan=radar['independent_metadata_scan']
                if scan['at_or_before_lidar_count']!=len(eligible) or scan['record_count']!=len(same_channel) or scan['earliest_timestamp_us']!=min(r['timestamp'] for r in same_channel) or radar['adapter_selected_sweep_count']!=int(actual['timestamp']<=sd['timestamp']):
                    raise InvalidArtifact('Radar audit disagrees with independent metadata scan')
                if not radar['independent_metadata_scan']['agrees_with_prev_chain'] or radar['metadata_traversal_errors']:
                    raise InvalidArtifact('Radar audit reports a traversal disagreement')
            if frame['adapter_reproduction']['message']!=failed['error']['message']:
                raise InvalidArtifact('Radar audit and benchmark error mismatch')
            linked.append(dict(sample_token=sample['token'],scene=frame['scene'],frame_index=frame['frame_index'],
                reference_timestamp_us=reference['timestamp_us'],rejected_channels=frame['rejected_radar_channels'],
                radars=[dict(channel=r['channel'],sample_data_token=r['sample_radar']['sample_data_token'],
                    timestamp_us=r['sample_radar']['timestamp_us'],delta_ms=r['sample_radar']['radar_minus_lidar_ms'],
                    prev_token=r['sample_radar']['prev_token'],file_exists=r['sample_radar']['file_exists'],
                    eligible_count=r['adapter_selected_sweep_count']) for r in frame['radars']],
                reason='Radar keyframe nằm sau LiDAR và prev rỗng; không có record quá khứ hợp lệ trong metadata. File vật lý tồn tại.',
                adapter_bug_detected=value['summary']['adapter_bug_detected']))
        return dict(status='verified',source='radar_startup_audit.json',sha256=digest(content),frames=linked)

    def context(self, identifier):
        folder=self.folder(identifier);paths={};hashes={};contents={}
        for name in ROOT_FILES:
            path=contained(folder/name,folder);content=path.read_bytes()
            paths[f'benchmark/{name}']=path;hashes[f'benchmark/{name}']=record(content);contents[name]=content
        manifest,progress,pooled,initial,latest=[decode(contents[name]) for name in ROOT_FILES[:5]]
        identity=manifest['identity'];entries=manifest['frames'];provenance=manifest['prediction_provenance']
        if identity['scope']!='mini_val' or identity['split']!='mini_val' or manifest['schema_version']!=1 or benchmark_fingerprint(identity)!=manifest['identity_sha256'] or benchmark_fingerprint(provenance)!=identity['prediction_provenance']:
            raise InvalidArtifact('Invalid benchmark identity/provenance/scope')
        if [{k:v for k,v in f.items() if k!='gt_initial'} for f in entries]!=identity['frames'] or len(entries)!=identity['expected_frame_count'] or len(entries)!=identity['dataset_frame_count']:
            raise InvalidArtifact('Manifest membership/count was changed')
        tokens=[f['sample_token'] for f in entries]
        if len(set(tokens))!=len(tokens) or not all(TOKEN.fullmatch(t) for t in tokens) or entries!=sorted(entries,key=lambda f:(f['scene'],f['frame_index'],f['sample_token'])):
            raise InvalidArtifact('Duplicate/invalid/unsorted manifest samples')
        manifest_hash=digest(contents['manifest.json'])
        if any(doc['manifest_sha256']!=manifest_hash for doc in (progress,pooled,initial,latest)):
            raise InvalidArtifact('Benchmark manifest hash mismatch')
        for report in (initial,latest):
            if report['provenance']!=provenance or any(report[key]!=provenance[key] for key in ('precision','autocast','batch_size','tf32')):
                raise InvalidArtifact('Benchmark report precision/model provenance mismatch')
        dataset=self.dataset(provenance);root,tables=dataset
        # Check official membership against saved dataset chains, not browser IDs.
        expected=[]
        for scene in sorted((s for s in tables['scene'].values() if s['name'] in identity['official_scene_names']),key=lambda s:s['name']):
            cursor=scene['first_sample_token'];index=0;seen=set()
            while cursor:
                if cursor in seen: raise InvalidArtifact('Cycle in nuScenes sample chain')
                seen.add(cursor);sample=tables['sample'][cursor]
                expected.append((scene['name'],index,cursor,sample['timestamp']));cursor=sample['next'];index+=1
        if identity['official_scene_names']!=['scene-0103','scene-0916'] or [(f['scene'],f['frame_index'],f['sample_token'],f['timestamp_us']) for f in entries]!=expected:
            raise InvalidArtifact('Manifest is not the complete mini_val scene membership')
        if set(progress['frames'])!=set(tokens):
            raise InvalidArtifact('Progress sample membership mismatch')
        return dict(folder=folder,paths=paths,hashes=hashes,contents=contents,manifest=manifest,progress=progress,
                    pooled=pooled,initial=initial,latest=latest,entries=entries,provenance=provenance,dataset=dataset,tokens=tokens)

    def snapshot(self, identifier):
        context=self.context(identifier)
        folder,paths,hashes,contents,manifest,progress,pooled,initial,latest,entries,provenance,dataset,tokens=[context[key] for key in
            ('folder','paths','hashes','contents','manifest','progress','pooled','initial','latest','entries','provenance','dataset','tokens')]
        protocol=pooled['protocol']
        frames=[];matrix=np.zeros((17,17),dtype=np.int64);missing=[]
        for entry in entries:
            state=progress['frames'][entry['sample_token']]
            if state['status']=='success':
                frame=self.frame_evidence(folder,entry,state,protocol,provenance,dataset,paths,hashes)
                matrix+=np.asarray(frame['metrics']['confusion_matrix'],dtype=np.int64)
            else:
                frame=dict(sample_token=entry['sample_token'],scene=entry['scene'],frame_index=entry['frame_index'],
                    timestamp_us=entry['timestamp_us'],status=state['status'],prediction_available=False,error=state.get('error'))
                missing.append(dict(sample_token=entry['sample_token'],scene=entry['scene'],frame_index=entry['frame_index'],status=state['status'],error=state.get('error')))
            frames.append(frame)
        successful=[f for f in frames if f['prediction_available']]
        if not successful: raise InvalidArtifact('Benchmark has no valid evaluated prediction')
        validate_metrics(pooled['metrics'],protocol['class_names'],max_voxels=640000*len(successful))
        if pooled['metrics']['confusion_matrix']!=matrix.tolist() or pooled['sample_tokens']!=[f['sample_token'] for f in successful] or pooled['missing_frames']!=missing:
            raise InvalidArtifact('Pooled statistics/membership do not equal the saved frame evidence')
        status='complete' if len(successful)==len(entries) else 'partial'
        failed=sum(f['status']=='failed' for f in frames)
        for doc in (pooled,latest):
            if doc['status']!=status or doc['successful_frame_count']!=len(successful) or doc['expected_frame_count']!=len(entries) or doc['failed_frame_count']!=failed or doc['metrics']!=pooled['metrics']:
                raise InvalidArtifact('Benchmark report/count/status disagrees with verified frames')
        if pooled['scene_counts']!=dict(Counter(f['scene'] for f in successful)):
            raise InvalidArtifact('Pooled scene counts mismatch')
        audit=dict(status='unavailable',frames=[],reason='No failed-frame audit needed')
        if missing:
            audit_path=contained(folder.parent/'radar_startup_audit.json',folder.parent)
            content=audit_path.read_bytes();paths['radar_startup_audit.json']=audit_path;hashes['radar_startup_audit.json']=record(content)
            audit=self.radar_audit(content,missing,dataset,provenance)
        latency=self.latency(initial,contents['timings.csv'])
        latency['source_report_sha256']=digest(contents['initial_report.json'])
        detail=dict(benchmark_id=identifier,status=status,split='mini_val',scope=pooled['metrics_scope'],
            expected_frame_count=len(entries),successful_frame_count=len(successful),failed_frame_count=failed,
            scene_counts=pooled['scene_counts'],metrics=pooled['metrics'],metric_unit='ratio_0_1',protocol=protocol,
            frames=[{k:v for k,v in f.items() if k not in ('cache_key','prediction_folder','evaluation','metrics')} for f in frames],
            radar_audit=audit,latency=latency,precision=initial['precision'],tf32=initial['tf32'],
            allocator=dict(initial['allocator'],source='initial_report.json'),reference_check=pooled['reference_check'],
            evidence_hashes={name:hashes['benchmark/'+name] for name in ROOT_FILES},
            validation=dict(tokens=True,geometry=True,protocol=True,artifact_hashes=True,pooled_counts=True),
            replay=True,inference_enabled=False,playback_note='Phát lại prediction từng frame đã lưu; chưa phải learned 4D flow.')
        core=dict(schema_version=1,kind='occfusion_benchmark_collection',collection_id=identifier,split='mini_val',
            status=status,expected_frame_count=len(entries),successful_frame_count=len(successful),failed_frame_count=failed,
            sample_tokens=tokens,evaluated_sample_tokens=[f['sample_token'] for f in successful],
            frames=[dict(sample_token=f['sample_token'],scene=f['scene'],frame_index=f['frame_index'],status=f['status'],
                         prediction_sha256=f.get('prediction_sha256'),evaluation_sha256=f.get('evaluation_sha256')) for f in frames],
            metric_unit='ratio_0_1',protocol_fingerprint=benchmark_fingerprint(protocol),
            token_sources=['manifest.json','metadata.json:sample_token','report.json:sample_token','evaluation.json:frames'],
            npz_contains_sample_token=False,latency_source='benchmark/initial_report.json',artifacts=hashes,
            review_scope='Collection of verified saved frames and pooled evidence; independent of single-run v1/v2 approvals',
            playback_note=detail['playback_note'])
        snapshot=CollectionSnapshot(detail,dict(core,fingerprint_sha256=fingerprint(core)),paths)
        # Private server-only data for frame rendering; never return folders to the browser.
        snapshot.frames={f['sample_token']:f for f in frames}
        return snapshot

    def frame(self, identifier, token):
        if not TOKEN.fullmatch(token): raise OutsideRoot('Invalid sample token')
        context=self.context(identifier)
        entry=next((f for f in context['entries'] if f['sample_token']==token),None)
        if entry is None: raise FileNotFoundError('Sample is not in this benchmark manifest')
        state=context['progress']['frames'][token]
        if state['status']!='success':
            missing=next((f for f in context['pooled']['missing_frames'] if f['sample_token']==token),None)
            if missing is None or missing['status']!=state['status'] or missing['error']!=state.get('error'):
                raise InvalidArtifact('Failed frame state disagrees with pooled evidence')
            path=contained(context['folder'].parent/'radar_startup_audit.json',context['folder'].parent)
            audit=self.radar_audit(path.read_bytes(),[missing],context['dataset'],context['provenance'])
            return dict(missing,prediction_available=False,replay=True,inference_enabled=False,
                        radar_audit=audit['frames'][0]),None
        frame=self.frame_evidence(context['folder'],entry,state,context['pooled']['protocol'],context['provenance'],
                                  context['dataset'],context['paths'],context['hashes'])
        with self.lock:
            key=frame['cache_key']
            if key not in self.rendered:
                loaded=self.runs.load(f'{identifier}-{token}',frame['prediction_folder'])
                for name in PRED_FILES:
                    path=contained(frame['prediction_folder']/name,self.folder(identifier))
                    if record(path.read_bytes())!=context['hashes'][f'frames/{token}/{name}']:
                        raise InvalidArtifact('Frame changed during playback load')
                self.rendered[key]=loaded
                while len(self.rendered)>self.settings.cache_entries: self.rendered.popitem(last=False)
            loaded=self.rendered[key];self.rendered.move_to_end(key)
        detail=copy.deepcopy(loaded.detail);url=frame['url']
        suffix='?bundle='+frame['cache_key']
        detail['occupancy']['url']=url+'/occupancy'+suffix;detail['lidar']['url']=url+'/lidar'+suffix;detail['bev_url']=url+'/bev'+suffix
        detail.update(evaluation=frame['evaluation'],review=dict(registered=False,state='collection_scope',permissions={},events=[]),
                      collection_id=identifier,bundle_fingerprint=frame['cache_key'],playback_note='Phát lại prediction từng frame đã lưu; chưa phải learned 4D flow.')
        return detail,loaded
