#!/usr/bin/env python3
"""Sequential, resumable nuScenes mini_val benchmark, using the verified CLI helpers."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import fcntl
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
import uuid

PROCESS_STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import infer_nuscenes_mini as inference
from tools import evaluate_occupancy as evaluator
import numpy as np


ARTIFACTS = evaluator.ARTIFACTS
TF32 = dict(matmul=False, cudnn=True)  # Recorded defaults of the verified FP32 baseline.
TIMING_FIELDS = ('session_id', 'sample_token', 'scene', 'frame_index', 'status',
                 'prediction_source', 'evaluation_source', 'warmup', 'input_validation_s',
                 'preprocessing_s', 'forward_s', 'postprocessing_export_s', 'evaluation_s',
                 'frame_wall_s', 'peak_allocated_bytes', 'peak_reserved_bytes',
                 'allocated_after_release_bytes')


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    return evaluator.json_object(Path(path).read_bytes())


def file_record(path):
    path = Path(path)
    return dict(sha256=inference.file_hash(path), bytes=path.stat().st_size)


def file_records(folder, names):
    return {name: file_record(Path(folder)/name) for name in names}


@contextmanager
def output_lock(folder, resume):
    folder = Path(folder)
    if resume and (not (folder/'manifest.json').is_file() or read_json(folder/'manifest.json').get('kind') != 'occfusion_occupancy_benchmark'):
        raise ValueError('--resume requires an existing benchmark manifest; other runs are never adopted')
    if folder.exists() and not resume and any(folder.iterdir()):
        raise ValueError('Output directory is not empty; choose a fresh benchmark directory or --resume')
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'.benchmark.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another benchmark process owns this output directory') from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def ordered_samples(catalog, pilot=False):
    frames = sorted((item for item in catalog.samples.values() if item['split'] == 'mini_val'),
                    key=lambda item: (item['scene']['name'], item['frame_index'], item['sample']['token']))
    if {item['scene']['name'] for item in frames} != set(catalog.splits['mini_val']):
        raise ValueError('Dataset does not contain the complete official mini_val scene list')
    if pilot:
        # Predeclared ordinal selection, independent of prediction/metric results.
        frames = [item for item in frames if item['frame_index'] == 1]
        if len(frames) != len(catalog.splits['mini_val']):
            raise ValueError('Pilot requires frame index 1 in every official mini_val scene')
    return frames


def sensor_identity(metadata, shared_fingerprint):
    sensors = metadata['sensors']
    records = list(sensors['keyframes'].values()) + sensors['lidar_past_sweeps']
    records += [record for sweeps in sensors['radar_sweeps'].values() for record in sweeps]
    unique = {record['sample_data_token']: record for record in records}
    # Hash actual bytes on each visit; mtime/size alone cannot authorize reuse.
    files = {token: dict(path=record['path'], **file_record(record['path']))
             for token, record in sorted(unique.items())}
    value = dict(shared_fingerprint=shared_fingerprint, metadata=metadata, sensor_files=files)
    return fingerprint(value), files


def metadata_for_frame(base, item, shared, cfg, environment, code):
    metadata = dict(base)
    bounds = np.asarray(cfg.point_cloud_range, dtype=float)
    grid = np.asarray(cfg.model.view_transformer.grid_size[0], dtype=int)*2
    from occfusion.nuscenes_dataset import NuScenesSegDataset
    metadata.update(
        result_source='prediction', dataset=dict(root=shared['data_root'], version='v1.0-mini', split='mini_val',
            split_source='nuscenes.utils.splits.create_splits_scenes'),
        scene=dict(name=item['scene']['name'], token=item['scene']['token'], index=item['scene_index']),
        frame_index=item['frame_index'], bounds=bounds.tolist(), grid_shape=grid.tolist(),
        voxel_size=((bounds[3:]-bounds[:3])/grid).tolist(), grid_axis_order=['x','y','z'],
        flatten_order='C (z varies fastest)', voxel_center='bounds_min + (index + 0.5) * voxel_size',
        class_names=['free', *NuScenesSegDataset.METAINFO['classes'][1:]],
        palette=NuScenesSegDataset.METAINFO['palette'], free_label=0, gt_ignore_label=255,
        visibility_mask=None, visibility_note='No visibility mask produced or inferred.',
        config=dict(path=shared['config_path'], sha256=shared['config_source_sha256'],
                    resolved_file='resolved_config.py', resolved_sha256=shared['resolved_config_sha256'],
                    baseline_path=str(ROOT/'configs/OccFusion.py'),
                    baseline_sha256=shared['source_hashes']['configs/OccFusion.py']),
        checkpoint=shared['checkpoint'], code=code, environment=environment,
        image_preprocessing=dict(color_order='BGR', resize=False, scale_factor_actual=1.0,
            mean=list(cfg.model.data_preprocessor.mean), std=list(cfg.model.data_preprocessor.std),
            pad_size_divisor=cfg.model.data_preprocessor.pad_size_divisor,
            note='Verified CLI loader/preprocessor, full-resolution images; no GT inputs.'),
        sensor_configuration=dict(use_camera=True,use_lidar=True,use_radar=True,use_occ3d=False,points_per_voxel=35),
        bev=dict(projection='highest non-free voxel along z; free if none',rows='y decreasing',columns='x increasing',width=int(grid[0]),height=int(grid[1])),
        confidence_note='Max softmax probability is not calibrated uncertainty.',
        metrics=dict(occupancy_iou=None,occupancy_miou=None,reason='Inference does not load GT; evaluation is separate.'),
    )
    return metadata


class GroundTruthIndex(evaluator.GroundTruthIndex):
    """Index names once; resolve only selected files, preserving root confinement."""
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.files = defaultdict(list)
        if self.root.is_dir():
            for path in self.root.rglob('*.npy'):
                self.files[path.name].append(path)

    def lookup(self, lidar_filename):
        key, path = super().lookup(lidar_filename)
        if path is not None:
            path = path.resolve()
            if not path.is_relative_to(self.root):
                raise ValueError(f'GT symlink escapes configured root: {path}')
        return key, path


class BaselineSession:
    """Exactly one model/checkpoint load; frame tensors never escape predict()."""
    def __init__(self, context, warmup_forwards):
        import torch
        from mmengine.registry import init_default_scope
        from mmdet3d.registry import MODELS
        from mmdet3d.utils import register_all_modules
        from mmdet3d.models import Base3DSegmentor
        self.torch, self.context = torch, context
        self.device = torch.device(context['shared']['device'])
        self.model = None
        self.completed_forwards = self.attempted_forwards = 0
        self.warmup_forwards = warmup_forwards
        self.initialization = dict(started_at_utc=utc(), model_load_count=0, checkpoint_load_count=0)
        started = time.perf_counter()
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no CPU fallback')
        if 'mode' not in inspect.signature(Base3DSegmentor.forward).parameters:
            raise RuntimeError('Installed framework lacks tensor forward API')
        actual_tf32 = dict(matmul=torch.backends.cuda.matmul.allow_tf32, cudnn=torch.backends.cudnn.allow_tf32)
        if actual_tf32 != TF32 or torch.get_default_dtype() != torch.float32:
            raise ValueError('Runtime precision/TF32 defaults differ from the verified baseline')
        self.initialization.update(tf32=actual_tf32, precision='float32', autocast=False, batch_size=1)
        register_all_modules(init_default_scope=False)
        init_default_scope('mmdet3d')
        self.model = MODELS.build(context['cfg'].model).cpu()
        self.initialization['model_load_count'] = 1
        inference.strict_load_checkpoint(self.model, Path(context['shared']['checkpoint']['path']), self.initialization)
        self.initialization['checkpoint_load_count'] = 1
        self.model.eval()
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.model.to(self.device)
            self.model.eval()  # Rebuild EfficientViT's plain-tensor attention caches on CUDA.
            self.model.data_preprocessor.cpu()
            torch.cuda.synchronize(self.device)
            props = torch.cuda.get_device_properties(self.device)
            self.initialization['gpu'] = dict(name=props.name, total_memory_bytes=props.total_memory)
        self.initialization['wall_s'] = time.perf_counter()-started
        self.initialization['allocator'] = self.allocator()

    def synchronize(self):
        if self.device.type == 'cuda':
            self.torch.cuda.synchronize(self.device)

    def allocator(self):
        if self.device.type != 'cuda':
            return dict(peak_allocated_bytes=None,peak_reserved_bytes=None)
        return dict(peak_allocated_bytes=self.torch.cuda.max_memory_allocated(self.device),
                    peak_reserved_bytes=self.torch.cuda.max_memory_reserved(self.device))

    def predict(self, item, base, inputs, sweeps, input_fingerprint, sensor_files, folder, report):
        prepared = logits = None
        torch, cfg = self.torch, self.context['cfg']
        metadata = metadata_for_frame(base,item,self.context['shared'],cfg,self.context['environment'],self.context['code'])
        metadata['benchmark_provenance'] = dict(shared_fingerprint=self.context['shared_fingerprint'],
            input_fingerprint=input_fingerprint, sensor_files=sensor_files)
        report.update(precision='float32',autocast=False,check_only=False,tf32=dict(TF32),
            sample_token=item['sample']['token'],scene_name=item['scene']['name'],scene_index=item['scene_index'],
            frame_index=item['frame_index'],split='mini_val',metrics=metadata['metrics'],timings_s={})
        started = time.perf_counter()
        try:
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(self.device)
            report['stage'] = 'preprocessing'
            stage = time.perf_counter()
            prepared, details = inference.prepare_frame(inputs,sweeps,cfg,self.model,report['warnings'])
            if self.device.type == 'cuda':
                prepared = inference.move_data(prepared,self.device)
            self.synchronize()
            report['timings_s']['preprocessing'] = time.perf_counter()-stage
            report['preprocessing'] = details
            metadata['image_preprocessing']['batch_shape'] = details['image_shape']
            metadata['sensors']['point_counts'] = details['point_counts']
            report['stage'] = 'forward'
            report['warmup'] = self.completed_forwards < self.warmup_forwards
            self.attempted_forwards += 1
            self.synchronize()
            stage = time.perf_counter()
            with torch.inference_mode():
                logits = self.model(inputs=prepared['inputs'],data_samples=prepared['data_samples'],mode='tensor')
            self.synchronize()
            report['timings_s']['forward'] = time.perf_counter()-stage
            self.completed_forwards += 1
            report['stage'] = 'postprocessing_export'
            stage = time.perf_counter()
            report['prediction'] = inference.export_predictions(logits,metadata,folder)
            report['timings_s']['postprocessing_export'] = time.perf_counter()-stage
            report['timings_s']['pipeline_total'] = time.perf_counter()-started
            report['vram'] = dict(**self.allocator(),scope='PyTorch allocator: model resident plus this frame pipeline; not physical VRAM measured')
            report['status'] = 'success'
        except Exception as error:
            report['status'] = 'cuda_out_of_memory' if isinstance(error,torch.cuda.OutOfMemoryError) else 'failed'
            raise
        finally:
            # No prepared batch/logits is stored on the session or accumulated across frames.
            del logits, prepared
            gc.collect()
            try:
                self.synchronize()
                report['vram'] = dict(**self.allocator(),scope='PyTorch allocator, not physical VRAM measured')
                if self.device.type == 'cuda':
                    if report.get('status') == 'cuda_out_of_memory':
                        torch.cuda.empty_cache()
                        report['oom_allocator_cache_released'] = True
                    report['allocated_after_release_bytes'] = torch.cuda.memory_allocated(self.device)
            except Exception as error:
                report['cleanup_error'] = str(error)
        metadata.update(status='success',precision='float32',autocast=False,tf32=dict(TF32),
                        warnings=report['warnings'],timings_s=report['timings_s'],vram=report['vram'],
                        timing_note='Frame pipeline excludes shared startup; warm-up forwards still contribute accuracy.')
        atomic_json(folder/'metadata.json',metadata)
        atomic_json(folder/'report.json',report)
        (folder/'resolved_config.py').write_text(self.context['resolved_config'],encoding='utf-8')

    def close(self):
        self.model = None
        gc.collect()
        if self.device.type == 'cuda':
            self.torch.cuda.empty_cache()


def prepare_context(args):
    from mmengine.config import Config
    started = time.perf_counter()
    config_path = inference.repo_path(args.config)
    cfg = Config.fromfile(str(config_path))
    reference = Config.fromfile(str(ROOT/'configs/OccFusion_infer_mini.py'))
    fields = ('model','point_cloud_range','camera_order','radar_order','inference_pipeline',
              'lidar_loader','lidar_sweep_loader','radar_loader','dataset_version')
    if any(cfg[key] != reference[key] for key in fields) or cfg.model != Config.fromfile(str(ROOT/'configs/OccFusion.py')).model:
        raise ValueError('Model, grid, images, sensors or preprocessing differ from verified baseline')
    checkpoint = inference.repo_path(args.checkpoint or cfg.load_from)
    data_root = inference.repo_path(args.data_root or cfg.data_root)
    inference.validate_paths(checkpoint,data_root,cfg.dataset_version)
    catalog = evaluator.DatasetCatalog(data_root)
    gt_index = GroundTruthIndex(args.gt_root)
    environment = dict(python=sys.version,executable=sys.executable,packages={name:version(name) for name in
        ('torch','torchvision','mmengine','mmcv','mmdet','mmdet3d','nuscenes-devkit','numpy')})
    for name in ('opencv-python','opencv-python-headless','Pillow','pyquaternion','scipy'):
        try:
            environment['packages'][name] = version(name)
        except PackageNotFoundError:
            environment['packages'][name] = None
    source_paths = sorted(set([*ROOT.glob('occfusion/*.py'), *ROOT.glob('configs/*.py'),
                               *ROOT.glob('configs/_base_/*.py'), ROOT/'tools/infer_nuscenes_mini.py']))
    resolved = cfg.pretty_text
    shared = dict(schema_version=1, precision='float32',autocast=False,batch_size=1,tf32=dict(TF32),device=args.device,
        checkpoint=dict(path=str(checkpoint),**file_record(checkpoint)),config_path=str(config_path),
        config_source_sha256=inference.file_hash(config_path),resolved_config_sha256=evaluator.sha256(resolved.encode()),
        data_root=str(data_root),dataset_version=cfg.dataset_version,environment=environment,
        dataset_table_hashes={name:inference.file_hash(data_root/cfg.dataset_version/(name+'.json')) for name in inference.TABLES},
        source_hashes={str(path.relative_to(ROOT)):inference.file_hash(path) for path in source_paths},
        adapter_source_sha256=evaluator.sha256((''.join(inspect.getsource(fn) for fn in
            (BaselineSession,metadata_for_frame,sensor_identity))).encode()))
    protocol = evaluator.protocol()
    return dict(cfg=cfg,catalog=catalog,gt_index=gt_index,shared=shared,shared_fingerprint=fingerprint(shared),
                protocol=protocol,protocol_fingerprint=fingerprint(protocol),resolved_config=resolved,
                environment=environment,code=inference.code_version(),setup_wall_s=time.perf_counter()-started)


def build_manifest(context, pilot):
    frames = []
    full = ordered_samples(context['catalog'])
    for item in ordered_samples(context['catalog'],pilot):
        sd, sample = item['lidar'], item['sample']
        key, path = context['gt_index'].lookup(sd['filename'])
        frames.append(dict(scene=item['scene']['name'],scene_index=item['scene_index'],frame_index=item['frame_index'],
            sample_token=sample['token'],timestamp_us=sample['timestamp'],reference_timestamp_us=sd['timestamp'],
            lidar_sample_data_token=sd['token'],lidar_filename=sd['filename'],gt_filename=key,
            gt_path=str(path) if path else None,gt_initial=file_record(path) if path else None))
    membership = [{key:value for key,value in frame.items() if key != 'gt_initial'} for frame in frames]
    identity = dict(scope='pilot_two_scenes' if pilot else 'mini_val',split='mini_val',
                    official_scene_names=sorted(context['catalog'].splits['mini_val']),
                    dataset_frame_count=len(full),expected_frame_count=len(frames),frames=membership,
                    prediction_provenance=context['shared_fingerprint'])
    return dict(schema_version=1,kind='occfusion_occupancy_benchmark',created_at_utc=utc(),
                identity=identity,identity_sha256=fingerprint(identity),frames=frames,
                prediction_provenance=context['shared'],initial_evaluation_protocol=context['protocol'])


def prediction_cache(folder, entry, context, input_fingerprint):
    """A commit receipt AND hashes AND provenance authorize reuse, never filenames alone."""
    folder = Path(folder)
    try:
        if folder.name.endswith('.pending'):
            raise ValueError('Unpublished prediction staging directory')
        receipt = read_json(folder/'prediction_commit.json')
        if receipt['status'] != 'success' or receipt['sample_token'] != entry['sample_token']:
            raise ValueError('Incomplete receipt or wrong sample token')
        if receipt['input_fingerprint'] != input_fingerprint or receipt['shared_fingerprint'] != context['shared_fingerprint']:
            raise ValueError('Prediction checkpoint/config/preprocessing provenance changed')
        if receipt['artifact_hashes'] != file_records(folder,[*ARTIFACTS,'resolved_config.py']):
            raise ValueError('Prediction artifact hash mismatch')
        metadata, report, labels, hashes = evaluator.load_prediction(folder)
        if metadata['sample_token'] != entry['sample_token'] or report['sample_token'] != entry['sample_token']:
            raise ValueError('Saved prediction sample token differs from manifest')
        context['catalog'].validate_sample(metadata,report)
        if metadata['checkpoint'] != context['shared']['checkpoint'] or metadata['config']['resolved_sha256'] != context['shared']['resolved_config_sha256'] or report['tf32'] != context['shared']['tf32']:
            raise ValueError('Saved prediction model/config/TF32 mismatch')
        if 'environment' in context['shared'] and metadata['environment'] != context['shared']['environment']:
            raise ValueError('Saved prediction preprocessing library versions differ')
        provenance = metadata['benchmark_provenance']
        if provenance['input_fingerprint'] != input_fingerprint or provenance['shared_fingerprint'] != context['shared_fingerprint']:
            raise ValueError('Saved preprocessing provenance mismatch')
        return dict(folder=folder,metadata=metadata,report=report,labels=labels,hashes=hashes,receipt=receipt),None
    except (OSError,ValueError,KeyError,TypeError) as error:
        return None,str(error)


def evaluation_cache(folder, prediction, gt_record, context, token):
    try:
        folder = Path(folder)
        if folder.name.endswith('.pending'):
            raise ValueError('Unpublished evaluation staging directory')
        commit = read_json(folder/'evaluation_commit.json')
        value = read_json(folder/'evaluation.json')
        if commit['status'] != 'success' or commit['sample_token'] != token:
            raise ValueError('Incomplete evaluation receipt')
        if commit['evaluation'] != file_record(folder/'evaluation.json') or commit['artifact_hashes'] != prediction['hashes']:
            raise ValueError('Evaluation/prediction hashes changed')
        if commit['gt'] != gt_record or commit['protocol_fingerprint'] != context['protocol_fingerprint']:
            raise ValueError('GT hash or evaluation protocol changed')
        frame = value['frames'][0]
        if value['status'] != 'success' or value['frame_count_evaluated'] != 1 or frame['sample_token'] != token or frame['gt_sha256'] != gt_record['sha256'] or frame['prediction_sha256'] != prediction['hashes']['occupancy.npz']['sha256'] or frame['artifact_hashes'] != prediction['hashes'] or value['protocol'] != context['protocol']:
            raise ValueError('Evaluation schema/token/provenance mismatch')
        matrix = np.asarray(value['metrics']['confusion_matrix'],dtype=np.int64)
        if evaluator.summarize(matrix) != value['metrics']:
            raise ValueError('Evaluation count evidence mismatch')
        return dict(folder=folder,value=value,matrix=matrix),None
    except (OSError,ValueError,KeyError,TypeError,IndexError) as error:
        return None,str(error)


def completed_candidates(parent, preferred=None):
    paths = []
    if preferred:
        paths.append(Path(preferred))
    if Path(parent).is_dir():
        paths += sorted((p for p in Path(parent).iterdir() if p.is_dir() and not p.name.endswith('.pending')),reverse=True)
    return list(dict.fromkeys(paths))


def pending_attempt(parent):
    parent = Path(parent)
    parent.mkdir(parents=True,exist_ok=True)
    folder = parent/(utc().replace(':','-')+'_'+uuid.uuid4().hex[:10]+'.pending')
    folder.mkdir()
    return folder


def publish(folder):
    final = folder.with_name(folder.name.removesuffix('.pending'))
    folder.rename(final)
    return final


def compute_evaluation(prediction, entry, gt_record, context, parent):
    content = Path(entry['gt_path']).read_bytes()
    if evaluator.sha256(content) != gt_record['sha256']:
        raise ValueError('GT changed during evaluation')
    dense, details = evaluator.sparse_to_dense(np.load(BytesIO(content),allow_pickle=False))
    matrix = evaluator.confusion_statistics(prediction['labels'],dense)
    metrics = evaluator.summarize(matrix)
    check = evaluator.compare_reference([dense],[prediction['labels']],metrics)
    metadata = prediction['metadata']
    frame = dict(run_dir=str(prediction['folder']),status='evaluated',metrics=metrics,
        sample_token=entry['sample_token'],artifact_hashes=prediction['hashes'],
        prediction_sha256=prediction['hashes']['occupancy.npz']['sha256'],gt_sha256=gt_record['sha256'],
        scene=entry['scene'],frame_index=entry['frame_index'],split='mini_val',
        timestamp_us=entry['timestamp_us'],reference_timestamp_us=entry['reference_timestamp_us'],
        lidar_sample_data_token=entry['lidar_sample_data_token'],lidar_filename=entry['lidar_filename'],
        gt_path=entry['gt_path'],gt_status='has_gt',gt_bytes=gt_record['bytes'],
        expected_gt_filename=entry['gt_filename'],gt_statistics=details,reference_check=check)
    value = dict(schema_version=1,status='success',generated_at_utc=utc(),protocol=context['protocol'],
        split='mini_val',split_source='nuscenes.utils.splits.create_splits_scenes',
        official_split_scene_names=context['catalog'].splits['mini_val'],dataset=metadata['dataset'],
        frame_count_requested=1,frame_count_evaluated=1,all_requested_frames_evaluated=True,
        metrics_scope='One benchmark frame; pooled benchmark result is a separate set-level artifact',
        sample_tokens=[entry['sample_token']],frames=[frame],metrics=metrics,pooled_reference_check=check)
    folder = pending_attempt(parent)
    atomic_json(folder/'evaluation.json',value)
    atomic_json(folder/'evaluation_commit.json',dict(status='success',sample_token=entry['sample_token'],
        evaluation=file_record(folder/'evaluation.json'),artifact_hashes=prediction['hashes'],
        gt=gt_record,protocol_fingerprint=context['protocol_fingerprint']))
    folder = publish(folder)
    return dict(folder=folder,value=value,matrix=matrix)


def timing_statistics(rows):
    def stats(values):
        return dict(sample_count=len(values),median_s=float(np.median(values)) if values else None,
                    p95_s=float(np.percentile(values,95)) if values else None)
    measured, malformed = [], 0
    for row in rows:
        if row.get('prediction_source') != 'new' or row.get('forward_s') in (None,''):
            continue
        try:
            value = float(row['forward_s'])
            if not np.isfinite(value) or value < 0 or row.get('warmup') is None:
                raise ValueError('Invalid timing record')
            measured.append(row)
        except (TypeError,ValueError):
            malformed += 1
    steady = [row for row in measured if str(row.get('warmup')).lower() not in ('true','1')]
    return dict(all_completed_forwards=stats([float(row['forward_s']) for row in measured]),
                excluding_marked_warmup=stats([float(row['forward_s']) for row in steady]),
                warmup_completed_count=len(measured)-len(steady),malformed_timing_rows_excluded=malformed,
                scope='Synchronized tensor forward only; actual new calls, no cache I/O; p95 NumPy linear percentile; units seconds')


def append_timing(path, row):
    new = not path.exists() or path.stat().st_size == 0
    if not new and path.stat().st_size:
        with path.open('rb') as previous:
            previous.seek(-1,os.SEEK_END)
            incomplete = previous.read(1) != b'\n'
        if incomplete:
            with path.open('a',encoding='utf-8') as stream:
                stream.write('\n')
    with path.open('a',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=TIMING_FIELDS,extrasaction='ignore')
        if new:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def read_timings(path):
    if not path.exists():
        return []
    with path.open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def save_summary(out, manifest, progress, matrices, current_session, final=False, reference_check=None):
    entries = manifest['frames']
    expected = [entry['sample_token'] for entry in entries]
    valid = [token for token in expected if token in matrices]
    missing = [dict(sample_token=entry['sample_token'],scene=entry['scene'],frame_index=entry['frame_index'],
        status=progress['frames'].get(entry['sample_token'],{}).get('status','pending'),
        error=progress['frames'].get(entry['sample_token'],{}).get('error')) for entry in entries if entry['sample_token'] not in matrices]
    status = ('complete' if len(valid)==len(expected) else 'partial') if final else 'running'
    # Only unique successful tokens in the fixed manifest contribute, never retry/cached duplicates.
    pooled = evaluator.summarize(sum((matrices[token] for token in valid),np.zeros((17,17),dtype=np.int64))) if valid else None
    set_result = dict(schema_version=1,status=status,scope=manifest['identity']['scope'],split='mini_val',
        expected_frame_count=len(expected),successful_frame_count=len(valid),
        failed_frame_count=sum(item['status']=='failed' for item in missing),missing_frames=missing,
        sample_tokens=valid,scene_counts=dict(Counter(entry['scene'] for entry in entries if entry['sample_token'] in matrices)),
        protocol=current_session['evaluation_protocol'],metrics=pooled,
        metrics_scope=f'Pooled over {len(valid)}/{len(expected)} fixed manifest frames; {manifest["identity"]["scope"]}, not full nuScenes validation',
        manifest_sha256=inference.file_hash(out/'manifest.json'),reference_check=reference_check,
        review_scope='Set-level pooled result; not attached to a single-frame run or approval')
    rows = read_timings(out/'timings.csv')
    warmup_tokens = {row['sample_token'] for row in rows if row.get('prediction_source') == 'new' and
                     row.get('forward_s') not in (None,'') and str(row.get('warmup')).lower() in ('true','1')}
    report = dict(schema_version=1,status=status,split='mini_val',scope=manifest['identity']['scope'],
        dataset_frame_count=manifest['identity']['dataset_frame_count'],expected_frame_count=len(expected),
        successful_frame_count=len(valid),failed_frame_count=set_result['failed_frame_count'],
        missing_frames=missing,metrics=pooled,session_id=current_session['id'],
        actual_scene_counts=set_result['scene_counts'],metrics_scope=set_result['metrics_scope'],
        current_invocation=current_session,sessions=progress['sessions'],
        latency_current_invocation=timing_statistics([row for row in rows if row['session_id']==current_session['id']]),
        latency_all_invocations=timing_statistics(rows),
        precision='float32',autocast=False,batch_size=1,tf32=dict(TF32),
        allocator=dict(peak_allocated_bytes=max((int(row['peak_allocated_bytes']) for row in rows if row.get('peak_allocated_bytes') not in ('',None)),default=None),
            peak_reserved_bytes=max((int(row['peak_reserved_bytes']) for row in rows if row.get('peak_reserved_bytes') not in ('',None)),default=None),
            scope='PyTorch allocator peaks with model resident; not physical VRAM measured'),
        accuracy_excludes_warmup=False,all_warmup_predictions_included_in_accuracy=warmup_tokens.issubset(matrices),
        warmup_frames_without_valid_evaluation=sorted(warmup_tokens-set(matrices)),reference_check=reference_check,
        manifest_sha256=set_result['manifest_sha256'],provenance=manifest['prediction_provenance'])
    progress.update(status=status,successful_frame_count=len(valid),last_updated_at_utc=utc())
    atomic_json(out/'progress.json',progress)
    atomic_json(out/'metrics_pooled.json',set_result)
    atomic_json(out/'benchmark_report.json',report)
    return report


def execute(args, context, session_factory=BaselineSession):
    out = inference.repo_path(args.out_dir)
    candidate = build_manifest(context,args.pilot)
    if args.resume:
        manifest = read_json(out/'manifest.json')
        if manifest.get('kind') != 'occfusion_occupancy_benchmark' or manifest['identity_sha256'] != fingerprint(manifest['identity']) or manifest['identity_sha256'] != candidate['identity_sha256']:
            raise ValueError('Resume manifest membership or prediction provenance differs; use a new output directory')
        stored_membership = [{key:value for key,value in entry.items() if key != 'gt_initial'} for entry in manifest['frames']]
        if stored_membership != manifest['identity']['frames'] or len(manifest['frames']) != manifest['identity']['expected_frame_count'] or fingerprint(manifest['prediction_provenance']) != manifest['identity']['prediction_provenance']:
            raise ValueError('Manifest frame list or embedded provenance was modified')
        progress = read_json(out/'progress.json') if (out/'progress.json').exists() else dict(frames={},sessions=[])
        if progress.get('manifest_sha256',inference.file_hash(out/'manifest.json')) != inference.file_hash(out/'manifest.json'):
            raise ValueError('Immutable manifest hash changed')
    else:
        manifest = candidate
        atomic_json(out/'manifest.json',manifest)
        progress = dict(frames={},sessions=[])
    progress['manifest_sha256'] = inference.file_hash(out/'manifest.json')
    identifier = uuid.uuid4().hex
    current = dict(id=identifier,pid=os.getpid(),started_at_utc=utc(),command=[sys.executable,*sys.argv],
        evaluate_only=args.evaluate_only,dataset_load_count=1,model_load_count=0,checkpoint_load_count=0,
        bootstrap_wall_s=context.get('bootstrap_wall_s',0.0),shared_setup_wall_s=context['setup_wall_s'],
        model_initialization=None,model_initialization_attempts=0,model_initialization_wall_s=0.0,
        warmup_forwards_per_process=args.warmup_forwards,predictions_new=0,predictions_cached=0,
        evaluations_new=0,evaluations_cached=0,failed_frames=0,
        evaluation_protocol=context['protocol'],benchmark_script_sha256=inference.file_hash(Path(__file__)))
    progress['sessions'].append(current)
    matrices, reference_gt, reference_pred = {}, [], []
    session = None
    process_started = time.perf_counter()
    save_summary(out,manifest,progress,matrices,current)
    try:
        for ordinal, entry in enumerate(manifest['frames'],1):
            token = entry['sample_token']
            previous = progress['frames'].get(token,{})
            state = dict(status='running',sample_token=token,session_id=identifier,started_at_utc=utc())
            progress['frames'][token] = state
            atomic_json(out/'progress.json',progress)
            row = dict(session_id=identifier,sample_token=token,scene=entry['scene'],frame_index=entry['frame_index'],
                       prediction_source='none',evaluation_source='none',warmup=False)
            started, stage = time.perf_counter(), 'input_validation'
            prediction = evaluation = None
            inputs = sweeps = base = None
            model_startup_in_frame = 0.0
            try:
                item = context['catalog'].samples[token]
                warnings = []
                inputs, sweeps, base = inference.make_frame_info(context['catalog'].nusc,item['sample'],context['cfg'],warnings)
                input_fp, sensors = sensor_identity(base,context['shared_fingerprint'])
                state['input_fingerprint'] = input_fp
                candidates = completed_candidates(out/'frames'/token,previous.get('prediction_dir'))
                reasons = []
                for folder in candidates:
                    prediction, reason = prediction_cache(folder,entry,context,input_fp)
                    if prediction:
                        break
                    reasons.append(dict(folder=str(folder),reason=reason))
                state['prediction_cache_rejections'] = reasons
                row['input_validation_s'] = time.perf_counter()-started
                if prediction:
                    current['predictions_cached'] += 1
                    row['prediction_source'] = 'cache'
                else:
                    if args.evaluate_only:
                        raise ValueError('No valid cached prediction; --evaluate-only forbids model loading/forward')
                    stage = 'model_initialization'
                    if session is None:
                        if current.get('model_initialization_error'):
                            raise RuntimeError('Model initialization already failed in this process; no reload attempted')
                        init_started = time.perf_counter()
                        current['model_initialization_attempts'] += 1
                        try:
                            session = session_factory(context,args.warmup_forwards)
                        except Exception as error:
                            current['model_initialization_error'] = f'{type(error).__name__}: {error}'
                            raise
                        finally:
                            model_startup_in_frame = time.perf_counter()-init_started
                            current['model_initialization_wall_s'] += model_startup_in_frame
                        current.update(model_load_count=1,checkpoint_load_count=1,model_initialization=session.initialization)
                    stage = 'prediction'
                    folder = pending_attempt(out/'frames'/token)
                    frame_report = dict(status='started',stage='preprocessing',warnings=warnings)
                    try:
                        session.predict(item,base,inputs,sweeps,input_fp,sensors,folder,frame_report)
                    finally:
                        row.update(warmup=frame_report.get('warmup',False),
                            **{key+'_s':value for key,value in frame_report.get('timings_s',{}).items() if key in ('preprocessing','forward','postprocessing_export')},
                            **{key:frame_report.get('vram',{}).get(key) for key in ('peak_allocated_bytes','peak_reserved_bytes')},
                            allocated_after_release_bytes=frame_report.get('allocated_after_release_bytes'))
                        row['prediction_source'] = 'new'
                        if frame_report.get('status') != 'success':
                            state['prediction_failure_report'] = frame_report
                    atomic_json(folder/'prediction_commit.json',dict(status='success',sample_token=token,
                        input_fingerprint=input_fp,shared_fingerprint=context['shared_fingerprint'],
                        artifact_hashes=file_records(folder,[*ARTIFACTS,'resolved_config.py'])))
                    folder = publish(folder)
                    prediction, reason = prediction_cache(folder,entry,context,input_fp)
                    if not prediction:
                        raise ValueError('New prediction failed commit validation: '+str(reason))
                    current['predictions_new'] += 1
                state.update(prediction_dir=str(prediction['folder']),prediction_sha256=prediction['hashes']['occupancy.npz']['sha256'])
                atomic_json(out/'progress.json',progress)  # Durable prediction even if evaluation is interrupted.
                stage = 'evaluation'
                if not entry['gt_path'] or not Path(entry['gt_path']).is_file():
                    raise FileNotFoundError('missing_gt: no matching SurroundOcc file')
                gt_record = file_record(entry['gt_path'])
                state.update(gt=gt_record,gt_changed_since_manifest=gt_record!=entry['gt_initial'])
                eval_started = time.perf_counter()
                rejections = []
                for folder in completed_candidates(out/'evaluations'/token,previous.get('evaluation_dir')):
                    evaluation, reason = evaluation_cache(folder,prediction,gt_record,context,token)
                    if evaluation:
                        break
                    rejections.append(dict(folder=str(folder),reason=reason))
                state['evaluation_cache_rejections'] = rejections
                if evaluation:
                    current['evaluations_cached'] += 1
                    row['evaluation_source'] = 'cache'
                else:
                    evaluation = compute_evaluation(prediction,entry,gt_record,context,out/'evaluations'/token)
                    current['evaluations_new'] += 1
                    row['evaluation_source'] = 'new'
                row['evaluation_s'] = time.perf_counter()-eval_started
                matrices[token] = evaluation['matrix']
                if args.verify_reference:
                    content = Path(entry['gt_path']).read_bytes()
                    if evaluator.sha256(content) != gt_record['sha256']:
                        raise ValueError('GT changed before reference comparison')
                    reference_gt.append(evaluator.sparse_to_dense(np.load(BytesIO(content),allow_pickle=False))[0])
                    reference_pred.append(prediction['labels'])
                state.update(status='success',evaluation_dir=str(evaluation['folder']),
                             evaluation_sha256=inference.file_hash(evaluation['folder']/'evaluation.json'))
            except Exception as error:
                matrices.pop(token,None)
                current['failed_frames'] += 1
                state.update(status='failed',error=dict(stage=state.get('prediction_failure_report',{}).get('stage',stage),
                    type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
                    cuda_out_of_memory=state.get('prediction_failure_report',{}).get('status')=='cuda_out_of_memory'))
                atomic_json(out/'failures'/token/(identifier+'.json'),state)
            finally:
                state.update(finished_at_utc=utc(),frame_wall_s=time.perf_counter()-started-model_startup_in_frame,
                             excluded_shared_model_initialization_s=model_startup_in_frame)
                row.update(status=state['status'],frame_wall_s=state['frame_wall_s'])
                append_timing(out/'timings.csv',row)
                current['wall_s'] = time.perf_counter()-process_started
                current.update(forward_attempts=session.attempted_forwards if session else 0,
                               completed_forwards=session.completed_forwards if session else 0)
                save_summary(out,manifest,progress,matrices,current)
                print(f'[{ordinal}/{len(manifest["frames"])}] {entry["scene"]} frame {entry["frame_index"]}: '
                      f'{state["status"]}; prediction={row["prediction_source"]}; evaluation={row["evaluation_source"]}; '
                      f'forward={row.get("forward_s")}; {state["frame_wall_s"]:.2f}s',flush=True)
                del prediction, evaluation, inputs, sweeps, base
        check = None
        if args.verify_reference and matrices:
            check = evaluator.compare_reference(reference_gt,reference_pred,
                evaluator.summarize(sum(matrices.values(),np.zeros((17,17),dtype=np.int64))))
        current.update(finished_at_utc=utc(),wall_s=time.perf_counter()-process_started,
                       forward_attempts=session.attempted_forwards if session else 0,
                       completed_forwards=session.completed_forwards if session else 0)
        result = save_summary(out,manifest,progress,matrices,current,True,check)
        return result
    finally:
        if session is not None:
            session.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/OccFusion_infer_mini.py')
    parser.add_argument('--data-root')
    parser.add_argument('--checkpoint')
    parser.add_argument('--gt-root',required=True)
    parser.add_argument('--out-dir',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--pilot',action='store_true',help='Predeclared frame index 1 in each official mini_val scene, before full benchmark.')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--evaluate-only',action='store_true',help='Never construct a model; require valid saved predictions.')
    parser.add_argument('--warmup-forwards',type=int,default=1,help='Mark first N completed forwards per process; keep them in accuracy.')
    parser.add_argument('--verify-reference',action='store_true',help='Also compare pooled result with original EvalMetric on CPU.')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'cpu|cuda:[0-9]+',args.device) or args.warmup_forwards < 0:
        parser.error('Use --device cpu/cuda:N and nonnegative --warmup-forwards')
    if args.evaluate_only and not args.resume:
        parser.error('--evaluate-only requires --resume')
    return args


def main(argv=None):
    args = parse_args(argv)
    out = inference.repo_path(args.out_dir)
    owned = False
    try:
        with output_lock(out,args.resume):
            owned = True
            started = time.perf_counter()
            context = prepare_context(args)
            context['setup_wall_s'] = time.perf_counter()-started
            context['bootstrap_wall_s'] = started-PROCESS_STARTED
            result = execute(args,context)
            print(json.dumps(dict(status=result['status'],scope=result['scope'],
                successful=result['successful_frame_count'],expected=result['expected_frame_count'],
                failed=result['failed_frame_count'],new=result['current_invocation']['predictions_new'],
                cached=result['current_invocation']['predictions_cached'],report=str(out/'benchmark_report.json')),ensure_ascii=False),flush=True)
            return 0 if result['status']=='complete' else 2
    except Exception as error:
        # Preserve existing manifests/artifacts on incompatible resume, too.
        path = None
        if owned:
            path = out/('process_failure_'+uuid.uuid4().hex+'.json')
            atomic_json(path,dict(status='failed',command=[sys.executable,*sys.argv],
                stage='initialization_or_orchestration',type=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
        print(f'Benchmark failed: {error}; report={path}',file=sys.stderr,flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
