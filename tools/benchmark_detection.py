#!/usr/bin/env python3
"""Resumable, sequential CenterPoint benchmark on the entire official mini_val."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import csv
import gc
from importlib.metadata import version
from importlib.util import find_spec
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.detection_benchmark_common import (
    BUNDLE_FILES, KIND, atomic_json, audit_frame, canonical, committed_prediction,
    exclusive_lock, file_record, file_records, fingerprint, gpu_lock_path,
    ordered_frames, read_json, utc, valid_predictions, validate_bundle,
    verify_files, verify_manifest,
)
from tools import evaluate_detection as evaluator

TIMING_FIELDS = ('invocation_id', 'sample_token', 'scene', 'frame_index', 'source',
    'status', 'warmup', 'forward_completed', 'input_validation_s', 'preprocessing_s',
    'forward_s', 'postprocessing_export_s', 'end_to_end_s', 'peak_allocated_bytes',
    'peak_reserved_bytes', 'allocated_after_release_bytes', 'prediction_directory')


def create_context(args):
    """CPU preparation: dataset/config loaded once, no model/checkpoint deserialization."""
    start = time.perf_counter()
    from mmengine.config import Config
    from nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    source = Path(args.profiles).resolve()
    profile = next(p for p in read_json(source)['profiles'] if p['id'] == args.profile)
    if (profile.get('task') != 'detection' or profile['precision'] != 'float32'
            or profile['autocast'] or profile['batch_size'] != 1 or profile['device'] != 'cuda:0'):
        raise ValueError('Benchmark requires the verified FP32/batch-1 CenterPoint detection profile')
    def absolute(value):
        path = Path(value)
        return (source.parent / path).resolve() if not path.is_absolute() else path.resolve()
    config, checkpoint, data = (absolute(profile[k]) for k in ('config', 'checkpoint', 'data_root'))
    files = {str(source): file_record(source)}
    for name, expected in profile['source_hashes'].items():
        path = ROOT / name
        record = file_record(path)
        if record['sha256'] != expected:
            raise ValueError('Pinned profile source changed: ' + name)
        files[str(path.resolve())] = record
    record = file_record(checkpoint)
    if record != dict(sha256=profile['checkpoint_sha256'], bytes=profile['checkpoint_bytes']):
        raise ValueError('Checkpoint differs from pinned profile')
    files[str(checkpoint)] = record
    tables = {}
    for name, expected in profile['dataset_table_hashes'].items():
        path = data / 'v1.0-mini' / (name + '.json')
        record = file_record(path)
        if record['sha256'] != expected:
            raise ValueError('Pinned dataset table changed: ' + name)
        files[str(path)] = tables[name] = record
    cfg = Config.fromfile(str(config))
    if version('mmdet3d') != '1.4.0':
        raise ValueError('The verified detector requires MMDetection3D 1.4.0')
    if (cfg.model.type != 'CenterPoint' or cfg.model.test_cfg.pts.nms_type != 'circle'
            or list(cfg.voxel_size) != [.2, .2, 8] or list(cfg.point_cloud_range) != profile['geometry']['bounds']):
        raise ValueError('Config is not the verified pillar02/circle baseline')
    resolved = cfg.dump()
    runtime_files = ['tools/benchmark_detection.py', 'tools/detection_benchmark_common.py',
                     'tools/infer_centerpoint_mini.py', 'tools/infer_nuscenes_mini.py']
    for name in runtime_files:
        files[str(ROOT / name)] = file_record(ROOT / name)
    package = Path(find_spec('mmdet3d').origin).parent
    installed = {}
    for name in ('datasets/transforms/loading.py', 'datasets/transforms/transforms_3d.py',
                 'datasets/transforms/formating.py', 'models/data_preprocessors/data_preprocessor.py',
                 'models/detectors/centerpoint.py', 'models/dense_heads/centerpoint_head.py',
                 'models/voxel_encoders/pillar_encoder.py', 'models/backbones/second.py',
                 'models/necks/second_fpn.py'):
        path = package / name
        files[str(path)] = installed[name] = file_record(path)
    test_pipeline = json.loads(json.dumps(cfg.test_pipeline))
    provenance = dict(profile_id=profile['id'], profile_file=str(source), data_root=str(data),
        dataset_version='v1.0-mini', config=dict(path=str(config), sha256=file_record(config)['sha256'],
            resolved_sha256=__import__('hashlib').sha256(resolved.encode()).hexdigest()),
        checkpoint=dict(path=str(checkpoint), **file_record(checkpoint)),
        reference_cli_sha256=file_record(ROOT / 'tools/infer_centerpoint_mini.py')['sha256'],
        runtime_sha256=file_record(Path(__file__))['sha256'], files=files,
        installed_pipeline_model_sources=installed, test_pipeline=test_pipeline,
        bounds=list(cfg.point_cloud_range), voxel_size=list(cfg.voxel_size), class_names=list(cfg.class_names),
        packages={name: version(name) for name in ('mmdet3d', 'mmengine', 'mmcv', 'mmdet', 'torch', 'numpy',
            'nuscenes-devkit', 'pyquaternion')},
        precision='float32', autocast=False, batch_size=1, tf32=profile['tf32'])
    pipeline_fp = fingerprint(dict(reference_cli_sha256=provenance['reference_cli_sha256'],
        test_pipeline=test_pipeline, installed_sources=installed, bounds=provenance['bounds'],
        voxel_size=provenance['voxel_size'], packages=provenance['packages']))
    provenance['pipeline_fingerprint'] = pipeline_fp
    nusc = NuScenes(version='v1.0-mini', dataroot=str(data), verbose=False)
    names, entries = ordered_frames(nusc)
    for entry in entries:
        try:
            _, metadata, audit, identity = audit_frame(nusc, entry, pipeline_fp)
            entry.update(input_metadata=metadata, input_audit=audit, input_fingerprint=identity)
        except Exception as error:
            entry.update(input_metadata=None, input_fingerprint=None,
                input_audit=dict(status='failed', error=dict(type=type(error).__name__, message=str(error))))
    protocol = evaluator.protocol()
    identity = dict(split='mini_val', split_source='nuscenes.utils.splits.create_splits_scenes',
        scene_names=names, frames=entries, prediction_fingerprint=fingerprint(provenance),
        evaluation_protocol_fingerprint=fingerprint(protocol))
    manifest = dict(schema_version=1, kind=KIND, created_at_utc=utc(), identity=identity,
        identity_sha256=fingerprint(identity), prediction_provenance=provenance,
        evaluation_protocol=protocol, dataset_table_hashes=tables,
        ordering='scene name ascending, sample chain ordinal; independent of prediction/metrics',
        mini_scene_membership_in_trainval={name: next((s for s in ('train', 'val')
            if name in create_splits_scenes()[s]), None) for name in names})
    deviations = [d for entry in entries for d in entry['input_audit'].get('loader_translation_difference_m', [])]
    import numpy as np
    audit = dict(status='passed' if all(e['input_audit']['status'] == 'passed' for e in entries) else 'failed',
        frame_count=len(entries), scene_counts={name: sum(e['scene'] == name for e in entries) for name in names},
        padded_frames=[dict(sample_token=e['sample_token'], scene=e['scene'], frame_index=e['frame_index'],
            padded_keyframe_copies=e['input_audit']['padded_keyframe_copies'])
            for e in entries if e['input_audit'].get('padded_keyframe_copies')],
        sweep_count=len(deviations), loader_vs_se3_translation_m=dict(max=max(deviations, default=0),
            median=float(np.median(deviations)) if deviations else None,
            p95=float(np.percentile(deviations, 95)) if deviations else None),
        worst_frame=max((dict(sample_token=e['sample_token'], scene=e['scene'], frame_index=e['frame_index'],
            maximum_m=e['input_audit']['max_loader_vs_se3_translation_m']) for e in entries
            if e['input_audit']['status'] == 'passed'), key=lambda v: v['maximum_m'], default=None),
        frames=[dict(sample_token=e['sample_token'], scene=e['scene'], frame_index=e['frame_index'],
            **e['input_audit']) for e in entries],
        limitation='Official loader unchanged. This audit covers this manifest, not every nuScenes sample.')
    return dict(manifest=manifest, input_audit=audit, nusc=nusc, cfg=cfg, resolved_config=resolved,
        setup_wall_s=time.perf_counter() - start)


class CenterPointSession:
    """One model per invocation; exactly one top-level predict call per new frame."""
    def __init__(self, context):
        start = time.perf_counter()
        import torch
        from mmdet3d.registry import MODELS
        from mmdet3d.utils import register_all_modules
        from mmengine.dataset import Compose
        from tools.infer_centerpoint_mini import initialize_cuda, strict_load_checkpoint
        self.torch, self.context, self.forward_calls = torch, context, 0
        cfg = context['cfg']
        provenance = context['manifest']['prediction_provenance']
        verify_files(provenance)
        register_all_modules(init_default_scope=True)
        self.model = MODELS.build(cfg.model)
        comparison = {}
        strict_load_checkpoint(self.model, Path(provenance['checkpoint']['path']), comparison)
        self.comparison = comparison['checkpoint_comparison']
        self.model.eval()
        cpu_s = time.perf_counter() - start
        start = time.perf_counter()
        self.device = torch.device('cuda:0')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no fallback or baseline changes')
        initialize_cuda(torch, self.device)
        self.model.to(self.device).eval()
        torch.cuda.synchronize(self.device)
        flags = dict(matmul=torch.backends.cuda.matmul.allow_tf32, cudnn=torch.backends.cudnn.allow_tf32)
        if flags != provenance['tf32']:
            raise ValueError('Actual TF32 flags differ from the verified profile')
        self.pipeline = Compose(cfg.test_pipeline)
        self.setup = dict(model_checkpoint_cpu_s=cpu_s, device_setup_s=time.perf_counter() - start,
            checkpoint_comparison=self.comparison, tf32=flags,
            allocator_setup=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device)))

    def run(self, entry, folder):
        import numpy as np
        from mmengine.dataset import pseudo_collate
        torch = self.torch
        context, folder = self.context, Path(folder)
        provenance = context['manifest']['prediction_provenance']
        start, phase = time.perf_counter(), 'input_validation'
        metadata = dict(entry['input_metadata'])
        code = dict(inference_script_sha256=provenance['runtime_sha256'],
            verified_single_frame_script_sha256=provenance['reference_cli_sha256'])
        shared = dict(config=provenance['config'], checkpoint=provenance['checkpoint'], code=code,
            framework=dict(mmdet3d=provenance['packages']['mmdet3d'], torch=provenance['packages']['torch']))
        report = dict(task='detection', status='failed', check_only=False, forward_calls=0,
            forward_completed=False, extra_warmup_forward_calls=0, latency_warmup=self.forward_calls == 0,
            precision='float32', autocast=False, tf32=provenance['tf32'], timings_s={}, vram={},
            sample_token=entry['sample_token'], scene_name=entry['scene'], frame_index=entry['frame_index'],
            checkpoint_comparison=self.comparison, **shared,
            metrics=dict(mAP=None, NDS=None, status='pending_dataset_evaluation',
                reason='Single-frame artifacts remain separate from the set-level benchmark evaluation.'))
        packed = batch = processed = output = prediction = boxes_3d = points = None
        try:
            t = time.perf_counter()
            inputs, base, _, input_fp = audit_frame(context['nusc'], entry, provenance['pipeline_fingerprint'])
            if input_fp != entry['input_fingerprint']:
                raise ValueError('Input changed after the manifest audit')
            metadata = dict(base, **shared, bounds=provenance['bounds'], voxel_size=provenance['voxel_size'],
                class_names=provenance['class_names'], box_convention=dict(center='geometric',
                    dimensions=['dx', 'dy', 'dz'], yaw='radians, CCW about +Z, zero along +X',
                    velocity=['vx', 'vy'], velocity_unit='m/s'),
                test_pipeline=provenance['test_pipeline'], result_source='prediction',
                model_source='CenterPoint PointPillars 0.2m circle NMS / LiDAR-only',
                metrics=report['metrics'], benchmark_provenance=dict(prediction_fingerprint=fingerprint(provenance),
                    input_fingerprint=input_fp))
            (folder / 'resolved_config.py').write_text(context['resolved_config'])
            report['timings_s']['input_validation'] = time.perf_counter() - t
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            pipeline_start = time.perf_counter()
            phase, t = 'preprocessing', time.perf_counter()
            packed = self.pipeline(inputs)
            batch = pseudo_collate([packed])
            points = packed['inputs']['points']
            if not points.numel() or not torch.isfinite(points).all():
                raise ValueError('Empty/non-finite preprocessed points')
            report['point_count'] = len(points)
            with torch.inference_mode():
                processed = self.model.data_preprocessor(batch, training=False)
                torch.cuda.synchronize(self.device)
                report['timings_s']['preprocessing'] = time.perf_counter() - t
                phase, t = 'forward', time.perf_counter()
                self.forward_calls += 1
                report['forward_calls'] = 1
                print('FORWARD', self.forward_calls, entry['scene'], entry['frame_index'], entry['sample_token'], flush=True)
                output = self.model(**processed, mode='predict')
                torch.cuda.synchronize(self.device)
                report['forward_completed'] = True
                report['timings_s']['forward'] = time.perf_counter() - t
            phase, t = 'postprocessing_export', time.perf_counter()
            prediction = output[0].pred_instances_3d
            boxes_3d = prediction.bboxes_3d
            centers, dims, tensor = (a.cpu().numpy() for a in (boxes_3d.gravity_center, boxes_3d.dims, boxes_3d.tensor))
            scores, labels = prediction.scores_3d.cpu().numpy(), prediction.labels_3d.cpu().numpy()
            if not all(np.isfinite(a).all() for a in (centers, dims, tensor, scores)):
                raise ValueError('Non-finite model output')
            boxes = [dict(center=c.tolist(), dimensions=d.tolist(), yaw=float(row[6]),
                velocity=row[7:9].tolist(), score=float(score), label=int(label))
                for c, d, row, score, label in zip(centers, dims, tensor, scores, labels)]
            doc = dict(schema_version=1, task='detection', sample_token=entry['sample_token'],
                timestamp_us=metadata['timestamp_us'], reference_timestamp_us=metadata['reference_timestamp_us'],
                coordinate_system=metadata['coordinate_system'], class_names=metadata['class_names'],
                box_convention=metadata['box_convention'], boxes=boxes, provenance=shared,
                postprocessing=dict(nms_type='circle',
                    config_score_threshold=self.context['cfg'].model.test_cfg.pts.score_threshold,
                    display_filter_applied=False))
            atomic_json(folder / 'detections.json', doc)
            report['box_count'], report['status'] = len(boxes), 'success'
            report['timings_s']['postprocessing_export'] = time.perf_counter() - t
            report['timings_s']['pipeline_total'] = time.perf_counter() - pipeline_start
        except Exception as error:
            report['error'] = dict(stage=phase, type=type(error).__name__, message=str(error),
                traceback=traceback.format_exc())
            raise
        finally:
            packed = batch = points = processed = output = prediction = boxes_3d = None
            gc.collect()
            try:
                report['vram'] = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
                    scope='Per-frame allocator peak with the single model resident; not physical VRAM')
                torch.cuda.synchronize(self.device)
                report['allocated_after_release_bytes'] = torch.cuda.memory_allocated(self.device)
            except Exception as cleanup_error:
                report['allocator_cleanup_error'] = dict(type=type(cleanup_error).__name__, message=str(cleanup_error))
            report['timings_s']['frame_until_report'] = time.perf_counter() - start
            metadata.update(status=report['status'], schema_version=1, precision='float32', autocast=False)
            atomic_json(folder / 'metadata.json', metadata)
            atomic_json(folder / 'report.json', report)
        return report

    def close(self):
        del self.model
        gc.collect()
        self.torch.cuda.synchronize(self.device)
        self.torch.cuda.empty_cache()


def append_timing(path, row):
    path = Path(path)
    new = not path.exists() or not path.stat().st_size
    if not new:
        with path.open('rb') as stream:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b'\n':
                raise ValueError('Truncated timing journal; preserve and repair explicitly')
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=TIMING_FIELDS, extrasaction='ignore')
        if new:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def timing_statistics(rows):
    import numpy as np
    measured = [r for r in rows if r['source'] == 'new' and str(r.get('forward_completed')).lower() == 'true']
    def distribution(items, key):
        values = [float(r[key]) for r in items if r.get(key) not in (None, '')]
        if any(not np.isfinite(v) or v < 0 for v in values):
            raise ValueError('Invalid measured latency')
        return dict(sample_count=len(values), median_s=float(np.median(values)) if values else None,
            p95_s=float(np.percentile(values, 95)) if values else None)
    steady = [r for r in measured if str(r['warmup']).lower() != 'true']
    return dict(all_completed_forwards=distribution(measured, 'forward_s'),
        excluding_marked_warmup=distribution(steady, 'forward_s'),
        end_to_end_excluding_marked_warmup=distribution(steady, 'end_to_end_s'),
        marked_warmup_count=len(measured) - len(steady),
        scope='New synchronized predict calls only; no cache I/O. NumPy linear p95; seconds.',
        end_to_end_scope='Frame input audit/hash, preprocessing, predict/decode/NMS, CPU export, validation and atomic commit; excludes common setup/full-split audit')


def commit_prediction(pending, entry, provenance, origin, source):
    names = (*BUNDLE_FILES, 'job_result.json') if origin == 'external_smoke' else BUNDLE_FILES
    validate_bundle(pending, entry, provenance, origin)
    commit = dict(status='complete', sample_token=entry['sample_token'], input_fingerprint=entry['input_fingerprint'],
        prediction_fingerprint=fingerprint(provenance), origin=origin, source=source,
        artifacts=file_records(pending, names), completed_at_utc=utc())
    commit['fingerprint_sha256'] = fingerprint(commit)
    atomic_json(pending / 'prediction_commit.json', commit)
    final = pending.with_name(pending.name.removeprefix('.pending-'))
    pending.rename(final)
    return final


def execute(args, context, session_factory=CenterPointSession, evaluation_function=evaluator.evaluate):
    start = time.perf_counter()
    out = Path(args.out_dir).resolve()
    journal = out / 'timings.csv'
    if journal.exists() and journal.stat().st_size:
        with journal.open('rb') as stream:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b'\n':
                raise ValueError('Truncated timing journal; no model will load. Preserve and repair explicitly.')
    candidate = context['manifest']
    if args.resume:
        manifest = read_json(out / 'manifest.json')
        verify_manifest(manifest)
        if manifest['identity_sha256'] != candidate['identity_sha256']:
            raise ValueError('Manifest/provenance/input changed; resume is refused')
        progress = read_json(out / 'progress.json')
    else:
        manifest = candidate
        atomic_json(out / 'manifest.json', manifest)
        atomic_json(out / 'input_audit.json', context['input_audit'])
        (out / 'resolved_config.py').write_text(context['resolved_config'])
        progress = dict(schema_version=1, status='prepared', frames={}, invocations=[])
    context['manifest'] = manifest
    entries = verify_manifest(manifest)
    provenance = manifest['prediction_provenance']
    invocation = dict(id=uuid.uuid4().hex, started_at_utc=utc(), command=[sys.executable, *sys.argv],
        mode='prepare_only' if args.prepare_only else 'evaluate_only' if args.evaluate_only else 'inference',
        setup_cpu_s=context['setup_wall_s'], model_load_count=0, model_initialization_attempts=0, forward_calls=0,
        predictions_new=0, predictions_cached=0, predictions_imported=0, predictions_recovered=0, extra_warmup_forwards=0,
        cache_rejections=[], errors=[], gpu_lock_path=str(gpu_lock_path()))
    progress['invocations'].append(invocation)
    def save():
        progress['updated_at_utc'] = utc()
        atomic_json(out / 'progress.json', progress)
    save()
    runtime, gpu_stack = None, ExitStack()
    if not args.prepare_only:
        try:
            for entry in entries:
                token = entry['sample_token']
                state = progress['frames'].get(token, {})
                source = 'cache'
                frame_start = time.perf_counter()
                if state.get('status') == 'complete':
                    try:
                        committed_prediction(out / state['prediction_directory'], entry, provenance)
                        invocation['predictions_cached'] += 1
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        state.update(status='invalid_cache', error=dict(stage='cache_validation', message=str(error)))
                        invocation['errors'].append(dict(sample_token=token, **state['error']))
                    save()
                    continue
                # Never re-infer a failed or ambiguously interrupted frame on resume.
                if state.get('status') in ('running', 'failed', 'interrupted', 'invalid_cache'):
                    if state['status'] == 'running':
                        # Recover a durable commit after a crash between filesystem publication and progress save.
                        try:
                            pending = (out / state['pending_directory']).resolve()
                            if not pending.is_relative_to(out):
                                raise ValueError('Pending output escaped benchmark root')
                            final = pending.with_name(pending.name.removeprefix('.pending-'))
                            published = final if final.is_dir() else pending
                            value = committed_prediction(published, entry, provenance)
                            if published == pending:
                                pending.rename(final)
                            state.update(status='complete', origin=value['commit']['origin'],
                                prediction_directory=str(final.relative_to(out)), recovered_without_forward=True)
                            invocation['predictions_cached'] += 1
                            invocation['predictions_recovered'] += 1
                            save()
                            continue
                        except (OSError, ValueError, KeyError, TypeError):
                            pass
                        state.update(status='interrupted', error=dict(stage='restart',
                            message='Previous launch intent has no complete publication; automatic retry refused'))
                        save()
                    continue
                if entry['input_audit']['status'] != 'passed':
                    progress['frames'][token] = dict(status='failed', error=entry['input_audit']['error'])
                    save()
                    continue
                imported = None
                if args.smoke_run:
                    path = Path(args.smoke_run).resolve()
                    try:
                        if read_json(path / 'report.json').get('sample_token') == token:
                            validate_bundle(path, entry, provenance, 'external_smoke')
                            pending = out / 'frames' / token / ('.pending-' + uuid.uuid4().hex)
                            pending.mkdir(parents=True)
                            for name in (*BUNDLE_FILES, 'job_result.json'):
                                shutil.copyfile(path / name, pending / name)
                            imported = commit_prediction(pending, entry, provenance, 'external_smoke',
                                dict(kind='external_smoke', directory=str(path), artifacts=file_records(path, (*BUNDLE_FILES, 'job_result.json'))))
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        invocation['cache_rejections'].append(dict(sample_token=token, source=str(path), reason=str(error)))
                if imported is not None:
                    progress['frames'][token] = dict(status='complete', origin='external_smoke',
                        prediction_directory=str(imported.relative_to(out)))
                    invocation['predictions_imported'] += 1
                    source = 'external_smoke_cache'
                    append_timing(out / 'timings.csv', dict(invocation_id=invocation['id'], sample_token=token,
                        scene=entry['scene'], frame_index=entry['frame_index'], source=source, status='complete',
                        prediction_directory=str(imported.relative_to(out))))
                    save()
                    continue
                if args.evaluate_only:
                    continue
                if context['input_audit']['status'] != 'passed':
                    raise ValueError('Full manifest input audit must pass before loading the model')
                if runtime is None:
                    gpu_stack.enter_context(exclusive_lock(gpu_lock_path()))
                    invocation['model_initialization_attempts'] += 1
                    save()
                    runtime = session_factory(context)
                    invocation['model_load_count'] += 1
                    invocation['model_initialization'] = runtime.setup
                frame_start = time.perf_counter()  # Common model setup is not a frame latency.
                pending = out / 'frames' / token / ('.pending-' + uuid.uuid4().hex)
                pending.mkdir(parents=True)
                state = dict(status='running', invocation_id=invocation['id'],
                    launch_intent_at_utc=utc(), pending_directory=str(pending.relative_to(out)))
                progress['frames'][token] = state
                save()  # Durable BEFORE the model can see this frame.
                error = None
                try:
                    runtime.run(entry, pending)
                    final = commit_prediction(pending, entry, provenance, 'benchmark', dict(invocation_id=invocation['id']))
                    state.update(status='complete', origin='benchmark', prediction_directory=str(final.relative_to(out)))
                    invocation['predictions_new'] += 1
                except Exception as caught:
                    error = caught
                    state.update(status='failed', error=dict(stage='inference_or_publication', type=type(caught).__name__, message=str(caught)))
                    invocation['errors'].append(dict(sample_token=token, **state['error']))
                invocation['forward_calls'] = runtime.forward_calls
                report_path = (out / state['prediction_directory'] if state['status'] == 'complete' else pending) / 'report.json'
                report = read_json(report_path) if report_path.exists() else {}
                if report.get('error'):
                    state['error'] = report['error']
                    if invocation['errors']:
                        invocation['errors'][-1].update(report['error'])
                times = report.get('timings_s', {})
                append_timing(out / 'timings.csv', dict(invocation_id=invocation['id'], sample_token=token,
                    scene=entry['scene'], frame_index=entry['frame_index'], source='new', status=state['status'],
                    warmup=report.get('latency_warmup'), forward_completed=report.get('forward_completed', False),
                    input_validation_s=times.get('input_validation'), preprocessing_s=times.get('preprocessing'),
                    forward_s=times.get('forward'), postprocessing_export_s=times.get('postprocessing_export'),
                    end_to_end_s=time.perf_counter() - frame_start,
                    **report.get('vram', {}), allocated_after_release_bytes=report.get('allocated_after_release_bytes'),
                    prediction_directory=state.get('prediction_directory')))
                save()
                if error is not None:
                    break  # No retry and no further model calls after a failed frame/OOM/export.
        except Exception as error:
            invocation['errors'].append(dict(stage='invocation', type=type(error).__name__, message=str(error),
                traceback=traceback.format_exc()))
        finally:
            if runtime is not None:
                try:
                    runtime.close()
                except Exception as error:
                    invocation['errors'].append(dict(stage='model_cleanup', type=type(error).__name__, message=str(error)))
            gpu_stack.close()
    evaluation = None
    if not args.prepare_only:
        try:
            evaluation = evaluation_function(out, manifest, progress, nusc=context['nusc'])
        except Exception as error:
            invocation['errors'].append(dict(stage='evaluation_cpu', type=type(error).__name__, message=str(error),
                traceback=traceback.format_exc()))
    predictions, missing = valid_predictions(out, manifest, progress)
    invocation.update(finished_at_utc=utc(), wall_s=time.perf_counter() - start,
        evaluation_source=evaluation['evaluation_source'] if evaluation else None)
    progress['status'] = ('prepared' if args.prepare_only and context['input_audit']['status'] == 'passed' else
        'preparation_failed' if args.prepare_only else
        'complete' if not missing and evaluation and evaluation['status'] == 'complete' else 'partial')
    save()
    rows = []
    if (out / 'timings.csv').is_file():
        with (out / 'timings.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
    gpu_reports = [str(Path('reports') / (i['id'] + '.json')) for i in progress['invocations'] if i['forward_calls']]
    measured = [r for r in rows if r['source'] == 'new']
    report = dict(schema_version=1, status=progress['status'], split='mini_val', task='detection',
        scope='Entire official mini_val manifest; not full nuScenes validation',
        expected_frames=len(entries), successful_frames=len(predictions), missing_frames=missing,
        failed_frames=sum(s.get('status') in ('failed', 'interrupted', 'invalid_cache') for s in progress['frames'].values()),
        scene_counts={name: sum(e['scene'] == name and e['sample_token'] in predictions for e in entries)
            for name in manifest['identity']['scene_names']},
        current_invocation=invocation, invocations=progress['invocations'], original_gpu_reports=gpu_reports,
        manifest_identity_sha256=manifest['identity_sha256'], manifest_file=file_record(out / 'manifest.json'),
        input_audit_summary={k: v for k, v in context['input_audit'].items() if k != 'frames'},
        prediction_provenance=provenance, evaluation_protocol=manifest['evaluation_protocol'],
        metrics=evaluation['metrics'] if evaluation else None,
        evaluation_file=file_record(out / 'evaluation.json') if (out / 'evaluation.json').exists() else None,
        latency_all_invocations=timing_statistics(rows),
        latency_current_invocation=timing_statistics([r for r in rows if r['invocation_id'] == invocation['id']]),
        allocator=dict(peak_allocated_bytes=max((int(r['peak_allocated_bytes']) for r in measured if r.get('peak_allocated_bytes')), default=None),
            peak_reserved_bytes=max((int(r['peak_reserved_bytes']) for r in measured if r.get('peak_reserved_bytes')), default=None),
            scope='Maximum per-frame PyTorch allocator peak, one model resident; not physical VRAM'),
        warmup_policy='First new manifest forward of each invocation marked for latency exclusion; no extra forwards; accuracy includes it',
        complete_only_if='All manifest predictions hash-valid AND official evaluator complete',
        review_scope='Collection-level detection result; no single-frame approval extended or created')
    atomic_json(out / 'reports' / (invocation['id'] + '.json'), report)
    atomic_json(out / 'benchmark_report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--profiles', default=str(ROOT / 'web/centerpoint_profiles.json'))
    parser.add_argument('--profile', default='centerpoint_pillar02_circle')
    parser.add_argument('--smoke-run', help='Optional original real smoke bundle; validates all inputs/provenance/receipt hashes')
    parser.add_argument('--resume', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args(argv)
    out = Path(args.out_dir).resolve()
    if args.evaluate_only and not args.resume:
        parser.error('--evaluate-only requires --resume')
    if args.resume and not (out / 'manifest.json').is_file():
        parser.error('--resume requires an existing detection benchmark manifest')
    if not args.resume and out.exists() and any(out.iterdir()):
        parser.error('Output is not empty; choose a fresh directory or --resume')
    with exclusive_lock(out / '.benchmark.lock'):
        context = create_context(args)
        result = execute(args, context)
    print(json.dumps(dict(status=result['status'], expected=result['expected_frames'],
        successful=result['successful_frames'], current_invocation=result['current_invocation'],
        metrics=result['metrics']), indent=2, ensure_ascii=False))
    return 0 if result['status'] in ('prepared', 'complete') else 2


if __name__ == '__main__':
    raise SystemExit(main())
