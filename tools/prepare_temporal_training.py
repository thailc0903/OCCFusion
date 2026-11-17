#!/usr/bin/env python3
"""Prepare missing mini_train GT and audit temporal training readiness on CPU.

No model/checkpoint bytes, inference, worker, queue or training execution.
Only --install-missing-gt may add verified missing GT; never overwrites GT.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.forecast_persistence import CPUOnlyImports, FORBIDDEN


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir',type=Path,default=ROOT/'data/downloads/surroundocc')
    parser.add_argument('--previous-root',type=Path,default=ROOT/'work/forecasting/persistence_v1')
    parser.add_argument('--data-root',type=Path,default=ROOT/'data/nuscenes')
    parser.add_argument('--gt-root',type=Path,default=ROOT/'data/nuscenes/occ_samples/nuscenes_occ/samples')
    parser.add_argument('--runs-root',type=Path,default=ROOT/'work_dirs')
    parser.add_argument('--out-dir',type=Path,default=ROOT/'work/forecasting/training_readiness_v1')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--inspect-only',action='store_true')
    mode.add_argument('--install-missing-gt',action='store_true')
    mode.add_argument('--resume-audit',action='store_true',help='Continue only sealed completed GT phases; no re-extraction/overwrite')
    return parser.parse_args(argv)


def run(args,out,started):
    import numpy as np
    from tools import evaluate_occupancy as occupancy
    from tools.forecasting import dataset as dataset_module
    from tools.forecasting.path_snapshot import DatasetPaths
    from tools import infer_nuscenes_mini as adapter
    from tools.forecasting.gt_archive import inspect_archives,stage_missing,publish_staged
    from tools.forecasting.protocol import file_record,fingerprint,write_json,protocol
    from tools.forecasting.training_readiness import (
        requested_train_gt,audit_all_gt,raw_inventory,sensor_config,cache_inventory,
        frame_summary,build_manifests,cache_plan,
        make_source,
    )
    read=lambda path:occupancy.json_object(Path(path).read_bytes())
    previous=args.previous_root.resolve()
    old_gt=read(previous/'gt_audit.json');old_inputs=read(previous/'input_audit.json')
    old_manifest=read(previous/'sequence_manifest.json');old_spec=read(previous/'protocol.json')
    spec=protocol();spec_fingerprint=fingerprint(spec)
    if spec!=old_spec or spec_fingerprint!=old_manifest['protocol_fingerprint']:
        raise ValueError('Forecasting protocol v1 changed; readiness must not silently change protocol')
    resume=None
    if args.resume_audit:
        resume=read(out/'resume_state.json')
        for name,expected in resume['completed_files'].items():
            if file_record(out/name)!=expected:raise ValueError('Completed phase changed before resume: '+name)
        if read(out/'protocol.json')!=spec:raise ValueError('Resume protocol mismatch')
    else:write_json(out/'protocol.json',spec)
    source=make_source(args.data_root,old_manifest['source_benchmark']['root'],args.gt_root)
    missing=requested_train_gt(old_gt,source.catalog)
    paths=sorted(args.archive_dir.resolve().glob('*.zip'))
    archive_stats={str(p.resolve()):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths}
    if args.resume_audit:
        inspection=read(out/'archive_inspection.json')
        for archive in inspection['archives']:
            if file_record(archive['path'])!=archive['file']:raise ValueError('Archive hash changed before resume')
        installation=read(out/'gt_installation.json')
        if installation['status']!='complete':raise ValueError('Cannot resume incomplete GT publication')
        for item in installation['files']:
            if file_record(item['destination'])!=item['file']:raise ValueError('Imported GT changed before resume')
        print('Resume sealed archive/GT phases: no extraction or publication repeated.',flush=True)
    else:
        print('Inspect/hash ZIP central directories; no payload extraction yet...',flush=True)
        index,archives=inspect_archives(paths)
        found=[r for r in missing if r['expected_gt_filename'] in index]
        inspection=dict(status='inspected',archives=archives,requested_mini_train_missing=len(missing),
            matching_requested_count=len(found),not_found=[r for r in missing if r['expected_gt_filename'] not in index],
            requested_uncompressed_bytes=sum(index[r['expected_gt_filename']][0]['uncompressed_bytes'] for r in found),
            existing_gt_basename_overlap=len(set(index)&set(source.gt_index.files)),
            source_policy=occupancy.GT_SOURCE,publisher_checksum_supplied=False,
            provenance_limit='Archive name/local SHA and decoder are evidence; no publisher-signed checksum or proof of original GT generation geometry.')
        write_json(out/'archive_inspection.json',inspection)
    if args.inspect_only:
        print(json.dumps(inspection,ensure_ascii=False),flush=True);return 0
    if inspection['not_found']:
        raise ValueError('Some requested missing mini_train labels are absent from the archive')
    if args.install_missing_gt:
        required=2*inspection['requested_uncompressed_bytes']+64*1024*1024
        if shutil.disk_usage(args.gt_root).free<required:
            raise OSError('Insufficient free space for staging + independent GT copies')
        print(f'Staging/decoding only {len(missing)} requested GT files...',flush=True)
        staged=stage_missing(index,missing,out/'staging/gt',args.gt_root,existing_index=source.gt_index)
        write_json(out/'staging_manifest.json',staged)
        if any((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=s for p,s in archive_stats.items()):
            raise ValueError('Archive modified during inspection/staging; no publication')
        installed=publish_staged(staged,args.gt_root,out/'gt_install_receipts')
        write_json(out/'gt_installation.json',installed)
        print(f"GT installed={installed['installed']}, identical existing kept={installed['identical_existing_kept']}",flush=True)
    elif not args.resume_audit:
        write_json(out/'gt_installation.json',dict(status='not_requested',installed=0,files=[]))
    print('Audit official GT coverage/content (CPU)...',flush=True)
    if args.resume_audit:
        gt_report=read(out/'gt_audit.json');gt={f['sample_token']:f for f in gt_report['frames']}
        if set(gt)!=set(source.catalog.samples):raise ValueError('GT resume token coverage mismatch')
        gt_root=args.gt_root.resolve()
        for token,value in gt.items():
            item=source.catalog.samples[token]
            if value['split']!=item['split'] or value['expected_gt_filename']!=Path(item['lidar']['filename']).name+'.npy':raise ValueError('GT resume token/split/filename mismatch')
            if value['status']=='valid_gt' and (not Path(value['gt_path']).is_relative_to(gt_root) or file_record(value['gt_path'])!=value['file']):raise ValueError('Previously decoded GT bytes changed')
    else:gt=audit_all_gt(source.catalog,args.gt_root,old_gt)
    gt_report=dict(kind='temporal_training_gt_audit',protocol=occupancy.protocol(),root=str(args.gt_root.resolve()),
        split_source='nuscenes.utils.splits.create_splits_scenes',frames=list(gt.values()),scenes={})
    for name in sorted({r['scene'] for r in gt.values()}):
        group=[r for r in gt.values() if r['scene']==name]
        gt_report['scenes'][name]=dict(split=group[0]['split'],expected=len(group),status_counts=dict(Counter(r['status'] for r in group)),
            missing_or_invalid=[r for r in group if r['status']!='valid_gt'])
    if not args.resume_audit:write_json(out/'gt_audit.json',gt_report)
    preferred=next(f['source_dir'] for f in old_inputs['frames'] if f['status']=='valid_prediction')
    cfg=sensor_config(Path(preferred)/'resolved_config.py',source.shared['resolved_config_sha256'])
    print('Audit all raw camera/LiDAR/radar metadata using the original CPU adapter...',flush=True)
    path_snapshot=DatasetPaths(source.catalog.root)
    with path_snapshot.use(source.catalog,adapter,dataset_module):
        raw=raw_inventory(source.catalog,cfg,path_snapshot)
    write_json(out/'raw_sensor_audit.json',dict(kind='temporal_training_raw_sensor_audit',frames=list(raw.values())))
    print('Validate all saved OCCFusion bundles; separate strict and legacy provenance...',flush=True)
    with path_snapshot.use(source.catalog,adapter,dataset_module):
        caches=cache_inventory(source,args.runs_root,old_inputs)
    write_json(out/'prediction_cache_inventory.json',caches)
    frames=frame_summary(source.catalog,raw,gt,caches)
    write_json(out/'frame_inventory.json',dict(frames=frames))
    manifests=build_manifests(source.catalog,frames,raw,gt,caches['selected'],spec_fingerprint)
    write_json(out/'train_manifest.json',manifests['mini_train'])
    write_json(out/'validation_manifest.json',manifests['mini_val'])
    old_valid={w['window_id'] for w in old_manifest['windows'] if w['status']=='valid'}
    new_valid={w['window_id'] for w in manifests['mini_val']['windows'] if w['status']=='ready'}
    if new_valid!=old_valid:
        raise ValueError('Validation v1 window membership drift: '+repr(dict(added=sorted(new_valid-old_valid),removed=sorted(old_valid-new_valid))))
    plan=cache_plan(frames,raw,manifests,caches)
    plan['baseline_identity']=dict(checkpoint=source.shared['checkpoint'],config_source_sha256=source.shared['config_source_sha256'],
        resolved_config_sha256=source.shared['resolved_config_sha256'],shared_fingerprint=source.shared_fingerprint,
        precision=source.shared['precision'],tf32=source.shared['tf32'],input_profile=str(ROOT/'web/inference_profiles.json'))
    write_json(out/'cache_generation_plan.json',plan)
    splits={}
    for split in ('mini_train','mini_val'):
        group=[f for f in frames if f['split']==split]
        splits[split]=dict(expected_frames=len(group),raw_keyframes_present=sum(f['raw_keyframes_present'] for f in group),
            metadata_adapter_valid_frames=sum(f['adapter_valid'] for f in group),gt_valid_frames=sum(f['gt_status']=='valid_gt' for f in group),
            valid_v1_prediction_cache_frames=sum(f['valid_v1_cache'] for f in group),legacy_valid_output_frames=sum(bool(f['legacy_valid_output_folders']) for f in group),
            candidate_windows=manifests[split]['candidate_windows'],ready_windows=manifests[split]['ready_windows'],
            potential_windows_after_cache=manifests[split]['potential_windows_after_cache'])
    scene_summary={}
    for scene in sorted({f['scene'] for f in frames}):
        group=[f for f in frames if f['scene']==scene];split=group[0]['split']
        windows=[w for w in manifests[split]['windows'] if w['scene']==scene]
        scene_summary[scene]=dict(split=split,frames=len(group),raw_keyframes_present=sum(f['raw_keyframes_present'] for f in group),
            adapter_valid=sum(f['adapter_valid'] for f in group),gt_valid=sum(f['gt_status']=='valid_gt' for f in group),
            valid_v1_cache=sum(f['valid_v1_cache'] for f in group),legacy_output=sum(bool(f['legacy_valid_output_folders']) for f in group),
            ready_windows=sum(w['status']=='ready' for w in windows),potential_after_cache=sum(w['potential_after_verified_cache_generation'] for w in windows))
    checkpoint_overlap={split:{scene:[name for name in ('train','val','test') if scene in source.catalog.splits.get(name,[])] for scene in source.catalog.splits[split]} for split in ('mini_train','mini_val')}
    bad_gt=splits['mini_train']['gt_valid_frames']!=splits['mini_train']['expected_frames']
    blockers=[]
    if bad_gt:blockers.append('missing_or_invalid_train_gt')
    if not splits['mini_train']['ready_windows']:blockers.append('no_train_window_with_three_valid_prediction_caches')
    loaded=[m for m in sys.modules if m.split('.')[0] in FORBIDDEN]
    if loaded:raise RuntimeError('CPU-only import contract violated: '+repr(loaded))
    files=[ROOT/'tools/prepare_temporal_training.py',ROOT/'tools/forecasting/gt_archive.py',ROOT/'tools/forecasting/training_readiness.py',ROOT/'tools/forecasting/path_snapshot.py']
    report=dict(kind='temporal_training_readiness_v1',status='blocked' if blockers else 'ready',audit_complete=True,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),protocol_fingerprint=spec_fingerprint,blockers=blockers,splits=splits,scenes=scene_summary,
        validation=dict(evaluation_only=True,included_in_training=False,previous_ready_windows=len(old_valid),unchanged_window_ids=new_valid==old_valid),
        official_full_split_membership=checkpoint_overlap,checkpoint_training_split_independently_verified=False,
        checkpoint_limit='Declared checkpoint identity retained without reading weights. Original training sample provenance is unavailable; do not claim a proven independent held-out evaluation or train temporal model on mini_val.',
        cache_plan_summary={k:v for k,v in plan.items() if k not in ('frames',)},
        provenance=dict(previous_artifacts={n:file_record(previous/n) for n in ('protocol.json','gt_audit.json','input_audit.json','sequence_manifest.json')},
            source_code={str(p.relative_to(ROOT)):file_record(p) for p in files},dataset_tables=source.dataset_tables,
            source_benchmark=old_manifest['source_benchmark']),
        execution=dict(device='cpu',python=sys.executable,numpy=np.__version__,inference_launches=0,model_loads=0,checkpoint_loads=0,
            checkpoint_bytes_read=False,model_forward_calls=0,GPU_used=False,training=False,worker_started=False,job_submission=False,
            forbidden_modules_loaded=loaded,wall_s=time.perf_counter()-started),
        filesystem_snapshot=dict(unique_paths_checked=len(path_snapshot.files),parent_directories_checked=len(path_snapshot.directories),
            stable_at_exit=True,method='Cache first actual lstat/stat/canonical-parent facts; verify all file and directory identities/sizes/mtimes again on exit; original metadata adapter outputs unchanged'),
        resumed_completed_GT_phases=args.resume_audit,
        limitations=['Raw sensor audit checks file existence/nonempty, timestamps and metadata adapter, not full image/point decoding or future forward success.',
                     'Existing saved predictions contain labels/max-softmax only, no full logits/features. Temporal architecture input requirements remain to be specified.',
                     'Training GT archive publisher authenticity/physical geometry not independently established by local SHA.',
                     'Potential windows require future verified cache generation; they are not ready training windows.'])
    write_json(out/'readiness_report.json',report)
    print(json.dumps(dict(status=report['status'],splits=splits,blockers=blockers),ensure_ascii=False),flush=True)
    return 2 if blockers else 0


def main(argv=None):
    args=parse_args(argv);out=args.out_dir.resolve()
    if not out.is_relative_to(ROOT/'work/forecasting') or out==ROOT/'work/forecasting':
        raise SystemExit('--out-dir must be a fresh subdirectory of work/forecasting')
    if args.resume_audit:
        if not (out/'resume_state.json').is_file() or (out/'readiness_report.json').exists():raise SystemExit('Resume requires interrupted audit with sealed GT phases, no final report')
    elif out.exists() and any(out.iterdir()):raise SystemExit('Output already exists; never overwrite readiness or persistence artifacts')
    if args.gt_root.resolve()!=ROOT/'data/nuscenes/occ_samples/nuscenes_occ/samples':
        raise SystemExit('Use the existing canonical GT destination; no alternate dataset/alias for this import')
    out.mkdir(parents=True,exist_ok=True)
    guard=CPUOnlyImports();sys.meta_path.insert(0,guard)
    try:return run(args,out,time.perf_counter())
    except Exception as error:
        from tools.forecasting.protocol import write_json
        write_json(out/'failure.json',dict(status='failed',error=f'{type(error).__name__}: {error}',
            note='No retry. Any completed GT receipts describe valid new files; existing data is never overwritten.',model_loaded=False,GPU_used=False))
        raise
    finally:sys.meta_path.remove(guard)


if __name__=='__main__':raise SystemExit(main())
