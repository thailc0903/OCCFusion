#!/usr/bin/env python3
"""CPU-only saved OCCFusion sequences and persistence forecasting v1.

No inference/worker/web/database/checkpoint imports or invocation. Output belongs
under work/forecasting, outside the saved-run viewer's configured work_dirs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.abc
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FORBIDDEN = {'torch', 'torchvision', 'mmcv', 'mmengine', 'mmdet', 'mmdet3d', 'occfusion'}


class CPUOnlyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise ImportError('Forecasting CPU contract forbids model/framework import: ' + fullname)
        return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/nuscenes')
    parser.add_argument('--benchmark-root', type=Path, default=ROOT / 'work_dirs/benchmark_mini_val')
    parser.add_argument('--gt-root', type=Path, default=ROOT / 'data/nuscenes/occ_samples/nuscenes_occ/samples')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'work/forecasting/persistence_v1')
    parser.add_argument('--pilot', action='store_true', help='Only the first valid window in deterministic manifest order; still audit all membership.')
    parser.add_argument('--audit-only', action='store_true', help='Only create protocol/manifest/input/GT audit, no forecast metrics.')
    return parser.parse_args(argv)


def save_npz(path, **arrays):
    import numpy as np
    pending = path.with_name(path.name + '.pending')
    try:
        with pending.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        path.hardlink_to(pending)
    finally:
        pending.unlink(missing_ok=True)


def example_plot(path, forecast, targets, palette):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap

    def bev(labels):
        occupied = (labels != 0) & (labels != 255)
        last = np.where(occupied, np.arange(labels.shape[2]), -1).max(axis=2)
        xy = np.take_along_axis(labels, np.maximum(last, 0)[..., None], axis=2)[..., 0]
        xy[last < 0] = 0
        return np.ma.array(xy.T, mask=~(labels != 255).any(axis=2).T)

    colors = np.asarray(palette, dtype=np.float64)[:17] / 255
    cmap = ListedColormap(colors); cmap.set_bad('#77818a')
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True)
    for h, target in enumerate(targets):
        for j, data in enumerate((forecast[h], target.labels)):
            axes[h, j].imshow(bev(data), origin='lower', extent=(-50, 50, -50, 50), cmap=cmap,
                              vmin=-.5, vmax=16.5, interpolation='nearest')
            axes[h, j].set_title(('Persistence forecast' if j == 0 else 'Offline future GT target') + f' +{(h + 1)*.5:.1f}s')
            axes[h, j].set_xlabel('anchor LiDAR x (m)'); axes[h, j].set_ylabel('anchor LiDAR y (m)')
    fig.suptitle('Saved prediction persistence; no learned temporal model. Highest occupied BEV; gray=unknown/OOB. Not a visibility map.')
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run(args, output, started):
    import numpy as np
    from tools import evaluate_occupancy as occupancy
    from tools.forecasting.dataset import SequenceDataset
    from tools.forecasting import evaluation
    from tools.forecasting.persistence import predict
    from tools.forecasting.protocol import HORIZONS_US, protocol, fingerprint, write_json, file_record
    from tools.forecasting.validation import check_window, future_gt_independence

    spec = protocol()
    write_json(output / 'protocol.json', spec)
    spec_hash = fingerprint(spec)
    dataset = SequenceDataset(args.data_root, args.benchmark_root, args.gt_root)
    print('Auditing saved predictions and all actually used sensors (CPU)...', flush=True)
    inputs = dataset.audit_predictions()
    write_json(output / 'input_audit.json', inputs)
    print('Auditing official mini_train/mini_val GT (CPU)...', flush=True)
    gt = dataset.audit_gt()
    write_json(output / 'gt_audit.json', gt)
    manifest = dataset.sequence_manifest(spec_hash)
    write_json(output / 'sequence_manifest.json', manifest)
    selected = [w for w in manifest['windows'] if w['status'] == 'valid']
    if args.pilot:
        selected = selected[:1]
    if args.audit_only:
        selected = []
    print(f"Windows: {manifest['valid_window_count']}/{manifest['candidate_window_count']} valid; evaluating {len(selected)}", flush=True)
    sums = {h: np.zeros((17, 17), dtype=np.int64) for h in HORIZONS_US}
    counts, voxel_counts, geometries, per_window = Counter(), Counter(), [], []
    examples = set()
    for scene in sorted({w['scene'] for w in selected}):
        examples.add(next(w['window_id'] for w in selected if w['scene'] == scene))
    if selected:
        examples.add(selected[-1]['window_id'])
    horizon_errors = defaultdict(list)
    for i, window in enumerate(selected):
        folder = output / 'windows' / window['window_id']
        folder.mkdir(parents=True)
        begin = time.perf_counter()
        history, history_masks = dataset.load_history(window)
        # Forecast is materialized BEFORE any offline future target/pose access.
        forecast = predict(history)
        save_npz(folder / 'forecast.npz', labels=forecast, valid=np.repeat(history.valid[-1][None], len(HORIZONS_US), axis=0))
        forecast_hash = file_record(folder / 'forecast.npz')
        independence = future_gt_independence(dataset, window, history, forecast) if window['window_id'] in examples else None
        targets, target_transforms = dataset.load_targets(window)
        metrics, masks = [], []
        for h, target, labels in zip(HORIZONS_US, targets, forecast):
            matrix = evaluation.statistics(labels, target.labels, target.in_bounds, target.known)
            summary = occupancy.summarize(matrix)
            reference = occupancy.compare_reference([target.labels], [labels], summary)
            sums[h] += matrix; counts[h] += 1; voxel_counts[h] += summary['valid_voxel_count']
            metrics.append(dict(horizon_us=h, metrics=summary, reference_check=reference))
            masks.append(dict(horizon_us=h, in_bounds=int(target.in_bounds.sum()), ignore_in_bounds=int((target.in_bounds & ~target.known).sum()),
                              out_of_bounds=int((~target.in_bounds).sum()), valid=int(target.valid.sum())))
        for timing in window['timing']:
            horizon_errors[(timing['role'], timing['nominal_offset_us'])].append(timing['error_us'])
        geometry = check_window(dataset, window)
        geometries.append(geometry)
        if window['window_id'] in examples:
            save_npz(folder / 'aligned_history.npz', labels=history.labels, valid=history.valid,
                     in_bounds=history_masks['in_bounds'], known=history_masks['known'], anchor_from_history=history_masks['anchor_from_history'])
            save_npz(folder / 'offline_targets.npz', labels=np.stack([t.labels for t in targets]),
                     in_bounds=np.stack([t.in_bounds for t in targets]), known=np.stack([t.known for t in targets]),
                     valid=np.stack([t.valid for t in targets]), anchor_from_future=target_transforms)
            example_plot(folder / 'forecast_vs_offline_target.png', forecast, targets, dataset.metadata[window['anchor']['sample_token']]['palette'])
        record = dict(kind='persistence_forecast_window', protocol_fingerprint=spec_hash, window=window,
                      forecast_file=forecast_hash, prediction_inputs='aligned saved history only; persistence copies last/anchor labels',
                      future_inputs_used=False, target_mask_statistics=masks, metrics_by_horizon=metrics,
                      future_gt_perturbation_check=independence,
                      history_valid_voxel_counts=[int(x.sum()) for x in history.valid],
                      geometry_check=geometry, example=window['window_id'] in examples,
                      cpu_window_pipeline_s=time.perf_counter() - begin)
        write_json(folder / 'window.json', record)
        write_json(folder / 'commit.json', dict(kind='forecasting_window_commit', status='success', protocol_fingerprint=spec_hash,
                   files={p.name: file_record(p) for p in folder.iterdir() if p.is_file()}))
        per_window.append(dict(window_id=window['window_id'], scene=window['scene'], frame_index=window['anchor']['frame_index'],
                               forecast=forecast_hash, target_mask_statistics=masks,
                               metrics_by_horizon=metrics, output=str(folder)))
        print(f"[{i+1}/{len(selected)}] {window['scene']}/{window['anchor']['frame_index']} token={window['window_id']} valid targets={[m['valid'] for m in masks]}", flush=True)
        del history, forecast, targets
    by_horizon = [dict(horizon_us=h, nominal_seconds=h/1e6, window_count=counts[h], valid_voxel_count=voxel_counts[h],
                       metrics=occupancy.summarize(sums[h]) if counts[h] else None,
                       scope='pooled integer confusion counts across evaluated valid windows only') for h in HORIZONS_US]
    metrics = dict(kind='occupancy_forecasting_evaluation', protocol_fingerprint=spec_hash,
                   metric_unit='fraction [0,1]', scope='pilot_first_valid_window' if args.pilot else 'all_valid_mini_val_windows',
                   candidate_windows=manifest['candidate_window_count'], valid_windows=manifest['valid_window_count'],
                   evaluated_windows=len(selected), horizons=by_horizon, windows=per_window,
                   flow=None, EPE=None, ADE=None, FDE=None, flow_status='not_implemented')
    write_json(output / 'metrics_by_horizon.json', metrics)
    write_json(output / 'geometry_validation.json', dict(status='passed' if geometries else 'not_evaluated',
               precision='float64', max_error_m=max((x['max_error_m'] for x in geometries), default=None),
               roundtrip_max_error_m=max((x['roundtrip_max_error_m'] for x in geometries), default=None),
               windows=geometries, quantization_note='Semantic source cell lookup is quantized at 0.5m; the float64 transform error above is not a quantization error or a flow estimate.'))
    sources = ['tools/forecast_persistence.py', 'tools/evaluate_occupancy.py', *[str(p.relative_to(ROOT)) for p in sorted((ROOT / 'tools/forecasting').glob('*.py'))]]
    loaded = sorted(name for name in sys.modules if name.split('.')[0] in FORBIDDEN)
    if loaded:
        raise RuntimeError('CPU import contract violated: ' + repr(loaded))
    timing = [dict(role=role, nominal_offset_us=offset, count=len(values), min_error_us=min(values), max_error_us=max(values),
                   max_abs_error_us=max(map(abs, values))) for (role, offset), values in sorted(horizon_errors.items())]
    report = dict(kind='rav25_persistence_forecasting_report', status='audit_only' if args.audit_only else 'pilot_complete' if args.pilot and selected else 'complete_for_valid_windows' if selected else 'no_valid_windows',
                  completed_at_utc=datetime.now(timezone.utc).isoformat(), protocol_fingerprint=spec_hash,
                  candidate_windows=manifest['candidate_window_count'], valid_windows=manifest['valid_window_count'], evaluated_windows=len(selected),
                  excluded_windows=manifest['excluded_window_count'], primary_exclusion_counts=manifest['primary_exclusion_counts'],
                  any_exclusion_counts=manifest['any_exclusion_counts'], scene_counts=manifest['scene_counts'],
                  selected_window_ids=[w['window_id'] for w in selected], timing_deviations=timing,
                  execution=dict(device='cpu', executable=sys.executable, numpy=np.__version__, nuscenes_devkit=version('nuscenes-devkit'),
                                 forbidden_modules_loaded=loaded, inference_launches=0, model_loads=0, checkpoint_loads=0,
                                 forward_calls=0, GPU_used=False, worker_started=False, training=False, wall_s=time.perf_counter()-started),
                  provenance=dict(source_benchmark=manifest['source_benchmark'], dataset_tables=dataset.dataset_tables,
                                  declared_checkpoint_identity_only=dataset.shared['checkpoint'], checkpoint_bytes_read=False,
                                  source_code={p:file_record(ROOT / p) for p in sources},
                                  config_identity=dict(path=dataset.shared['config_path'], sha256=dataset.shared['config_source_sha256'],resolved_sha256=dataset.shared['resolved_config_sha256'])),
                  production_modified=False, old_artifacts_or_approvals_modified=False,
                  unimplemented=dict(learned_temporal_model=True, flow=True, EPE=True, ADE=True, FDE=True),
                  outputs={p.name:file_record(p) for p in output.iterdir() if p.is_file()},
                  note='Forecasting outputs are separate and unapproved; not a viewer run, full-validation result or learned 4D flow.')
    write_json(output / 'forecasting_report.json', report)
    print(json.dumps({k:report[k] for k in ('status', 'candidate_windows', 'valid_windows', 'excluded_windows', 'evaluated_windows', 'primary_exclusion_counts')}, ensure_ascii=False), flush=True)
    return 0 if selected or args.audit_only else 2


def main(argv=None):
    args = parse_args(argv)
    output = args.out_dir.expanduser().resolve()
    # Restrict outputs to this separate namespace; no accidental publication into work_dirs/state/data.
    if not output.is_relative_to(ROOT / 'work/forecasting') or output == ROOT / 'work/forecasting':
        raise SystemExit('--out-dir must be a fresh subdirectory of work/forecasting')
    if output.exists() and any(output.iterdir()):
        raise SystemExit('Output is not empty; use a new forecasting directory. No overwrite/resume inference.')
    output.mkdir(parents=True, exist_ok=True)
    guard = CPUOnlyImports(); sys.meta_path.insert(0, guard)
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/occfusion-matplotlib')
    started = time.perf_counter()
    try:
        return run(args, output, started)
    except Exception as error:
        # An incomplete output never gets a successful report or commit.
        from tools.forecasting.protocol import write_json
        write_json(output / 'failure.json', dict(status='failed', error=f'{type(error).__name__}: {error}',
                   model_loaded=False, GPU_used=False, no_retry=True))
        raise
    finally:
        sys.meta_path.remove(guard)


if __name__ == '__main__':
    raise SystemExit(main())
