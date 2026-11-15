"""Read-only detection collection playback. No CLI, evaluator or ML imports."""
from collections import Counter, OrderedDict
import csv
from io import StringIO
import json
from pathlib import PurePosixPath
import re

import numpy as np

from .benchmarks import CollectionSnapshot, TOKEN, fingerprint, record
from .detections import CLASSES, validate_detection
from .evaluations import decode
from .store import contained, InvalidArtifact, LoadedRun, quaternion_pose

KIND = 'centerpoint_detection_benchmark'
FILES = ('detections.json', 'metadata.json', 'report.json', 'resolved_config.py')
TABLES = {'attribute', 'calibrated_sensor', 'category', 'ego_pose', 'instance', 'log',
          'map', 'sample', 'sample_annotation', 'sample_data', 'scene', 'sensor', 'visibility'}
DEFAULTS = dict(car='vehicle.parked', truck='vehicle.parked', trailer='vehicle.parked',
    construction_vehicle='vehicle.parked', bus='vehicle.moving', pedestrian='pedestrian.moving',
    motorcycle='cycle.without_rider', bicycle='cycle.without_rider', barrier='', traffic_cone='')


def require(condition, message):
    if not condition:
        raise InvalidArtifact(message)


def ratio(value):
    require(type(value) in (float, int) and np.isfinite(value) and -1e-12 <= value <= 1 + 1e-12,
            'Detection metric must be a finite fraction, not percent')
    return value


def counts(value):
    require(set(value) == set(CLASSES) and all(type(v) is int and v >= 0 for v in value.values()),
            'Detection support must contain nonnegative counts for all 10 classes')


def protocol_check(protocol):
    require(protocol['name'] == 'detection_cvpr_2019' and
        protocol['evaluator'] == 'nuscenes.eval.detection.evaluate.DetectionEval', 'Unsupported detection protocol/evaluator')
    expected = dict(class_range=dict(car=50, truck=50, bus=50, trailer=50, construction_vehicle=50,
        pedestrian=40, motorcycle=40, bicycle=40, traffic_cone=30, barrier=30),
        dist_fcn='center_distance', dist_ths=[.5, 1., 2., 4.], dist_th_tp=2.,
        min_recall=.1, min_precision=.1, max_boxes_per_sample=500, mean_ap_weight=5)
    require(protocol['configuration'] == expected, 'detection_cvpr_2019 configuration mismatch')
    policy = protocol['attribute_policy']
    require(policy['threshold_mps'] == .2 and policy['defaults'] == DEFAULTS and
        policy['uses_gt_attributes'] is False, 'Unsupported predicted attribute policy')
    conversion = protocol['box_conversion']
    require(conversion['input_center'] == 'geometric' and conversion['input_dimensions'] == ['dx', 'dy', 'dz']
        and conversion['output_size'] == ['dy', 'dx', 'dz'] and
        conversion['extra_display_score_filter'] is False and conversion['extra_formatting_range_filter'] is False,
        'Detection box conversion/display-filter protocol mismatch')


def qmultiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j, w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def attribute(name, velocity):
    moving = np.linalg.norm(velocity) > .2
    if moving and name in ('car', 'truck', 'bus', 'trailer', 'construction_vehicle'):
        return 'vehicle.moving'
    if moving and name in ('bicycle', 'motorcycle'):
        return 'cycle.with_rider'
    if not moving and name == 'pedestrian':
        return 'pedestrian.standing'
    if not moving and name == 'bus':
        return 'vehicle.stopped'
    return DEFAULTS[name]


def check_global(local, global_rows, meta, tables):
    """Check saved geometry/score/attributes, without computing any GT metric."""
    require(len(local['boxes']) == len(global_rows), 'Saved/global detection box count mismatch')
    key = meta['sensors']['keyframes']['LIDAR_TOP']
    cal = tables['calibrated_sensor'][key['calibrated_sensor_token']]
    ego = tables['ego_pose'][key['ego_pose_token']]
    rotation = qmultiply(ego['rotation'], cal['rotation'])
    matrix = np.asarray(meta['coordinate_system']['lidar_to_global'])
    for box, saved in zip(local['boxes'], global_rows):
        name = CLASSES[box['label']]
        center = matrix[:3, :3] @ box['center'] + matrix[:3, 3]
        velocity = (matrix[:3, :3] @ [*box['velocity'], 0.])[:2]
        yaw = box['yaw'] / 2
        orientation = qmultiply(rotation, [np.cos(yaw), 0, 0, np.sin(yaw)])
        orientation /= np.linalg.norm(orientation)
        actual = np.asarray(saved['rotation'])
        require(saved['sample_token'] == local['sample_token'] and saved['detection_name'] == name
            and saved['detection_score'] == box['score']
            and np.allclose(saved['translation'], center, rtol=0, atol=1e-6)
            and np.allclose(saved['size'], [box['dimensions'][1], box['dimensions'][0], box['dimensions'][2]], rtol=0, atol=1e-6)
            and np.allclose(saved['velocity'], velocity, rtol=0, atol=1e-6)
            and (np.allclose(actual, orientation, rtol=0, atol=1e-6) or np.allclose(actual, -orientation, rtol=0, atol=1e-6))
            and saved['attribute_name'] == attribute(name, velocity),
            'nuScenes global boxes do not match saved native-LiDAR prediction')


class DetectionBenchmarkStore:
    def __init__(self, parent):
        self.parent, self.settings, self.runs = parent, parent.settings, parent.runs
        self.validated = OrderedDict()

    def context(self, identifier):
        folder = self.parent.folder(identifier)
        paths, hashes = {}, {}
        def load(relative, arc=None):
            path = contained(folder / relative, folder)
            data = path.read_bytes()
            arc = arc or 'benchmark/' + str(relative)
            paths[arc], hashes[arc] = path, record(data)
            return data
        manifest = decode(load('manifest.json'))
        require(manifest['kind'] == KIND and manifest['schema_version'] == 1, 'Not a detection benchmark manifest')
        identity, provenance, protocol = (manifest[k] for k in ('identity', 'prediction_provenance', 'evaluation_protocol'))
        require(fingerprint(identity) == manifest['identity_sha256']
            and fingerprint(provenance) == identity['prediction_fingerprint']
            and fingerprint(protocol) == identity['evaluation_protocol_fingerprint'], 'Detection manifest fingerprint mismatch')
        protocol_check(protocol)
        require(provenance['dataset_version'] == 'v1.0-mini' and identity['split'] == 'mini_val'
            and provenance['class_names'] == CLASSES and provenance['precision'] == 'float32'
            and provenance['autocast'] is False and provenance['batch_size'] == 1, 'Unsupported detection dataset/model contract')
        require(record(load('resolved_config.py'))['sha256'] == provenance['config']['resolved_sha256'],
            'Resolved detection config hash mismatch')
        table_hashes = manifest['dataset_table_hashes']
        require(set(table_hashes) == TABLES, 'Incomplete nuScenes metadata provenance')
        root, tables = self.parent.dataset(dict(dataset_version='v1.0-mini',
            dataset_table_hashes={k: v['sha256'] for k, v in table_hashes.items()}))
        entries = identity['frames']
        names = identity['scene_names']
        # These are the published split scene names; membership/order comes from real sample chains.
        require(names == ['scene-0103', 'scene-0916'], 'Not the complete official mini_val scene set')
        lidar_keys = {}
        for sd in tables['sample_data'].values():
            cal = tables['calibrated_sensor'][sd['calibrated_sensor_token']]
            if sd['is_key_frame'] and tables['sensor'][cal['sensor_token']]['channel'] == 'LIDAR_TOP':
                require(sd['sample_token'] not in lidar_keys, 'Duplicate LiDAR keyframe for sample')
                lidar_keys[sd['sample_token']] = sd
        expected = []
        for scene in sorted((s for s in tables['scene'].values() if s['name'] in names), key=lambda s: s['name']):
            token, previous, timestamp, seen = scene['first_sample_token'], '', -1, set()
            while token:
                require(token not in seen, 'Cycle in nuScenes scene sample chain')
                seen.add(token)
                sample = tables['sample'][token]
                require(sample['prev'] == previous and sample['timestamp'] > timestamp and sample['scene_token'] == scene['token'],
                    'Invalid nuScenes sample chain')
                sd = lidar_keys[token]
                expected.append((scene['name'], len(seen)-1, token, sample['timestamp'], sd['timestamp'], sd['token'], sd['filename']))
                previous, timestamp, token = token, sample['timestamp'], sample['next']
            require(len(seen) == scene['nbr_samples'] and previous == scene['last_sample_token'], 'Scene sample count mismatch')
        actual = [(e['scene'], e['frame_index'], e['sample_token'], e['timestamp_us'], e['reference_timestamp_us'],
                   e['lidar_sample_data_token'], e['lidar_filename']) for e in entries]
        require(actual == expected and len({e['sample_token'] for e in entries}) == len(entries)
            and all(TOKEN.fullmatch(e['sample_token']) for e in entries), 'Detection manifest token/order/coverage mismatch')
        tokens = [e['sample_token'] for e in entries]
        candidates = []
        for path in sorted((folder / 'reports').glob('*.json')):
            if not TOKEN.fullmatch(path.stem):
                continue
            value = decode(contained(path, folder).read_bytes())
            inv = value['current_invocation']
            if inv.get('forward_calls', 0) > 0 and inv.get('model_load_count', 0) > 0:
                candidates.append((path, value))
        require(len(candidates) == 1, 'Original GPU report missing or ambiguous; never use cache/resume latency')
        gpu_path, gpu = candidates[0]
        gpu_relative = gpu_path.relative_to(folder)
        gpu = decode(load(gpu_relative))
        invocation = gpu['current_invocation']
        require(gpu_path.stem == invocation['id'] and invocation['mode'] == 'inference'
            and gpu['manifest_identity_sha256'] == manifest['identity_sha256']
            and gpu['prediction_provenance'] == provenance and gpu['evaluation_protocol'] == protocol
            and gpu['status'] == 'complete' and gpu['successful_frames'] == gpu['expected_frames'] == len(entries)
            and gpu['failed_frames'] == 0 and not gpu['missing_frames'] and not invocation['errors'],
            'Original GPU report does not match the complete detection manifest')
        # The original report pins the exact immutable evaluation bytes, not a mutable cache pointer.
        evaluations = []
        for path in sorted((folder / 'evaluations').glob('*/evaluation.json')):
            if TOKEN.fullmatch(path.parent.name):
                data = contained(path, folder).read_bytes()
                if record(data) == gpu['evaluation_file']:
                    evaluations.append(path)
        require(len(evaluations) == 1, 'Immutable evaluation pinned by GPU report is absent/hash-mismatched')
        eval_path = evaluations[0]
        evaluation = decode(load(eval_path.relative_to(folder), 'benchmark/evaluation/evaluation.json'))
        commit_path = eval_path.with_name('evaluation_commit.json')
        commit = decode(load(commit_path.relative_to(folder), 'benchmark/evaluation/evaluation_commit.json'))
        require(commit['status'] == 'complete' and fingerprint({k:v for k,v in commit.items() if k != 'fingerprint_sha256'})
            == commit['fingerprint_sha256'], 'Detection evaluation commit fingerprint mismatch')
        require(set(commit['files']) == {'evaluation.json', 'official/metrics_summary.json', 'official/metrics_details.json'},
            'Unexpected detection evaluation snapshot files')
        official = {}
        for name, declared in commit['files'].items():
            data = load((eval_path.parent / name).relative_to(folder), 'benchmark/evaluation/' + name)
            require(record(data) == declared, 'Detection evaluation snapshot hash mismatch: ' + name)
            if name.startswith('official/'):
                official[name] = data
        global_content = load('predictions_nuscenes.json')
        predictions = decode(global_content)
        ev_identity = evaluation['identity']
        require(commit['identity'] == ev_identity and ev_identity['manifest_identity_sha256'] == manifest['identity_sha256']
            and ev_identity['protocol_sha256'] == fingerprint(protocol)
            and ev_identity['prediction'] == record(global_content) and ev_identity['gt_table_hashes'] == table_hashes
            and ev_identity['split'] == 'mini_val' and evaluation['protocol'] == protocol
            and evaluation['status'] == 'complete' and evaluation['task'] == 'detection'
            and evaluation['sample_tokens'] == tokens and set(predictions['results']) == set(tokens)
            and set(ev_identity['frame_artifacts']) == set(tokens) and evaluation['coverage'] ==
                dict(expected=len(tokens), predicted=len(tokens), ground_truth=len(tokens), missing=[], complete=True),
            'Detection evaluation prediction/GT/protocol/token hash mismatch')
        require(predictions['meta'] == dict(use_camera=False, use_lidar=True, use_radar=False, use_map=False, use_external=False),
            'Detection modality is not LiDAR-only')
        metrics, support = evaluation['metrics'], evaluation['support']
        require(gpu['metrics'] == metrics, 'GPU report/evaluation metric mismatch')
        self.metrics(metrics, support, official['official/metrics_summary.json'])
        audit = decode(load('input_audit.json'))
        require(audit['status'] == 'passed' and audit['frame_count'] == len(tokens)
            and {k:v for k,v in audit.items() if k != 'frames'} == gpu['input_audit_summary'],
            'Detection loader audit differs from the GPU report')
        return dict(folder=folder, paths=paths, hashes=hashes, load=load, manifest=manifest,
            provenance=provenance, protocol=protocol, entries=entries, tokens=tokens, dataset=(root,tables),
            gpu=gpu, gpu_relative=str(gpu_relative), evaluation=evaluation, audit=audit,
            predictions=predictions['results'], sensor_hashes={})

    @staticmethod
    def metrics(metrics, support, raw):
        # JSON parsing/consistency only. No GT loading, matching, evaluator or metric generation.
        official = json.loads(raw, parse_constant=lambda _: None)
        expected = dict(mAP=official['mean_ap'], NDS=official['nd_score'],
            per_class_ap=official['mean_dist_aps'], per_class_ap_by_distance=official['label_aps'],
            tp_errors=official['tp_errors'], per_class_tp_errors=official['label_tp_errors'])
        require(metrics == expected, 'Normalized metrics differ from hash-verified official summary')
        for field in ('per_class_ap', 'per_class_ap_by_distance', 'per_class_tp_errors'):
            require(set(metrics[field]) == set(CLASSES), 'Metrics must retain all 10 detection classes')
        ratio(metrics['mAP']); ratio(metrics['NDS'])
        for name in CLASSES:
            ratio(metrics['per_class_ap'][name])
            by_distance = metrics['per_class_ap_by_distance'][name]
            require(set(by_distance) == {'0.5','1.0','2.0','4.0'}, 'Detection AP distance thresholds differ')
            require(np.isclose(np.mean([ratio(v) for v in by_distance.values()]), metrics['per_class_ap'][name], atol=1e-12, rtol=0),
                'Saved per-class AP statistics disagree')
            for key, value in metrics['per_class_tp_errors'][name].items():
                undefined = name == 'traffic_cone' and key in ('orient_err','vel_err','attr_err') or name == 'barrier' and key in ('vel_err','attr_err')
                require((value is None) == undefined, 'Undefined TP errors must remain null')
                if value is not None:
                    require(type(value) in (float,int) and np.isfinite(value) and value >= 0, 'Invalid TP error')
        require(np.isclose(np.mean(list(metrics['per_class_ap'].values())), metrics['mAP'], atol=1e-12, rtol=0), 'Saved mAP averaging mismatch')
        require(np.isclose((5*metrics['mAP'] + sum(max(0,1-v) for v in metrics['tp_errors'].values()))/10,
            metrics['NDS'], atol=1e-12, rtol=0), 'Saved NDS summary mismatch')
        for key in ('gt_before_official_filters','gt_after_official_filters','predictions_after_official_filters'):
            counts(support[key])
        require(support['display_score_filter_applied'] is False
            and all(support['gt_after_official_filters'][n] <= support['gt_before_official_filters'][n] for n in CLASSES),
            'GT support/display filter mismatch')

    def frame_evidence(self, context, entry):
        folder, load, hashes = (context[k] for k in ('folder','load','hashes'))
        token, provenance = entry['sample_token'], context['provenance']
        expected = context['evaluation']['identity']['frame_artifacts'][token]
        attempts = []
        for candidate in sorted((folder / 'frames' / token).iterdir()):
            if candidate.is_dir() and TOKEN.fullmatch(candidate.name):
                path = contained(candidate / 'prediction_commit.json', folder)
                if path.is_file():
                    receipt = decode(path.read_bytes())
                    if receipt.get('artifacts') == expected:
                        attempts.append((candidate, receipt))
        require(len(attempts) == 1, token + ': missing/ambiguous committed prediction matching evaluation')
        attempt, receipt = attempts[0]
        names = (*FILES, 'job_result.json') if receipt['origin'] == 'external_smoke' else FILES
        require(set(expected) == set(names) and receipt['status'] == 'complete' and receipt['sample_token'] == token
            and receipt['input_fingerprint'] == entry['input_fingerprint']
            and receipt['prediction_fingerprint'] == fingerprint(provenance)
            and fingerprint({k:v for k,v in receipt.items() if k != 'fingerprint_sha256'}) == receipt['fingerprint_sha256'],
            token + ': prediction commit token/provenance mismatch')
        files = {}
        for name in names:
            data = load((attempt / name).relative_to(folder), f'frames/{token}/{name}')
            require(record(data) == expected[name], token + ': prediction artifact hash mismatch: ' + name)
            files[name] = data
        load((attempt / 'prediction_commit.json').relative_to(folder), f'frames/{token}/prediction_commit.json')
        meta, report, doc = (decode(files[n]) for n in ('metadata.json','report.json','detections.json'))
        root, tables = context['dataset']
        # Re-read/hash real bytes on every visit. Only expensive decoded geometry
        # checks are cached, keyed by artifact + input + metadata/protocol content.
        input_hashes = {}
        for sensor in [meta['sensors']['keyframes']['LIDAR_TOP'], *meta['sensors']['lidar_sweeps']]:
            sd = tables['sample_data'][sensor['sample_data_token']]
            if sd['token'] not in context['sensor_hashes']:
                data = contained(root / sd['filename'], root).read_bytes()
                require(data and len(data) % 20 == 0 and np.isfinite(np.frombuffer(data,dtype='<f4')).all(), 'Invalid LiDAR sweep file')
                context['sensor_hashes'][sd['token']] = record(data)
            actual = context['sensor_hashes'][sd['token']]
            require(actual == dict(sha256=sensor['sha256'],bytes=sensor['bytes']), 'Detection sensor input hash mismatch')
            input_hashes[sd['token']] = actual
        key = fingerprint(dict(artifacts=expected, inputs=input_hashes,
            manifest_identity=context['manifest']['identity_sha256'], commit=receipt['fingerprint_sha256'],
            tables=context['manifest']['dataset_table_hashes'], protocol=context['protocol'],
            global_prediction=context['evaluation']['identity']['prediction']))
        def result():
            return dict(entry=entry, metadata=meta, report=report, document=doc, origin=receipt['origin'],
                fingerprint=fingerprint(expected), artifacts=expected, folder=attempt)
        with self.parent.lock:
            if key in self.validated:
                self.validated.move_to_end(key)
                return result()
        validate_detection(attempt, self.settings, files, catalog=self.runs.catalog)
        require(not report.get('fixture') and report.get('forward_calls') == 1 and report.get('forward_completed') is True,
            'Fixture/incomplete detection cannot be played as benchmark prediction')
        require(all(meta.get(k) == v for k,v in entry['input_metadata'].items()), token + ': input metadata mismatch')
        require(meta['config'] == provenance['config'] and meta['checkpoint'] == provenance['checkpoint']
            and meta['test_pipeline'] == provenance['test_pipeline'] and report['tf32'] == provenance['tf32'],
            'Detection frame baseline model/pipeline mismatch')
        require(fingerprint(dict(metadata=entry['input_metadata'], pipeline_fingerprint=provenance['pipeline_fingerprint']))
            == entry['input_fingerprint'], 'Detection input fingerprint mismatch')
        expected_code = provenance['reference_cli_sha256'] if receipt['origin'] == 'external_smoke' else provenance['runtime_sha256']
        require(meta['code']['inference_script_sha256'] == expected_code, 'Detection frame code hash mismatch')
        if receipt['origin'] == 'external_smoke':
            publication = decode(files['job_result.json'])
            require(publication['task'] == 'detection' and publication['execution_mode'] == 'real'
                and publication['sample_token'] == token and publication['status'] == 'complete'
                and publication['artifacts'] == {n:expected[n] for n in FILES}
                and fingerprint({k:v for k,v in publication.items() if k != 'fingerprint_sha256'}) == publication['fingerprint_sha256'],
                'Original smoke receipt is invalid')
        else:
            require(meta['benchmark_provenance'] == dict(prediction_fingerprint=fingerprint(provenance),
                input_fingerprint=entry['input_fingerprint']), 'Detection benchmark preprocessing provenance mismatch')
        sensors = meta['sensors']
        previous = sensors['keyframes']['LIDAR_TOP']['sample_data_token']
        last = meta['reference_timestamp_us']
        deviations = []
        world = np.asarray(meta['coordinate_system']['lidar_to_global'])
        records = [sensors['keyframes']['LIDAR_TOP'], *sensors['lidar_sweeps']]
        require(len(sensors['lidar_sweeps']) <= 9, 'Too many detection sweeps')
        for index, sensor in enumerate(records):
            sd = tables['sample_data'][sensor['sample_data_token']]
            cal = tables['calibrated_sensor'][sd['calibrated_sensor_token']]
            ego = tables['ego_pose'][sd['ego_pose_token']]
            require(tables['sensor'][cal['sensor_token']]['channel'] == 'LIDAR_TOP'
                and tables['sample'][sd['sample_token']]['scene_token'] == entry['scene_token']
                and sd['timestamp'] == sensor['timestamp_us']
                and (not index or tables['sample_data'][previous]['prev'] == sd['token'] and sd['timestamp'] < last),
                'Detection LiDAR sweep timestamp/scene/prev chain mismatch')
            declared = PurePosixPath(sensor['path']).relative_to(PurePosixPath(meta['dataset']['root']))
            require(str(declared) == sd['filename'], 'Detection sensor filename mismatch')
            require(context['sensor_hashes'][sd['token']] == dict(sha256=sensor['sha256'], bytes=sensor['bytes']), 'Detection sensor input hash mismatch')
            pose = quaternion_pose(ego['rotation'], ego['translation']) @ quaternion_pose(cal['rotation'], cal['translation'])
            physical = np.linalg.inv(world) @ pose
            require(np.allclose(sensor['sensor_to_reference_lidar'], physical, rtol=0, atol=1e-6), 'Detection sweep physical transform mismatch')
            if index:
                delta = -np.linalg.inv(physical)[:3,3] - physical[:3,3]
                require(np.allclose(sensor['official_loader_translation_difference_m'], delta, rtol=0, atol=1e-6),
                    'Loader/SE(3) audit vector mismatch')
                deviations.append(float(np.linalg.norm(sensor['official_loader_translation_difference_m'])))
            previous, last = sd['token'], sd['timestamp']
        require(np.allclose(deviations, entry['input_audit']['loader_translation_difference_m'], rtol=0, atol=1e-12)
            and len(sensors['lidar_sweeps']) == entry['input_audit']['past_sweeps']
            and entry['input_audit']['padded_keyframe_copies'] == (9 if not deviations else 0),
            'Detection input audit/padding mismatch')
        if len(sensors['lidar_sweeps']) < 9:
            require(not tables['sample_data'][previous]['prev'], 'Available past history was omitted')
        check_global(doc, context['predictions'][token], meta, tables)
        with self.parent.lock:
            self.validated[key] = True
            while len(self.validated) > 256:
                self.validated.popitem(last=False)
        return result()

    @staticmethod
    def latency(context, frames):
        content = context['load']('timings.csv')
        gpu, identifier = context['gpu'], context['gpu']['current_invocation']['id']
        inv = gpu['current_invocation']
        rows = list(csv.DictReader(StringIO(content.decode(), newline='')))
        require(all(None not in row and all(v is not None for v in row.values()) for row in rows), 'Malformed detection timing CSV')
        rows = [r for r in rows if r['invocation_id'] == identifier and r['source'] == 'new']
        require(all(r['forward_completed'].lower() == 'true' and r['status'] == 'complete' for r in rows),
            'Original timing includes failed/incomplete forward')
        tokens = [r['sample_token'] for r in rows]
        new_tokens = {f['entry']['sample_token'] for f in frames if f['origin'] == 'benchmark'}
        imported = sum(f['origin'] == 'external_smoke' for f in frames)
        require(len(tokens) == len(set(tokens)) == inv['forward_calls'] == inv['predictions_new']
            and set(tokens) == new_tokens and inv['predictions_cached'] == imported
            and inv['model_load_count'] == 1 and inv['extra_warmup_forwards'] == 0,
            'Original GPU timing/cache/forward count mismatch')
        warmup = [r for r in rows if r['warmup'].lower() == 'true']
        require(len(warmup) == 1 and warmup[0] == rows[0], 'First new manifest forward must be the only marked warm-up')
        def stats(selected, field):
            values = [float(r[field]) for r in selected]
            require(values and np.isfinite(values).all() and min(values) >= 0, 'Invalid saved detection latency')
            return dict(sample_count=len(values), median_s=float(np.median(values)), p95_s=float(np.percentile(values,95)))
        steady = [r for r in rows if r['warmup'].lower() != 'true']
        calculated = dict(all_completed_forwards=stats(rows,'forward_s'),
            excluding_marked_warmup=stats(steady,'forward_s'),
            end_to_end_excluding_marked_warmup=stats(steady,'end_to_end_s'))
        for key, value in calculated.items():
            saved = gpu['latency_current_invocation'][key]
            require(value['sample_count'] == saved['sample_count'] and all(np.isclose(value[k],saved[k],atol=1e-12,rtol=0)
                for k in ('median_s','p95_s')), 'Original GPU report disagrees with detection CSV latency')
        require(gpu['latency_current_invocation']['marked_warmup_count'] == len(warmup), 'Warm-up count mismatch')
        for key in ('peak_allocated_bytes','peak_reserved_bytes'):
            require(max(int(r[key]) for r in rows) == gpu['allocator'][key], 'Allocator report/CSV mismatch')
        return dict(**calculated, sample_count=len(steady), all_forward_sample_count=len(rows), warmup_count=len(warmup),
            new_prediction_count=inv['predictions_new'], smoke_cache_count=imported,
            source_report=context['gpu_relative'], source_report_sha256=context['hashes']['benchmark/'+context['gpu_relative']]['sha256'],
            source_session_id=identifier, scope=gpu['latency_current_invocation']['scope'],
            end_to_end_scope=gpu['latency_current_invocation']['end_to_end_scope'], units='seconds',
            startup=dict(common_cpu_s=inv['setup_cpu_s'], model_checkpoint_cpu_s=inv['model_initialization']['model_checkpoint_cpu_s'],
                device_setup_s=inv['model_initialization']['device_setup_s'], invocation_wall_s=inv['wall_s']))

    @staticmethod
    def audit(context):
        entries, audit = context['entries'], context['audit']
        expected = [dict(sample_token=e['sample_token'], scene=e['scene'], frame_index=e['frame_index'], **e['input_audit']) for e in entries]
        require(audit['frames'] == expected, 'Loader audit frame membership mismatch')
        values = [v for e in entries for v in e['input_audit']['loader_translation_difference_m']]
        require(audit['sweep_count'] == len(values) and all(e['input_audit']['status'] == 'passed' for e in entries),
            'Loader audit sweep count/status mismatch')
        stats = audit['loader_vs_se3_translation_m']
        require(np.isclose(stats['max'],max(values),atol=1e-12,rtol=0)
            and np.isclose(stats['median'],np.median(values),atol=1e-12,rtol=0)
            and np.isclose(stats['p95'],np.percentile(values,95),atol=1e-12,rtol=0), 'Loader audit summary mismatch')
        worst = max(entries, key=lambda e: e['input_audit']['max_loader_vs_se3_translation_m'])
        require(audit['worst_frame'] == dict(sample_token=worst['sample_token'], scene=worst['scene'],
            frame_index=worst['frame_index'], maximum_m=worst['input_audit']['max_loader_vs_se3_translation_m']),
            'Loader audit worst frame mismatch')
        return {k:v for k,v in audit.items() if k != 'frames'}

    def snapshot(self, identifier):
        context = self.context(identifier)
        frames = [self.frame_evidence(context, entry) for entry in context['entries']]
        evaluation, gpu, provenance = (context[k] for k in ('evaluation','gpu','provenance'))
        support = evaluation['support']
        require(sum(len(f['document']['boxes']) for f in frames) == support['saved_prediction_box_count']
            and sum(support['predictions_after_official_filters'].values()) <= support['saved_prediction_box_count'],
            'Saved/official-filtered detection box count mismatch')
        latency, audit = self.latency(context, frames), self.audit(context)
        summaries = [dict(sample_token=f['entry']['sample_token'], scene=f['entry']['scene'],
            frame_index=f['entry']['frame_index'], timestamp_us=f['entry']['timestamp_us'],
            reference_timestamp_us=f['entry']['reference_timestamp_us'], status='success', prediction_available=True,
            box_count=len(f['document']['boxes']), origin=f['origin'],
            prediction_sha256=f['artifacts']['detections.json']['sha256'],
            url=f'/api/benchmarks/{identifier}/frames/{f["entry"]["sample_token"]}') for f in frames]
        detail = dict(task='detection', benchmark_id=identifier, status='complete', split='mini_val',
            scope='Full official mini_val metadata membership; not all nuScenes validation',
            expected_frame_count=len(frames), successful_frame_count=len(frames), failed_frame_count=0,
            scene_counts=dict(Counter(e['scene'] for e in context['entries'])),
            model_source=frames[0]['metadata']['model_source'], model=provenance,
            sensor_configuration=frames[0]['metadata']['sensor_configuration'], metrics=evaluation['metrics'],
            classes=[dict(name=n, ap=evaluation['metrics']['per_class_ap'][n],
                gt_count=support['gt_after_official_filters'][n], gt_count_before_filters=support['gt_before_official_filters'][n],
                has_gt=support['gt_after_official_filters'][n] > 0,
                predicted_count_after_filters=support['predictions_after_official_filters'][n],
                tp_errors=evaluation['metrics']['per_class_tp_errors'][n]) for n in context['protocol']['configuration']['class_range']],
            support=dict(support, official_filtered_box_count=sum(support['predictions_after_official_filters'].values())),
            metric_unit='ratio_0_1', protocol=context['protocol'], frames=summaries, latency=latency,
            loader_audit=audit, precision=provenance['precision'], tf32=provenance['tf32'],
            allocator=dict(gpu['allocator'], source=context['gpu_relative']),
            evaluation_source=dict(snapshot=context['evaluation']['evaluation_directory']+'/evaluation.json',
                immutable=True, evaluator=context['protocol']['evaluator'], devkit_version=context['protocol']['devkit_version']),
            evidence_hashes=context['hashes'], validation=dict(tokens=True, geometry=True, protocol=True,
                artifact_hashes=True, global_boxes=True, saved_official_metrics=True),
            replay=True, inference_enabled=False,
            metric_source_export='benchmark/evaluation/evaluation.json',
            latency_source_export='benchmark/'+context['gpu_relative'],
            playback_note='Phát lại prediction từng frame đã lưu; chưa phải learned 4D flow.')
        core = dict(schema_version=2, kind='centerpoint_detection_benchmark_collection', task='detection',
            collection_id=identifier, split='mini_val', status='complete', expected_frame_count=len(frames),
            successful_frame_count=len(frames), failed_frame_count=0, sample_tokens=context['tokens'],
            evaluated_sample_tokens=context['tokens'], frames=[dict(sample_token=f['entry']['sample_token'],
                scene=f['entry']['scene'], frame_index=f['entry']['frame_index'], origin=f['origin'],
                artifacts=f['artifacts']) for f in frames],
            protocol=context['protocol'], protocol_fingerprint=fingerprint(context['protocol']),
            metric_unit='ratio_0_1', metric_source=detail['metric_source_export'],
            latency_source=detail['latency_source_export'], dataset_table_hashes=context['manifest']['dataset_table_hashes'],
            token_sources=['manifest.json:identity.frames','detections.json','metadata.json','report.json','immutable evaluation snapshot'],
            artifacts=context['hashes'], review_scope='Detection set-level snapshot; independent of occupancy and single-frame approvals',
            mutable_pointers_excluded=['progress.json','benchmark_report.json','evaluation.json'],
            playback_note=detail['playback_note'])
        snapshot = CollectionSnapshot(detail, dict(core, fingerprint_sha256=fingerprint(core)), context['paths'])
        return snapshot

    def frame(self, identifier, token):
        require(bool(TOKEN.fullmatch(token)), 'Invalid detection sample token')
        context = self.context(identifier)
        entry = next((e for e in context['entries'] if e['sample_token'] == token), None)
        if entry is None:
            raise FileNotFoundError('Sample is not in this detection benchmark')
        frame = self.frame_evidence(context, entry)
        meta, report, doc = (frame[k] for k in ('metadata','report','document'))
        points, info = self.runs.catalog.lidar(meta, meta['bounds'])
        base = f'/api/benchmarks/{identifier}/frames/{token}'
        bundle = fingerprint(dict(frame=frame['fingerprint'], manifest=context['manifest']['identity_sha256']))
        detail = dict(task='detection', collection_id=identifier, scene=meta['scene'], frame_index=entry['frame_index'],
            sample_token=token, timestamp_us=meta['timestamp_us'], reference_timestamp_us=meta['reference_timestamp_us'],
            prediction_available=True, replay=True, inference_enabled=False, bounds=meta['bounds'],
            voxel_size=meta['voxel_size'], coordinate_system=meta['coordinate_system'],
            classes=[dict(id=i,name=n,color=[255,210,60],count=sum(b['label']==i for b in doc['boxes']),background=False) for i,n in enumerate(CLASSES)],
            boxes=doc['boxes'], box_count=len(doc['boxes']), box_convention=meta['box_convention'],
            origin=frame['origin'], source=meta['model_source'], precision=report['precision'],
            checkpoint=meta['checkpoint'], timings_s=report['timings_s'], allocator=report['vram'],
            lidar=dict(**info, url=base+'/lidar?bundle='+bundle), bundle_fingerprint=bundle,
            artifacts=frame['artifacts'], metrics_scope='Dataset mAP/NDS belongs to the collection; no per-frame mAP',
            playback_note='Prediction + LiDAR đã lưu; không tạo job/publication receipt; chưa phải learned 4D flow.')
        loaded = LoadedRun(detail, lidar=points.tobytes())
        loaded.metadata = meta
        return detail, loaded
