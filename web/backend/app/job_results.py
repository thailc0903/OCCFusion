"""Validate and publish one-frame CLI outputs without importing model code."""
import hashlib
import json
from pathlib import Path
import re

from .evaluations import ARTIFACTS, EvaluationStore, decode, digest
from .review import canonical
from .store import contained, InvalidArtifact, RunStore
from .task_types import task_of,DETECTION,DETECTION_FILES


PREDICTION_FILES=(*ARTIFACTS,'resolved_config.py')


def file_hash(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):value.update(block)
    return value.hexdigest()


def file_records(folder,task='occupancy'):
    return {name:dict(sha256=file_hash(contained(folder/name,folder)),bytes=(folder/name).stat().st_size) for name in (DETECTION_FILES if task==DETECTION else PREDICTION_FILES)}


def verify_publication(folder,expected_id=None):
    """Only job-prefixed folders require this receipt; legacy bundles stay unchanged."""
    folder=Path(folder)
    marker=decode(contained(folder/'job_result.json',folder).read_bytes())
    identifier=marker['job_id']
    if not re.fullmatch('[0-9a-f]{32}',identifier) or marker['run_id']!='infer_job_'+identifier or folder.name!=marker['run_id'] or marker['status']!='complete' or expected_id is not None and identifier!=expected_id:
        raise InvalidArtifact('Job publication identity/status mismatch')
    core={k:v for k,v in marker.items() if k!='fingerprint_sha256'}
    if digest(canonical(core).encode())!=marker['fingerprint_sha256'] or marker['artifacts']!=file_records(folder,marker.get('task','occupancy')):
        raise InvalidArtifact('Job publication artifact hash mismatch')
    if marker.get('task')==DETECTION:
        return marker
    evaluation=marker['evaluation']
    if evaluation['directory']!='eval_job_'+identifier:
        raise InvalidArtifact('Job evaluation directory mismatch')
    path=contained(folder.parent/evaluation['directory']/'evaluation.json',folder.parent)
    if file_hash(path)!=evaluation['sha256'] or path.stat().st_size!=evaluation['bytes']:
        raise InvalidArtifact('Job publication evaluation hash mismatch')
    return marker


class JobResults:
    def __init__(self,settings):
        self.settings=settings;self.store=RunStore(settings);self.evaluations=EvaluationStore(settings)

    def prediction(self,folder,job):
        profile=json.loads(job['profile_json'])
        if task_of(profile)==DETECTION:
            from .detections import validate_detection
            meta,report,doc=validate_detection(folder,self.settings)
            if meta['sample_token']!=job['sample_token'] or meta['scene']['name']!=job['scene_name'] or meta['frame_index']!=job['frame_index']:
                raise InvalidArtifact('Detection output does not match job token/scene/frame')
            if meta['checkpoint']['sha256']!=profile['checkpoint_sha256'] or meta['checkpoint']['bytes']!=profile['checkpoint_bytes'] or meta['config']['sha256']!=file_hash(Path(profile['config'])):
                raise InvalidArtifact('Detection config/checkpoint provenance mismatch')
            if meta['code']['inference_script_sha256']!=profile['source_hashes']['tools/infer_centerpoint_mini.py'] or report['tf32']!=profile['tf32']:
                raise InvalidArtifact('Detection source/TF32 mismatch')
            return meta,report,{name:(folder/name).read_bytes() for name in DETECTION_FILES}
        files={name:contained(folder/name,folder).read_bytes() for name in ARTIFACTS}
        meta,report=decode(files['metadata.json']),decode(files['report.json'])
        loaded=self.store.load(job['run_id'],folder)
        if not loaded.detail['prediction_available'] or meta['sample_token']!=job['sample_token'] or meta['scene']['name']!=job['scene_name'] or meta['frame_index']!=job['frame_index']:
            raise InvalidArtifact('Job output token/scene/frame does not match its request')
        if meta['dataset']['version']!=profile['dataset_version'] or meta['dataset'].get('split',job['split'])!=job['split'] or report.get('split',job['split'])!=job['split']:
            raise InvalidArtifact('Job dataset/split mismatch')
        if meta['checkpoint']['sha256']!=profile['checkpoint_sha256'] or meta['checkpoint']['bytes']!=profile['checkpoint_bytes']:
            raise InvalidArtifact('Job checkpoint provenance mismatch')
        if meta['config']['sha256']!=profile['source_hashes']['configs/OccFusion_infer_mini.py'] or meta['config']['baseline_sha256']!=profile['source_hashes']['configs/OccFusion.py'] or meta['code']['inference_script_sha256']!=profile['source_hashes']['tools/infer_nuscenes_mini.py']:
            raise InvalidArtifact('Job config/CLI source provenance mismatch')
        if meta['config']['resolved_sha256']!=file_hash(folder/'resolved_config.py'):
            raise InvalidArtifact('Resolved config hash mismatch')
        if report['precision']!='float32' or report['autocast'] is not False or report['tf32']!=profile['tf32']:
            raise InvalidArtifact('Job precision/TF32 differs from baseline')
        comparison=report['checkpoint_comparison']
        if not comparison['compatible'] or not comparison['strict_load_succeeded'] or any(comparison[key] for key in ('missing_keys','unexpected_keys','shape_mismatches')):
            raise InvalidArtifact('Checkpoint was not strictly matched')
        if any(meta[key]!=value for key,value in profile['geometry'].items()):
            raise InvalidArtifact('Job output geometry/classes differ from baseline')
        if any(meta[key]!=value for key,value in profile['input_contract'].items()):
            raise InvalidArtifact('Job image/sensor/coordinate contract differs from baseline')
        if any(meta['coordinate_system'][key]!=value for key,value in profile['coordinate_contract'].items()):
            raise InvalidArtifact('Job coordinate convention differs from baseline')
        return meta,report,files

    def evaluation(self,path,job,meta,files):
        content=path.read_bytes();value=decode(content);profile=json.loads(job['profile_json'])
        if value['execution']['model_loaded'] is not False or value['execution']['device']!='cpu':
            raise InvalidArtifact('Evaluation must be CPU-only')
        if value['split']!=job['split'] or value['protocol']['name']!='surroundocc-200x200x16-occfusion-v1' or value['protocol']['semantic_mean_class_count']!=16 or value['protocol']['union_zero_iou']!=0:
            raise InvalidArtifact('Evaluation protocol/split mismatch')
        if any(value['protocol'][key]!=expected for key,expected in profile['geometry'].items()):
            raise InvalidArtifact('Evaluation geometry/classes mismatch')
        result=self.evaluations.match(dict(id=digest(content),label='job evaluation',sha256=digest(content),bytes=len(content),document=value),meta,files)
        if result['status'] not in ('matched','missing_gt'):
            raise InvalidArtifact('Evaluation cannot be bound to job output: '+result['reason'])
        if result['status']=='matched':
            frame=value['frames'][0];gt=contained(Path(frame['gt_path']),Path(profile['gt_root']))
            expected=Path(meta['sensors']['keyframes']['LIDAR_TOP']['path']).name+'.npy'
            if gt.name!=expected or file_hash(gt)!=frame['gt_sha256'] or gt.stat().st_size!=frame['gt_bytes']:
                raise InvalidArtifact('Ground truth filename/hash changed after evaluation')
        elif value['metrics'] is not None:
            raise InvalidArtifact('Missing-GT evaluation must not contain replacement metrics')
        return result['status'],value
