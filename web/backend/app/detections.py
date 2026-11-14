"""Independent detection schema, validation and native-LiDAR pairing; no Torch."""
from pathlib import Path
import numpy as np
from .evaluations import decode,digest
from .store import contained,InvalidArtifact,SensorCatalog
from .task_types import DETECTION_FILES

CLASSES=['car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone']
CONVENTION=dict(center='geometric',dimensions=['dx','dy','dz'],yaw='radians, CCW about +Z, zero along +X',velocity=['vx','vy'],velocity_unit='m/s')

def _validate_detection(folder,settings,files=None):
    files=files if files is not None else {n:contained(folder/n,folder).read_bytes() for n in DETECTION_FILES}
    meta,report,doc=(decode(files[n]) for n in ('metadata.json','report.json','detections.json'))
    if meta.get('schema_version')!=1 or doc.get('schema_version')!=1:raise InvalidArtifact('Unsupported detection schema version')
    if any(v.get('task')!='detection' for v in (meta,report,doc)) or meta['status']!='success' or report['status']!='success' or report.get('check_only'):
        raise InvalidArtifact('Not a complete detection prediction')
    if meta['result_source']!='prediction' or meta['class_names']!=CLASSES or meta['box_convention']!=CONVENTION:
        raise InvalidArtifact('Detection classes/box convention unsupported')
    for key in ('sample_token','timestamp_us','reference_timestamp_us','coordinate_system','class_names','box_convention'):
        if doc[key]!=meta[key]:raise InvalidArtifact('Detection '+key+' mismatch')
    if meta['sample_token']!=report['sample_token'] or meta['frame_index']!=report['frame_index'] or meta['scene']['name']!=report['scene_name']:
        raise InvalidArtifact('Detection metadata/report sample mismatch')
    if meta['coordinate_system']['name']!='LIDAR_TOP at reference_timestamp_us' or meta['coordinate_system']['unit']!='metres':raise InvalidArtifact('Detection must use native LIDAR_TOP metres')
    if meta['bounds']!=[-51.2,-51.2,-5.0,51.2,51.2,3.0] or meta['voxel_size']!=[.2,.2,8]:raise InvalidArtifact('Detector spatial contract differs')
    if meta['sensor_configuration']!=dict(use_camera=False,use_lidar=True,use_radar=False):raise InvalidArtifact('Detector must be LiDAR-only')
    for key in ('config','checkpoint','code'):
        if doc['provenance'][key]!=meta[key] or meta[key]!=report[key]:raise InvalidArtifact('Detection '+key+' provenance mismatch')
    if digest(files['resolved_config.py'])!=meta['config']['resolved_sha256']:raise InvalidArtifact('Detection resolved config hash mismatch')
    comparison=report['checkpoint_comparison']
    if not comparison['compatible'] or not comparison['strict_load_succeeded'] or any(comparison[k] for k in ('missing_keys','unexpected_keys','shape_mismatches')):raise InvalidArtifact('Detection checkpoint not strictly compatible')
    if report['metrics'].get('mAP') is not None or report['metrics'].get('NDS') is not None or report['metrics']['status']!='pending_dataset_evaluation':raise InvalidArtifact('Single-frame detection metrics must remain pending')
    if meta['metrics']!=report['metrics'] or report['precision']!='float32' or report['autocast'] is not False:raise InvalidArtifact('Detection metric/precision disagreement')
    boxes=doc['boxes']
    if not isinstance(boxes,list) or len(boxes)>500 or len(boxes)!=report['box_count']:raise InvalidArtifact('Detection box count invalid')
    for box in boxes:
        for key,length in [('center',3),('dimensions',3),('velocity',2)]:
            a=np.asarray(box[key]);
            if a.shape!=(length,) or a.dtype.kind not in 'fiu' or not np.isfinite(a).all():raise InvalidArtifact('Invalid detection '+key)
        if any(x<=0 for x in box['dimensions']) or not np.isfinite(box['yaw']) or not np.isfinite(box['score']) or not 0<=box['score']<=1 or type(box['label']) is not int or not 0<=box['label']<10:
            raise InvalidArtifact('Invalid detection dimensions/yaw/score/label')
    if doc['postprocessing']['nms_type']!='circle' or doc['postprocessing']['display_filter_applied'] is not False:raise InvalidArtifact('Detection postprocessing protocol mismatch')
    SensorCatalog(settings).lidar(meta,meta['bounds'])
    return meta,report,doc

def validate_detection(folder,settings,files=None):
    try:return _validate_detection(folder,settings,files)
    except (TypeError,IndexError,OverflowError) as e:raise InvalidArtifact('Malformed detection schema: '+str(e)) from e

def pair(meta,occupancy):
    for key in ('sample_token','timestamp_us','reference_timestamp_us'):
        if meta[key]!=occupancy[key]:raise InvalidArtifact('Detection/occupancy '+key+' mismatch')
    if meta['dataset']['version']!=occupancy['dataset']['version']:raise InvalidArtifact('Detection/occupancy dataset mismatch')
    for key in ('name','unit','lidar_to_ego','lidar_to_global'):
        a,b=meta['coordinate_system'][key],occupancy['coordinate_system'][key]
        if (a!=b if isinstance(a,str) else not np.allclose(a,b,rtol=0,atol=1e-6)):raise InvalidArtifact('Detection/occupancy coordinate mismatch')
    if meta['sensors']['keyframes']['LIDAR_TOP']['sample_data_token']!=occupancy['sensors']['keyframes']['LIDAR_TOP']['sample_data_token']:raise InvalidArtifact('Detection/occupancy LiDAR identity mismatch')

class DetectionStore:
    def __init__(self,settings,store):self.settings,self.store=settings,store

    def list(self):
        items=[]
        if not self.settings.runs_root.exists():return items
        for folder in sorted(self.settings.runs_root.iterdir()):
            if not folder.is_dir() or folder.name.startswith('.'):continue
            try:
                report=decode(contained(folder/'report.json',self.settings.runs_root).read_bytes())
                if report.get('task')=='detection' and report['status']=='success':
                    items.append(dict(run_id=folder.name,sample_token=report['sample_token'],scene_name=report['scene_name'],frame_index=report['frame_index'],box_count=report['box_count']))
            except (OSError,ValueError,KeyError):continue
        return items

    def get(self,identifier,occupancy=None):
        folder=self.store.folder(identifier);meta,report,doc=validate_detection(folder,self.settings)
        if occupancy is not None:pair(meta,occupancy)
        lidar,info=self.store.catalog.lidar(meta,meta['bounds'])
        return dict(run_id=identifier,task='detection',fixture=bool(report.get('fixture',False)),prediction_available=True,replay=True,scene=meta['scene'],frame_index=meta['frame_index'],
            sample_token=meta['sample_token'],timestamp_us=meta['timestamp_us'],reference_timestamp_us=meta['reference_timestamp_us'],
            bounds=meta['bounds'],voxel_size=meta['voxel_size'],classes=[dict(id=i,name=n,color=[255,210,60],count=sum(b['label']==i for b in doc['boxes']),background=False) for i,n in enumerate(CLASSES)],
            boxes=doc['boxes'],box_count=len(doc['boxes']),box_convention=CONVENTION,source=meta['model_source'],precision=report['precision'],checkpoint=meta['checkpoint'],
            timings_s=report['timings_s'],allocator=report['vram'],metrics=report['metrics'],ground_truth=meta['ground_truth'],
            artifacts={n:dict(sha256=digest((folder/n).read_bytes()),bytes=(folder/n).stat().st_size) for n in DETECTION_FILES},
            lidar=dict(**info,url=f'/api/detections/{identifier}/lidar'),evaluation=dict(status='pending_dataset_evaluation',matched=None,candidates=[],reason='Detection dataset evaluation pending; single frame is insufficient.'),
            review_note='Detection is a separate revision/snapshot; occupancy approval/export does not include these boxes.')
