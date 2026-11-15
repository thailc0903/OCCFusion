"""Registered source pixels, explicit privacy reviews, saved prediction projection.

Original dataset/inference files are read-only. All registrations, privacy events,
full-size derivatives and export snapshots live in the configured web state.
"""
from contextlib import contextmanager
import fcntl
from io import BytesIO
import json
from pathlib import Path
import re
import secrets

import numpy as np
from PIL import Image, ImageDraw
from fastapi import HTTPException

from .auth import require_role
from .benchmarks import CollectionSnapshot, IDENTIFIER, TOKEN, fingerprint, record
from .camera_projection import CHANNELS, PROTOCOL, project_box, project_points
from .camera_sensors import camera_record, image_size, require
from .database import utc_now
from .review import canonical
from .store import contained, InvalidArtifact, read_json


def json_bytes(value):
    return (canonical(value)+'\n').encode()


def camera_json(content):
    def reject(value):
        raise InvalidArtifact('Nonfinite camera provenance JSON: '+value)
    value=json.loads(content,parse_constant=reject)
    require(isinstance(value,dict),'Camera provenance/policy/review must be a JSON object')
    return value


def make_derivative(content,masks):
    with Image.open(BytesIO(content)) as original:
        image=original.convert('RGB')
    # convert() copies image.info; strip it explicitly instead of relying on
    # the current release JPEGs having no EXIF/ICC metadata.
    image.info.clear()
    draw=ImageDraw.Draw(image)
    for x0,y0,x1,y1 in masks:draw.rectangle((x0,y0,x1-1,y1-1),fill=(0,0,0))
    buffer=BytesIO();image.save(buffer,format='PNG')
    return buffer.getvalue()


def atomic(path,content,immutable=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    if immutable and path.exists():
        require(path.read_bytes()==content,'Immutable camera evidence changed')
        return
    temporary=path.with_suffix('.'+secrets.token_hex(12)+'.tmp')
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


class CameraStore:
    def __init__(self,settings,runs,reviews,benchmarks):
        self.settings,self.runs,self.reviews,self.benchmarks=settings,runs,reviews,benchmarks
        self.root=settings.state_directory/'cameras'
        # CollectionReviews exports only immutable, verified snapshot copies.
        self.roots=(self.root/'bundles',)

    @staticmethod
    def identifier(kind,identifier,token=None):
        require(bool(IDENTIFIER.fullmatch(identifier)),'Invalid camera scope ID')
        if kind=='run':return 'camera-run-'+identifier
        require(kind=='benchmark' and token is not None and bool(TOKEN.fullmatch(token)),'Invalid camera benchmark sample')
        return 'camera-frame-'+identifier+'-'+token

    @staticmethod
    def scope(identifier):
        if identifier.startswith('camera-run-'):
            name=identifier[len('camera-run-'):]
            require(bool(IDENTIFIER.fullmatch(name)),'Invalid camera run scope')
            return 'run',name,None
        require(identifier.startswith('camera-frame-'),'Invalid camera collection ID')
        name,token=identifier[len('camera-frame-'):].rsplit('-',1)
        require(bool(IDENTIFIER.fullmatch(name)) and bool(TOKEN.fullmatch(token)),'Invalid camera frame scope')
        return 'benchmark',name,token

    def folder(self,identifier):
        self.scope(identifier)
        return contained(self.root/'bundles'/identifier,self.root)

    def source(self,identifier):
        kind,name,token=self.scope(identifier)
        if kind=='run':
            snapshot=self.reviews.snapshot(name)
            meta=snapshot.metadata
            artifacts={n:(self.runs.artifact(self.runs.folder(name),n),record(data)) for n,data in snapshot.files.items()}
            boxes=json.loads(snapshot.files['detections.json'])['boxes'] if 'detections.json' in snapshot.files else []
        else:
            detail,loaded=self.benchmarks.frame(name,token)
            require(detail['prediction_available'],'Camera scope has no saved prediction')
            meta=loaded.metadata;artifacts=loaded.camera_artifacts
            boxes=detail.get('boxes',[])
        # Read exact hash-checked prediction bytes, not a second unchecked copy.
        files={}
        for filename,(path,expected) in artifacts.items():
            data=path.read_bytes()
            require(record(data)==expected,'Camera parent prediction artifact changed: '+filename)
            files[filename]=data
        frame=self.runs.catalog.camera_frame(meta)
        return dict(identifier=identifier,kind=kind,name=name,token=meta['sample_token'],metadata=meta,frame=frame,
            prediction_files=files,prediction_hashes={n:record(data) for n,data in files.items()},
            prediction_fingerprint=fingerprint({n:record(data) for n,data in files.items()}),boxes=boxes)

    def policy(self,version):
        data=self.settings.camera_policy_path.read_bytes();value=camera_json(data)
        require(value['schema_version']==1 and value['id']=='nuscenes-release-publisher-privacy-v1'
            and value['dataset_version']==version and value['human_review_default']=='not_reviewed'
            and value['does_not_guarantee_all_pii_removed'] is True,'Camera source policy missing or unsupported')
        return value,data

    def pin_path(self,sd_token):
        require(bool(TOKEN.fullmatch(sd_token)),'Invalid camera sample_data token')
        return contained(self.root/'sources'/(sd_token+'.json'),self.root)

    def register_sources(self,identifier):
        source=self.source(identifier);frame=source['frame']
        policy,policy_content=self.policy(source['metadata']['dataset']['version'])
        registered=[]
        for channel in CHANNELS:
            camera=camera_record(frame,channel);evidence=camera['evidence']
            path=self.pin_path(evidence['sample_data_token'])
            expected=dict(evidence=evidence,policy=policy,policy_file=record(policy_content))
            if path.exists():
                old=read_json(path)
                require(all(old[k]==v for k,v in expected.items()),channel+': registered source changed; not overwritten')
                require(fingerprint({k:v for k,v in old.items() if k!='fingerprint_sha256'})==old['fingerprint_sha256'],'Camera source registration fingerprint mismatch')
            else:
                core=dict(schema_version=1,**expected,registered_at=utc_now(),
                    registered_by='local source registration CLI; not an Engineer privacy inspection')
                atomic(path,json_bytes(dict(core,fingerprint_sha256=fingerprint(core))),immutable=True)
            registered.append(dict(channel=channel,sample_data_token=evidence['sample_data_token'],source_image=evidence['source_image']))
        return registered

    def registered_camera(self,source,channel):
        camera=camera_record(source['frame'],channel);evidence=camera['evidence']
        policy,content=self.policy(evidence['dataset_version'])
        path=self.pin_path(evidence['sample_data_token'])
        if not path.is_file():raise InvalidArtifact('source_provenance_missing: register this exact release image before serving it')
        pin_content=path.read_bytes();pin=camera_json(pin_content)
        require(pin['evidence']==evidence and pin['policy']==policy and pin['policy_file']==record(content),
            'Registered source image/calibration/metadata/policy hash mismatch')
        require(fingerprint({k:v for k,v in pin.items() if k!='fingerprint_sha256'})==pin['fingerprint_sha256'],
            'Camera source registration fingerprint mismatch')
        camera.update(pin=pin,pin_content=pin_content,policy_content=content)
        return camera

    def privacy(self,camera):
        evidence=camera['evidence'];token=evidence['sample_data_token']
        directory=contained(self.root/'privacy'/token,self.root)
        current=directory/'current.json'
        if not current.exists():return dict(status='not_reviewed',revision=None,masks=[],reviewed_by=None,derivative=None)
        pointer=read_json(contained(current,self.root));revision=pointer['revision']
        require(bool(re.fullmatch(r'[0-9a-f]{64}',revision)),'Invalid privacy revision pointer')
        path=contained(directory/'events'/(revision+'.json'),self.root)
        content=path.read_bytes();value=camera_json(content)
        require(fingerprint({k:v for k,v in value.items() if k!='fingerprint_sha256'})==revision==value['fingerprint_sha256'],
            'Privacy review event hash mismatch')
        require(value['camera_fingerprint']==camera['pin']['fingerprint_sha256'] and value['source_image']==evidence['source_image']
            and value['sample_data_token']==token and value['status']=='engineer_reviewed',
            'Privacy review no longer matches registered camera source')
        derivative_path=contained(self.root/'derivatives'/(value['derivative']['sha256']+'.png'),self.root)
        image=derivative_path.read_bytes()
        require(record(image)==value['derivative'] and image_size(image)==(evidence['width'],evidence['height']),
            'Privacy derivative hash/dimensions mismatch')
        return dict(value,revision=revision,event_content=content,derivative_path=derivative_path,content=image)

    @contextmanager
    def privacy_lock(self,token):
        path=contained(self.root/'privacy'/token/'write.lock',self.root);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:yield
            finally:fcntl.flock(lock,fcntl.LOCK_UN)

    def review_image(self,identifier,channel,body,user):
        require_role(user,'engineer')
        require(body.reviewed is True,'An explicit inspection acknowledgement is required for this image')
        source=self.source(identifier);camera=self.registered_camera(source,channel)
        evidence=camera['evidence'];w,h=evidence['width'],evidence['height']
        require(body.camera_fingerprint==camera['pin']['fingerprint_sha256'],'Stale source camera fingerprint')
        masks=[]
        for mask in body.masks:
            x0,y0,x1,y1=mask
            require(all(type(x) is int for x in mask) and 0<=x0<x1<=w and 0<=y0<y1<=h,
                'Masks must be nonempty integer rectangles in original-image pixels')
            masks.append([x0,y0,x1,y1])
        token=evidence['sample_data_token']
        with self.privacy_lock(token):
            previous=self.privacy(camera)
            if body.previous_revision!=previous['revision']:raise HTTPException(409,'Privacy state changed; reload before saving')
            content=make_derivative(camera['content'],masks)
            derivative=record(content)
            atomic(contained(self.root/'derivatives'/(derivative['sha256']+'.png'),self.root),content,immutable=True)
            core=dict(schema_version=1,status='engineer_reviewed',sample_token=evidence['sample_token'],sample_data_token=token,
                channel=channel,camera_fingerprint=camera['pin']['fingerprint_sha256'],source_image=evidence['source_image'],
                policy_sha256=camera['pin']['policy_file']['sha256'],masks=masks,mask_coordinates='original pixels, [x0,y0,x1,y1), solid black',
                derivative=derivative,width=w,height=h,processing='RGB PNG re-encoding strips source metadata; optional explicit supplemental black masks',
                reviewed_by=dict(id=user['user_id'],username=user['username'],role=user['role']),reviewed_at=utc_now(),
                comment=body.comment,previous_revision=previous['revision'],all_pii_removed_guaranteed=False)
            revision=fingerprint(core);value=dict(core,fingerprint_sha256=revision)
            atomic(contained(self.root/'privacy'/token/'events'/(revision+'.json'),self.root),json_bytes(value),immutable=True)
            atomic(contained(self.root/'privacy'/token/'current.json',self.root),json_bytes(dict(revision=revision)))
        return dict(status='engineer_reviewed',revision=revision,derivative=derivative)

    def channel(self,source,channel,base):
        camera=self.registered_camera(source,channel);privacy=self.privacy(camera);evidence=camera['evidence']
        display=privacy['derivative'] or evidence['source_image']
        version=fingerprint(dict(camera=camera['pin']['fingerprint_sha256'],privacy=privacy['revision'],display=display))
        return camera,privacy,dict(evidence,status='available',reason=None,camera_fingerprint=camera['pin']['fingerprint_sha256'],
            image_fingerprint=version,image_url=base+'/'+channel+'/image?version='+version,
            projection_url=base+'/'+channel+'/projection?version='+version,
            privacy=dict(status=privacy['status'],revision=privacy['revision'],masks=privacy['masks'],
                reviewed_by=privacy['reviewed_by'],reviewed_at=privacy.get('reviewed_at'),
                source='publisher release processing, not automatic application PII detection',
                source_policy=camera['pin']['policy'],derivative=privacy['derivative']),
            export_eligible=privacy['status']=='engineer_reviewed')

    def listing(self,identifier,base):
        source=self.source(identifier);rows=[]
        for channel in CHANNELS:
            try:rows.append(self.channel(source,channel,base)[2])
            except (OSError,ValueError,KeyError) as error:
                rows.append(dict(channel=channel,status='blocked',reason=str(error),image_url=None,
                    privacy=dict(status='not_verified',revision=None),export_eligible=False))
        return dict(scope_id=identifier,sample_token=source['token'],scene=source['metadata']['scene'],frame_index=source['metadata']['frame_index'],
            channels=rows,projection_protocol=PROTOCOL,prediction_box_count=len(source['boxes']),replay=True,
            privacy_note='Publisher release provenance is separate from individual Engineer inspection; no guarantee that every PII region was removed.')

    def image(self,identifier,channel,version,base):
        source=self.source(identifier)
        camera,privacy,detail=self.channel(source,channel,base)
        if version!=detail['image_fingerprint']:raise HTTPException(409,'Camera image/privacy version changed; reload the frame')
        return (privacy.get('content') or camera['content']),('image/png' if privacy['derivative'] else 'image/jpeg')

    def projection(self,source,camera):
        evidence=camera['evidence'];matrix=evidence['camera_from_lidar'];intrinsic=evidence['intrinsic']
        width,height=evidence['width'],evidence['height']
        points=project_points(source['frame']['points'],matrix,intrinsic,width,height)
        boxes=[dict(index=i,**project_box(box,matrix,intrinsic,width,height)) for i,box in enumerate(source['boxes'])]
        core=dict(schema_version=1,sample_token=source['token'],sample_data_token=evidence['sample_data_token'],
            channel=evidence['channel'],width=width,height=height,camera_fingerprint=camera['pin']['fingerprint_sha256'],
            prediction_fingerprint=source['prediction_fingerprint'],prediction_artifacts=source['prediction_hashes'],
            lidar_source=evidence['lidar_source'],protocol=PROTOCOL,
            implementation_sha256=record(Path(__file__).with_name('camera_projection.py').read_bytes())['sha256'],
            fields=['lidar_point_index','u','v','camera_depth_m'],point_count=len(points),
            points=points.tolist(),box_count=len(boxes),boxes=boxes,box_source='saved prediction only',
            class_names=source['metadata']['class_names'],camera_from_lidar=matrix,intrinsic=intrinsic,
            camera_timestamp_us=evidence['camera_timestamp_us'],lidar_timestamp_us=evidence['lidar_timestamp_us'])
        return dict(core,fingerprint_sha256=fingerprint(core))

    def project(self,identifier,channel,version,base):
        source=self.source(identifier);camera,_,detail=self.channel(source,channel,base)
        if version!=detail['image_fingerprint']:raise HTTPException(409,'Camera image/privacy version changed; reload the frame')
        return self.projection(source,camera)

    def snapshot(self,identifier):
        source=self.source(identifier);files=dict(source=source['prediction_files'])
        content={('prediction/'+n):data for n,data in files['source'].items()}
        channels=[]
        for channel in CHANNELS:
            camera=self.registered_camera(source,channel);privacy=self.privacy(camera)
            require(privacy['status']=='engineer_reviewed',channel+': camera export requires explicit individual Engineer review')
            content['cameras/'+channel+'/image.png']=privacy['content']
            content['cameras/'+channel+'/source_provenance.json']=camera['pin_content']
            content['cameras/'+channel+'/privacy_review.json']=privacy['event_content']
            content['cameras/'+channel+'/projection.json']=json_bytes(self.projection(source,camera))
            channels.append(dict(channel=channel,sample_data_token=camera['evidence']['sample_data_token'],
                source_image=camera['evidence']['source_image'],derivative=privacy['derivative'],privacy_revision=privacy['revision']))
        content['camera_policy.json']=camera['policy_content']
        core=dict(schema_version=1,kind='rav25_camera_frame_collection',task='camera',collection_id=identifier,
            sample_tokens=[source['token']],sample_token=source['token'],scope=dict(kind=source['kind'],id=source['name']),
            status='complete',expected_frame_count=1,successful_frame_count=1,failed_frame_count=0,channel_count=len(CHANNELS),
            channels=channels,projection_protocol=PROTOCOL,metric_unit=None,metric_source=None,latency_source=None,
            prediction_fingerprint=source['prediction_fingerprint'],artifacts={n:record(data) for n,data in content.items()},
            review_scope='Separate camera derivative/projection snapshot; no extension of existing prediction/benchmark approvals',
            source_images_included=False,playback_note='Saved images/predictions; ego-motion compensation does not compensate object motion; not learned 4D flow')
        manifest=dict(core,fingerprint_sha256=fingerprint(core))
        paths={}
        directory=self.folder(identifier)/manifest['fingerprint_sha256']
        for name,data in content.items():
            path=contained(directory/name,self.roots[0]);atomic(path,data,immutable=True);paths[name]=path
        detail=dict(benchmark_id=identifier,task='camera',status='complete',expected_frame_count=1,successful_frame_count=1,
            metrics={},metric_source_export=None,latency_source_export=None,metric_unit=None,playback_note=core['playback_note'])
        return CollectionSnapshot(detail,manifest,paths)
