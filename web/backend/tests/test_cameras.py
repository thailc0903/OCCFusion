"""Camera/provenance/projection tests. Isolated copied pixels and SQLite only."""
from contextlib import contextmanager
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import zipfile

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from web.backend.app.config import Settings, REPO
from web.backend.app.main import create_app
from web.backend.app.benchmarks import fingerprint,record
from web.backend.app.camera_projection import project_points,project_box,box_corners,camera_from_lidar,clip_image
from web.backend.app.camera_sensors import TABLES
from web.backend.tests.auth_support import ensure_user,login

SAMPLE='3950bd41f74548429c0f7700ff3d8269'
BASE='/api/runs/detection/cameras'
DETECTION=REPO/'work/centerpoint-smoke-wyp1nl__/runs/infer_job_65e1b30833e343c3b434da8dc71423f9'


def transition(client,path,info,decision=None):
    body=dict(comment='Explicit QA camera snapshot',revision_id=info['revision_id'],fingerprint=info['fingerprint'])
    if decision:body['decision']=decision
    response=client.post(path,json=body);assert response.status_code==200,response.text
    return response.json()


@contextmanager
def changed(path,content):
    previous=path.read_bytes();path.write_bytes(content)
    try:yield
    finally:path.write_bytes(previous)


def legacy_rows(app):
    with app.state.database.connect() as c:
        return {t:[tuple(r) for r in c.execute('SELECT * FROM '+t+' ORDER BY 1')]
            for t in ('users','app_settings','runs','revisions','review_events','inference_jobs')}


@pytest.fixture(scope='module')
def corpus(tmp_path_factory):
    root=tmp_path_factory.mktemp('camera-web');runs=root/'runs';data=root/'data';version=data/'v1.0-mini'
    version.mkdir(parents=True);runs.mkdir()
    shutil.copytree(DETECTION,runs/'detection')
    shutil.copytree(REPO/'work_dirs/infer_mini_val_scene0103_f1',runs/'occupancy')
    for name in TABLES:shutil.copy2(REPO/'data/nuscenes/v1.0-mini'/(name+'.json'),version/(name+'.json'))
    tables={name:json.loads((version/(name+'.json')).read_text()) for name in TABLES}
    for sd in tables['sample_data']:
        if sd['sample_token']==SAMPLE and sd['is_key_frame']:
            path=Path(sd['filename'])
            if path.parts[1] not in ('LIDAR_TOP','CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT'):continue
            target=data/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/'data/nuscenes'/path,target)
    policy=root/'policy.json';shutil.copy2(REPO/'web/camera_policy.json',policy)
    settings=Settings(runs,(data,),state_root=root/'state',camera_policy_path=policy)
    app=create_app(settings);ensure_user(app);ensure_user(app,'test_reviewer','reviewer')
    with TestClient(app) as engineer,TestClient(app) as reviewer:
        login(engineer);login(reviewer,'test_reviewer')
        old=engineer.post('/api/runs/detection/register',json=dict(comment='Independent original detection approval')).json()
        old=transition(engineer,'/api/runs/detection/review/submit',old)
        old=transition(reviewer,'/api/runs/detection/review/decision',old,'approved')
        scope=app.state.cameras.identifier('run','detection')
        app.state.cameras.register_sources(scope)
        yield SimpleNamespace(app=app,engineer=engineer,reviewer=reviewer,root=root,runs=runs,data=data,
            settings=settings,scope=scope,legacy=legacy_rows(app),old=old)


@pytest.fixture
def system(corpus):
    s=corpus
    with s.app.state.database.transaction() as c:
        c.execute('DELETE FROM collection_events');c.execute('UPDATE collections SET current_revision_id=NULL')
        c.execute('DELETE FROM collection_revisions');c.execute('DELETE FROM collections')
    for folder in ('privacy','derivatives','bundles'):
        shutil.rmtree(s.settings.state_directory/'cameras'/folder,ignore_errors=True)
    yield s
    assert legacy_rows(s.app)==s.legacy
    assert not any(n.split('.')[0] in ('torch','mmcv','mmdet','mmdet3d','occfusion') for n in sys.modules)


def listing(s):
    r=s.engineer.get(BASE);assert r.status_code==200,r.text
    return r.json()


def inspect(s,camera,masks=None):
    r=s.engineer.post(BASE+'/'+camera['channel']+'/privacy',json=dict(camera_fingerprint=camera['camera_fingerprint'],
        previous_revision=camera['privacy']['revision'],reviewed=True,masks=masks or [],comment='QA individual image inspection; no claim of exhaustive PII detection'))
    assert r.status_code==200,r.text
    return r.json()


def approve(s):
    for camera in listing(s)['channels']:inspect(s,camera)
    r=s.engineer.post(BASE+'/register',json=dict(comment='Separate six-camera derivative revision'))
    assert r.status_code==200,r.text
    info=transition(s.engineer,BASE+'/review/submit',r.json())
    return transition(s.reviewer,BASE+'/review/decision',info,'approved')


def test_projection_known_pixels_native_rotation_and_behind_camera():
    k=np.array([[100,0,80],[0,100,45],[0,0,1]])
    points=np.array([[0,0,10],[1,2,10],[0,0,-10],[0,0,.5],[100,0,10]])
    result=project_points(points,np.eye(4),k,160,90)
    np.testing.assert_allclose(result,[[0,80,45,10],[1,90,65,10]])
    q=[np.sqrt(.5),0,0,np.sqrt(.5)]
    transform=camera_from_lidar(dict(rotation=q,translation=[1,0,0]),dict(rotation=[1,0,0,0],translation=[5,2,0]),
        dict(rotation=[1,0,0,0],translation=[0,0,0]),dict(rotation=[1,0,0,0],translation=[4,2,0]))
    point=(transform@np.array([1,0,10,1]))[:3]
    np.testing.assert_allclose(point,[2,1,10],atol=1e-14)
    assert not np.allclose(transform,np.eye(4))


def test_geometric_box_near_plane_image_edge_clipping():
    box=dict(center=[10,20,3],dimensions=[4,2,6],yaw=np.pi/2,label=0,score=.9)
    corners=box_corners(box)
    np.testing.assert_allclose(corners.min(axis=0),[9,18,0]);np.testing.assert_allclose(corners.max(axis=0),[11,22,6])
    k=np.array([[50,0,80],[0,50,45],[0,0,1]])
    box.update(center=[0,0,.1],dimensions=[2,2,2],yaw=.2)
    value=project_box(box,np.eye(4),k,160,90)
    assert value['segments']
    segments=np.asarray(value['segments']).reshape(-1,2)
    assert np.isfinite(segments).all() and (segments>=0).all() and (segments<=[160,90]).all()
    box['center']=[0,0,-5];assert project_box(box,np.eye(4),k,160,90)['segments']==[]
    np.testing.assert_allclose(clip_image(np.array([-5.,5]),np.array([15.,5]),10,10),[[0,5],[10,5]])


def test_derivative_strips_metadata_even_if_source_has_exif():
    from web.backend.app.cameras import make_derivative
    original=Image.new('RGB',(16,12),color=(120,80,40))
    exif=Image.Exif();exif[315]='Synthetic unit-test metadata, not an actual identity';exif[274]=1
    buffer=BytesIO();original.save(buffer,format='JPEG',exif=exif,icc_profile=b'unit-test-profile')
    source=buffer.getvalue()
    with Image.open(BytesIO(source)) as image:assert image.getexif() and image.info.get('icc_profile')
    content=make_derivative(source,[[2,3,6,8]])
    with Image.open(BytesIO(content)) as image:
        assert image.size==(16,12) and not image.getexif()
        assert 'exif' not in image.info and 'icc_profile' not in image.info
        assert (np.asarray(image)[3:8,2:6]==0).all()


def test_real_six_images_projection_timestamps_and_no_automatic_human_review(system):
    s=system;value=listing(s);assert value['sample_token']==SAMPLE and len(value['channels'])==6
    assert value['prediction_box_count']==151
    for c in value['channels']:
        assert c['status']=='available',c
        assert (c['width'],c['height'])==(1600,900) and c['privacy']['status']=='not_reviewed'
        assert c['privacy']['reviewed_by'] is None and not c['export_eligible']
        assert c['camera_minus_lidar_ms']==(c['camera_timestamp_us']-c['lidar_timestamp_us'])/1000
        image=s.engineer.get(c['image_url']);assert image.status_code==200
        assert record(image.content)==c['source_image']
    c=value['channels'][0];r=s.engineer.get(c['projection_url']);assert r.status_code==200,r.text
    p=r.json();assert p['sample_token']==SAMPLE and p['sample_data_token']==c['sample_data_token']
    assert p['box_count']==151 and p['box_source']=='saved prediction only' and p['protocol']['precision']=='float64'
    assert p['point_count']>0 and np.asarray(p['points']).shape==(p['point_count'],4)
    assert s.engineer.get('/api/health').json()['model_modules_loaded'] is False


def test_source_missing_policy_and_registration_block_pixels(system):
    s=system;c=listing(s)['channels'][0];path=s.app.state.cameras.pin_path(c['sample_data_token'])
    previous=path.read_bytes();path.unlink()
    try:
        value=listing(s)['channels'][0];assert value['status']=='blocked' and 'source_provenance_missing' in value['reason']
        assert s.engineer.get(c['image_url']).status_code==422
    finally:path.write_bytes(previous)
    with changed(s.settings.camera_policy_path,b'{}'):
        assert all(c['status']=='blocked' for c in listing(s)['channels'])
        assert s.engineer.get(c['image_url']).status_code==422


@pytest.mark.parametrize('target',['source','policy','privacy'])
def test_malformed_provenance_has_explicit_blocked_reason(system,target):
    s=system;c=listing(s)['channels'][0]
    if target=='source':path=s.app.state.cameras.pin_path(c['sample_data_token'])
    elif target=='policy':path=s.settings.camera_policy_path
    else:
        event=inspect(s,c)
        path=s.settings.state_directory/'cameras/privacy'/c['sample_data_token']/'events'/(event['revision']+'.json')
    with changed(path,b'[]'):
        front=listing(s)['channels'][0]
        assert front['status']=='blocked' and 'must be a JSON object' in front['reason']
        assert s.engineer.get(c['image_url']).status_code==422


@pytest.mark.parametrize('field',['token','intrinsic','extrinsic','size'])
def test_inference_camera_token_calibration_dimensions_mismatch(system,field):
    s=system;path=s.runs/'occupancy/metadata.json';value=json.loads(path.read_bytes())
    c=value['sensors']['keyframes']['CAM_FRONT']
    if field=='token':c['sample_data_token']='0'*32
    elif field=='intrinsic':c['intrinsic'][0][0]+=2
    elif field=='extrinsic':c['sensor_to_ego'][0][3]+=1
    else:c['width']+=1
    with changed(path,json.dumps(value).encode()):
        r=s.engineer.get('/api/runs/occupancy/cameras');assert r.status_code==200,r.text
        front=next(c for c in r.json()['channels'] if c['channel']=='CAM_FRONT')
        assert front['status']=='blocked' and not front['image_url']


def test_source_hash_missing_image_and_outside_root_are_not_served(system,tmp_path):
    s=system;c=listing(s)['channels'][0];pin=json.loads(s.app.state.cameras.pin_path(c['sample_data_token']).read_text())
    path=s.data/pin['evidence']['filename']
    with changed(path,path.read_bytes()+b' '):
        assert listing(s)['channels'][0]['status']=='blocked'
        assert s.engineer.get(c['image_url']).status_code==422
    previous=path.read_bytes();path.unlink()
    try:
        assert 'No such file' in listing(s)['channels'][0]['reason']
        assert s.engineer.get(c['image_url']).status_code==404
        outside=tmp_path/'same.jpg';outside.write_bytes(previous);path.symlink_to(outside)
        assert s.engineer.get(c['image_url']).status_code==403
    finally:path.unlink(missing_ok=True);path.write_bytes(previous)


def test_auth_role_csrf_and_no_approval_expansion(system):
    s=system;c=listing(s)['channels'][0]
    with TestClient(s.app) as anon:
        assert anon.get(BASE).status_code==401 and anon.get(c['image_url']).status_code==401
    assert s.reviewer.get('/api/runs/occupancy/cameras').status_code==403
    assert s.reviewer.post(BASE+'/'+c['channel']+'/privacy',json={}).status_code in (403,422)
    body=dict(camera_fingerprint=c['camera_fingerprint'],previous_revision=None,reviewed=True,masks=[],comment='Bad CSRF')
    assert s.engineer.post(BASE+'/'+c['channel']+'/privacy',json=body,headers={'X-CSRF-Token':'wrong'}).status_code==403
    assert s.engineer.post(BASE+'/review/decision',json=dict(comment='Denied',decision='approved',revision_id=1,fingerprint='0'*64)).status_code==403
    assert s.engineer.post(BASE+'/register',json=dict(comment='Unreviewed images')).status_code==409
    assert s.engineer.post(BASE+'/export').status_code==409
    assert s.engineer.get('/api/runs/detection/review').json()['approval_valid'] is True


@pytest.mark.parametrize('masks,reviewed', [([[0,0,0,20]],True),([[-1,0,10,10]],True),
    ([[0,0,1601,10]],True),([[1.5,0,10,10]],True),([],False)])
def test_invalid_masks_or_missing_inspection_ack_refused(system,masks,reviewed):
    s=system;c=listing(s)['channels'][0]
    r=s.engineer.post(BASE+'/'+c['channel']+'/privacy',json=dict(camera_fingerprint=c['camera_fingerprint'],
        previous_revision=None,reviewed=reviewed,masks=masks,comment='Invalid input'))
    assert r.status_code==422,r.text
    assert listing(s)['channels'][0]['privacy']['status']=='not_reviewed'


def test_full_size_derivative_mask_source_hash_and_stale_version(system):
    s=system;c=listing(s)['channels'][0];source=s.engineer.get(c['image_url']).content
    inspect(s,c,[[20,30,40,50]])
    updated=listing(s)['channels'][0]
    assert updated['source_image']==c['source_image'] and updated['privacy']['status']=='engineer_reviewed'
    assert s.engineer.get(c['image_url']).status_code==409
    assert s.engineer.get(c['projection_url']).status_code==409
    image=s.engineer.get(updated['image_url']);assert image.status_code==200 and image.headers['content-type']=='image/png'
    with Image.open(BytesIO(image.content)) as decoded:
        assert decoded.size==(1600,900) and not decoded.getexif()
        assert (np.asarray(decoded)[30:50,20:40]==0).all()
    assert record(source)==updated['source_image']
    r=s.engineer.post(BASE+'/'+c['channel']+'/privacy',json=dict(camera_fingerprint=c['camera_fingerprint'],
        previous_revision=None,reviewed=True,masks=[],comment='Stale write'))
    assert r.status_code==409


def test_separate_revision_export_hashes_restart_and_privacy_change(system):
    s=system;info=approve(s)
    assert info['manifest']['kind']=='rav25_camera_frame_collection'
    assert s.engineer.get('/api/runs/detection/review').json()['fingerprint']==s.old['fingerprint']
    r=s.engineer.post(BASE+'/export');assert r.status_code==200,r.text
    with zipfile.ZipFile(BytesIO(r.content)) as z:
        manifest=json.loads(z.read('manifest.json'));review=json.loads(z.read('review.json'))
        assert review['kind']=='camera_frame_collection_review' and review['metrics'] is None
        assert manifest['source_images_included'] is False and len(manifest['channels'])==6
        assert not any(n.endswith('.jpg') for n in z.namelist())
        for name,expected in manifest['artifacts'].items():assert record(z.read(name))==expected
        for c in manifest['channels']:
            assert record(z.read('cameras/'+c['channel']+'/image.png'))==c['derivative']
    restarted=create_app(s.settings)
    with TestClient(restarted) as client:
        login(client);assert client.get(BASE+'/review').json()['approval_valid']
    c=listing(s)['channels'][0];inspect(s,c,[[1,1,9,9]])
    assert not s.engineer.get(BASE+'/review').json()['approval_valid']
    assert s.engineer.post(BASE+'/export').status_code==409
    revised=transition(s.engineer,BASE+'/review/revise',info)
    assert revised['state']=='draft' and revised['revision_number']==2
    assert revised['revisions'][0]['state']=='approved'
    assert s.engineer.post(BASE+'/export').status_code==409


@pytest.mark.parametrize('target',['derivative','privacy_event','prediction','projection_snapshot'])
def test_tampering_blocks_export_and_metric_approval_does_not_expand(system,target):
    s=system;info=approve(s);c=listing(s)['channels'][0]
    root=s.settings.state_directory/'cameras'
    if target=='derivative':path=root/'derivatives'/(c['privacy']['derivative']['sha256']+'.png')
    elif target=='privacy_event':path=root/'privacy'/c['sample_data_token']/'events'/(c['privacy']['revision']+'.json')
    elif target=='prediction':path=s.runs/'detection/detections.json'
    else:path=next((root/'bundles'/s.scope/info['fingerprint']).glob('cameras/*/projection.json'))
    with changed(path,path.read_bytes()+b' '):
        # Semantically identical whitespace also changes the approved bytes.
        assert s.engineer.post(BASE+'/export').status_code==409


def test_saved_benchmark_frame_cameras_without_queue_receipt(tmp_path):
    runs=tmp_path/'runs';name='benchmark_detection_mini_val';runs.mkdir()
    shutil.copytree(REPO/'work_dirs'/name,runs/name)
    settings=Settings(runs,(REPO/'data/nuscenes',),state_root=tmp_path/'state')
    app=create_app(settings);ensure_user(app);ensure_user(app,'test_reviewer','reviewer')
    with TestClient(app) as engineer,TestClient(app) as reviewer:
        login(engineer);login(reviewer,'test_reviewer')
        base='/api/benchmarks/'+name+'/frames/'+SAMPLE+'/cameras'
        scope=app.state.cameras.identifier('benchmark',name,SAMPLE)
        app.state.cameras.register_sources(scope)
        r=engineer.get(base);assert r.status_code==200,r.text
        assert all(c['status']=='available' for c in r.json()['channels'])
        assert reviewer.get(base).status_code==403
        assert engineer.get(base.replace(SAMPLE,'0'*32)).status_code==404
        assert not list((runs/name/'frames').glob('*/job_result.json'))
