from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import shutil
import uuid
import zipfile

from fastapi.testclient import TestClient
import numpy as np
import pytest

from web.backend.app.config import REPO,Settings
from web.backend.app.jobs import Profiles
from web.backend.app.job_worker import Worker,Commands
from web.backend.app.main import create_app
from web.backend.app.detections import validate_detection,pair
from web.backend.tests.auth_support import ensure_user,login
from web.backend.tests.detection_support import FixtureDetectionCommands,make_fixture

@pytest.fixture
def detection(tmp_path):
    runs=tmp_path/'runs';runs.mkdir()
    shutil.copytree(REPO/'work_dirs/infer_mini_val_scene0103_f1',runs/'occupancy')
    settings=Settings(runs,(REPO/'data/nuscenes',),state_root=tmp_path/'state',job_profiles_path=REPO/'web/centerpoint_profiles.json',gpu_lock_path=tmp_path/'gpu.lock')
    app=create_app(settings);ensure_user(app);ensure_user(app,'test_reviewer','reviewer')
    with TestClient(app) as eng,TestClient(app) as rev:
        login(eng);login(rev,'test_reviewer');yield app,eng,rev,settings

def create(eng):
    r=eng.post('/api/jobs',json=dict(profile_id='centerpoint_pillar02_circle',scene_name='scene-0103',frame_index=1,request_key=uuid.uuid4().hex));assert r.status_code==202,r.text
    return r.json()

def done(app,eng,settings):
    job=create(eng);Worker(app.state.database,settings,FixtureDetectionCommands()).run(once=True)
    value=eng.get('/api/jobs/'+job['id']).json();assert value['status']=='succeeded',value
    return value,app.state.jobs.paths(job['id'])['run']

def test_legacy_profile_fingerprint_and_dispatch(detection):
    app,eng,_,settings=detection
    old=Profiles(replace(settings,job_profiles_path=REPO/'web/inference_profiles.json')).get('occfusion_clr_fp32')
    current=app.state.jobs.profiles.get('occfusion_clr_fp32')
    assert old==current and 'task' not in current
    occ={'profile_json':json.dumps(current),'scene_name':'scene-0103','frame_index':1,'split':'mini_val'}
    det=app.state.jobs.profiles.get('centerpoint_pillar02_circle');job=dict(occ,profile_json=json.dumps(det))
    paths=dict(prediction=Path('/tmp/pred'),evaluation=Path('/tmp/eval'))
    assert 'infer_nuscenes_mini.py' in Commands().inference(occ,paths)[2]
    assert 'evaluate_occupancy.py' in Commands().evaluation(occ,paths)[2]
    assert 'infer_centerpoint_mini.py' in Commands().inference(job,paths)[2]
    assert Commands().evaluation(job,paths) is None

def test_detection_worker_has_own_validator_publication_and_no_occupancy_artifact(detection):
    app,eng,rev,settings=detection
    job,folder=done(app,eng,settings)
    assert job['task']=='detection' and job['evaluation_status']=='pending_dataset_evaluation' and job['inference_launch_count']==1
    assert job['result_url'].startswith('/detections?run=')
    assert not (folder/'occupancy.npz').exists() and not app.state.jobs.paths(job['id'])['eval_run'].exists()
    r=eng.get('/api/detections/'+folder.name);assert r.status_code==200,r.text
    value=r.json();assert value['box_count']==2 and value['metrics']['mAP'] is None and value['review']['state']=='unregistered'
    assert rev.get('/api/detections/'+folder.name).status_code==403
    assert folder.name not in [r['run_id'] for r in eng.get('/api/runs').json()['runs']]
    joined=eng.get('/api/runs/occupancy/detections').json();assert len(joined['matches'])==1
    assert eng.get(value['lidar']['url']).status_code==200
    assert eng.post('/api/runs/'+folder.name+'/export').status_code==409
    with app.state.database.connect() as c:assert c.execute('SELECT COUNT(*) FROM revisions').fetchone()[0]==0

@pytest.mark.parametrize('change',['token','timestamp','label','dimensions','score','config','resolved','coordinate'])
def test_bad_detection_rejected(detection,tmp_path,change):
    _,_,_,settings=detection;folder=tmp_path/'fixture';make_fixture(folder)
    path=folder/'detections.json';value=json.loads(path.read_text())
    if change=='token':value['sample_token']='0'*32
    elif change=='timestamp':value['reference_timestamp_us']+=1
    elif change=='label':value['boxes'][0]['label']=10
    elif change=='dimensions':value['boxes'][0]['dimensions'][0]=0
    elif change=='score':value['boxes'][0]['score']=float('nan')
    elif change=='config':value['provenance']['config']['sha256']='0'*64
    elif change=='resolved':(folder/'resolved_config.py').write_text('tampered')
    elif change=='coordinate':value['coordinate_system']['unit']='centimetres'
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError,KeyError)):validate_detection(folder,settings)

def test_empty_boxes_and_pairing_are_explicit(detection,tmp_path):
    _,_,_,settings=detection;folder=tmp_path/'empty';make_fixture(folder,empty=True)
    meta,report,doc=validate_detection(folder,settings);assert doc['boxes']==[] and report['box_count']==0
    occ=json.loads((settings.runs_root/'occupancy/metadata.json').read_bytes());pair(meta,occ)
    for field in ['sample_token','reference_timestamp_us']:
        wrong=dict(occ);wrong[field]='wrong' if field=='sample_token' else wrong[field]+1
        with pytest.raises(ValueError):pair(meta,wrong)

def test_detection_revision_review_export_hash_and_restart(detection):
    app,eng,rev,settings=detection;job,folder=done(app,eng,settings);base='/api/runs/'+folder.name
    r=eng.post(base+'/register',json={'comment':'QA synthetic detection snapshot'});assert r.status_code==200,r.text
    draft=r.json();assert draft['manifest']['schema_version']==3 and 'occupancy.npz' not in draft['manifest']['artifacts']
    body=dict(comment='Review exact detection hashes',revision_id=draft['revision_id'],fingerprint=draft['fingerprint'])
    assert eng.post(base+'/review/decision',json=dict(body,decision='approved')).status_code==403
    assert eng.post(base+'/review/submit',json=body).status_code==200
    assert rev.post(base+'/review/decision',json=dict(body,decision='approved')).status_code==200
    exported=eng.post(base+'/export');assert exported.status_code==200,exported.text[:100]
    with zipfile.ZipFile(BytesIO(exported.content)) as archive:
        assert json.loads(archive.read('detections.json'))['boxes'][0]['center']==[5.,10.,-1.]
        assert json.loads(archive.read('review.json'))['metrics']['mAP'] is None
    restarted=create_app(settings)
    with TestClient(restarted) as client:
        login(client);assert client.get(base+'/review').json()['approval_valid']
    path=folder/'detections.json';path.write_bytes(path.read_bytes()+b' ')
    assert eng.get('/api/detections/'+folder.name).status_code==422
    assert eng.post(base+'/export').status_code in (409,422)

def test_anonymous_and_reviewer_job_permissions(detection):
    app,eng,rev,_=detection
    with TestClient(app) as anon:
        assert anon.get('/api/detections').status_code==401
        assert anon.get('/api/runs/occupancy/detections').status_code==401
    job=create(eng)
    assert rev.get('/api/jobs/'+job['id']).status_code==403
    assert rev.post('/api/jobs',json=dict(profile_id='centerpoint_pillar02_circle',scene_name='scene-0103',frame_index=1,request_key=uuid.uuid4().hex)).status_code==403


def test_cuda_initialization_precedes_allocator_reset_without_using_gpu():
    from types import SimpleNamespace
    from tools.infer_centerpoint_mini import initialize_cuda
    calls=[]
    def record(name):
        def method(device):calls.append((name,device))
        return method
    fake=SimpleNamespace(cuda=SimpleNamespace(set_device=record('set_device'),synchronize=record('synchronize'),reset_peak_memory_stats=record('reset')))
    initialize_cuda(fake,'cuda:0')
    assert calls==[('set_device','cuda:0'),('synchronize','cuda:0'),('reset','cuda:0')]
