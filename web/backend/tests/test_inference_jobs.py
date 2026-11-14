"""Queue/worker tests use real artifact fixtures and no model/GPU execution."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from io import BytesIO
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
import zipfile

from fastapi.testclient import TestClient
import pytest

from web.backend.app.config import REPO, Settings
from web.backend.app.database import Database
from web.backend.app.job_worker import Worker, gpu_lock, process_ticks
from web.backend.app.main import create_app
from web.backend.tests.auth_support import ensure_user, login
from web.backend.tests.job_support import FixtureCommands, SOURCE, EVALUATION, settings_to_file


def wait_for(predicate,timeout=15):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        value=predicate()
        if value:return value
        time.sleep(.05)
    raise AssertionError('Timed out waiting for worker state')


@pytest.fixture
def queued(tmp_path):
    assert SOURCE.is_dir() and EVALUATION.is_file(), 'Saved genuine inference/evaluation fixtures required'
    runs=tmp_path/'runs';runs.mkdir()
    settings=Settings(runs,(REPO/'data/nuscenes',),state_root=tmp_path/'state',gpu_lock_path=tmp_path/'gpu.lock')
    app=create_app(settings)
    for name,role in [('test_engineer','engineer'),('other_engineer','engineer'),('test_reviewer','reviewer')]:ensure_user(app,name,role)
    with TestClient(app) as engineer,TestClient(app) as reviewer,TestClient(app) as other:
        login(engineer);login(reviewer,'test_reviewer');login(other,'other_engineer')
        yield app,engineer,reviewer,other,settings


def create(client,**overrides):
    body=dict(profile_id='occfusion_clr_fp32',scene_name='scene-0103',frame_index=1,request_key=uuid.uuid4().hex)
    body.update(overrides)
    response=client.post('/api/jobs',json=body)
    diagnostic={}
    if response.status_code!=202:
        token=client.headers.get('x-csrf-token','')
        diagnostic=dict(cookie_matches_header=client.cookies.get('rav25_csrf')==token,
                        csrf_age_s=time.time()-int(token.split('.')[0]) if token else None)
    assert response.status_code==202,(response.text,diagnostic)
    return response.json(),body


def worker(app,settings,**kwargs):return Worker(app.state.database,settings,FixtureCommands(**kwargs),poll_s=.03)


def detail(client,job):return client.get('/api/jobs/'+job['id']).json()


def test_permissions_csrf_paths_and_fast_idempotent_submission(queued):
    app,eng,rev,other,settings=queued
    with TestClient(app) as anon:
        assert anon.get('/api/jobs').status_code==401
        assert anon.get('/api/jobs/options').status_code==401
        assert anon.post('/api/jobs',json={}).status_code==401
    assert rev.get('/api/jobs/options').status_code==403
    assert rev.get('/api/jobs').status_code==403
    started=time.monotonic();job,body=create(eng)
    assert time.monotonic()-started<3
    assert job['status']=='queued' and job['inference_launch_count']==0
    assert not app.state.jobs.paths(job['id'])['stage'].exists()
    assert eng.post('/api/jobs',json=body).json()['id']==job['id']
    assert len(eng.get('/api/jobs').json()['jobs'])==1
    assert eng.post('/api/jobs',json=dict(body,frame_index=2)).status_code==409
    for client in (rev,other):
        assert client.get('/api/jobs/'+job['id']).status_code==403
        assert client.post('/api/jobs/'+job['id']+'/cancel').status_code==403
    assert other.get('/api/jobs').json()['jobs']==[]
    assert rev.post('/api/jobs',json=dict(body,request_key=uuid.uuid4().hex)).status_code==403
    for change in ({'out_dir':'/tmp/injected'}, {'device':'cpu'}, {'frame_index':True}, {'frame_index':'1'}, {'frame_index':999}, {'scene_name':'scene-9999'}):
        assert eng.post('/api/jobs',json=dict(body,**change)).status_code==422
    assert eng.get('/api/jobs/'+job['id']+'/log?offset=-1').status_code==422
    assert eng.get('/api/jobs/'+job['id']+'/log?kind=../../x').status_code==422
    assert eng.post('/api/jobs',json=body,headers={'X-CSRF-Token':'wrong'}).status_code==403
    assert eng.post('/api/jobs/'+job['id']+'/cancel',headers={'Origin':'https://other.invalid'}).status_code==403
    health=eng.get('/api/health').json()
    assert health['model_modules_loaded'] is False and health['inference_enabled'] is False and health['job_submission_enabled'] is True
    assert not any(name.split('.')[0] in ('torch','occfusion','mmcv','mmdet3d') for name in sys.modules)


def test_options_match_official_metadata_and_profile(queued):
    app,eng,_,_,_=queued
    assert app.state.jobs.profiles.get('occfusion_clr_fp32')['python']=='/home/hatis/.venvs/rav25/bin/python'
    options=eng.get('/api/jobs/options').json();profile=options['profiles'][0]
    assert profile['id']=='occfusion_clr_fp32' and profile['precision']=='float32'
    assert profile['tf32']==dict(matmul=False,cudnn=True)
    scenes=options['scenes_by_profile'][profile['id']]
    val=[s for s in scenes if s['split']=='mini_val']
    assert [s['name'] for s in val]==['scene-0103','scene-0916']
    assert sum(len(s['frames']) for s in val)==81
    assert val[0]['frames'][1]['sample_token']=='3950bd41f74548429c0f7700ff3d8269'


def test_success_validates_then_publishes_viewer_without_approval(queued):
    app,eng,rev,_,settings=queued
    job,_=create(eng);w=worker(app,settings)
    assert w.run(max_jobs=1)==1
    result=detail(eng,job)
    assert result['status']=='succeeded',result
    assert result['execution_mode']=='fixture' and result['inference_launch_count']==1
    assert result['evaluation_status']=='matched' and result['result']['approved'] is False
    assert result['events'][-1]['phase']=='succeeded'
    run_id='infer_job_'+job['id'];paths=app.state.jobs.paths(job['id'])
    assert paths['run'].is_dir() and paths['eval_run'].is_dir()
    assert not paths['prediction'].exists()
    for name in ('occupancy.npz','metadata.json','report.json','bev.png','resolved_config.py'):
        assert (paths['run']/name).read_bytes()==(SOURCE/name).read_bytes()
    assert (paths['eval_run']/'evaluation.json').read_bytes()==EVALUATION.read_bytes()
    view=eng.get('/api/runs/'+run_id)
    assert view.status_code==200,view.text
    value=view.json();assert value['evaluation']['status']=='matched' and value['sample_token']==job['sample_token']
    assert value['review']['state']=='unregistered'
    assert eng.get(value['occupancy']['url']).status_code==200
    assert eng.get(value['lidar']['url']).status_code==200
    assert eng.get(value['bev_url']).status_code==200
    assert eng.post('/api/runs/'+run_id+'/export').status_code==409
    assert rev.get('/api/runs/'+run_id).status_code==403
    with app.state.database.connect() as c:
        assert c.execute('SELECT COUNT(*) FROM runs').fetchone()[0]==0
        assert c.execute('SELECT COUNT(*) FROM revisions').fetchone()[0]==0
    restarted=create_app(settings)
    with TestClient(restarted) as c:
        login(c);assert detail(c,job)['status']=='succeeded'
        assert c.get('/api/runs/'+run_id).json()['evaluation']['status']=='matched'
    assert worker(app,settings).run(once=True)==0
    assert detail(eng,job)['inference_launch_count']==1


@pytest.mark.parametrize('mode,stage',[('fail','forward'),('missing','artifact_validation'),('wrong-token','artifact_validation')])
def test_cli_failure_missing_outputs_wrong_sample_are_private(queued,mode,stage):
    app,eng,_,_,settings=queued
    job,_=create(eng);worker(app,settings,inference_mode=mode).run(once=True)
    value=detail(eng,job)
    assert value['status']=='failed' and value['error']['stage']==stage,value
    assert value['result_url'] is None and value['inference_launch_count']==1
    assert not app.state.jobs.paths(job['id'])['run'].exists()
    assert eng.get('/api/runs').json()['runs']==[]
    assert 'TEST FIXTURE' in eng.get('/api/jobs/'+job['id']+'/log').json()['text']


@pytest.mark.parametrize('mode',['wrong-token','wrong-hash'])
def test_invalid_evaluation_never_publishes(queued,mode):
    app,eng,_,_,settings=queued
    job,_=create(eng);worker(app,settings,evaluation_mode=mode).run(once=True)
    value=detail(eng,job)
    assert value['status']=='failed' and value['error']['stage']=='evaluation_validation',value
    assert not app.state.jobs.paths(job['id'])['run'].exists()
    assert eng.get('/api/runs').json()['runs']==[]
    assert app.state.reviews.evaluations.paths()=={}


def test_queued_and_running_cancel_terminate_cli_and_keep_partial(queued):
    app,eng,_,_,settings=queued
    first,_=create(eng)
    assert eng.post('/api/jobs/'+first['id']+'/cancel').json()['status']=='canceled'
    assert worker(app,settings).run(once=True)==0
    assert detail(eng,first)['inference_launch_count']==0
    second,_=create(eng);w=worker(app,settings,inference_mode='slow')
    with ThreadPoolExecutor(1) as pool:
        future=pool.submit(w.run_once)
        pid=wait_for(lambda: app.state.jobs.row(second['id'])['child_pid'])
        assert eng.post('/api/jobs/'+second['id']+'/cancel').json()['cancel_requested']
        assert future.result(timeout=15)
        assert process_ticks(pid) is None
    value=detail(eng,second)
    assert value['status']=='canceled' and value['inference_launch_count']==1
    assert eng.post('/api/jobs/'+second['id']+'/cancel').status_code==409
    with gpu_lock(settings.gpu_lock_path) as fd:assert fd is not None


def test_queue_and_cross_process_gpu_lane(queued):
    app,eng,_,_,settings=queued
    first,_=create(eng);second,_=create(eng)
    # Hold a lane in this process; a separate worker process may not claim a job.
    config=settings.state_directory/'test-settings.json';settings_to_file(settings,config)
    script='from web.backend.tests.job_support import read_settings,FixtureCommands; from web.backend.app.job_worker import Worker; from web.backend.app.database import Database; import sys; s=read_settings(sys.argv[1]); print(Worker(Database(s.state_directory),s,FixtureCommands()).run(once=True))'
    with gpu_lock(settings.gpu_lock_path) as fd:
        assert fd is not None
        result=subprocess.run([sys.executable,'-c',script,str(config)],cwd=REPO,capture_output=True,text=True,timeout=15)
        assert result.returncode==0 and result.stdout.strip()=='0',result.stderr
    assert detail(eng,first)['inference_launch_count']==0
    w=worker(app,settings,inference_mode='slow')
    with ThreadPoolExecutor(1) as pool:
        future=pool.submit(w.run_once)
        wait_for(lambda: app.state.jobs.row(first['id'])['child_pid'])
        assert worker(app,settings).run_once() is False
        assert detail(eng,second)['status']=='queued' and detail(eng,second)['inference_launch_count']==0
        # A separate SQLite cannot bypass the shared OS GPU lane.
        other_settings=replace(settings,state_root=settings.state_directory.parent/'other-state')
        other_app=create_app(other_settings);ensure_user(other_app)
        with TestClient(other_app) as independent:
            login(independent);independent_job,_=create(independent)
            assert worker(other_app,other_settings).run_once() is False
            assert detail(independent,independent_job)['inference_launch_count']==0
            independent.post('/api/jobs/'+independent_job['id']+'/cancel')
        # Restart only the backend; the live worker/CLI remains the same job.
        restarted=create_app(settings)
        with TestClient(restarted) as c:
            login(c);value=detail(c,first)
            assert value['status']=='running' and value['inference_launch_count']==1
        eng.post('/api/jobs/'+first['id']+'/cancel');future.result(timeout=15)
    assert worker(app,settings).run_once()
    assert detail(eng,second)['status']=='succeeded'
    assert detail(eng,first)['status']=='canceled'


def test_hard_worker_crash_kills_child_and_restart_does_not_reinfer(queued):
    app,eng,_,_,settings=queued
    job,_=create(eng)
    config=settings.state_directory/'test-settings.json';settings_to_file(settings,config)
    process=subprocess.Popen([sys.executable,'-m','web.backend.tests.job_support','worker','--target',str(config),'--mode','slow'],cwd=REPO)
    try:
        pid=wait_for(lambda: app.state.jobs.row(job['id'])['child_pid'])
        # Wait until prctl shim has exec'd the fake CLI, to test parent-death kill.
        wait_for(lambda:(app.state.jobs.paths(job['id'])['stage']/'inference.log').stat().st_size>0)
        process.kill();process.wait(timeout=10)
        wait_for(lambda: process_ticks(pid) is None)
        assert worker(app,settings).run_once() is False
        result=detail(eng,job)
        assert result['status']=='failed' and result['error']['stage']=='worker_interrupted',result
        assert result['inference_launch_count']==1
        assert app.state.jobs.paths(job['id'])['stage'].is_dir()
        assert not app.state.jobs.paths(job['id'])['run'].exists()
        assert worker(app,settings).run_once() is False
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=10)


def test_publication_crash_recovers_without_launching_again(queued):
    app,eng,_,_,settings=queued
    job,_=create(eng);worker(app,settings).run(once=True)
    with app.state.database.connect() as c:
        c.execute("UPDATE inference_workers SET pid=99999999,started_ticks='dead'")
        c.execute("UPDATE inference_jobs SET status='running',phase='publishing',result_json=NULL WHERE id=?",(job['id'],))
    assert worker(app,settings).run(once=True)==0
    value=detail(eng,job)
    assert value['status']=='succeeded' and value['inference_launch_count']==1
    assert any(event['phase']=='recovered' for event in value['events'])


def test_profile_source_drift_fails_before_any_cli(queued,tmp_path):
    app,eng,_,_,settings=queued
    profile=json.loads(settings.job_profiles_path.read_text());profile['profiles'][0]['source_hashes']['tools/infer_nuscenes_mini.py']='0'*64
    path=tmp_path/'profiles.json'
    # Resolve admin paths before relocating the profile fixture.
    p=app.state.jobs.profiles.get('occfusion_clr_fp32')
    for key in ('python','config','checkpoint','data_root','gt_root'):profile['profiles'][0][key]=p[key]
    path.write_text(json.dumps(profile))
    settings=replace(settings,job_profiles_path=path);altered=create_app(settings)
    with TestClient(altered) as c:
        login(c);job,_=create(c);worker(altered,settings).run(once=True)
        result=detail(c,job)
        assert result['status']=='failed' and result['error']['stage']=='preflight'
        assert result['inference_launch_count']==0


def test_job_artifact_or_evaluation_mutation_blocks_viewer(queued):
    app,eng,_,_,settings=queued
    job,_=create(eng);worker(app,settings).run(once=True)
    paths=app.state.jobs.paths(job['id']);url='/api/runs/infer_job_'+job['id']
    assert eng.get(url).status_code==200
    path=paths['eval_run']/'evaluation.json';original=path.read_bytes();path.write_bytes(original+b' ')
    assert eng.get(url).status_code==422
    path.write_bytes(original)
    assert eng.get(url).status_code==200
    (paths['run']/'bev.png').write_bytes((paths['run']/'bev.png').read_bytes()+b' ')
    assert eng.get(url).status_code==422


def test_job_completion_still_requires_review_revision_for_evaluation_export(queued):
    app,eng,rev,_,settings=queued
    job,_=create(eng);worker(app,settings).run(once=True)
    url='/api/runs/infer_job_'+job['id']
    info=eng.post(url+'/register',json=dict(comment='Register a new job result')).json()
    assert info['state']=='draft' and info['manifest']['schema_version']==1
    assert eng.post(url+'/export').status_code==409
    evaluation=eng.get(url).json()['evaluation']['matched']
    info=eng.post(url+'/review/evaluation',json=dict(comment='Bind evaluation to new revision',
        revision_id=info['revision_id'],fingerprint=info['fingerprint'],evaluation_id=evaluation['id'])).json()
    assert info['state']=='draft' and info['manifest']['schema_version']==2
    transition=dict(comment='Check exact job snapshot',revision_id=info['revision_id'],fingerprint=info['fingerprint'])
    assert eng.post(url+'/review/decision',json=dict(transition,decision='approved')).status_code==403
    info=eng.post(url+'/review/submit',json=transition).json();assert info['state']=='submitted'
    info=rev.post(url+'/review/decision',json=dict(transition,decision='approved')).json();assert info['approval_valid']
    exported=eng.post(url+'/export');assert exported.status_code==200,exported.text[:100]
    with zipfile.ZipFile(BytesIO(exported.content)) as z:
        assert z.read('evaluation.json')==EVALUATION.read_bytes()
        assert json.loads(z.read('manifest.json'))['fingerprint_sha256']==info['fingerprint']
    path=app.state.jobs.paths(job['id'])['eval_run']/'evaluation.json';path.write_bytes(path.read_bytes()+b' ')
    # Job publication integrity fails before a revision can be read/exported.
    assert eng.post(url+'/export').status_code==422


def test_missing_gt_is_success_without_replacement_metrics(queued,tmp_path):
    from web.backend.app.job_worker import Commands
    app,eng,_,_,settings=queued
    empty=tmp_path/'empty-gt';empty.mkdir()
    profile=json.loads(settings.job_profiles_path.read_text());p=app.state.jobs.profiles.get('occfusion_clr_fp32')
    for key in ('python','config','checkpoint','data_root','gt_root'):profile['profiles'][0][key]=p[key]
    profile['profiles'][0]['gt_root']=str(empty)
    path=tmp_path/'profiles.json';path.write_text(json.dumps(profile))
    settings=replace(settings,job_profiles_path=path);app=create_app(settings)
    class MissingGT(FixtureCommands):
        evaluation=Commands.evaluation
    with TestClient(app) as c:
        login(c);job,_=create(c)
        Worker(app.state.database,settings,MissingGT(),poll_s=.05).run(once=True)
        result=detail(c,job)
        assert result['status']=='succeeded',result
        assert result['evaluation_status']=='missing_gt'
        document=json.loads((app.state.jobs.paths(job['id'])['eval_run']/'evaluation.json').read_bytes())
        assert document['metrics'] is None and document['execution']['model_loaded'] is False
        view=c.get('/api/runs/infer_job_'+job['id']).json()
        assert view['evaluation']['status']=='missing_gt' and view['evaluation']['matched'] is None
