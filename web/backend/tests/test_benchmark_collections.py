"""Real saved benchmark copies; no model/inference/GT evaluator imports."""
from contextlib import contextmanager
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import zipfile

from fastapi.testclient import TestClient
import numpy as np
import pytest

from web.backend.app.config import Settings
from web.backend.app.main import create_app
from web.backend.tests.auth_support import ensure_user, login

ROOT=Path(__file__).resolve().parents[3]
BENCH='/api/benchmarks/benchmark_mini_val'
COL='/api/collections/benchmark_mini_val'
TOKEN='3950bd41f74548429c0f7700ff3d8269'
FAILED='3e8750f331d7499e9b5123e9eb70f2e2'


def legacy_rows(app):
    with app.state.database.connect() as c:
        return {name:[tuple(row) for row in c.execute('SELECT * FROM '+name+' ORDER BY 1')]
                for name in ('users','app_settings','runs','revisions','review_events','revision_evaluations')}


def transition(client,path,info,decision=None):
    response=client.post(path,json=dict(comment='Review exact saved collection',revision_id=info['revision_id'],
        fingerprint=info['fingerprint'],**({'decision':decision} if decision else {})))
    assert response.status_code==200,response.text
    return response.json()


@pytest.fixture(scope='module')
def corpus(tmp_path_factory):
    base=tmp_path_factory.mktemp('benchmark-web');runs=base/'runs';runs.mkdir()
    shutil.copytree(ROOT/'work_dirs/benchmark_mini_val',runs/'benchmark_mini_val')
    shutil.copyfile(ROOT/'work_dirs/radar_startup_audit.json',runs/'radar_startup_audit.json')
    for source,target in [('infer_mini_cuda_v2','legacy_v1'),('infer_mini_val_scene0103_f1','legacy_v2'),('eval_mini_val_scene0103_f1','legacy_evaluation')]:
        shutil.copytree(ROOT/'work_dirs'/source,runs/target)
    settings=Settings(runs,(ROOT/'data/nuscenes',),state_root=base/'state')
    app=create_app(settings);ensure_user(app);ensure_user(app,'test_reviewer','reviewer')
    with TestClient(app) as engineer,TestClient(app) as reviewer:
        login(engineer);login(reviewer,'test_reviewer')
        # Existing v1 and v2 approvals are deliberately created only in this QA DB.
        for identifier in ('legacy_v1','legacy_v2'):
            path='/api/runs/'+identifier
            info=engineer.post(path+'/register',json={'comment':'Legacy independent run'}).json()
            info=transition(engineer,path+'/review/submit',info)
            info=transition(reviewer,path+'/review/decision',info,'approved')
            if identifier=='legacy_v2':
                evaluation=engineer.get(path).json()['evaluation']['matched']['id']
                response=engineer.post(path+'/review/evaluation',json=dict(comment='Legacy evaluation revision',
                    revision_id=info['revision_id'],fingerprint=info['fingerprint'],evaluation_id=evaluation))
                assert response.status_code==200,response.text
                info=transition(engineer,path+'/review/submit',response.json())
                transition(reviewer,path+'/review/decision',info,'approved')
        yield SimpleNamespace(app=app,engineer=engineer,reviewer=reviewer,settings=settings,runs=runs,
                              folder=runs/'benchmark_mini_val',legacy=legacy_rows(app))


@pytest.fixture
def system(corpus):
    with corpus.app.state.database.transaction() as c:
        c.execute('DELETE FROM collection_events');c.execute('UPDATE collections SET current_revision_id=NULL')
        c.execute('DELETE FROM collection_revisions');c.execute('DELETE FROM collections')
    yield corpus
    assert legacy_rows(corpus.app)==corpus.legacy
    assert not any(n.split('.')[0] in ('torch','mmcv','mmdet3d','occfusion') for n in sys.modules)


def draft(s):
    response=s.engineer.post(COL+'/register',json={'comment':'Review partial 79/81 collection explicitly'})
    assert response.status_code==200,response.text
    return response.json()


def approve(s):
    info=transition(s.engineer,COL+'/review/submit',draft(s))
    return transition(s.reviewer,COL+'/review/decision',info,'approved')


@contextmanager
def changed(path,content):
    original=path.read_bytes();stat=path.stat()
    path.write_bytes(content);os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    try: yield
    finally:
        path.write_bytes(original);os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))


def test_actual_partial_metrics_original_latency_audit_and_membership(system):
    s=system;response=s.engineer.get(BENCH);assert response.status_code==200,response.text
    data=response.json();source=json.loads((s.folder/'metrics_pooled.json').read_text())
    assert data['status']=='partial' and data['successful_frame_count']==79 and data['expected_frame_count']==81
    assert data['metrics']==source['metrics'] and len(data['metrics']['per_class_semantic'])==16
    assert np.asarray(data['metrics']['confusion_matrix']).shape==(17,17)
    original=json.loads((s.folder/'initial_report.json').read_text())
    assert data['latency']['sample_count']==78 and data['latency']['source_report']=='initial_report.json'
    assert data['latency']['median_s']==original['latency_current_invocation']['excluding_marked_warmup']['median_s']
    assert data['latency']['p95_s']==original['latency_current_invocation']['excluding_marked_warmup']['p95_s']
    latest=json.loads((s.folder/'benchmark_report.json').read_text())
    assert latest['latency_current_invocation']['all_completed_forwards']['sample_count']==0
    assert data['radar_audit']['status']=='verified' and len(data['radar_audit']['frames'])==2
    assert data['replay'] and not data['inference_enabled'] and 'learned 4D flow' in data['playback_note']
    assert len(data['frames'])==81


def test_frame_playback_binding_and_failed_frame_never_has_prediction(system):
    s=system;response=s.engineer.get(BENCH+'/frames/'+TOKEN);assert response.status_code==200,response.text
    data=response.json();assert data['sample_token']==TOKEN and data['evaluation']['status']=='matched'
    assert len(data['evaluation']['candidates'])==1
    response=s.engineer.get(data['occupancy']['url']);assert response.status_code==200,response.text
    points=np.frombuffer(response.content,dtype='<f4').reshape(-1,5)
    assert len(points)==data['occupancy']['point_count'] and np.isfinite(points).all()
    assert s.engineer.get(data['lidar']['url']).status_code==200
    assert s.engineer.get(data['bev_url']).content.startswith(b'\x89PNG')
    assert s.engineer.get(BENCH+'/frames/'+TOKEN+'/occupancy?bundle='+'0'*64).status_code==409
    failed=s.engineer.get(BENCH+'/frames/'+FAILED).json()
    assert failed['status']=='failed' and not failed['prediction_available'] and 'metrics' not in failed
    assert failed['radar_audit']['rejected_channels']==['RADAR_FRONT','RADAR_BACK_RIGHT']
    assert s.engineer.get(BENCH+'/frames/'+FAILED+'/occupancy').status_code==409
    assert s.engineer.get(BENCH+'/frames/'+'0'*32).status_code==404


def test_auth_role_csrf_and_unapproved_export(system):
    s=system
    with TestClient(s.app) as anonymous:
        for path in ('/api/benchmarks',BENCH,BENCH+'/frames/'+TOKEN,COL+'/review'):
            assert anonymous.get(path).status_code==401
        assert anonymous.post(COL+'/export').status_code==401
    assert s.reviewer.get(BENCH).status_code==403
    assert s.reviewer.get('/api/benchmarks').json()['benchmarks']==[]
    assert s.reviewer.post(COL+'/register',json={'comment':'Reviewer cannot own'}).status_code==403
    assert s.engineer.post(COL+'/export').status_code==409
    info=draft(s);assert s.engineer.post(COL+'/export').status_code==409
    body=dict(comment='Cannot approve as engineer',decision='approved',revision_id=info['revision_id'],fingerprint=info['fingerprint'])
    assert s.engineer.post(COL+'/review/decision',json=body).status_code==403
    assert s.engineer.post(COL+'/review/submit',json=body,headers={'X-CSRF-Token':'wrong'}).status_code==403


def test_collection_approval_export_restart_and_old_approvals_remain_independent(system):
    s=system;info=approve(s);assert info['approval_valid'] and info['manifest']['kind']=='occfusion_benchmark_collection'
    assert info['manifest']['successful_frame_count']==79 and len(info['manifest']['sample_tokens'])==81
    exported=s.engineer.post(COL+'/export');assert exported.status_code==200,exported.text[:500]
    with zipfile.ZipFile(BytesIO(exported.content)) as archive:
        manifest=json.loads(archive.read('manifest.json'));review=json.loads(archive.read('review.json'))
        assert manifest==info['manifest'] and review['status']=='partial' and review['successful_frame_count']==79
        assert review['metric_source']=='benchmark/metrics_pooled.json'
        assert review['latency_source']=='benchmark/initial_report.json'
        for name,value in manifest['artifacts'].items():
            content=archive.read(name);assert hashlib.sha256(content).hexdigest()==value['sha256'] and len(content)==value['bytes']
        assert not any(name.startswith('frames/'+FAILED+'/') for name in archive.namelist())
    assert not list((s.settings.state_directory/'collection-exports').glob('*.zip'))
    before=s.engineer.get(COL+'/review').json()
    restarted=create_app(s.settings)
    with TestClient(restarted) as client:
        client.cookies.update(s.engineer.cookies);client.headers.update(s.engineer.headers)
        assert client.get('/api/auth/session').json()['user']['username']=='test_engineer'
        after=client.get(COL+'/review').json()
        assert after['approval_valid'] and after['fingerprint']==before['fingerprint'] and after['events']==before['events']
    old1=s.engineer.get('/api/runs/legacy_v1/review').json();old2=s.engineer.get('/api/runs/legacy_v2/review').json()
    assert old1['manifest']['schema_version']==1 and old1['approval_valid']
    assert old2['manifest']['schema_version']==2 and old2['approval_valid']
    assert old1['fingerprint']!=info['fingerprint']!=old2['fingerprint']
    revised=transition(s.engineer,COL+'/review/revise',info)
    assert revised['state']=='draft' and revised['revisions'][0]['state']=='approved'
    assert s.engineer.post(COL+'/export').status_code==409 and s.reviewer.get(BENCH).status_code==403
    assert s.reviewer.post(COL+'/review/decision',json=dict(comment='Stale',decision='approved',revision_id=info['revision_id'],fingerprint=info['fingerprint'])).status_code==409


@pytest.mark.parametrize('name',['occupancy.npz','metadata.json','report.json','bev.png','evaluation.json','metrics_pooled.json','initial_report.json','radar_startup_audit.json','timings.csv'])
def test_changed_bytes_even_same_stat_block_approval_and_export(system,name):
    s=system;info=approve(s);progress=json.loads((s.folder/'progress.json').read_text())['frames'][TOKEN]
    if name=='evaluation.json': folder=s.folder/'evaluations'/TOKEN/Path(progress['evaluation_dir']).name
    elif name in ('occupancy.npz','metadata.json','report.json','bev.png'): folder=s.folder/'frames'/TOKEN/Path(progress['prediction_dir']).name
    else: folder=s.runs if name=='radar_startup_audit.json' else s.folder
    path=folder/name;content=bytearray(path.read_bytes());content[-1]=32 if content[-1]==10 else content[-1]^1
    with changed(path,bytes(content)):
        integrity=s.engineer.get(COL+'/review').json()
        assert not integrity['approval_valid'] and not integrity['artifact_matches']
        assert s.engineer.post(COL+'/export').status_code==409
    assert s.engineer.get(COL+'/review').json()['approval_valid']


@pytest.mark.parametrize('field',['token','geometry','protocol'])
def test_semantic_mismatch_rejected_before_join_even_with_updated_receipt(system,field):
    s=system;progress=json.loads((s.folder/'progress.json').read_text())['frames'][TOKEN]
    pred=s.folder/'frames'/TOKEN/Path(progress['prediction_dir']).name
    if field=='protocol':
        path=s.folder/'metrics_pooled.json';value=json.loads(path.read_text());value['protocol']['union_zero_iou']=1
        with changed(path,json.dumps(value).encode()): assert s.engineer.get(BENCH).status_code==422
        return
    path=pred/'metadata.json';value=json.loads(path.read_text())
    if field=='token': value['sample_token']='0'*32
    else: value['grid_axis_order']=['y','x','z']
    content=json.dumps(value).encode();receipt_path=pred/'prediction_commit.json';receipt=json.loads(receipt_path.read_text())
    receipt['artifact_hashes']['metadata.json']={'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
    with changed(path,content),changed(receipt_path,json.dumps(receipt).encode()):
        assert s.engineer.get(BENCH).status_code==422


def test_sidecar_tampering_and_symlink_escape_are_blocked(system,tmp_path):
    s=system;info=approve(s)
    path=s.settings.state_directory/'collection-manifests/benchmark_mini_val'/ (info['fingerprint']+'.json')
    with changed(path,b'{}'):
        assert not s.engineer.get(COL+'/review').json()['manifest_matches']
        assert s.engineer.post(COL+'/export').status_code==409
    outside=tmp_path/'outside';outside.mkdir();(outside/'manifest.json').write_text('{"kind":"occfusion_occupancy_benchmark"}')
    link=s.runs/'escape';link.symlink_to(outside,target_is_directory=True)
    try: assert s.engineer.get('/api/benchmarks/escape').status_code==403
    finally: link.unlink()


@pytest.mark.parametrize('decision',['needs_changes','rejected'])
def test_other_review_decisions_and_owner_notes_persist(system,decision):
    s=system;info=draft(s)
    assert s.engineer.post(COL+'/review/notes',json={'comment':'Explain partial radar startup coverage'}).status_code==200
    info=transition(s.engineer,COL+'/review/submit',info)
    result=transition(s.reviewer,COL+'/review/decision',info,decision)
    assert result['state']==decision and not result['approval_valid']
    assert s.engineer.post(COL+'/export').status_code==409
