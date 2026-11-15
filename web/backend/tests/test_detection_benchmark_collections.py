"""Saved-artifact playback/review tests only. No inference or GT evaluator."""
from contextlib import contextmanager
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import zipfile

from fastapi.testclient import TestClient
import numpy as np
import pytest

from web.backend.app.config import Settings, REPO
from web.backend.app.main import create_app
from web.backend.app.benchmarks import fingerprint
from web.backend.tests.auth_support import ensure_user, login

NAME='benchmark_detection_mini_val'
BENCH='/api/benchmarks/'+NAME
COL='/api/collections/'+NAME
TOKEN='3950bd41f74548429c0f7700ff3d8269'


def transition(client, path, info, decision=None):
    body=dict(comment='QA exact saved detection collection',revision_id=info['revision_id'],fingerprint=info['fingerprint'])
    if decision: body['decision']=decision
    response=client.post(path,json=body)
    assert response.status_code==200,response.text
    return response.json()


def legacy_rows(app):
    with app.state.database.connect() as c:
        return {name:[tuple(r) for r in c.execute('SELECT * FROM '+name+' ORDER BY 1')]
            for name in ('users','app_settings','runs','revisions','review_events','inference_jobs')}


@pytest.fixture(scope='module')
def corpus(tmp_path_factory):
    base=tmp_path_factory.mktemp('detection-collection-web');runs=base/'runs';runs.mkdir()
    shutil.copytree(REPO/'work_dirs'/NAME,runs/NAME)
    shutil.copytree(REPO/'work_dirs/benchmark_mini_val',runs/'benchmark_mini_val')
    shutil.copyfile(REPO/'work_dirs/radar_startup_audit.json',runs/'radar_startup_audit.json')
    shutil.copytree(REPO/'work_dirs/infer_mini_val_scene0103_f1',runs/'occupancy')
    shutil.copytree(REPO/'work/centerpoint-smoke-wyp1nl__/runs/infer_job_65e1b30833e343c3b434da8dc71423f9',runs/'single_detection')
    settings=Settings(runs,(REPO/'data/nuscenes',),state_root=base/'state')
    app=create_app(settings);ensure_user(app);ensure_user(app,'test_reviewer','reviewer')
    with TestClient(app) as engineer,TestClient(app) as reviewer:
        login(engineer);login(reviewer,'test_reviewer')
        info=engineer.post('/api/runs/single_detection/register',json={'comment':'Existing independent single-frame approval in QA'}).json()
        info=transition(engineer,'/api/runs/single_detection/review/submit',info)
        transition(reviewer,'/api/runs/single_detection/review/decision',info,'approved')
        yield SimpleNamespace(app=app,engineer=engineer,reviewer=reviewer,settings=settings,folder=runs/NAME,
            runs=runs,legacy=legacy_rows(app))


@pytest.fixture
def system(corpus):
    with corpus.app.state.database.transaction() as c:
        c.execute('DELETE FROM collection_events');c.execute('UPDATE collections SET current_revision_id=NULL')
        c.execute('DELETE FROM collection_revisions');c.execute('DELETE FROM collections')
    yield corpus
    assert legacy_rows(corpus.app)==corpus.legacy
    assert not any(name.split('.')[0] in ('torch','mmcv','mmdet','mmdet3d','occfusion') for name in sys.modules)
    assert 'tools.evaluate_detection' not in sys.modules and 'tools.benchmark_detection' not in sys.modules


@contextmanager
def changed(path, content):
    previous=path.read_bytes()
    path.write_bytes(content)
    try: yield
    finally: path.write_bytes(previous)


def draft(s):
    response=s.engineer.post(COL+'/register',json={'comment':'QA whole detection set, independent of single-frame approvals'})
    assert response.status_code==200,response.text
    return response.json()


def approve(s):
    info=transition(s.engineer,COL+'/review/submit',draft(s))
    return transition(s.reviewer,COL+'/review/decision',info,'approved')


def test_complete_metrics_units_support_original_latency_and_loader(system):
    s=system;response=s.engineer.get(BENCH);assert response.status_code==200,response.text
    value=response.json()
    gpu=json.loads((s.folder/'reports/367b21268aed4eb4852c6b4d221a59c5.json').read_text())
    assert value['task']=='detection' and value['status']=='complete'
    assert value['successful_frame_count']==value['expected_frame_count']==81 and value['failed_frame_count']==0
    assert value['metrics']==gpu['metrics'] and value['metric_unit']=='ratio_0_1'
    assert 0<value['metrics']['mAP']<1 and 0<value['metrics']['NDS']<1
    assert len(value['classes'])==10 and sum(not c['has_gt'] for c in value['classes'])==3
    assert value['support']['saved_prediction_box_count']>value['support']['official_filtered_box_count']
    assert value['classes'][-1]['tp_errors']['vel_err'] is None
    assert value['latency']['source_report']=='reports/367b21268aed4eb4852c6b4d221a59c5.json'
    assert value['latency']['sample_count']==79 and value['latency']['new_prediction_count']==80 and value['latency']['smoke_cache_count']==1
    assert value['latency']['excluding_marked_warmup']==gpu['latency_current_invocation']['excluding_marked_warmup']
    assert value['loader_audit']['loader_vs_se3_translation_m']['max']==gpu['input_audit_summary']['loader_vs_se3_translation_m']['max']
    assert value['evaluation_source']['immutable']
    assert value['review']['state']=='unregistered'
    assert s.engineer.get('/api/health').json()['model_modules_loaded'] is False


def test_frame_without_publication_receipt_lidar_and_occupancy_pair(system):
    s=system
    token='3e8750f331d7499e9b5123e9eb70f2e2'
    response=s.engineer.get(BENCH+'/frames/'+token);assert response.status_code==200,response.text
    value=response.json();assert value['sample_token']==token and value['origin']=='benchmark'
    assert 'run_id' not in value and value['task']=='detection' and value['box_count']>0
    points=s.engineer.get(value['lidar']['url']);assert points.status_code==200,points.text
    assert len(points.content)==value['lidar']['point_count']*4*4
    assert np.isfinite(np.frombuffer(points.content,dtype='<f4')).all()
    assert s.engineer.get(BENCH+'/frames/'+token+'/occupancy').status_code==409
    assert s.engineer.get(BENCH+'/frames/'+token+'/lidar?bundle='+'0'*64).status_code==409
    matches=s.engineer.get(BENCH+'/frames/'+TOKEN+'/overlays').json()['matches']
    assert any(m.get('run_id')=='occupancy' for m in matches)
    assert any(m.get('benchmark_id')=='benchmark_mini_val' for m in matches)
    overlay=s.engineer.get(BENCH+'/frames/'+TOKEN+'?occupancy_run_id=occupancy')
    assert overlay.status_code==200,overlay.text
    assert overlay.json()['occupancy_overlay']['sample_token']==TOKEN
    collection_overlay=s.engineer.get(BENCH+'/frames/'+TOKEN+'?occupancy_benchmark_id=benchmark_mini_val')
    assert collection_overlay.status_code==200,collection_overlay.text
    value=collection_overlay.json()['occupancy_overlay']
    assert value['sample_token']==TOKEN and value['collection_id']=='benchmark_mini_val'
    assert s.engineer.get(value['occupancy']['url']).status_code==200
    assert s.engineer.get(BENCH+'/frames/'+token+'?occupancy_run_id=occupancy').status_code==422
    assert s.engineer.get(BENCH+'/frames/'+'0'*32).status_code==404
    assert all(not (path/'job_result.json').exists() for path in (s.folder/'frames'/token).iterdir() if path.is_dir())


def test_auth_csrf_engineer_approve_denied_and_single_frame_approval_is_independent(system):
    s=system
    with TestClient(s.app) as anon:
        for path in ('/api/benchmarks',BENCH,BENCH+'/frames/'+TOKEN,COL+'/review'):
            assert anon.get(path).status_code==401
    assert s.reviewer.get(BENCH).status_code==403
    assert s.reviewer.post(COL+'/register',json={'comment':'Denied'}).status_code==403
    assert s.engineer.post(COL+'/export').status_code==409
    info=draft(s)
    assert s.engineer.post(COL+'/review/decision',json=dict(comment='Denied',decision='approved',
        revision_id=info['revision_id'],fingerprint=info['fingerprint'])).status_code==403
    assert s.engineer.post(COL+'/review/submit',json=dict(comment='Bad CSRF',revision_id=info['revision_id'],
        fingerprint=info['fingerprint']),headers={'X-CSRF-Token':'wrong'}).status_code==403


def test_collection_revision_export_hashes_and_restart(system):
    s=system;info=approve(s)
    assert info['manifest']['kind']=='centerpoint_detection_benchmark_collection'
    assert info['manifest']['schema_version']==2 and info['manifest']['task']=='detection'
    response=s.engineer.post(COL+'/export');assert response.status_code==200,response.text[:200]
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        manifest=json.loads(archive.read('manifest.json'));review=json.loads(archive.read('review.json'))
        assert review['metric_source']=='benchmark/evaluation/evaluation.json'
        assert review['latency_source']=='benchmark/reports/367b21268aed4eb4852c6b4d221a59c5.json'
        assert len(json.loads(archive.read('benchmark/predictions_nuscenes.json'))['results'])==81
        assert not any(name.endswith('occupancy.npz') for name in archive.namelist())
        from web.backend.app.benchmarks import record
        for name,expected in manifest['artifacts'].items():
            assert record(archive.read(name))==expected
    restarted=create_app(s.settings)
    with TestClient(restarted) as client:
        login(client)
        assert client.get(COL+'/review').json()['approval_valid']
    revised=transition(s.engineer,COL+'/review/revise',info)
    assert revised['state']=='draft' and revised['revision_number']==2 and not revised['approval_valid']
    assert s.engineer.post(COL+'/export').status_code==409


def test_mutable_cache_pointers_do_not_extend_or_invalidate_detection_approval(system):
    s=system;info=approve(s)
    paths=[s.folder/name for name in ('progress.json','benchmark_report.json','evaluation.json')]
    from contextlib import ExitStack
    with ExitStack() as stack:
        for path in paths:
            value=json.loads(path.read_text());value['qa_cache_read_annotation']='Mutable pointer, not approved evidence'
            stack.enter_context(changed(path,json.dumps(value).encode()))
        response=s.engineer.get(BENCH);assert response.status_code==200,response.text
        review=s.engineer.get(COL+'/review').json()
        assert review['fingerprint']==info['fingerprint'] and review['approval_valid']


@pytest.mark.parametrize('target',['frame','evaluation','official','global','gpu_latency'])
def test_artifact_hash_or_report_mismatch_blocks_metrics_and_export(system,target):
    s=system;approve(s)
    if target=='frame':
        path=next((s.folder/'frames'/TOKEN).glob('*/detections.json'))
    elif target=='evaluation':
        path=next((s.folder/'evaluations').glob('*/evaluation.json'))
    elif target=='official':
        path=next((s.folder/'evaluations').glob('*/official/metrics_summary.json'))
    elif target=='global':
        path=s.folder/'predictions_nuscenes.json'
    else:
        path=s.folder/'reports/367b21268aed4eb4852c6b4d221a59c5.json'
    content=path.read_bytes()+b' '
    if target=='gpu_latency':
        value=json.loads(path.read_bytes());value['latency_current_invocation']['excluding_marked_warmup']['median_s']=999
        content=json.dumps(value).encode()
    with changed(path,content):
        response=s.engineer.get(BENCH)
        assert response.status_code==422,response.text[:200]
        assert 'metrics' not in response.json()
        assert s.engineer.post(COL+'/export').status_code==409


@pytest.mark.parametrize('field',['protocol','token'])
def test_protocol_and_token_coverage_tamper_refused(system,field):
    s=system;path=s.folder/'manifest.json';value=json.loads(path.read_text())
    if field=='protocol':
        value['evaluation_protocol']['name']='different_protocol'
        value['identity']['evaluation_protocol_fingerprint']=fingerprint(value['evaluation_protocol'])
    else:
        value['identity']['frames'].pop()
    value['identity_sha256']=fingerprint(value['identity'])
    with changed(path,json.dumps(value).encode()):
        assert s.engineer.get(BENCH).status_code==422


def test_symlink_cannot_escape_benchmark_root_even_with_identical_bytes(system,tmp_path):
    s=system;path=next((s.folder/'frames'/TOKEN).glob('*/detections.json'))
    previous=path.read_bytes();outside=tmp_path/'identical.json';outside.write_bytes(previous)
    path.unlink();path.symlink_to(outside)
    try:
        assert s.engineer.get(BENCH).status_code in (403,422)
    finally:
        path.unlink();path.write_bytes(previous)
