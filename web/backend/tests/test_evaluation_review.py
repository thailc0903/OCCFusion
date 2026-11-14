"""Evaluation/revision integration using copies of the saved real mini_val run."""
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

from fastapi.testclient import TestClient
import pytest

from web.backend.app.config import Settings
from web.backend.app.main import create_app
from web.backend.tests.auth_support import ensure_user, login


ROOT = Path(__file__).resolve().parents[3]
NAMES = ('occupancy.npz', 'metadata.json', 'report.json', 'bev.png')
RUN = '/api/runs/val_prediction'


@pytest.fixture
def evaluated(tmp_path):
    source = ROOT/'work_dirs/infer_mini_val_scene0103_f1'
    if not source.is_dir():
        pytest.skip('Real saved mini_val artifacts are unavailable')
    runs = tmp_path/'runs'
    folder = runs/'val_prediction'
    shutil.copytree(source, folder)
    target = runs/'saved_evaluation/evaluation.json'
    target.parent.mkdir()
    shutil.copyfile(ROOT/'work_dirs/eval_mini_val_scene0103_f1/evaluation.json', target)
    settings = Settings(runs, (ROOT/'data/nuscenes',), state_root=tmp_path/'state')
    app = create_app(settings)
    ensure_user(app)
    ensure_user(app, 'test_reviewer', 'reviewer')
    with TestClient(app) as engineer, TestClient(app) as reviewer:
        login(engineer)
        login(reviewer, 'test_reviewer')
        yield app, engineer, reviewer, folder, target, settings


def draft(client):
    response = client.post(RUN+'/register', json={'comment': 'Register four-artifact snapshot'})
    assert response.status_code == 200, response.text
    return response.json()


def submit(engineer, info):
    response = engineer.post(RUN+'/review/submit', json=dict(comment='Review this exact snapshot', revision_id=info['revision_id'], fingerprint=info['fingerprint']))
    assert response.status_code == 200, response.text
    return response.json()


def approve(reviewer, info):
    response = reviewer.post(RUN+'/review/decision', json=dict(comment='Approve exact snapshot', decision='approved', revision_id=info['revision_id'], fingerprint=info['fingerprint']))
    assert response.status_code == 200, response.text
    return response.json()


def attach(client, info, evaluation_id):
    return client.post(RUN+'/review/evaluation', json=dict(comment='Include saved evaluation in a new revision',
        revision_id=info['revision_id'], fingerprint=info['fingerprint'], evaluation_id=evaluation_id))


def matching(client):
    value = client.get(RUN).json()['evaluation']
    assert value['status'] == 'matched', value
    return value['matched']


def test_match_uses_real_ratios_class_evidence_and_fresh_bytes(evaluated):
    _, engineer, _, _, path, _ = evaluated
    source = json.loads(path.read_text())
    match = matching(engineer)
    assert match['metrics']['occupancy_iou'] == source['metrics']['occupancy_iou']
    assert match['metrics']['semantic_miou'] == source['metrics']['semantic_miou']
    assert match['metric_unit'] == 'ratio_0_1'
    assert len(match['metrics']['per_class_semantic']) == 16
    assert match['frame_count_evaluated'] == 1 and match['split'] == 'mini_val'
    stat = path.stat()
    changed = path.read_bytes().replace(b'3950bd41f74548429c0f7700ff3d8269', b'00000000000000000000000000000000')
    path.write_bytes(changed)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    response = engineer.get(RUN).json()['evaluation']
    assert response['status'] == 'token_mismatch' and response['matched'] is None


@pytest.mark.parametrize('change,expected', [('token','token_mismatch'),('hash','hash_mismatch'),('ratio','invalid'),('protocol','invalid')])
def test_bad_evaluation_is_visible_and_cannot_be_attached(evaluated, change, expected):
    _, engineer, _, _, path, _ = evaluated
    info = draft(engineer)
    identifier = matching(engineer)['id']
    value = json.loads(path.read_text())
    if change == 'token':
        value['frames'][0]['sample_token'] = 'wrong-token'
    elif change == 'hash':
        value['frames'][0]['prediction_sha256'] = '0'*64
    elif change == 'ratio':
        value['frames'][0]['metrics']['occupancy_iou'] *= 100
    else:
        value['protocol']['union_zero_iou'] = 1
    path.write_text(json.dumps(value))
    evidence = engineer.get(RUN).json()['evaluation']
    assert evidence['status'] == expected and evidence['matched'] is None
    assert attach(engineer, info, identifier).status_code == 409


def test_v1_approval_does_not_expand_v2_requires_review_and_exports_exact_evaluation(evaluated):
    _, engineer, reviewer, folder, path, _ = evaluated
    before = {name:(folder/name).read_bytes() for name in NAMES}
    evaluation = path.read_bytes()
    info = approve(reviewer, submit(engineer, draft(engineer)))
    old_fingerprint, old_manifest = info['fingerprint'], info['manifest']
    with zipfile.ZipFile(BytesIO(engineer.post(RUN+'/export').content)) as z:
        assert 'evaluation.json' not in z.namelist()
        assert json.loads(z.read('manifest.json')) == old_manifest
    response = attach(engineer, info, matching(engineer)['id'])
    assert response.status_code == 200, response.text
    v2 = response.json()
    assert v2['revision_number'] == 2 and v2['state'] == 'draft'
    assert v2['manifest']['schema_version'] == 2 and v2['fingerprint'] != old_fingerprint
    assert v2['manifest']['artifacts']['evaluation.json']['sha256'] == hashlib.sha256(evaluation).hexdigest()
    core = {key:value for key,value in v2['manifest'].items() if key != 'fingerprint_sha256'}
    assert hashlib.sha256(json.dumps(core,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest() == v2['fingerprint']
    assert v2['revisions'][0]['state'] == 'approved' and v2['revisions'][0]['fingerprint'] == old_fingerprint
    assert engineer.post(RUN+'/export').status_code == 409
    assert reviewer.get(RUN).status_code == 403
    assert attach(engineer, info, matching(engineer)['id']).status_code == 409
    approved = approve(reviewer, submit(engineer, v2))
    assert approved['approval_valid']
    response = engineer.post(RUN+'/export')
    assert response.status_code == 200
    with zipfile.ZipFile(BytesIO(response.content)) as z:
        assert set(z.namelist()) == {*NAMES,'evaluation.json','manifest.json','review.json'}
        assert z.read('evaluation.json') == evaluation
        for name in NAMES:
            assert z.read(name) == before[name]
        review = json.loads(z.read('review.json'))
        assert review['metric_source'] == 'evaluation.json' and review['metric_unit'] == 'ratio_0_1'
        assert review['metrics']['occupancy_iou'] == json.loads(evaluation)['metrics']['occupancy_iou']
    assert {name:(folder/name).read_bytes() for name in NAMES} == before
    assert path.read_bytes() == evaluation
    revised = engineer.post(RUN+'/review/revise', json={'comment':'Carry evaluation into another draft'}).json()
    assert revised['manifest']['schema_version'] == 2 and revised['evaluation_included']


def test_malformed_frame_list_is_invalid_without_leaking_stale_metrics(evaluated):
    _, engineer, _, _, path, _ = evaluated
    matching(engineer)
    value = json.loads(path.read_text())
    value['frames'].append('invalid frame entry')
    path.write_text(json.dumps(value))
    response = engineer.get(RUN)
    assert response.status_code == 200
    assert response.json()['evaluation']['status'] == 'invalid'
    assert response.json()['evaluation']['matched'] is None


@pytest.mark.parametrize('name', [*NAMES,'evaluation.json'])
def test_each_v2_source_change_blocks_export_even_with_same_stat(evaluated, name):
    _, engineer, reviewer, folder, path, _ = evaluated
    info = draft(engineer)
    v2 = attach(engineer, info, matching(engineer)['id']).json()
    approve(reviewer, submit(engineer, v2))
    target = path if name == 'evaluation.json' else folder/name
    stat = target.stat()
    data = bytearray(target.read_bytes())
    if name == 'evaluation.json':
        # Valid JSON with changed bytes, no mtime/size change.
        index = data.index(b'\n')
        data[index] = ord(' ')
    else:
        data[-1] ^= 1
    target.write_bytes(data)
    os.utime(target,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    review = engineer.get(RUN+'/review').json()
    assert not review['approval_valid'] and name in review['changed_artifacts']
    assert engineer.post(RUN+'/export').status_code == 409


def test_additive_v1_migration_keeps_accounts_sessions_history_and_approval(evaluated):
    app, engineer, reviewer, _, _, settings = evaluated
    old = approve(reviewer, submit(engineer, draft(engineer)))
    old_manifest = app.state.reviews.manifest_path('val_prediction',old['fingerprint']).read_bytes()
    tables = ['users','app_settings','sessions','runs','revisions','review_events']
    with app.state.database.connect() as c:
        c.execute('DROP TABLE revision_evaluations')
        c.execute('PRAGMA user_version=1')
        before = {table:[tuple(row) for row in c.execute('select * from '+table+' order by rowid')] for table in tables}
    restarted = create_app(settings)
    with restarted.state.database.connect() as c:
        assert c.execute('PRAGMA user_version').fetchone()[0] == 4
        after = {table:[tuple(row) for row in c.execute('select * from '+table+' order by rowid')] for table in tables}
    assert before == after
    with TestClient(restarted) as client:
        client.cookies.update(engineer.cookies)
        client.headers.update(engineer.headers)
        assert client.get('/api/auth/session').json()['user']['role'] == 'engineer'
        preserved = client.get(RUN+'/review').json()
        assert preserved['approval_valid'] and preserved['fingerprint'] == old['fingerprint']
        assert client.post(RUN+'/export').status_code == 200
    assert restarted.state.reviews.manifest_path('val_prediction',old['fingerprint']).read_bytes() == old_manifest


def test_evaluation_ids_cannot_escape_configured_roots(evaluated):
    _, engineer, _, _, path, _ = evaluated
    info = draft(engineer)
    identifier = matching(engineer)['id']
    outside = path.parents[2]/'outside.json'
    path.rename(outside)
    path.symlink_to(outside)
    assert attach(engineer,info,identifier).status_code == 409
    assert engineer.post(RUN+'/review/evaluation',json=dict(comment='no paths',revision_id=info['revision_id'],fingerprint=info['fingerprint'],evaluation_id='/etc/passwd')).status_code == 422
