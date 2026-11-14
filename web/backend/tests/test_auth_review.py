from dataclasses import replace
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import zipfile

from fastapi.testclient import TestClient
import pytest

from web.backend.app.auth import SESSION_COOKIE, CSRF_COOKIE, digest, verify_password
from web.backend.app.main import create_app
from web.backend.tests.auth_support import ensure_user, login, TEST_PASSWORD
from web.backend.tests.test_viewer import fixture


@pytest.fixture
def actors(fixture):
    settings, folder, _ = fixture
    app = create_app(settings)
    ensure_user(app)
    ensure_user(app, 'test_reviewer', 'reviewer')
    with TestClient(app) as engineer, TestClient(app) as reviewer:
        login(engineer)
        login(reviewer, 'test_reviewer')
        yield app, engineer, reviewer, folder, settings


def register(engineer):
    response = engineer.post('/api/runs/prediction/register', json={'comment': 'Register verified inference files'})
    assert response.status_code == 200, response.text
    return response.json()


def submit(engineer, info):
    response = engineer.post('/api/runs/prediction/review/submit', json=dict(
        comment='Ready for independent review', revision_id=info['revision_id'], fingerprint=info['fingerprint']))
    assert response.status_code == 200, response.text
    return response.json()


def decide(client, info, decision='approved'):
    return client.post('/api/runs/prediction/review/decision', json=dict(
        comment='Checked coordinate alignment and saved artifacts', decision=decision,
        revision_id=info['revision_id'], fingerprint=info['fingerprint']))


def test_anonymous_and_login_csrf_are_blocked(fixture):
    app = create_app(fixture[0])
    with TestClient(app) as client:
        for path in ('/api/runs', '/api/runs/prediction', '/api/runs/prediction/occupancy',
                     '/api/runs/prediction/lidar', '/api/runs/prediction/bev', '/api/runs/prediction/review'):
            assert client.get(path).status_code == 401
        assert client.post('/api/runs/prediction/export').status_code == 401
        assert client.post('/api/auth/login', json=dict(username='any', password='any')).status_code == 403
        assert client.get('/api/auth/session').json()['user'] is None
        assert client.get('/api/health').json()['inference_enabled'] is False


def test_password_cookie_csrf_logout_and_expiry(actors):
    app, engineer, reviewer, _, _ = actors
    with app.state.database.connect() as connection:
        rows = connection.execute('SELECT username,password_hash FROM users ORDER BY id').fetchall()
        stored_sessions = connection.execute('SELECT token_hash FROM sessions').fetchall()
    assert rows[0]['password_hash'] != rows[1]['password_hash']
    assert all(verify_password(TEST_PASSWORD, row['password_hash']) for row in rows)
    assert TEST_PASSWORD not in app.state.database.path.read_bytes().decode('latin1')
    assert all(row['token_hash'] != engineer.cookies.get(SESSION_COOKIE) for row in stored_sessions)
    response = login(engineer)
    cookies = response.headers.get_list('set-cookie')
    assert all('HttpOnly' in value and 'SameSite=strict' in value and 'Max-Age=28800' in value for value in cookies)
    assert any(value.startswith(SESSION_COOKIE+'=') for value in cookies)
    assert any(value.startswith(CSRF_COOKIE+'=') for value in cookies)
    token = engineer.headers.pop('x-csrf-token')
    assert engineer.post('/api/runs/prediction/register', json={'comment': 'missing CSRF'}).status_code == 403
    engineer.headers['X-CSRF-Token'] = token
    assert engineer.post('/api/runs/prediction/register', json={'comment': 'bad origin'}, headers={'Origin': 'https://evil.example'}).status_code == 403
    assert engineer.post('/api/runs/prediction/register', json={'comment': 'other session'}, headers={'X-CSRF-Token': reviewer.headers['x-csrf-token']}).status_code == 403
    assert engineer.post('/api/runs/prediction/register', json={'comment': 'planted signed cookie'}, headers={
        'Cookie': f'{SESSION_COOKIE}={engineer.cookies.get(SESSION_COOKIE)}; {CSRF_COOKIE}={reviewer.cookies.get(CSRF_COOKIE)}',
        'X-CSRF-Token': reviewer.headers['x-csrf-token'],
    }).status_code == 403
    old_cookie = engineer.cookies.get(SESSION_COOKIE)
    assert engineer.post('/api/auth/logout').status_code == 200
    assert engineer.get('/api/runs').status_code == 401
    assert app.state.identity.authenticate(old_cookie) is None
    login(engineer)
    with app.state.database.connect() as connection:
        connection.execute('UPDATE sessions SET expires_at=? WHERE token_hash=?',
                           (time.time()-1, digest(engineer.cookies.get(SESSION_COOKIE))))
    assert engineer.get('/api/runs').status_code == 401


def test_roles_review_workflow_export_and_source_immutability(actors):
    app, engineer, reviewer, folder, _ = actors
    names = ['occupancy.npz', 'metadata.json', 'report.json', 'bev.png']
    before = {name: (folder/name).read_bytes() for name in names}
    assert reviewer.get('/api/runs').json()['runs'] == []
    assert reviewer.get('/api/runs/prediction').status_code == 403
    assert reviewer.post('/api/runs/prediction/register', json={'comment': 'unauthorized'}).status_code == 403
    info = register(engineer)
    assert info['state'] == 'draft' and not info['approval_valid']
    assert reviewer.get('/api/runs/prediction/occupancy').status_code == 403
    assert engineer.post('/api/runs/prediction/export').status_code == 409
    manifest = info['manifest']
    core = {key: value for key, value in manifest.items() if key != 'fingerprint_sha256'}
    assert hashlib.sha256(json.dumps(core,sort_keys=True,separators=(',', ':'),ensure_ascii=False).encode()).hexdigest() == info['fingerprint']
    assert manifest['sample_token_sources'] == ['metadata.json:sample_token', 'report.json:sample_token']
    assert manifest['npz_contains_sample_token'] is False
    for name in names:
        assert manifest['artifacts'][name]['sha256'] == hashlib.sha256(before[name]).hexdigest()
    assert engineer.post('/api/runs/prediction/review/notes', json={'comment': 'Engineering note'}).status_code == 200
    info = submit(engineer, info)
    assert len(reviewer.get('/api/runs').json()['runs']) == 1
    assert reviewer.get('/api/runs/prediction').status_code == 200
    assert decide(engineer, info).status_code == 403
    assert engineer.post('/api/runs/prediction/export').status_code == 409
    response = decide(reviewer, info)
    assert response.status_code == 200, response.text
    approved = response.json()
    assert approved['state'] == 'approved' and approved['approval_valid']
    assert decide(reviewer, info).status_code == 409
    archive_response = engineer.post('/api/runs/prediction/export')
    assert archive_response.status_code == 200
    with zipfile.ZipFile(BytesIO(archive_response.content)) as archive:
        assert set(archive.namelist()) == {*names, 'manifest.json', 'review.json'}
        for name in names:
            assert archive.read(name) == before[name]
        assert json.loads(archive.read('manifest.json')) == manifest
        review = json.loads(archive.read('review.json'))
        assert review['status'] == 'approved' and review['fingerprint_sha256'] == info['fingerprint']
        assert review['metrics']['occupancy_iou'] is None
        assert {event['actor_role'] for event in review['events']} == {'engineer', 'reviewer'}
    assert {name: (folder/name).read_bytes() for name in names} == before
    assert (app.state.reviews.manifest_root/'prediction'/f'{info["fingerprint"]}.json').is_file()


@pytest.mark.parametrize('filename', ['occupancy.npz', 'metadata.json', 'report.json', 'bev.png'])
def test_artifact_change_invalidates_approval_even_with_same_size_and_mtime(actors, filename):
    _, engineer, reviewer, folder, _ = actors
    info = submit(engineer, register(engineer))
    assert decide(reviewer, info).status_code == 200
    path = folder / filename
    stat = path.stat()
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    status = engineer.get('/api/runs/prediction/review').json()
    assert not status['artifact_matches'] and not status['approval_valid']
    assert filename in status['changed_artifacts']
    assert engineer.post('/api/runs/prediction/export').status_code == 409


def test_artifact_change_before_decision_and_sidecar_tamper_are_rejected(actors):
    app, engineer, reviewer, folder, _ = actors
    info = submit(engineer, register(engineer))
    path = folder / 'bev.png'
    before = path.read_bytes()
    path.write_bytes(before+b'changed')
    assert decide(reviewer, info).status_code == 409
    path.write_bytes(before)
    assert decide(reviewer, info).status_code == 200
    sidecar = app.state.reviews.manifest_root/'prediction'/f'{info["fingerprint"]}.json'
    sidecar.write_text('{}')
    assert engineer.get('/api/runs/prediction/review').json()['manifest_matches'] is False
    assert engineer.post('/api/runs/prediction/export').status_code == 409


@pytest.mark.parametrize('decision', ['rejected', 'needs_changes'])
def test_revisions_and_negative_decisions(actors, decision):
    _, engineer, reviewer, _, _ = actors
    info = submit(engineer, register(engineer))
    assert decide(reviewer, info, decision).status_code == 200
    assert engineer.post('/api/runs/prediction/export').status_code == 409
    draft = engineer.post('/api/runs/prediction/review/revise', json={'comment': 'Address review feedback'}).json()
    assert draft['state'] == 'draft' and draft['revision_number'] == 2
    assert reviewer.get('/api/runs').json()['runs'] == []
    assert reviewer.get('/api/runs/prediction').status_code == 403
    assert decide(reviewer, draft).status_code == 409
    assert engineer.post('/api/runs/prediction/review/submit', json=dict(
        comment='stale', revision_id=info['revision_id'], fingerprint=info['fingerprint'])).status_code == 409
    info = submit(engineer, draft)
    assert decide(reviewer, info).status_code == 200


def test_owner_checks_and_accounts_history_sessions_survive_restart(actors):
    app, engineer, reviewer, _, settings = actors
    info = submit(engineer, register(engineer))
    assert decide(reviewer, info).status_code == 200
    ensure_user(app, 'another_engineer')
    with TestClient(app) as another:
        login(another, 'another_engineer')
        assert another.post('/api/runs/prediction/review/revise', json={'comment': 'not owner'}).status_code == 403
        assert another.post('/api/runs/prediction/review/notes', json={'comment': 'not owner'}).status_code == 403
    restarted = create_app(settings)
    with TestClient(restarted) as client:
        client.cookies.update(engineer.cookies)
        client.headers.update(engineer.headers)
        assert client.get('/api/auth/session').json()['user']['username'] == 'test_engineer'
        history = client.get('/api/runs/prediction/review').json()
        assert history['state'] == 'approved' and history['approval_valid']
        assert len(history['events']) == 3
        assert client.post('/api/runs/prediction/export').status_code == 200
        assert login(client).status_code == 200


def test_cli_provisioning_and_login_rate_limit(fixture):
    settings, _, _ = fixture
    root = Path(__file__).resolve().parents[3]
    password = secrets.token_urlsafe(24)
    environment = dict(os.environ, VIEWER_STATE_ROOT=str(settings.state_directory))
    result = subprocess.run([sys.executable,str(root/'tools/create_viewer_user.py'),
                             '--username','cli_engineer','--role','engineer','--password-stdin'],
                            input=password+'\n', text=True, capture_output=True, env=environment)
    assert result.returncode == 0, result.stderr
    assert password not in result.stdout+result.stderr
    app = create_app(settings)
    with TestClient(app) as client:
        login(client, 'cli_engineer', password)
        assert client.post('/api/auth/logout').status_code == 200
        csrf = client.get('/api/auth/session').json()['csrf_token']
        for _ in range(5):
            assert client.post('/api/auth/login', json=dict(username='unknown',password='wrong'),
                               headers={'X-CSRF-Token':csrf}).status_code == 401
        assert client.post('/api/auth/login', json=dict(username='unknown',password='wrong'),
                           headers={'X-CSRF-Token':csrf}).status_code == 429


def test_secure_cookie_configuration(fixture):
    settings = replace(fixture[0], cookie_secure=True,
                       allowed_origins=('https://localhost:3000',))
    app = create_app(settings)
    ensure_user(app)
    with TestClient(app, base_url='https://localhost:3000') as client:
        csrf = client.get('/api/auth/session').json()['csrf_token']
        response = client.post('/api/auth/login', json=dict(username='test_engineer',password=TEST_PASSWORD),
                               headers={'X-CSRF-Token':csrf,'Origin':'https://localhost:3000'})
        assert response.status_code == 200
        assert all('Secure' in cookie and 'HttpOnly' in cookie for cookie in response.headers.get_list('set-cookie'))
        assert client.get('/api/runs').status_code == 200
