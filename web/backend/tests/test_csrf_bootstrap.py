"""Deterministic overlapping bootstrap schedules, using private QA SQLite only."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.cookies import SimpleCookie
from threading import Barrier, Event
from types import SimpleNamespace
import time

from fastapi import Response
from fastapi.testclient import TestClient
import pytest
from starlette.requests import Request

from web.backend.app import auth
from web.backend.app.auth import CSRF_COOKIE, SESSION_COOKIE, Identity, digest
from web.backend.app.config import Settings
from web.backend.app.database import Database
from web.backend.app.main import create_app
from web.backend.tests.auth_support import ensure_user, login


@pytest.fixture
def authenticated(tmp_path):
    settings = Settings(tmp_path/'runs', (tmp_path/'dataset',), state_root=tmp_path/'state')
    app = create_app(settings)
    ensure_user(app)
    with TestClient(app) as client:
        old = client.get('/api/auth/session').json()['csrf_token']
        login(client)
        yield app, client, settings, old


def request_for(raw_session, csrf=None, method='GET', **headers):
    cookie = f'{SESSION_COOKIE}={raw_session}'
    if csrf is not None:
        cookie += f'; {CSRF_COOKIE}={csrf}'
    return Request(dict(type='http', method=method, path='/api/auth/session',
        headers=[(b'cookie', cookie.encode()), *[(k.lower().encode(), v.encode()) for k, v in headers.items()]]))


def overlap(identities, raw_session, cookies):
    """Both authenticate before either bootstrap, then A commits before B.

    Return responses without delivering them, so the test can deliver B then A.
    Events/barrier fix the schedule; no sleep, probabilistic stress or retries.
    Separate Identity instances model independent backend processes.
    """
    snapshots_read = Barrier(2)
    a_finished = Event()

    def run(index):
        identity = identities[index]
        session = identity.authenticate(raw_session)
        assert session is not None
        snapshots_read.wait(timeout=10)
        if index == 1:
            assert a_finished.wait(timeout=10)
        response = Response()
        try:
            body = identity.bootstrap(request_for(raw_session, cookies[index]), response, session)
            return body, response, session
        finally:
            if index == 0:
                a_finished.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, index) for index in range(2)]
        return [future.result(timeout=15) for future in futures]


def deliver(response, browser_cookie):
    for header in response.headers.getlist('set-cookie'):
        cookies = SimpleCookie(header)
        if CSRF_COOKIE in cookies:
            browser_cookie = cookies[CSRF_COOKIE].value
    return browser_cookie


@pytest.mark.parametrize('cookie_state', ['old', 'missing'])
def test_overlapping_bootstrap_reversed_responses_allow_valid_post(authenticated, cookie_state):
    app, client, settings, old = authenticated
    raw_session = client.cookies.get(SESSION_COOKIE)
    cookie = old if cookie_state == 'old' else None
    assert cookie is None or app.state.identity.valid_csrf(cookie)
    # Both requests read the same original hash before A and B bootstrap.
    peer = Identity(Database(settings.state_directory), settings)
    a, b = overlap((app.state.identity, peer), raw_session, (cookie, cookie))
    assert a[2] == b[2]
    # Browser receives B first and A last, including body/header token state.
    browser_cookie = deliver(b[1], cookie)
    browser_cookie = deliver(a[1], browser_cookie)
    header_token = a[0]['csrf_token']
    assert app.state.identity.valid_csrf(header_token)
    assert browser_cookie == header_token
    response = client.post('/api/auth/logout', headers={
        'Cookie': f'{SESSION_COOKIE}={raw_session}; {CSRF_COOKIE}={browser_cookie}',
        'X-CSRF-Token': header_token, 'Origin': 'http://localhost:3000',
    })
    # Before the fix: exactly 403 "CSRF token does not belong to this session".
    assert response.status_code == 200, response.text
    assert app.state.identity.authenticate(raw_session) is None


@pytest.mark.parametrize('states', [
    ('current', 'current'), ('current', 'old'), ('old', 'current'),
    ('current', 'missing'), ('missing', 'current'), ('old', 'missing'),
    ('missing', 'old'), ('invalid', 'invalid'),
])
@pytest.mark.parametrize('delivery_order', [(0, 1), (1, 0)])
def test_all_bootstrap_responses_converge_without_updating_session(authenticated, states, delivery_order):
    app, client, settings, old = authenticated
    identity = app.state.identity
    raw = client.cookies.get(SESSION_COOKIE)
    before = identity.authenticate(raw)
    current = client.headers['x-csrf-token']
    choices = dict(current=current, old=old, missing=None, invalid='not-a-signed-token')
    cookies = tuple(choices[state] for state in states)
    peer = Identity(Database(settings.state_directory), settings)
    responses = overlap((identity, peer), raw, cookies)
    assert identity.authenticate(raw) == before
    browser_cookie = cookies[0]
    for index in delivery_order:
        browser_cookie = deliver(responses[index][1], browser_cookie)
    header = responses[delivery_order[-1]][0]['csrf_token']
    # New login already supplies the stable token: focus/bootstrap never rotates it.
    equal = current == header == browser_cookie
    assert equal
    accepted = client.post('/api/auth/logout', headers={
        'Cookie': f'{SESSION_COOKIE}={raw}; {CSRF_COOKIE}={browser_cookie}',
        'X-CSRF-Token': header,
    })
    assert accepted.status_code == 200, accepted.text


def test_legacy_session_and_token_survive_bootstrap_and_restart(authenticated):
    app, client, settings, _ = authenticated
    identity = app.state.identity
    raw = client.cookies.get(SESSION_COOKIE)
    legacy = identity.mint_csrf()
    # Model a pre-fix v4 session in QA; never use production as a fixture.
    with app.state.database.connect() as connection:
        connection.execute('UPDATE sessions SET csrf_hash=? WHERE token_hash=?', (digest(legacy), digest(raw)))
    before = identity.authenticate(raw)
    identity.check_csrf(request_for(raw, legacy, 'POST', **{'X-CSRF-Token': legacy}), before)
    first_response = Response()
    body = identity.bootstrap(request_for(raw, legacy), first_response, before)
    assert identity.authenticate(raw) == before
    stable = body['csrf_token']
    restarted = create_app(settings)
    assert restarted.state.identity.authenticate(raw) == before
    assert restarted.state.identity.session_csrf(before) == stable
    with TestClient(restarted) as new_client:
        cookies = f'{SESSION_COOKIE}={raw}; {CSRF_COOKIE}={stable}'
        info = new_client.get('/api/auth/session', headers={'Cookie': cookies})
        same = info.json()['csrf_token'] == stable
        assert same and info.json()['user']['username'] == 'test_engineer'
        assert 'set-cookie' not in info.headers
        # An in-flight pre-upgrade client can still write with its signed legacy
        # token after bootstrap. No DB hash or approval needs to be migrated.
        response = new_client.post('/api/auth/logout', headers={
            'Cookie': f'{SESSION_COOKIE}={raw}; {CSRF_COOKIE}={legacy}',
            'X-CSRF-Token': legacy, 'Origin': 'http://localhost:3000',
        })
        assert response.status_code == 200, response.text
        assert new_client.get('/api/jobs', headers={'Cookie': cookies}).status_code == 401


@pytest.mark.parametrize('attack', [
    'other_session', 'header_mismatch', 'missing_cookie', 'signature',
    'origin', 'referer', 'cross_site', 'anonymous_token',
])
def test_stable_csrf_still_rejects_invalid_writes(authenticated, attack):
    app, client, _, _ = authenticated
    raw = client.cookies.get(SESSION_COOKIE)
    token = client.headers['x-csrf-token']
    header, cookie = token, token
    extras = {'Origin': 'http://localhost:3000'}
    with TestClient(app) as other:
        login(other)  # Same user, different live session: binding must be per session.
        foreign = other.headers['x-csrf-token']
        if attack == 'other_session':
            header = cookie = foreign
            assert app.state.identity.valid_csrf(header)
        elif attack == 'header_mismatch':
            header = foreign
        elif attack == 'missing_cookie':
            cookie = None
        elif attack == 'signature':
            header = cookie = token[:-1] + ('1' if token[-1] != '1' else '2')
        elif attack == 'origin':
            extras['Origin'] = 'https://evil.example'
        elif attack == 'referer':
            client.headers.pop('origin')
            extras = {'Referer': 'https://evil.example/page'}
        elif attack == 'cross_site':
            extras['Sec-Fetch-Site'] = 'cross-site'
        elif attack == 'anonymous_token':
            header = cookie = app.state.identity.mint_csrf()
        cookie_header = f'{SESSION_COOKIE}={raw}' + (f'; {CSRF_COOKIE}={cookie}' if cookie else '')
        response = client.post('/api/auth/logout', headers={
            'Cookie': cookie_header, 'X-CSRF-Token': header, **extras,
        })
        assert response.status_code == 403, response.text
        if attack in ('other_session', 'anonymous_token'):
            assert response.json()['detail'] == 'CSRF token does not belong to this session'
        assert app.state.identity.authenticate(raw) is not None
    # Separate deliberate action with valid credentials, not an automatic retry.
    good = client.post('/api/auth/logout', headers={
        'Cookie': f'{SESSION_COOKIE}={raw}; {CSRF_COOKIE}={token}',
        'X-CSRF-Token': token, 'Origin': 'http://localhost:3000',
    })
    assert good.status_code == 200, good.text


@pytest.mark.parametrize('secure', [False, True])
def test_recovered_bootstrap_cookie_flags(authenticated, secure):
    _, _, settings, _ = authenticated
    app = create_app(replace(settings, cookie_secure=secure))
    with TestClient(app, base_url='https://testserver' if secure else 'http://testserver') as client:
        login(client)
        current = client.headers['x-csrf-token']
        client.cookies.delete(CSRF_COOKIE)
        response = client.get('/api/auth/session')
        same = response.json()['csrf_token'] == current
        assert same
        cookies = response.headers.get_list('set-cookie')
        assert len(cookies) == 1
        cookie = SimpleCookie(cookies[0])[CSRF_COOKIE]
        assert cookie['httponly'] and cookie['samesite'] == 'strict'
        assert bool(cookie['secure']) == secure
        assert cookie['path'] == '/' and cookie['max-age'] == str(settings.session_ttl_seconds)
        assert not cookie['domain']


def test_session_and_token_expiry_agree_at_fractional_deadline(authenticated, monkeypatch):
    app, client, _, _ = authenticated
    raw = client.cookies.get(SESSION_COOKIE)
    deadline = int(time.time()) + 120.75
    with app.state.database.connect() as connection:
        connection.execute('UPDATE sessions SET expires_at=? WHERE token_hash=?', (deadline, digest(raw)))
    clock = SimpleNamespace(time=lambda: deadline - .25)
    monkeypatch.setattr(auth, 'time', clock)
    snapshot = app.state.identity.authenticate(raw)
    response = client.get('/api/auth/session')
    token = response.json()['csrf_token']
    assert response.json()['user'] is not None and app.state.identity.valid_csrf(token)
    app.state.identity.check_csrf(request_for(raw, token, 'POST', **{'X-CSRF-Token': token}), snapshot)
    clock.time = lambda: deadline
    assert not app.state.identity.valid_csrf(token)
    assert app.state.identity.authenticate(raw) is None
    cookies = f'{SESSION_COOKIE}={raw}; {CSRF_COOKIE}={token}'
    assert client.post('/api/auth/logout', headers={'Cookie': cookies, 'X-CSRF-Token': token}).status_code == 401
    assert client.get('/api/auth/session', headers={'Cookie': cookies}).json()['user'] is None


def test_late_bootstrap_after_logout_cannot_resurrect_session(authenticated):
    app, client, _, old = authenticated
    raw = client.cookies.get(SESSION_COOKIE)
    stale_snapshot = app.state.identity.authenticate(raw)
    assert client.post('/api/auth/logout').status_code == 200
    response = Response()
    late = app.state.identity.bootstrap(request_for(raw, old), response, stale_snapshot)
    assert app.state.identity.authenticate(raw) is None
    token = late['csrf_token']
    cookies = f'{SESSION_COOKIE}={raw}; {CSRF_COOKIE}={token}'
    assert client.post('/api/auth/logout', headers={'Cookie': cookies, 'X-CSRF-Token': token}).status_code == 401
    with app.state.database.connect() as connection:
        assert connection.execute('SELECT COUNT(*) FROM sessions').fetchone()[0] == 0


def test_legacy_signature_ttl_and_nonce_namespace(authenticated, monkeypatch):
    app, _, settings, _ = authenticated
    # Legacy token_urlsafe nonces can start s1_ but never contain ':'.
    issued = int(time.time())
    legacy = app.state.identity.sign_csrf(f'{issued}.s1_' + 'a' * 40)
    clock = SimpleNamespace(time=lambda: issued + .5)
    monkeypatch.setattr(auth, 'time', clock)
    assert app.state.identity.valid_csrf(legacy)
    clock.time = lambda: issued + settings.session_ttl_seconds + 1
    assert not app.state.identity.valid_csrf(legacy)
    clock.time = lambda: issued - 1
    assert not app.state.identity.valid_csrf(legacy)
    for invalid in (None, b'bytes', '', 'x.y.z', '1.s1:' + 'x' * 300, '1.nonce.💥'):
        assert not app.state.identity.valid_csrf(invalid)
