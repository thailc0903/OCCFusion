"""Ephemeral test credentials; never seed production accounts."""
from contextlib import contextmanager
import secrets

from fastapi.testclient import TestClient


TEST_PASSWORD = secrets.token_urlsafe(24)


def ensure_user(app, username='test_engineer', role='engineer'):
    with app.state.database.connect() as connection:
        exists = connection.execute('SELECT id FROM users WHERE username=?', (username,)).fetchone()
    if not exists:
        app.state.identity.create_user(username, TEST_PASSWORD, role)


def login(client, username='test_engineer', password=TEST_PASSWORD):
    bootstrap = client.get('/api/auth/session').json()
    response = client.post('/api/auth/login', json=dict(username=username, password=password),
                           headers={'X-CSRF-Token': bootstrap['csrf_token'], 'Origin': 'http://localhost:3000'})
    assert response.status_code == 200, response.text
    client.headers.update({'X-CSRF-Token': response.json()['csrf_token'], 'Origin': 'http://localhost:3000'})
    return response


@contextmanager
def engineer_client(app):
    ensure_user(app)
    with TestClient(app) as client:
        login(client)
        yield client
