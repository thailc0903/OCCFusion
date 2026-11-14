"""stdlib scrypt passwords, opaque expiring sessions and signed CSRF tokens."""
import base64
import hashlib
import hmac
import re
import secrets
import time
from urllib.parse import urlsplit

from fastapi import HTTPException

from .database import utc_now


SESSION_COOKIE = 'rav25_session'
CSRF_COOKIE = 'rav25_csrf'
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**17, 8, 1


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def encode(value):
    return base64.urlsafe_b64encode(value).decode('ascii')


def hash_password(password):
    if not 12 <= len(password) <= 1024:
        raise ValueError('Password must contain 12..1024 characters')
    salt = secrets.token_bytes(32)
    hashed = hashlib.scrypt(password.encode('utf-8'), salt=salt, n=SCRYPT_N,
                            r=SCRYPT_R, p=SCRYPT_P, dklen=64, maxmem=256*1024*1024)
    return f'scrypt-v1${encode(salt)}${encode(hashed)}'


def verify_password(password, stored):
    try:
        version, salt, expected = stored.split('$')
        if version != 'scrypt-v1':
            return False
        salt = base64.b64decode(salt, altchars=b'-_', validate=True)
        expected = base64.b64decode(expected, altchars=b'-_', validate=True)
        if len(salt) != 32 or len(expected) != 64:
            return False
        actual = hashlib.scrypt(password.encode('utf-8'), salt=salt, n=SCRYPT_N,
                                r=SCRYPT_R, p=SCRYPT_P, dklen=64, maxmem=256*1024*1024)
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def require_role(user, role):
    if user['role'] != role:
        raise HTTPException(403, f'This action requires the {role} role')


class Identity:
    def __init__(self, database, settings):
        self.db, self.settings = database, settings
        self.dummy_hash = f'scrypt-v1${encode(secrets.token_bytes(32))}${encode(secrets.token_bytes(64))}'

    def create_user(self, username, password, role):
        username = username.strip().lower()
        if not re.fullmatch(r'[a-z0-9][a-z0-9_.-]{2,63}', username):
            raise ValueError('Username must be 3..64 ASCII letters/digits/._-')
        if role not in ('engineer', 'reviewer'):
            raise ValueError('Role must be engineer or reviewer')
        hashed = hash_password(password)
        with self.db.transaction() as connection:
            if connection.execute('SELECT id FROM users WHERE username=?', (username,)).fetchone():
                raise ValueError('Username already exists')
            cursor = connection.execute('INSERT INTO users(username,password_hash,role,created_at) VALUES(?,?,?,?)',
                                        (username, hashed, role, utc_now()))
            return dict(id=cursor.lastrowid, username=username, role=role)

    def authenticate(self, raw_token):
        if not raw_token or len(raw_token) > 256:
            return None
        with self.db.connect() as connection:
            row = connection.execute('''SELECT s.*, u.username,u.role FROM sessions s
                JOIN users u ON u.id=s.user_id WHERE s.token_hash=?''', (digest(raw_token),)).fetchone()
            if not row or row['expires_at'] <= time.time():
                if row:
                    connection.execute('DELETE FROM sessions WHERE token_hash=?', (row['token_hash'],))
                return None
            return dict(row)

    @staticmethod
    def public_user(session):
        return dict(id=session['user_id'], username=session['username'], role=session['role'])

    def mint_csrf(self):
        payload = f'{int(time.time())}.{secrets.token_urlsafe(32)}'
        signature = hmac.new(self.db.csrf_secret, payload.encode(), hashlib.sha256).hexdigest()
        return f'{payload}.{signature}'

    def valid_csrf(self, token):
        try:
            issued, nonce, signature = token.split('.')
            payload = f'{issued}.{nonce}'
            expected = hmac.new(self.db.csrf_secret, payload.encode(), hashlib.sha256).hexdigest()
            return len(nonce) >= 32 and 0 <= time.time()-int(issued) <= self.settings.session_ttl_seconds and hmac.compare_digest(signature, expected)
        except (ValueError, TypeError, AttributeError):
            return False

    def check_csrf(self, request, session):
        origin = request.headers.get('origin')
        referer = request.headers.get('referer')
        if not origin and referer:
            parts = urlsplit(referer)
            origin = f'{parts.scheme}://{parts.netloc}'
        if origin and origin not in self.settings.allowed_origins:
            raise HTTPException(403, 'Request origin is not allowed')
        if request.headers.get('sec-fetch-site') == 'cross-site':
            raise HTTPException(403, 'Cross-site writes are forbidden')
        token = request.headers.get('x-csrf-token', '')
        cookie = request.cookies.get(CSRF_COOKIE, '')
        if not token or len(token) > 256 or not token.isascii() or not cookie.isascii() or not hmac.compare_digest(token, cookie) or not self.valid_csrf(token):
            raise HTTPException(403, 'Missing or invalid CSRF token')
        if session and not hmac.compare_digest(digest(token), session['csrf_hash']):
            raise HTTPException(403, 'CSRF token does not belong to this session')

    def set_csrf(self, response, token):
        response.set_cookie(CSRF_COOKIE, token, max_age=self.settings.session_ttl_seconds,
                            httponly=True, secure=self.settings.cookie_secure, samesite='strict', path='/')

    def bootstrap(self, request, response, session):
        token = request.cookies.get(CSRF_COOKIE, '')
        if not self.valid_csrf(token) or (session and digest(token) != session['csrf_hash']):
            token = self.mint_csrf()
            if session:
                with self.db.connect() as connection:
                    connection.execute('UPDATE sessions SET csrf_hash=? WHERE token_hash=?',
                                       (digest(token), session['token_hash']))
            self.set_csrf(response, token)
        response.headers['Cache-Control'] = 'no-store'
        if not session:
            response.delete_cookie(SESSION_COOKIE, path='/', secure=self.settings.cookie_secure,
                                   httponly=True, samesite='strict')
        with self.db.connect() as connection:
            needs_setup = connection.execute('SELECT count(*) FROM users').fetchone()[0] == 0
        return dict(user=self.public_user(session) if session else None, csrf_token=token,
                    expires_at=session['expires_at'] if session else None, needs_setup=needs_setup)

    def login(self, username, password, request, response):
        username = username.strip().lower()
        attempt_key = digest(f'{request.client.host if request.client else "local"}\0{username}')
        now = time.time()
        with self.db.transaction() as connection:
            connection.execute('DELETE FROM login_attempts WHERE created_at<?', (now-60,))
            if connection.execute('SELECT count(*) FROM login_attempts WHERE key=?', (attempt_key,)).fetchone()[0] >= 5:
                raise HTTPException(429, 'Too many login attempts; try again in one minute')
            connection.execute('INSERT INTO login_attempts(key,created_at) VALUES(?,?)', (attempt_key, now))
            row = connection.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
        if not verify_password(password, row['password_hash'] if row else self.dummy_hash) or not row:
            raise HTTPException(401, 'Invalid username or password')
        raw_token, csrf = secrets.token_urlsafe(32), self.mint_csrf()
        expires = now + self.settings.session_ttl_seconds
        old = request.cookies.get(SESSION_COOKIE)
        with self.db.transaction() as connection:
            connection.execute('DELETE FROM sessions WHERE expires_at<=?', (now,))
            if old:
                connection.execute('DELETE FROM sessions WHERE token_hash=?', (digest(old),))
            connection.execute('INSERT INTO sessions VALUES(?,?,?,?,?)',
                               (digest(raw_token), row['id'], digest(csrf), now, expires))
            connection.execute('DELETE FROM login_attempts WHERE key=?', (attempt_key,))
        response.set_cookie(SESSION_COOKIE, raw_token, max_age=self.settings.session_ttl_seconds,
                            httponly=True, secure=self.settings.cookie_secure, samesite='strict', path='/')
        self.set_csrf(response, csrf)
        response.headers['Cache-Control'] = 'no-store'
        return dict(user=dict(id=row['id'], username=row['username'], role=row['role']),
                    csrf_token=csrf, expires_at=expires, needs_setup=False)

    def logout(self, request, response):
        raw_token = request.cookies.get(SESSION_COOKIE)
        with self.db.connect() as connection:
            connection.execute('DELETE FROM sessions WHERE token_hash=?', (digest(raw_token),))
        for name in (SESSION_COOKIE, CSRF_COOKIE):
            response.delete_cookie(name, path='/', secure=self.settings.cookie_secure,
                                   httponly=True, samesite='strict')
