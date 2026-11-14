"""Persistent local users, sessions, review revisions and append-only events."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import secrets
import sqlite3


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = directory / 'viewer.sqlite3'
        with self.connect() as connection:
            if connection.execute('PRAGMA user_version').fetchone()[0] > 4:
                raise ValueError('Viewer database schema is newer than this application')
            connection.execute('PRAGMA journal_mode=WAL')
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY, username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('engineer','reviewer')),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
                    csrf_hash TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS login_attempts (key TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS login_attempt_key ON login_attempts(key,created_at);
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL REFERENCES users(id),
                    created_at TEXT NOT NULL, current_revision_id INTEGER REFERENCES revisions(id)
                );
                CREATE TABLE IF NOT EXISTS revisions (
                    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
                    number INTEGER NOT NULL, fingerprint TEXT NOT NULL, manifest_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('draft','submitted','approved','rejected','needs_changes')),
                    created_at TEXT NOT NULL, submitted_at TEXT,
                    approved_by INTEGER REFERENCES users(id), approved_at TEXT,
                    UNIQUE(run_id,number)
                );
                CREATE TABLE IF NOT EXISTS review_events (
                    id INTEGER PRIMARY KEY, revision_id INTEGER NOT NULL REFERENCES revisions(id),
                    actor_id INTEGER NOT NULL REFERENCES users(id),
                    actor_username TEXT NOT NULL, actor_role TEXT NOT NULL,
                    action TEXT NOT NULL, from_state TEXT, to_state TEXT NOT NULL,
                    comment TEXT NOT NULL, created_at TEXT NOT NULL, fingerprint TEXT NOT NULL
                );
            ''')
            # Additive v1 -> v2. Existing rows and manifest bytes stay unchanged.
            connection.execute('BEGIN IMMEDIATE')
            try:
                connection.execute('''CREATE TABLE IF NOT EXISTS revision_evaluations (
                    revision_id INTEGER PRIMARY KEY REFERENCES revisions(id),
                    evaluation_id TEXT NOT NULL, sha256 TEXT NOT NULL,
                    byte_count INTEGER NOT NULL
                )''')
                connection.execute('''CREATE TABLE IF NOT EXISTS collections (
                    collection_id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL REFERENCES users(id),
                    created_at TEXT NOT NULL, current_revision_id INTEGER REFERENCES collection_revisions(id)
                )''')
                connection.execute('''CREATE TABLE IF NOT EXISTS collection_revisions (
                    id INTEGER PRIMARY KEY, collection_id TEXT NOT NULL REFERENCES collections(collection_id),
                    number INTEGER NOT NULL, fingerprint TEXT NOT NULL, manifest_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('draft','submitted','approved','rejected','needs_changes')),
                    created_at TEXT NOT NULL, submitted_at TEXT,
                    approved_by INTEGER REFERENCES users(id), approved_at TEXT,
                    UNIQUE(collection_id,number)
                )''')
                connection.execute('''CREATE TABLE IF NOT EXISTS collection_events (
                    id INTEGER PRIMARY KEY, revision_id INTEGER NOT NULL REFERENCES collection_revisions(id),
                    actor_id INTEGER NOT NULL REFERENCES users(id), actor_username TEXT NOT NULL,
                    actor_role TEXT NOT NULL, action TEXT NOT NULL, from_state TEXT, to_state TEXT NOT NULL,
                    comment TEXT NOT NULL, created_at TEXT NOT NULL, fingerprint TEXT NOT NULL
                )''')
                connection.execute('''CREATE TABLE IF NOT EXISTS inference_jobs (
                    id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL REFERENCES users(id),
                    request_key TEXT NOT NULL, profile_id TEXT NOT NULL, profile_json TEXT NOT NULL,
                    scene_name TEXT NOT NULL, frame_index INTEGER NOT NULL, sample_token TEXT NOT NULL,
                    split TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('queued','running','succeeded','failed','canceled')),
                    phase TEXT NOT NULL, created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0, worker_id TEXT,
                    child_pid INTEGER, child_started_ticks TEXT, inference_launch_count INTEGER NOT NULL DEFAULT 0,
                    run_id TEXT NOT NULL UNIQUE, evaluation_id TEXT NOT NULL,
                    evaluation_status TEXT, result_json TEXT, error_json TEXT, execution_mode TEXT NOT NULL DEFAULT 'real',
                    UNIQUE(owner_id,request_key)
                )''')
                connection.execute('''CREATE UNIQUE INDEX IF NOT EXISTS one_running_inference_job
                    ON inference_jobs(status) WHERE status='running' ''')
                connection.execute('''CREATE TABLE IF NOT EXISTS inference_job_events (
                    id INTEGER PRIMARY KEY, job_id TEXT NOT NULL REFERENCES inference_jobs(id),
                    actor_id INTEGER REFERENCES users(id), actor_type TEXT NOT NULL,
                    phase TEXT NOT NULL, message TEXT NOT NULL, created_at TEXT NOT NULL
                )''')
                connection.execute('''CREATE TABLE IF NOT EXISTS inference_workers (
                    id TEXT PRIMARY KEY, pid INTEGER NOT NULL, started_ticks TEXT NOT NULL,
                    started_at TEXT NOT NULL, heartbeat_at REAL NOT NULL, status TEXT NOT NULL,
                    execution_mode TEXT NOT NULL
                )''')
                connection.execute('PRAGMA user_version=4')
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            connection.execute('INSERT OR IGNORE INTO app_settings(key,value) VALUES(?,?)',
                               ('csrf_signing_key', secrets.token_hex(32)))
            self.csrf_secret = bytes.fromhex(connection.execute(
                'SELECT value FROM app_settings WHERE key=?', ('csrf_signing_key',)).fetchone()['value'])
        try:
            self.path.chmod(0o600)
        except PermissionError:
            pass

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys=ON')
        connection.execute('PRAGMA busy_timeout=15000')
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self):
        with self.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
