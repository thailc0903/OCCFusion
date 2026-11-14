#!/usr/bin/env python3
"""Real local QA with isolated SQLite, artifact copies and ephemeral secrets.

Run with .venv-web/bin/python. Uses ports 8001/3001 and .next-qa; it does not
stop the user's viewer or mutate the four original inference artifacts.
"""
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import zipfile

import httpx


ROOT = Path(__file__).resolve().parents[1]
NAMES = ('occupancy.npz', 'metadata.json', 'report.json', 'bev.png')


def fingerprints(folder):
    return {name: hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in NAMES}


def wait_ready(url, process):
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError(f'Service exited before readiness: {url}; see QA logs')
        try:
            response = httpx.get(url, timeout=2)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(.25)
    raise RuntimeError(f'Service did not become ready: {url}')


def stop(process):
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    source = ROOT/'work_dirs/infer_mini_cuda_v2'
    before = fingerprints(source)
    val_source = ROOT/'work_dirs/infer_mini_val_scene0103_f1'
    val_before = fingerprints(val_source)
    evaluation_source = ROOT/'work_dirs/eval_mini_val_scene0103_f1/evaluation.json'
    evaluation_hash = hashlib.sha256(evaluation_source.read_bytes()).hexdigest()
    (ROOT/'work').mkdir(exist_ok=True)
    for port in (8001, 3001):
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'QA port {port} is already in use; stop that QA process first')
    qa = Path(tempfile.mkdtemp(prefix='review-qa-', dir=ROOT/'work'))
    run_root = qa/'runs'
    run_root.mkdir()
    for name in ('infer_mini_cuda_v2', 'infer_mini_check', 'infer_mini_cuda',
                 'infer_mini_val_scene0103_f1', 'eval_mini_val_scene0103_f1', 'eval_existing_missing_gt'):
        shutil.copytree(ROOT/'work_dirs'/name, run_root/name)
    for name in ('auth_tamper', 'auth_negative'):
        shutil.copytree(source, run_root/name)
    environment = dict(os.environ,
        VIEWER_RUNS_ROOT=str(run_root), VIEWER_STATE_ROOT=str(qa/'state'),
        VIEWER_EVALUATION_ROOTS=json.dumps([str(run_root)]),
        VIEWER_ALLOWED_ORIGINS=json.dumps(['http://localhost:3001','http://127.0.0.1:3001',
                                          'http://localhost:8001','http://127.0.0.1:8001']),
        VIEWER_API_ORIGIN='http://127.0.0.1:8001', VIEWER_NEXT_DIST_DIR='.next-qa',
        NEXT_TELEMETRY_DISABLED='1',
    )
    node_directory = ROOT/'.cache/toolchains/node/bin'
    if node_directory.is_dir():
        environment['PATH'] = str(node_directory) + os.pathsep + environment.get('PATH','')
    npm = shutil.which('npm', path=environment['PATH'])
    node = shutil.which('node', path=environment['PATH'])
    if not npm or not node:
        raise RuntimeError('Node/npm is unavailable')
    passwords = {role: secrets.token_urlsafe(28) for role in ('engineer','reviewer')}
    for role in passwords:
        subprocess.run([sys.executable, str(ROOT/'tools/create_viewer_user.py'),
                        '--username',f'qa_{role}','--role',role,'--password-stdin'],
                       input=passwords[role]+'\n', text=True, check=True, env=environment, cwd=ROOT)
    print(f'QA directory: {qa}', flush=True)
    subprocess.run([npm,'run','build'],cwd=ROOT/'web/frontend',env=environment,check=True)
    backend = frontend = None
    backend_log = (qa/'backend.log').open('ab')
    frontend_log = (qa/'frontend.log').open('ab')
    backend_args = [sys.executable,'-m','uvicorn','web.backend.app.main:app','--host','127.0.0.1','--port','8001']
    try:
        backend = subprocess.Popen(backend_args,cwd=ROOT,env=environment,stdout=backend_log,stderr=subprocess.STDOUT)
        frontend = subprocess.Popen([node,str(ROOT/'web/frontend/node_modules/next/dist/bin/next'),
                                     'start',str(ROOT/'web/frontend'),'--hostname','127.0.0.1','--port','3001'],
                                    cwd=ROOT,env=environment,stdout=frontend_log,stderr=subprocess.STDOUT)
        wait_ready('http://127.0.0.1:8001/api/health',backend)
        wait_ready('http://127.0.0.1:3001/',frontend)
        with httpx.Client(base_url='http://127.0.0.1:8001',timeout=20) as persistent:
            csrf = persistent.get('/api/auth/session').json()['csrf_token']
            login = persistent.post('/api/auth/login',json=dict(username='qa_engineer',password=passwords['engineer']),headers={'X-CSRF-Token':csrf})
            login.raise_for_status()
            test_environment = dict(environment, VIEWER_URL='http://127.0.0.1:3001',
                                    VIEWER_E2E_RUNS_ROOT=str(run_root),VIEWER_E2E_OUTPUT=str(qa))
            for role in passwords:
                test_environment[f'VIEWER_E2E_{role.upper()}_USERNAME'] = f'qa_{role}'
                test_environment[f'VIEWER_E2E_{role.upper()}_PASSWORD'] = passwords[role]
            subprocess.run([npm,'run','test:e2e'],cwd=ROOT/'web/frontend',env=test_environment,check=True)
            info = persistent.get('/api/runs/infer_mini_cuda_v2/review').json()
            assert info['state'] == 'approved' and info['approval_valid']
            events_before = len(info['events'])
            print('Restarting the actual QA backend to verify persisted accounts, sessions and history',flush=True)
            stop(backend)
            backend = subprocess.Popen(backend_args,cwd=ROOT,env=environment,stdout=backend_log,stderr=subprocess.STDOUT)
            wait_ready('http://127.0.0.1:8001/api/health',backend)
            session = persistent.get('/api/auth/session').json()
            assert session['user']['username'] == 'qa_engineer'
            restored = persistent.get('/api/runs/infer_mini_cuda_v2/review').json()
            assert restored['approval_valid'] and len(restored['events']) == events_before
            response = persistent.post('/api/runs/infer_mini_cuda_v2/export',headers={'X-CSRF-Token':session['csrf_token']})
            response.raise_for_status()
            with zipfile.ZipFile(BytesIO(response.content)) as archive:
                assert set(archive.namelist()) == {*NAMES,'manifest.json','review.json'}
                assert {name:hashlib.sha256(archive.read(name)).hexdigest() for name in NAMES} == before
                manifest = json.loads(archive.read('manifest.json'))
                review = json.loads(archive.read('review.json'))
                assert manifest['fingerprint_sha256'] == restored['fingerprint']
                assert manifest['npz_contains_sample_token'] is False
                assert review['metrics']['occupancy_iou'] is None and review['metrics']['occupancy_miou'] is None
            with httpx.Client(base_url='http://127.0.0.1:8001',timeout=20) as checker:
                csrf = checker.get('/api/auth/session').json()['csrf_token']
                assert checker.post('/api/auth/login',json=dict(username='qa_reviewer',password=passwords['reviewer']),headers={'X-CSRF-Token':csrf}).status_code == 200
            assert fingerprints(source) == before
            assert fingerprints(val_source) == val_before
            assert hashlib.sha256(evaluation_source.read_bytes()).hexdigest() == evaluation_hash
            val_review = persistent.get('/api/runs/infer_mini_val_scene0103_f1/review').json()
            assert val_review['approval_valid'] and val_review['revision_number'] == 2
            assert val_review['revisions'][0]['state'] == 'approved'
            val_export = persistent.post('/api/runs/infer_mini_val_scene0103_f1/export',headers={'X-CSRF-Token':session['csrf_token']})
            val_export.raise_for_status()
            with zipfile.ZipFile(BytesIO(val_export.content)) as archive:
                assert set(archive.namelist()) == {*NAMES,'evaluation.json','manifest.json','review.json'}
                assert {name:hashlib.sha256(archive.read(name)).hexdigest() for name in NAMES} == val_before
                assert hashlib.sha256(archive.read('evaluation.json')).hexdigest() == evaluation_hash
                val_manifest = json.loads(archive.read('manifest.json'))
                assert val_manifest['schema_version'] == 2
                assert val_manifest['artifacts']['evaluation.json']['sha256'] == evaluation_hash
            val_detail = persistent.get('/api/runs/infer_mini_val_scene0103_f1').json()
            assert val_detail['evaluation']['status'] == 'matched'
            (qa/'evaluation-approved-after-restart.zip').write_bytes(val_export.content)
            evidence = dict(status='passed',qa_directory=str(qa),
                            backend_process_restarted=True,accounts_persisted=True,
                            session_persisted=True,history_events_before_restart=events_before,
                            history_events_after_restart=len(restored['events']),
                            approved_fingerprint=restored['fingerprint'],
                            export_hashes=before,original_artifacts_unchanged=True,
                            metrics=review['metrics'], evaluation_review=dict(
                                revision_number=2, manifest_schema_version=2,
                                fingerprint=val_review['fingerprint'], evaluation_sha256=evaluation_hash,
                                source_artifact_hashes=val_before, source_files_unchanged=True,
                                metrics=val_detail['evaluation']['matched']['metrics'],
                                legacy_revision_approved=True, export_includes_exact_evaluation=True))
            (ROOT/'work_dirs/web_review_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
            (ROOT/'work_dirs/web_evaluation_review_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
            print('WEB_REVIEW_QA_PASSED',json.dumps(dict(status='passed',qa_directory=str(qa),
                  backend_process_restarted=True,legacy_revision_valid=True,evaluation_revision_valid=True,
                  evidence='work_dirs/web_evaluation_review_evidence.json'),ensure_ascii=False),flush=True)
    finally:
        stop(frontend); stop(backend)
        frontend_log.close(); backend_log.close()


if __name__ == '__main__':
    main()
