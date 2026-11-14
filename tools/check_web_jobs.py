#!/usr/bin/env python3
"""Isolated web-job QA. Default: fake CLI; --real-smoke: one explicitly authorized GPU job."""
import argparse
from dataclasses import replace
import hashlib
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

import httpx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.check_web_review import stop,wait_ready
from web.backend.app.config import Settings
from web.backend.tests.job_support import settings_to_file


def audit_sources():
    records=json.loads((ROOT/'work_dirs/web_jobs_preservation_before.json').read_text())['source_files']
    changed=[]
    for name,expected in records.items():
        actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if actual!=expected:changed.append(name)
    if changed:raise RuntimeError('Protected original sources changed: '+str(changed))
    return len(records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-build',action='store_true')
    parser.add_argument('--real-smoke',action='store_true',help='Exactly one scene-0103 frame 1 GPU job; no retries')
    args=parser.parse_args()
    baseline_count=audit_sources()
    for port in (8003,3003):
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1',port))==0:raise RuntimeError(f'QA port {port} is occupied')
    if args.skip_build:
        value=json.loads((ROOT/'web/frontend/.next-qa/routes-manifest.json').read_text())['rewrites']
        routes=value if isinstance(value,list) else sum(value.values(),[])
        if not any(row.get('destination')=='http://127.0.0.1:8003/api/:path*' for row in routes):
            raise RuntimeError('QA build targets another API; build without --skip-build first')
    qa=Path(tempfile.mkdtemp(prefix='job-smoke-' if args.real_smoke else 'jobs-qa-',dir=ROOT/'work'))
    runs=ROOT/'work_dirs' if args.real_smoke else qa/'runs';runs.mkdir(exist_ok=True)
    if not args.real_smoke:
        for name in ('infer_mini_cuda_v2','eval_existing_missing_gt'):
            shutil.copytree(ROOT/'work_dirs'/name,runs/name)
    settings=replace(Settings.load(),runs_root=runs,state_root=qa/'state',evaluation_roots=(runs,),benchmark_roots=(runs,),
        allowed_origins=('http://localhost:3003','http://127.0.0.1:3003','http://127.0.0.1:8003'))
    settings_file=qa/'settings.json';settings_to_file(settings,settings_file)
    env=dict(os.environ,VIEWER_RUNS_ROOT=str(runs),VIEWER_STATE_ROOT=str(qa/'state'),
        VIEWER_EVALUATION_ROOTS=json.dumps([str(runs)]),VIEWER_BENCHMARK_ROOTS=json.dumps([str(runs)]),
        VIEWER_ALLOWED_ORIGINS=json.dumps(settings.allowed_origins),VIEWER_API_ORIGIN='http://127.0.0.1:8003',
        VIEWER_NEXT_DIST_DIR='.next-qa',NEXT_TELEMETRY_DISABLED='1',VIEWER_E2E_OUTPUT=str(qa),
        VIEWER_URL='http://127.0.0.1:3003',VIEWER_JOB_TEST_SETTINGS=str(settings_file),
        VIEWER_JOB_TEST_MODE='real-smoke' if args.real_smoke else 'fixture')
    env['PATH']=str(ROOT/'.cache/toolchains/node/bin')+os.pathsep+env.get('PATH','')
    npm=shutil.which('npm',path=env['PATH']);node=shutil.which('node',path=env['PATH'])
    if not node or not npm:raise RuntimeError('Node/npm unavailable')
    passwords={role:secrets.token_urlsafe(28) for role in ('engineer','reviewer')}
    for role,password in passwords.items():
        subprocess.run([sys.executable,str(ROOT/'tools/create_viewer_user.py'),'--username','qa_'+role,'--role',role,'--password-stdin'],
            input=password+'\n',text=True,cwd=ROOT,env=env,check=True)
        env[f'VIEWER_E2E_{role.upper()}_USERNAME']='qa_'+role
        env[f'VIEWER_E2E_{role.upper()}_PASSWORD']=password
    print('Web job QA directory:',qa,flush=True)
    if not args.skip_build:subprocess.run([npm,'run','build'],cwd=ROOT/'web/frontend',env=env,check=True)
    # An exclusive sentinel prevents accidental repetition of the one authorized
    # smoke through rerunning this harness after a browser/assertion failure.
    if args.real_smoke:
        with (ROOT/'work_dirs/web_job_gpu_smoke_intent.json').open('x') as stream:
            json.dump(dict(qa_directory=str(qa),scene='scene-0103',frame=1,max_jobs=1,created_at=time.time()),stream,indent=2)
    backend=frontend=worker=None
    logs={kind:(qa/(kind+'.log')).open('ab') for kind in ('backend','frontend','worker')}
    backend_command=[sys.executable,'-m','uvicorn','web.backend.app.main:app','--host','127.0.0.1','--port','8003']
    try:
        backend=subprocess.Popen(backend_command,cwd=ROOT,env=env,stdout=logs['backend'],stderr=subprocess.STDOUT)
        frontend=subprocess.Popen([node,str(ROOT/'web/frontend/node_modules/next/dist/bin/next'),'start',str(ROOT/'web/frontend'),
            '--hostname','127.0.0.1','--port','3003'],cwd=ROOT,env=env,stdout=logs['frontend'],stderr=subprocess.STDOUT)
        wait_ready('http://127.0.0.1:8003/api/health',backend);wait_ready('http://127.0.0.1:3003/jobs',frontend)
        direct=httpx.get('http://127.0.0.1:8003/api/health').json()
        assert httpx.get('http://127.0.0.1:3003/api/health').json()['workspace_id']==direct['workspace_id'],'Proxy must target isolated QA SQLite'
        assert direct['model_modules_loaded'] is False
        if args.real_smoke:
            worker=subprocess.Popen([sys.executable,'-u',str(ROOT/'tools/run_inference_worker.py'),'--max-jobs','1'],cwd=ROOT,env=env,stdout=logs['worker'],stderr=subprocess.STDOUT)
        spec='jobs-smoke.spec.ts' if args.real_smoke else 'jobs.spec.ts'
        subprocess.run([npm,'run','test:e2e','--',spec],cwd=ROOT/'web/frontend',env=env,check=True)
        with httpx.Client(base_url='http://127.0.0.1:8003',timeout=120) as client:
            csrf=client.get('/api/auth/session').json()['csrf_token']
            response=client.post('/api/auth/login',json=dict(username='qa_engineer',password=passwords['engineer']),headers={'X-CSRF-Token':csrf});response.raise_for_status()
            before=client.get('/api/jobs').json()['jobs']
            if args.real_smoke:
                assert len(before)==1 and before[0]['status']=='succeeded' and before[0]['execution_mode']=='real'
                assert before[0]['inference_launch_count']==1
                worker.wait(timeout=15);assert worker.returncode==0
            else:
                assert {j['status'] for j in before}=={'canceled','failed','succeeded'}
                assert all(j['execution_mode']=='fixture' or j['inference_launch_count']==0 for j in before)
            stop(backend)
            backend=subprocess.Popen(backend_command,cwd=ROOT,env=env,stdout=logs['backend'],stderr=subprocess.STDOUT)
            wait_ready('http://127.0.0.1:8003/api/health',backend)
            assert client.get('/api/auth/session').json()['user']['username']=='qa_engineer'
            after=client.get('/api/jobs').json()['jobs'];assert before==after
            for j in after:
                if j['status']=='succeeded':
                    view=client.get('/api/runs/infer_job_'+j['id']);view.raise_for_status()
                    assert view.json()['review']['registered'] is False
            assert client.get('/api/health').json()['model_modules_loaded'] is False
        evidence=dict(status='passed',mode=env['VIEWER_JOB_TEST_MODE'],qa_directory=str(qa),run_root=str(runs),
            backend_restarted=True,accounts_session_jobs_events_preserved=True,jobs=after,
            protected_original_files=audit_sources(),protected_original_files_unchanged=True,
            web_model_modules_loaded=False,playwright_report=str(qa/'browser-results.json'),
            screenshots=[str(p) for p in sorted(qa.glob('*.png'))])
        destination=ROOT/'work_dirs'/('web_job_gpu_smoke_evidence.json' if args.real_smoke else 'web_jobs_fixture_evidence.json')
        destination.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
        print('WEB_JOBS_QA_PASSED',json.dumps(dict(status='passed',mode=evidence['mode'],qa_directory=str(qa),jobs=len(after),protected_files=baseline_count)),flush=True)
    finally:
        stop(worker);stop(frontend);stop(backend)
        for stream in logs.values():stream.close()
        report=ROOT/'web/frontend/test-results/browser-results.json'
        if report.exists():shutil.copyfile(report,qa/'browser-results.json')
        for directory in (qa,ROOT/'web/frontend/test-results'):
            for file in directory.rglob('*'):
                if file.is_file() and file.suffix in ('.json','.md','.txt','.log'):
                    original=file.read_text(errors='replace');sanitized=original
                    for password in passwords.values():sanitized=sanitized.replace(password,'[REDACTED_QA_PASSWORD]')
                    if original!=sanitized:file.write_text(sanitized)


if __name__=='__main__':main()
