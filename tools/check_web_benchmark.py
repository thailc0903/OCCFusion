#!/usr/bin/env python3
"""CPU/Chromium QA of saved benchmark collections in an isolated SQLite/root."""
import argparse
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
import signal
import time
import zipfile

import httpx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.check_web_review import stop,wait_ready


def source_hashes():
    records=json.loads((ROOT/'work_dirs/web_benchmark_preservation_before.json').read_text())['source_files']
    return {path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in records}


def verify_export(path,info):
    with zipfile.ZipFile(path) as archive:
        manifest=json.loads(archive.read('manifest.json'));review=json.loads(archive.read('review.json'))
        assert manifest['kind']=='occfusion_benchmark_collection' and manifest['fingerprint_sha256']==info['fingerprint']
        assert manifest['expected_frame_count']==81 and manifest['successful_frame_count']==79 and manifest['failed_frame_count']==2
        assert review['status']=='partial' and review['metric_source']=='benchmark/metrics_pooled.json'
        assert review['latency_source']=='benchmark/initial_report.json' and 'learned 4D flow' in review['playback_note']
        assert len(archive.namelist())==len(manifest['artifacts'])+2
        for name,expected in manifest['artifacts'].items():
            digest=hashlib.sha256();size=0
            with archive.open(name) as stream:
                for chunk in iter(lambda:stream.read(2*1024*1024),b''):
                    digest.update(chunk);size+=len(chunk)
            assert digest.hexdigest()==expected['sha256'] and size==expected['bytes'],name
        return dict(file_count=len(archive.namelist()),artifact_hashes_checked=len(manifest['artifacts']),
                    fingerprint=manifest['fingerprint_sha256'],status=review['status'],metrics=review['metrics'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--skip-build',action='store_true')
    args=parser.parse_args();before=source_hashes()
    if args.skip_build:
        routes=json.loads((ROOT/'web/frontend/.next-qa/routes-manifest.json').read_text())['rewrites']
        rewrites=routes if isinstance(routes,list) else sum(routes.values(),[])
        if not any(r.get('destination')=='http://127.0.0.1:8002/api/:path*' for r in rewrites):
            raise RuntimeError('QA build targets a different API. Run without --skip-build to compile the correct QA origin.')
    for port in (8002,3002):
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1',port))==0: raise RuntimeError(f'QA port {port} is occupied')
    qa=Path(tempfile.mkdtemp(prefix='benchmark-qa-',dir=ROOT/'work'));runs=qa/'runs';runs.mkdir()
    shutil.copytree(ROOT/'work_dirs/benchmark_mini_val',runs/'benchmark_mini_val')
    shutil.copyfile(ROOT/'work_dirs/radar_startup_audit.json',runs/'radar_startup_audit.json')
    for name in ('infer_mini_cuda_v2','infer_mini_val_scene0103_f1','eval_mini_val_scene0103_f1'):
        shutil.copytree(ROOT/'work_dirs'/name,runs/name)
    env=dict(os.environ,VIEWER_RUNS_ROOT=str(runs),VIEWER_STATE_ROOT=str(qa/'state'),
        VIEWER_BENCHMARK_ROOTS=json.dumps([str(runs)]),VIEWER_EVALUATION_ROOTS=json.dumps([str(runs)]),
        VIEWER_ALLOWED_ORIGINS=json.dumps(['http://localhost:3002','http://127.0.0.1:3002','http://127.0.0.1:8002']),
        VIEWER_API_ORIGIN='http://127.0.0.1:8002',VIEWER_NEXT_DIST_DIR='.next-qa',NEXT_TELEMETRY_DISABLED='1')
    env['PATH']=str(ROOT/'.cache/toolchains/node/bin')+os.pathsep+env.get('PATH','')
    npm=shutil.which('npm',path=env['PATH']);node=shutil.which('node',path=env['PATH'])
    if not npm or not node: raise RuntimeError('Node/npm unavailable')
    passwords={role:secrets.token_urlsafe(28) for role in ('engineer','reviewer')}
    for role,password in passwords.items():
        subprocess.run([sys.executable,str(ROOT/'tools/create_viewer_user.py'),'--username','qa_'+role,'--role',role,'--password-stdin'],
            input=password+'\n',text=True,check=True,cwd=ROOT,env=env)
    print(f'Benchmark QA root: {qa}',flush=True)
    if not args.skip_build: subprocess.run([npm,'run','build'],cwd=ROOT/'web/frontend',env=env,check=True)
    backend=frontend=None
    backend_log=(qa/'backend.log').open('ab');frontend_log=(qa/'frontend.log').open('ab')
    # A failed browser test can request thread stacks from this isolated process.
    # Stacks contain function locations, never credentials or HTTP request bodies.
    command=[sys.executable,'-c',"import faulthandler,signal,uvicorn; faulthandler.register(signal.SIGUSR1,all_threads=True); uvicorn.run('web.backend.app.main:app',host='127.0.0.1',port=8002)"]
    try:
        backend=subprocess.Popen(command,cwd=ROOT,env=env,stdout=backend_log,stderr=subprocess.STDOUT)
        frontend=subprocess.Popen([node,str(ROOT/'web/frontend/node_modules/next/dist/bin/next'),'start',str(ROOT/'web/frontend'),
            '--hostname','127.0.0.1','--port','3002'],cwd=ROOT,env=env,stdout=frontend_log,stderr=subprocess.STDOUT)
        wait_ready('http://127.0.0.1:8002/api/health',backend);wait_ready('http://127.0.0.1:3002/benchmarks',frontend)
        direct=httpx.get('http://127.0.0.1:8002/api/health').json()
        proxied=httpx.get('http://127.0.0.1:3002/api/health').json()
        if proxied.get('workspace_id')!=direct['workspace_id']:
            raise RuntimeError('Frontend proxy does not target the isolated QA backend; refusing to log in')
        with httpx.Client(base_url='http://127.0.0.1:8002',timeout=180) as client:
            csrf=client.get('/api/auth/session').json()['csrf_token']
            login=client.post('/api/auth/login',json=dict(username='qa_engineer',password=passwords['engineer']),headers={'X-CSRF-Token':csrf})
            login.raise_for_status()
            detail=client.get('/api/benchmarks/benchmark_mini_val');detail.raise_for_status();detail=detail.json()
            assert detail['successful_frame_count']==79 and detail['latency']['sample_count']==78
            assert not client.get('/api/health').json()['model_modules_loaded']
            browser_env=dict(env,VIEWER_URL='http://127.0.0.1:3002',VIEWER_E2E_BENCHMARK_ROOT=str(runs),VIEWER_E2E_OUTPUT=str(qa))
            for role,password in passwords.items():
                browser_env[f'VIEWER_E2E_{role.upper()}_USERNAME']='qa_'+role
                browser_env[f'VIEWER_E2E_{role.upper()}_PASSWORD']=password
            try:
                subprocess.run([npm,'run','test:e2e','--','benchmark.spec.ts'],cwd=ROOT/'web/frontend',env=browser_env,check=True)
            except subprocess.CalledProcessError:
                backend.send_signal(signal.SIGUSR1)
                time.sleep(.2)
                raise
            info=client.get('/api/collections/benchmark_mini_val/review').json()
            assert info['approval_valid'] and info['state']=='approved'
            ui_export=verify_export(qa/'collection-approved.zip',info)
            events_before=info['events']
            stop(backend);backend=subprocess.Popen(command,cwd=ROOT,env=env,stdout=backend_log,stderr=subprocess.STDOUT)
            wait_ready('http://127.0.0.1:8002/api/health',backend)
            session=client.get('/api/auth/session').json();assert session['user']['username']=='qa_engineer'
            restored=client.get('/api/collections/benchmark_mini_val/review').json()
            assert restored['approval_valid'] and restored['events']==events_before
            exported=client.post('/api/collections/benchmark_mini_val/export',headers={'X-CSRF-Token':session['csrf_token']});exported.raise_for_status()
            output=qa/'collection-after-restart.zip';output.write_bytes(exported.content)
            after_restart=verify_export(output,restored)
            assert not client.get('/api/health').json()['model_modules_loaded']
            assert source_hashes()==before
            expected=json.loads((ROOT/'work_dirs/web_benchmark_preservation_before.json').read_text())['source_files']
            assert before==expected
            evidence=dict(status='passed',qa_directory=str(qa),benchmark_status=detail['status'],expected_frames=81,successful_frames=79,
                backend_process_restarted=True,accounts_sessions_and_history_persisted=True,history_events=len(events_before),
                original_source_file_count=len(before),original_artifacts_unchanged=True,model_modules_loaded=False,
                inference_enabled=False,original_gpu_latency=detail['latency'],ui_export=ui_export,restart_export=after_restart,
                playwright_result=str(ROOT/'web/frontend/test-results/browser-results.json'),
                screenshots=[str(ROOT/'work_dirs'/name) for name in ('web_benchmark_overview.png','web_benchmark_frame3d.png','web_benchmark_failed_frame.png','web_benchmark_approved.png','web_benchmark_hash_mismatch.png')])
            (ROOT/'work_dirs/web_benchmark_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
            print('WEB_BENCHMARK_QA_PASSED',json.dumps({k:evidence[k] for k in ('status','qa_directory','original_artifacts_unchanged','backend_process_restarted')}),flush=True)
    finally:
        stop(frontend);stop(backend);frontend_log.close();backend_log.close()
        # Playwright error-context snapshots can contain password input values
        # even with tracing disabled. Redact ephemeral secrets before retaining QA files.
        for directory in (qa,ROOT/'web/frontend/test-results'):
            for file in directory.rglob('*'):
                if file.is_file() and file.suffix in ('.json','.md','.txt','.log'):
                    original=file.read_text(errors='replace');sanitized=original
                    for password in passwords.values(): sanitized=sanitized.replace(password,'[REDACTED_QA_PASSWORD]')
                    if sanitized!=original: file.write_text(sanitized)


if __name__=='__main__': main()
