#!/usr/bin/env python3
"""CSRF Chromium QA on private SQLite/artifact copies; never starts a worker."""
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_web_review import stop, wait_ready


BROWSER = r"""
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = process.env.CSRF_QA_ROOT;
const output = process.env.CSRF_QA_OUTPUT;
const base = 'http://127.0.0.1:3003';
const {chromium} = require(path.join(root, 'web/frontend/node_modules/playwright'));
const {expect} = require(path.join(root, 'web/frontend/node_modules/@playwright/test'));
const wait = expect.configure({timeout: 30000});
const deferred = () => { let resolve; const promise = new Promise(r => resolve = r); return {promise, resolve}; };
const body = () => ({profile_id:'occfusion_clr_fp32', scene_name:'scene-0103', frame_index:1, request_key:require('node:crypto').randomUUID().replaceAll('-','')});

(async () => {
 const browser = await chromium.launch({headless:true,args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 try {
  const context = await browser.newContext({viewport:{width:1520,height:1080},locale:'vi-VN'});
  const page = await context.newPage(); const errors = []; page.on('pageerror', e => errors.push(e.message));
  const bootstrap = page.waitForResponse(r => r.url().endsWith('/api/auth/session'));
  await page.goto(base+'/jobs'); const oldToken = (await (await bootstrap).json()).csrf_token;
  await page.getByRole('textbox',{name:'Tên tài khoản'}).fill('qa_engineer');
  await page.getByLabel('Mật khẩu',{exact:true}).fill(process.env.CSRF_QA_PASSWORD);
  await page.getByRole('button',{name:'Đăng nhập',exact:true}).click();
  await wait(page.getByTestId('job-form-token')).toHaveText('3950bd41f74548429c0f7700ff3d8269');
  const loginToken = (await (await page.request.get(base+'/api/auth/session')).json()).csrf_token;
  const checks = [];
  for (const cookieState of ['old','missing']) {
   if (cookieState === 'old') await context.addCookies([{name:'rav25_csrf',value:oldToken,url:base,httpOnly:true,sameSite:'Strict'}]);
   else await context.clearCookies({name:'rav25_csrf'});
   const firstFetched = deferred(); const secondReceived = deferred(); const order = []; const cookieChecks=[];
   await page.exposeFunction('csrfSecondReceived_'+cookieState, () => secondReceived.resolve());
   const pattern = '**/api/auth/session?csrf_qa=*';
   await page.route(pattern, async route => {
    const tag = new URL(route.request().url()).searchParams.get('csrf_qa');
    const originalCookie = (await route.request().headerValue('cookie')) || '';
    cookieChecks.push(cookieState==='old' ? originalCookie.includes('rav25_csrf='+oldToken) : !originalCookie.includes('rav25_csrf='));
    if (tag==='B') await firstFetched.promise;
    const response = await route.fetch();
    if (tag==='A') { firstFetched.resolve(); await secondReceived.promise; }
    order.push(tag); await route.fulfill({response});
   });
   const results = await page.evaluate(async state => {
    const a = fetch('/api/auth/session?csrf_qa=A',{cache:'no-store'}).then(r=>r.json());
    const b = fetch('/api/auth/session?csrf_qa=B',{cache:'no-store'}).then(async r=>{
     const result=await r.json(); await window['csrfSecondReceived_'+state](); return result;
    });
    return Promise.all([a,b]);
   },cookieState);
   await page.unroute(pattern);
   assert.deepEqual(order,['B','A']); assert.equal(cookieChecks.length,2); assert.ok(cookieChecks.every(Boolean));
   const csrf = (await context.cookies()).find(c=>c.name==='rav25_csrf');
   assert.ok(results[0].csrf_token===results[1].csrf_token && csrf.value===results[0].csrf_token && csrf.value===loginToken,'Bootstrap did not converge');
   assert.ok(csrf.httpOnly && csrf.sameSite==='Strict');
   // Use the unmodified application's memory/header handling for a real write.
   // No worker exists in this QA workspace: creation queues, cancellation ends it.
   const created = page.waitForResponse(r=>r.request().method()==='POST' && r.url().endsWith('/api/jobs'));
   await page.getByRole('button',{name:'Tạo job inference',exact:true}).click();
   const response = await created; assert.equal(response.status(),202);
   const job = await response.json(); await wait(page.getByTestId('job-id')).toHaveText(job.id);
   await wait(page.getByTestId('job-status')).toHaveText('Chờ');
   await page.getByRole('button',{name:'Hủy job',exact:true}).click();
   await wait(page.getByTestId('job-status')).toHaveText('Đã hủy');
   await wait(page.getByTestId('launch-count')).toHaveText('0');
   checks.push({cookie_state:cookieState,response_order:order,request_cookie_states_verified:true,converged:true,post_status:202,job_id:job.id,inference_launch_count:0});
  }
  const sessionCookie=(await context.cookies()).find(c=>c.name==='rav25_session').value;
  const other=await browser.newContext();
  try {
   const anonymous=await (await other.request.get(base+'/api/auth/session')).json();
   const signedIn=await other.request.post(base+'/api/auth/login',{data:{username:'qa_engineer',password:process.env.CSRF_QA_PASSWORD},headers:{'X-CSRF-Token':anonymous.csrf_token,Origin:base}});
   assert.equal(signedIn.status(),200);const foreign=(await signedIn.json()).csrf_token;
   const denied=await page.request.post(base+'/api/jobs',{data:body(),headers:{Cookie:'rav25_session='+sessionCookie+'; rav25_csrf='+foreign,'X-CSRF-Token':foreign,Origin:base}});
   assert.equal(denied.status(),403);assert.equal((await denied.json()).detail,'CSRF token does not belong to this session');
  } finally { await other.close(); }
  const badOrigin=await page.request.post(base+'/api/jobs',{data:body(),headers:{'X-CSRF-Token':loginToken,Origin:'https://evil.example'}});assert.equal(badOrigin.status(),403);
  const noCsrf=await page.request.post(base+'/api/jobs',{data:body()});assert.equal(noCsrf.status(),403);
  const approve=await page.request.post(base+'/api/runs/prediction/review/decision',{data:{comment:'QA permission check',decision:'approved',revision_id:1,fingerprint:'0'.repeat(64)},headers:{'X-CSRF-Token':loginToken,Origin:base}});assert.equal(approve.status(),403);
  await page.evaluate(()=>window.scrollTo(0,0));await page.screenshot({path:path.join(output,'csrf-jobs.png'),fullPage:true});
  await page.goto(base+'/'); await wait(page.locator('canvas')).toBeVisible();
  await wait(page.getByTestId('review-state')).toContainText('unregistered');
  await page.getByLabel('Nhận xét review',{exact:true}).fill('QA: stable authenticated CSRF after overlapping bootstrap.');
  await page.getByRole('button',{name:'Đăng ký run',exact:true}).click();await wait(page.getByTestId('review-state')).toContainText('draft');
  await page.screenshot({path:path.join(output,'csrf-viewer-review.png'),fullPage:true});
  await page.getByRole('button',{name:'Đăng xuất',exact:true}).click();await wait(page.getByRole('heading',{name:'Đăng nhập',exact:true})).toBeVisible();
  const protectedResponse=await page.request.get(base+'/api/jobs');assert.equal(protectedResponse.status(),401);
  const health=await (await page.request.get(base+'/api/health')).json();assert.equal(health.model_modules_loaded,false);
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(output,'browser.json'),JSON.stringify({status:'passed',schedules:checks,foreign_session_status:403,bad_origin_status:403,missing_csrf_status:403,engineer_approval_status:403,logout_status:401,viewer_loaded:true,review_write_state:'draft',page_errors:errors,model_modules_loaded:false},null,2));
  console.log('CSRF_BROWSER_PASSED',checks.length,'deterministic reversed-response schedules; no CLI launch');
 } finally { await browser.close(); }
})().catch(error=>{console.error(error.message);process.exitCode=1;});
"""


def main():
    for port in (8003, 3003):
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'QA port {port} is occupied; production will not be stopped')
    manifest = ROOT/'web/frontend/.next-qa/routes-manifest.json'
    if not manifest.is_file():
        raise RuntimeError('Build unchanged frontend with VIEWER_API_ORIGIN=http://127.0.0.1:8003 VIEWER_NEXT_DIST_DIR=.next-qa npm --prefix web/frontend run build first')
    value = json.loads(manifest.read_text())['rewrites']
    routes = value if isinstance(value, list) else sum(value.values(), [])
    if not any(row.get('destination') == 'http://127.0.0.1:8003/api/:path*' for row in routes):
        raise RuntimeError('QA build has another API origin; refusing to send credentials')
    qa = Path(tempfile.mkdtemp(prefix='csrf-qa-', dir=ROOT/'work'))
    runs = qa/'runs'
    shutil.copytree(ROOT/'work_dirs/infer_mini_cuda_v2', runs/'prediction')
    password = secrets.token_urlsafe(28)
    env = dict(os.environ, VIEWER_RUNS_ROOT=str(runs), VIEWER_STATE_ROOT=str(qa/'state'),
        VIEWER_EVALUATION_ROOTS=json.dumps([str(runs)]), VIEWER_BENCHMARK_ROOTS=json.dumps([str(runs)]),
        VIEWER_ALLOWED_ORIGINS=json.dumps(['http://127.0.0.1:8003', 'http://127.0.0.1:3003']),
        VIEWER_API_ORIGIN='http://127.0.0.1:8003', VIEWER_NEXT_DIST_DIR='.next-qa', NEXT_TELEMETRY_DISABLED='1',
        CSRF_QA_ROOT=str(ROOT), CSRF_QA_OUTPUT=str(qa), CSRF_QA_PASSWORD=password,
        PLAYWRIGHT_BROWSERS_PATH=str(ROOT/'.cache/ms-playwright'))
    env['PATH'] = str(ROOT/'.cache/toolchains/node/bin') + os.pathsep + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = str(ROOT/'.cache/browser-libs/usr/lib/x86_64-linux-gnu') + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    subprocess.run([sys.executable, str(ROOT/'tools/create_viewer_user.py'), '--username', 'qa_engineer', '--role', 'engineer', '--password-stdin'],
        input=password+'\n', text=True, cwd=ROOT, env=env, check=True)
    print('CSRF QA workspace:', qa, flush=True)
    backend = frontend = None
    logs = {kind:(qa/(kind+'.log')).open('ab') for kind in ('backend', 'frontend')}
    command = [sys.executable, '-m', 'uvicorn', 'web.backend.app.main:app', '--host', '127.0.0.1', '--port', '8003']
    try:
        backend = subprocess.Popen(command, cwd=ROOT, env=env, stdout=logs['backend'], stderr=subprocess.STDOUT)
        frontend = subprocess.Popen([shutil.which('node', path=env['PATH']), str(ROOT/'web/frontend/node_modules/next/dist/bin/next'),
            'start', str(ROOT/'web/frontend'), '--hostname', '127.0.0.1', '--port', '3003'],
            cwd=ROOT, env=env, stdout=logs['frontend'], stderr=subprocess.STDOUT)
        wait_ready('http://127.0.0.1:8003/api/health', backend)
        wait_ready('http://127.0.0.1:3003/jobs', frontend)
        direct = httpx.get('http://127.0.0.1:8003/api/health').json()
        assert httpx.get('http://127.0.0.1:3003/api/health').json()['workspace_id'] == direct['workspace_id']
        browser_script = qa/'browser.cjs'
        browser_script.write_text(BROWSER)
        subprocess.run([shutil.which('node', path=env['PATH']), str(browser_script)], cwd=ROOT, env=env, check=True)
        with httpx.Client(base_url='http://127.0.0.1:8003') as client:
            csrf = client.get('/api/auth/session').json()['csrf_token']
            signed = client.post('/api/auth/login', json=dict(username='qa_engineer', password=password), headers={'X-CSRF-Token':csrf})
            signed.raise_for_status()
            before = client.get('/api/auth/session').json()
            history = client.get('/api/runs/prediction/review').json()
            jobs = client.get('/api/jobs').json()['jobs']
            stop(backend)
            backend = subprocess.Popen(command, cwd=ROOT, env=env, stdout=logs['backend'], stderr=subprocess.STDOUT)
            wait_ready('http://127.0.0.1:8003/api/health', backend)
            assert client.get('/api/auth/session').json() == before
            assert client.get('/api/runs/prediction/review').json() == history
            assert client.get('/api/jobs').json()['jobs'] == jobs
            assert client.post('/api/auth/logout', headers={'X-CSRF-Token':before['csrf_token']}).status_code == 200
        with sqlite3.connect(qa/'state/viewer.sqlite3') as connection:
            count, launches = connection.execute('SELECT COUNT(*),COALESCE(SUM(inference_launch_count),0) FROM inference_jobs').fetchone()
            assert count == 2 and launches == 0
            assert connection.execute('SELECT COUNT(*) FROM inference_workers').fetchone()[0] == 0
        browser = json.loads((qa/'browser.json').read_text())
        evidence = dict(status='passed', qa_directory=str(qa), browser=browser, backend_restarted=True,
            account_session_csrf_review_jobs_preserved=True, job_count=count, inference_launch_count=launches,
            worker_started=False, frontend_source_changed=False, reused_unchanged_frontend_build=str(manifest),
            screenshots=[str(qa/'csrf-jobs.png'), str(qa/'csrf-viewer-review.png')])
        (ROOT/'work_dirs/csrf_fix/browser_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
        print('CSRF_QA_PASSED: browser, backend process restart, zero worker/CLI/GPU', flush=True)
    finally:
        stop(frontend)
        stop(backend)
        for stream in logs.values():
            stream.close()


if __name__ == '__main__':
    main()
