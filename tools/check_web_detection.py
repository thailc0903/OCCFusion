#!/usr/bin/env python3
"""Isolated detector QA; fixture first, opt-in one GPU smoke with no retries."""
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

import httpx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.check_web_review import stop,wait_ready

BROWSER=r"""
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=process.env.DET_ROOT,out=process.env.DET_OUT,base='http://127.0.0.1:3005';
const {chromium}=require(root+'/web/frontend/node_modules/playwright');
const expect=require(root+'/web/frontend/node_modules/@playwright/test').expect.configure({timeout:40000});
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 try{
  const page=await browser.newPage({viewport:{width:1520,height:1080},locale:'vi-VN'}),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base+'/jobs');await page.getByRole('textbox',{name:'Tên tài khoản'}).fill('qa_engineer');await page.getByLabel('Mật khẩu',{exact:true}).fill(process.env.DET_PASSWORD);await page.getByRole('button',{name:'Đăng nhập',exact:true}).click();
  await expect(page.getByLabel('Profile',{exact:true})).toBeVisible();await page.getByLabel('Profile',{exact:true}).selectOption('centerpoint_pillar02_circle');await page.getByLabel('Scene',{exact:true}).selectOption('scene-0103');await page.getByLabel('Frame',{exact:true}).selectOption('1');
  await expect(page.getByTestId('job-form-token')).toHaveText('3950bd41f74548429c0f7700ff3d8269');
  const created=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');await page.getByRole('button',{name:'Tạo job inference',exact:true}).click();const response=await created;assert.equal(response.status(),202);const job=await response.json();fs.writeFileSync(path.join(out,'created-job.json'),JSON.stringify(job,null,2));
  let info;
  for(let i=0;i<180;i++){
   info=await(await page.request.get(base+'/api/jobs/'+job.id)).json();fs.writeFileSync(path.join(out,'latest-job.json'),JSON.stringify(info,null,2));
   if(info.status==='failed'||info.status==='canceled')throw new Error('Detection failed: '+JSON.stringify(info.error));
   if(info.status==='succeeded')break;await page.waitForTimeout(2000);
  }
  assert.equal(info.status,'succeeded');assert.equal(info.inference_launch_count,1);assert.equal(info.task,'detection');assert.equal(info.evaluation_status,'pending_dataset_evaluation');
  await expect(page.getByTestId('job-status')).toHaveText('Thành công');await page.getByRole('link',{name:'Mở kết quả trong viewer →'}).click();
  await expect(page.getByLabel('Chọn detection run')).toHaveValue('infer_job_'+job.id);await expect(page.getByTestId('detection-token')).toHaveText(job.sample_token);
  await expect(page.locator('canvas')).toBeVisible();const detail=await(await page.request.get(base+'/api/detections/infer_job_'+job.id)).json();assert.equal(detail.metrics.mAP,null);assert.equal(detail.metrics.status,'pending_dataset_evaluation');
  const expected=detail.boxes.filter(b=>b.score>=.35).length;await expect(page.locator('canvas')).toHaveAttribute('data-rendered-boxes',String(expected));await expect(page.locator('canvas')).toHaveAttribute('data-webgl-ready','true');
  assert.equal(detail.fixture,process.env.DET_MODE==='fixture');await expect(page.getByTestId('review-state')).toContainText('unregistered');
  const before=detail.artifacts['detections.json'].sha256;
  await page.getByLabel('Hiển thị detection box',{exact:true}).uncheck();await expect(page.locator('canvas')).toHaveAttribute('data-rendered-boxes','0');await page.getByLabel('Hiển thị detection box',{exact:true}).check();
  await page.getByLabel('Detection minimum score',{exact:true}).focus();await page.getByLabel('Detection minimum score',{exact:true}).press('End');await expect(page.locator('canvas')).toHaveAttribute('data-rendered-boxes',String(detail.boxes.filter(b=>b.score>=1).length));
  await page.getByLabel('Detection minimum score',{exact:true}).press('Home');await expect(page.locator('canvas')).toHaveAttribute('data-rendered-boxes',String(detail.boxes.length));
  await page.getByText('Lọc lớp detection',{exact:true}).click();await page.getByLabel('Detection class car',{exact:true}).uncheck();await expect(page.locator('canvas')).toHaveAttribute('data-rendered-boxes',String(detail.boxes.filter(b=>b.label!==0).length));await page.getByLabel('Detection class car',{exact:true}).check();
  await page.getByRole('button',{name:'Reset camera',exact:true}).click();await page.evaluate(()=>window.scrollTo(0,0));await page.screenshot({path:path.join(out,'detection.png'),fullPage:true});
  await page.goto(base+'/?run=occupancy');await expect(page.locator('canvas')).toBeVisible();await expect(page.getByLabel('Detection overlay run')).toBeVisible();await page.getByLabel('Detection overlay run').selectOption('infer_job_'+job.id);
  await expect(page.locator('canvas')).toHaveAttribute('data-rendered-boxes',String(expected));await expect(page.locator('.viewport-count')).toContainText('điểm keyframe');
  await page.evaluate(()=>window.scrollTo(0,0));await page.screenshot({path:path.join(out,'occupancy-overlay.png'),fullPage:true});
  const after=await(await page.request.get(base+'/api/detections/infer_job_'+job.id)).json();assert.equal(after.artifacts['detections.json'].sha256,before);
  const health=await(await page.request.get(base+'/api/health')).json();assert.equal(health.model_modules_loaded,false);
  assert.equal((await(await page.request.get(base+'/api/jobs/'+job.id)).json()).inference_launch_count,1);
  // Geometry oracle for a known geometric center/dimensions/yaw, independent of actual boxes.
  const ts=require(root+'/web/frontend/node_modules/typescript'),vm=require('node:vm');const source=fs.readFileSync(root+'/web/frontend/lib/detections.ts','utf8');const exports={};vm.runInNewContext(ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS}}).outputText,{exports,Math});
  const corners=exports.boxCorners({center:[10,20,3],dimensions:[4,2,6],yaw:Math.PI/2});const mins=[0,1,2].map(a=>Math.min(...corners.map(c=>c[a]))),maxs=[0,1,2].map(a=>Math.max(...corners.map(c=>c[a])));assert.deepEqual(mins,[9,18,0]);assert.deepEqual(maxs,[11,22,6]);
  assert.deepEqual(errors,[]);fs.writeFileSync(path.join(out,'browser-evidence.json'),JSON.stringify({status:'passed',job_id:job.id,box_count:detail.box_count,visible_default:expected,geometry_oracle:true,filters_do_not_mutate_artifact:true,webgl:true,job:info,detail,page_errors:errors},null,2));
  console.log('DETECTION_BROWSER_PASSED',detail.box_count,'boxes, mode',process.env.DET_MODE);
 }finally{await browser.close();}
})().catch(e=>{console.error(e.message);process.exitCode=1;});
"""

def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group();group.add_argument('--gpu-smoke',action='store_true');group.add_argument('--gpu-smoke-after-init-fix',action='store_true',help='One additional attempt explicitly authorized after preserving failed initial smoke');args=parser.parse_args()
    after_init_fix=args.gpu_smoke_after_init_fix
    args.gpu_smoke=args.gpu_smoke or after_init_fix
    out=ROOT/'work_dirs/centerpoint_milestone'
    if args.gpu_smoke:
        assert json.loads((out/'fixture_evidence.json').read_text())['status']=='passed','Fixture browser must pass first'
        assert json.loads((out/'backend_result.json').read_text())['returncode']==0,'Backend regression must pass before GPU'
        assert json.loads((out/'cpu_geometry.json').read_text())['status']=='passed'
    for port in (8005,3005):
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('QA port occupied')
    routes=json.loads((ROOT/'web/frontend/.next-qa/routes-manifest.json').read_text())['rewrites'];routes=routes if isinstance(routes,list) else sum(routes.values(),[])
    assert any(r['destination']=='http://127.0.0.1:8005/api/:path*' for r in routes),'QA build origin must be 8005'
    qa=Path(tempfile.mkdtemp(prefix='centerpoint-smoke-' if args.gpu_smoke else 'centerpoint-fixture-',dir=ROOT/'work'));runs=qa/'runs'
    shutil.copytree(ROOT/'work_dirs/infer_mini_val_scene0103_f1',runs/'occupancy')
    shutil.copytree(ROOT/'work_dirs/eval_mini_val_scene0103_f1',runs/'occupancy_evaluation')
    env=dict(os.environ,VIEWER_RUNS_ROOT=str(runs),VIEWER_STATE_ROOT=str(qa/'state'),VIEWER_JOB_PROFILES=str(ROOT/'web/centerpoint_profiles.json'),
        VIEWER_EVALUATION_ROOTS=json.dumps([str(runs)]),VIEWER_BENCHMARK_ROOTS=json.dumps([str(runs)]),VIEWER_ALLOWED_ORIGINS=json.dumps(['http://127.0.0.1:8005','http://127.0.0.1:3005']),
        VIEWER_API_ORIGIN='http://127.0.0.1:8005',VIEWER_NEXT_DIST_DIR='.next-qa',NEXT_TELEMETRY_DISABLED='1',DET_ROOT=str(ROOT),DET_OUT=str(qa),DET_MODE='real' if args.gpu_smoke else 'fixture')
    env['DET_PASSWORD']=secrets.token_urlsafe(28);env['PATH']=str(ROOT/'.cache/toolchains/node/bin')+os.pathsep+env.get('PATH','');env['PLAYWRIGHT_BROWSERS_PATH']=str(ROOT/'.cache/ms-playwright');env['LD_LIBRARY_PATH']=str(ROOT/'.cache/browser-libs/usr/lib/x86_64-linux-gnu')+os.pathsep+env.get('LD_LIBRARY_PATH','')
    subprocess.run([sys.executable,str(ROOT/'tools/create_viewer_user.py'),'--username','qa_engineer','--role','engineer','--password-stdin'],input=env['DET_PASSWORD']+'\n',text=True,cwd=ROOT,env=env,check=True)
    if args.gpu_smoke:
        if after_init_fix:assert (out/'gpu_smoke_intent.json').is_file(),'Initial attempt must remain preserved'
        with (out/('gpu_smoke_after_init_fix_intent.json' if after_init_fix else 'gpu_smoke_intent.json')).open('x') as stream:json.dump(dict(qa_directory=str(qa),max_jobs=1,scene='scene-0103',frame=1,after_init_fix=after_init_fix),stream,indent=2)
    print('Detection QA:',qa,flush=True)
    logs={k:(qa/(k+'.log')).open('ab') for k in ('backend','frontend','worker')};backend=frontend=worker=None
    try:
        backend=subprocess.Popen([sys.executable,'-m','uvicorn','web.backend.app.main:app','--host','127.0.0.1','--port','8005'],cwd=ROOT,env=env,stdout=logs['backend'],stderr=subprocess.STDOUT)
        frontend=subprocess.Popen([shutil.which('node',path=env['PATH']),str(ROOT/'web/frontend/node_modules/next/dist/bin/next'),'start',str(ROOT/'web/frontend'),'--hostname','127.0.0.1','--port','3005'],cwd=ROOT,env=env,stdout=logs['frontend'],stderr=subprocess.STDOUT)
        wait_ready('http://127.0.0.1:8005/api/health',backend);wait_ready('http://127.0.0.1:3005/jobs',frontend)
        assert httpx.get('http://127.0.0.1:8005/api/health').json()['workspace_id']==httpx.get('http://127.0.0.1:3005/api/health').json()['workspace_id']
        if args.gpu_smoke:cmd=[sys.executable,'-u',str(ROOT/'tools/run_inference_worker.py'),'--max-jobs','1']
        else:cmd=[sys.executable,'-u','-c','from web.backend.app.config import Settings;from web.backend.app.database import Database;from web.backend.app.job_worker import Worker;from web.backend.tests.detection_support import FixtureDetectionCommands;s=Settings.load();Worker(Database(s.state_directory),s,FixtureDetectionCommands()).run(max_jobs=1)']
        worker=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=logs['worker'],stderr=subprocess.STDOUT)
        script=qa/'browser.cjs';script.write_text(BROWSER)
        subprocess.run([shutil.which('node',path=env['PATH']),str(script)],cwd=ROOT,env=env,check=True)
        worker.wait(timeout=15);assert worker.returncode==0
        evidence=json.loads((qa/'browser-evidence.json').read_text());evidence.update(qa_directory=str(qa),run_root=str(runs),model_gpu_smoke=args.gpu_smoke)
        (out/('gpu_smoke_evidence.json' if args.gpu_smoke else 'fixture_evidence.json')).write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    finally:
        stop(worker);stop(frontend);stop(backend)
        for log in logs.values():log.close()

if __name__=='__main__':main()
