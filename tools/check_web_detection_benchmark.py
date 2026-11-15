#!/usr/bin/env python3
"""Isolated saved-artifact Chromium QA. No worker, model, inference or evaluator."""
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
import zipfile

import httpx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.check_web_review import stop,wait_ready

BROWSER=r"""
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=process.env.DET_ROOT,out=process.env.DET_OUT,base='http://127.0.0.1:3006',id='benchmark_detection_mini_val';
const {chromium}=require(root+'/web/frontend/node_modules/playwright');
const expect=require(root+'/web/frontend/node_modules/@playwright/test').expect.configure({timeout:180000});
const bench='/api/benchmarks/'+id,col='/api/collections/'+id;
async function sign(page,user,password){
 await page.goto(base+'/benchmarks?benchmark='+id);
 await page.getByRole('textbox',{name:'Tên tài khoản'}).fill(user);
 await page.getByLabel('Mật khẩu',{exact:true}).fill(password);
 await page.getByRole('button',{name:'Đăng nhập',exact:true}).click();
}
async function post(page,url,body){
 const s=await(await page.request.get(base+'/api/auth/session')).json();
 return page.request.post(base+url,{data:body,headers:{'X-CSRF-Token':s.csrf_token,Origin:base}});
}
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 try{
  const context=await browser.newContext({viewport:{width:1520,height:1080},locale:'vi-VN',acceptDownloads:true});
  const page=await context.newPage(),errors=[],jobs=[];
  page.setDefaultTimeout(180000);page.setDefaultNavigationTimeout(180000);
  page.on('pageerror',e=>errors.push(e.message));
  page.on('request',r=>{if(r.method()==='POST'&&r.url().includes('/api/jobs'))jobs.push(r.url());});
  await sign(page,'qa_engineer',process.env.DET_PASSWORD);
  await expect(page.getByTestId('detection-map')).toBeVisible();
  const data=await(await page.request.get(base+bench)).json();
  const percent=new Intl.NumberFormat('vi-VN',{style:'percent',maximumFractionDigits:4});
  await expect(page.getByTestId('benchmark-status')).toHaveText('COMPLETE');
  await expect(page.getByTestId('benchmark-coverage')).toHaveText('81/81 frame hợp lệ');
  await expect(page.getByTestId('detection-map')).toHaveText(percent.format(data.metrics.mAP));
  await expect(page.getByTestId('detection-nds')).toHaveText(percent.format(data.metrics.NDS));
  assert.equal(data.classes.length,10);
  await expect(page.locator('[data-testid^="detection-class-"]')).toHaveCount(10);
  await expect(page.getByTestId('detection-no-gt-class')).toHaveCount(data.classes.filter(c=>!c.has_gt).length);
  await expect(page.getByTestId('detection-latency-source')).toContainText(data.latency.source_report);
  await expect(page.getByTestId('detection-timing-counts')).toContainText('80 forward mới');
  await expect(page.getByTestId('detection-timing-counts')).toContainText('79 mẫu');
  await expect(page.getByTestId('detection-loader-max')).toHaveText(data.loader_audit.loader_vs_se3_translation_m.max.toFixed(9)+' m');
  await page.getByText('TP errors · giá trị không áp dụng được giữ null',{exact:true}).click();
  await expect(page.getByRole('table',{name:'Detection TP errors'})).toContainText('Không áp dụng');
  await page.getByText('TP errors · giá trị không áp dụng được giữ null',{exact:true}).click();
  await expect(page.locator('canvas')).toHaveAttribute('data-webgl-ready','true');
  await expect(page.getByTestId('detection-visible-boxes')).not.toHaveText('0');
  await page.screenshot({path:path.join(out,'detection-overview.png'),fullPage:true});
  const token='3950bd41f74548429c0f7700ff3d8269';
  await page.getByLabel('Chọn frame benchmark detection',{exact:true}).selectOption(token);
  await expect(page.getByTestId('detection-benchmark-token')).toHaveText(token);
  await expect(page.getByLabel('Detection benchmark occupancy overlay').locator('option[value="occupancy"]')).toHaveCount(1);
  await page.getByLabel('Detection benchmark occupancy overlay').selectOption('occupancy');
  await expect(page.locator('.viewport-count')).toContainText('voxel occupancy');
  await expect(page.getByLabel('Detection benchmark occupancy overlay').locator('option[value="benchmark:benchmark_mini_val"]')).toHaveCount(1);
  await page.getByLabel('Detection benchmark occupancy overlay').selectOption('benchmark:benchmark_mini_val');
  await expect(page.locator('.viewport-count')).toContainText('voxel occupancy');
  const frame=await(await page.request.get(base+bench+'/frames/'+token)).json(),hash=frame.artifacts['detections.json'].sha256;
  await page.getByLabel('Detection benchmark minimum score').focus();await page.getByLabel('Detection benchmark minimum score').press('End');
  await expect(page.getByTestId('detection-visible-boxes')).toHaveText(String(frame.boxes.filter(b=>b.score>=1).length));
  await page.getByLabel('Detection benchmark minimum score').press('Home');
  await expect(page.getByTestId('detection-visible-boxes')).toHaveText(String(frame.box_count));
  await page.getByText('Lọc lớp detection',{exact:true}).click();
  await page.getByLabel('Benchmark detection class car',{exact:true}).uncheck();
  await expect(page.getByTestId('detection-visible-boxes')).toHaveText(String(frame.boxes.filter(b=>b.label!==0).length));
  await page.getByLabel('Benchmark detection class car',{exact:true}).check();
  await page.getByLabel('Detection benchmark boxes',{exact:true}).uncheck();
  await expect(page.getByTestId('detection-visible-boxes')).toHaveText('0');
  await page.getByLabel('Detection benchmark boxes',{exact:true}).check();
  await page.getByRole('button',{name:'Reset camera',exact:true}).click();
  await page.locator('.detection-benchmark-playback').screenshot({path:path.join(out,'detection-occupancy-overlay.png')});
  assert.equal((await(await page.request.get(base+bench+'/frames/'+token)).json()).artifacts['detections.json'].sha256,hash);
  await page.getByLabel('Chọn scene benchmark detection').selectOption('scene-0916');
  await expect(page.getByTestId('detection-benchmark-token')).toHaveText(data.frames.find(f=>f.scene==='scene-0916').sample_token);
  await expect(page.getByLabel('Detection benchmark occupancy overlay')).toHaveValue('');
  await expect(page.locator('.viewport-count')).not.toContainText('voxel occupancy');
  await page.getByRole('button',{name:'Mở frame có sai khác lớn nhất',exact:true}).click();
  await expect(page.getByTestId('detection-benchmark-token')).toHaveText(data.loader_audit.worst_frame.sample_token);
  await page.getByLabel('Task benchmark',{exact:true}).selectOption('occupancy');
  await expect(page.getByTestId('pooled-iou')).toBeVisible();await expect(page.getByTestId('detection-map')).toHaveCount(0);
  await expect(page.getByTestId('benchmark-coverage')).toHaveText('79/81 frame hợp lệ');
  await page.getByLabel('Chọn frame benchmark',{exact:true}).selectOption('3e8750f331d7499e9b5123e9eb70f2e2');
  await expect(page.getByTestId('failed-frame')).toBeVisible();
  await page.getByLabel('Task benchmark',{exact:true}).selectOption('detection');
  await expect(page.getByTestId('detection-map')).toHaveText(percent.format(data.metrics.mAP));
  assert.equal((await post(page,col+'/export')).status(),409);
  await page.getByLabel('Nhận xét collection',{exact:true}).fill('QA saved detection collection; independent approval');
  await page.getByRole('button',{name:'Đăng ký collection',exact:true}).click();
  await expect(page.getByTestId('collection-state')).toHaveText('draft');
  let info=await(await page.request.get(base+col+'/review')).json();
  assert.equal((await post(page,col+'/review/decision',{comment:'Engineer denied',decision:'approved',revision_id:info.revision_id,fingerprint:info.fingerprint})).status(),403);
  await page.getByLabel('Nhận xét collection',{exact:true}).fill('QA submit all 81 saved frames with immutable evaluation/GPU report');
  await page.getByRole('button',{name:'Gửi review collection',exact:true}).click();
  await expect(page.getByTestId('collection-state')).toHaveText('submitted');
  const rc=await browser.newContext({viewport:{width:1520,height:1080},locale:'vi-VN'}),reviewer=await rc.newPage();
  reviewer.setDefaultTimeout(180000);reviewer.setDefaultNavigationTimeout(180000);
  await sign(reviewer,'qa_reviewer',process.env.DET_REVIEW_PASSWORD);
  await expect(reviewer.getByTestId('collection-state')).toHaveText('submitted');
  await reviewer.getByLabel('Nhận xét collection',{exact:true}).fill('QA approve exact detection collection fingerprint');
  await reviewer.getByRole('button',{name:'Phê duyệt collection',exact:true}).click();
  await expect(reviewer.getByTestId('collection-state')).toHaveText('approved');
  await reviewer.getByTestId('collection-review').screenshot({path:path.join(out,'detection-collection-approved.png')});
  await page.getByRole('button',{name:'Đọc lại artifact',exact:true}).click();
  await expect(page.getByTestId('collection-state')).toHaveText('approved');
  const download=page.waitForEvent('download');await page.getByRole('button',{name:'Xuất collection đã duyệt',exact:true}).click();
  await(await download).saveAs(path.join(out,'detection-collection-approved.zip'));
  const badRoot=path.join(process.env.DET_RUNS,id),manifest=JSON.parse(fs.readFileSync(path.join(badRoot,'manifest.json')));
  const first=manifest.identity.frames[0].sample_token,parent=path.join(badRoot,'frames',first);
  const attempt=fs.readdirSync(parent).find(n=>/^[0-9a-f]{32}$/.test(n)),file=path.join(parent,attempt,'detections.json'),previous=fs.readFileSync(file);
  try{
   fs.appendFileSync(file,' ');
   await page.getByRole('button',{name:'Đọc lại artifact',exact:true}).click();
   await expect(page.locator('.error-panel[role="alert"]')).toContainText('Không thể xác minh collection');
   await expect(page.getByTestId('detection-map')).toHaveCount(0);
   assert.equal((await post(page,col+'/export')).status(),409);
   await page.screenshot({path:path.join(out,'detection-hash-mismatch.png'),fullPage:true});
  }finally{fs.writeFileSync(file,previous);}
  const health=await(await page.request.get(base+'/api/health')).json();
  assert.equal(health.model_modules_loaded,false);assert.deepEqual(jobs,[]);assert.deepEqual(errors,[]);
  info=await(await page.request.get(base+col+'/review')).json();assert.equal(info.approval_valid,true);
  fs.writeFileSync(path.join(out,'browser-evidence.json'),JSON.stringify({status:'passed',benchmark:id,coverage:81,
   metrics:data.metrics,source_report:data.latency.source_report,metric_units:true,task_switch:true,frame_switch:true,
   overlay:true,webgl:true,filters_preserve_hash:true,hash_mismatch_blocks_metric_export:true,
   review_fingerprint:info.fingerprint,page_errors:errors,job_requests:jobs},null,2));
  await context.close();await rc.close();console.log('DETECTION_BENCHMARK_BROWSER_PASSED');
 }finally{await browser.close();}
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
"""


def verify_zip(path):
    with zipfile.ZipFile(path) as archive:
        manifest=json.loads(archive.read('manifest.json'));review=json.loads(archive.read('review.json'))
        assert manifest['kind']=='centerpoint_detection_benchmark_collection'
        assert review['state']=='approved' and manifest['successful_frame_count']==81
        assert len(json.loads(archive.read('benchmark/predictions_nuscenes.json'))['results'])==81
        assert not any(name.endswith('occupancy.npz') for name in archive.namelist())
        for name,expected in manifest['artifacts'].items():
            data=archive.read(name)
            assert hashlib.sha256(data).hexdigest()==expected['sha256'] and len(data)==expected['bytes'],name
        return dict(entries=len(archive.namelist()),fingerprint=manifest['fingerprint_sha256'],
            files_verified=len(manifest['artifacts']),review=review)


def main():
    evidence_root=ROOT/'work_dirs/web_detection_benchmark_qa'
    for port in (8006,3006):
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('QA port already occupied: '+str(port))
    routes=json.loads((ROOT/'web/frontend/.next-detection-benchmark-qa/routes-manifest.json').read_text())['rewrites']
    routes=routes if isinstance(routes,list) else sum(routes.values(),[])
    assert any(r['destination']=='http://127.0.0.1:8006/api/:path*' for r in routes)
    qa=Path(tempfile.mkdtemp(prefix='detection-benchmark-web-',dir=ROOT/'work'));runs=qa/'runs'
    for name in ('benchmark_detection_mini_val','benchmark_mini_val'):
        shutil.copytree(ROOT/'work_dirs'/name,runs/name)
    shutil.copyfile(ROOT/'work_dirs/radar_startup_audit.json',runs/'radar_startup_audit.json')
    shutil.copytree(ROOT/'work_dirs/infer_mini_val_scene0103_f1',runs/'occupancy')
    env=dict(os.environ,VIEWER_RUNS_ROOT=str(runs),VIEWER_STATE_ROOT=str(qa/'state'),
        VIEWER_EVALUATION_ROOTS=json.dumps([str(runs)]),VIEWER_BENCHMARK_ROOTS=json.dumps([str(runs)]),
        VIEWER_ALLOWED_ORIGINS=json.dumps(['http://127.0.0.1:8006','http://127.0.0.1:3006']),
        VIEWER_API_ORIGIN='http://127.0.0.1:8006',VIEWER_NEXT_DIST_DIR='.next-detection-benchmark-qa',
        NEXT_TELEMETRY_DISABLED='1',DET_ROOT=str(ROOT),DET_OUT=str(qa),DET_RUNS=str(runs))
    env['PATH']=str(ROOT/'.cache/toolchains/node/bin')+os.pathsep+env.get('PATH','')
    env['PLAYWRIGHT_BROWSERS_PATH']=str(ROOT/'.cache/ms-playwright')
    env['LD_LIBRARY_PATH']=str(ROOT/'.cache/browser-libs/usr/lib/x86_64-linux-gnu')+os.pathsep+env.get('LD_LIBRARY_PATH','')
    for user,role,key in [('qa_engineer','engineer','DET_PASSWORD'),('qa_reviewer','reviewer','DET_REVIEW_PASSWORD')]:
        env[key]=secrets.token_urlsafe(28)
        subprocess.run([sys.executable,'tools/create_viewer_user.py','--username',user,'--role',role,'--password-stdin'],
            cwd=ROOT,env=env,input=env[key]+'\n',text=True,check=True)
    logs={name:(qa/(name+'.log')).open('ab') for name in ('backend','frontend')}
    backend=frontend=None
    command=[sys.executable,'-m','uvicorn','web.backend.app.main:app','--host','127.0.0.1','--port','8006']
    print('Saved detection benchmark QA:',qa,flush=True)
    try:
        backend=subprocess.Popen(command,cwd=ROOT,env=env,stdout=logs['backend'],stderr=subprocess.STDOUT)
        frontend=subprocess.Popen([shutil.which('node',path=env['PATH']),str(ROOT/'web/frontend/node_modules/next/dist/bin/next'),
            'start',str(ROOT/'web/frontend'),'--hostname','127.0.0.1','--port','3006'],
            cwd=ROOT,env=env,stdout=logs['frontend'],stderr=subprocess.STDOUT)
        wait_ready('http://127.0.0.1:8006/api/health',backend);wait_ready('http://127.0.0.1:3006/benchmarks',frontend)
        script=qa/'browser.cjs';script.write_text(BROWSER)
        subprocess.run([shutil.which('node',path=env['PATH']),str(script)],cwd=ROOT,env=env,check=True)
        exported=verify_zip(qa/'detection-collection-approved.zip')
        stop(backend)
        backend=subprocess.Popen(command,cwd=ROOT,env=env,stdout=logs['backend'],stderr=subprocess.STDOUT)
        wait_ready('http://127.0.0.1:8006/api/health',backend)
        with httpx.Client(base_url='http://127.0.0.1:8006',timeout=180) as client:
            csrf=client.get('/api/auth/session').json()['csrf_token']
            response=client.post('/api/auth/login',json=dict(username='qa_engineer',password=env['DET_PASSWORD']),
                headers={'X-CSRF-Token':csrf,'Origin':'http://127.0.0.1:8006'})
            assert response.status_code==200,response.text
            info=client.get('/api/collections/benchmark_detection_mini_val/review').json()
            assert info['approval_valid'] and info['fingerprint']==exported['fingerprint']
            assert any(e['action']=='export' for e in info['events'])
        result=json.loads((qa/'browser-evidence.json').read_text())
        result.update(qa_directory=str(qa),worker_started=False,inference_or_evaluator_run=False,
            restart='passed: account/review/approval/history persisted',export=dict(path=str(qa/'detection-collection-approved.zip'),
                fingerprint=exported['fingerprint'],entries=exported['entries'],files_verified=exported['files_verified']))
        (evidence_root/'browser_evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(status='passed',qa=str(qa),export_files=exported['files_verified']),indent=2))
    finally:
        stop(frontend);stop(backend)
        for stream in logs.values():stream.close()


if __name__=='__main__':
    main()
