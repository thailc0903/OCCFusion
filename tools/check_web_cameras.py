#!/usr/bin/env python3
"""Saved-camera Chromium QA on private state/ports/build; no worker/model/evaluator."""
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
const root=process.env.CAMERA_ROOT,out=process.env.CAMERA_OUT,base='http://127.0.0.1:3007';
const {chromium}=require(root+'/web/frontend/node_modules/playwright');
const expect=require(root+'/web/frontend/node_modules/@playwright/test').expect.configure({timeout:180000});
const endpoint='/api/runs/detection/cameras',sample='3950bd41f74548429c0f7700ff3d8269';
const plan=JSON.parse(fs.readFileSync(root+'/work_dirs/web_camera_qa/manual_inspection_plan.json'));
async function post(page,url,body){
 const session=await(await page.request.get(base+'/api/auth/session')).json();
 return page.request.post(base+url,{data:body,headers:{'X-CSRF-Token':session.csrf_token,Origin:base}});
}
async function sign(page,user,password){
 await page.goto(base+'/detections?run=detection');await page.getByRole('textbox',{name:'Tên tài khoản'}).fill(user);
 await page.getByLabel('Mật khẩu',{exact:true}).fill(password);await page.getByRole('button',{name:'Đăng nhập',exact:true}).click();
 await expect(page.getByTestId('detection-token')).toHaveText(sample);
}
async function imagesReady(page){
 await expect(page.locator('[data-testid="camera-gallery"] img')).toHaveCount(6);
 await expect.poll(()=>page.locator('[data-testid="camera-gallery"] img').evaluateAll(nodes=>nodes.every(n=>n.complete&&n.naturalWidth===1600&&n.naturalHeight===900))).toBe(true);
}
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 try{
  const context=await browser.newContext({viewport:{width:1520,height:1080},locale:'vi-VN',acceptDownloads:true});
  const page=await context.newPage(),errors=[],jobs=[];
  page.setDefaultTimeout(180000);page.setDefaultNavigationTimeout(180000);
  page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(r.method()==='POST'&&r.url().includes('/api/jobs'))jobs.push(r.url());});
  const health=await(await page.request.get(base+'/api/health')).json();
  const direct=await(await page.request.get('http://127.0.0.1:8007/api/health')).json();
  assert.equal(health.workspace_id,direct.workspace_id);assert.equal(health.model_modules_loaded,false);
  assert.equal((await page.request.get(base+endpoint)).status(),401);
  await sign(page,'qa_engineer',process.env.CAMERA_PASSWORD);
  await page.getByRole('button',{name:'Mở sáu camera',exact:true}).click();await imagesReady(page);
  const initial=await(await page.request.get(base+endpoint)).json();
  assert(initial.channels.every(c=>c.privacy.status==='not_reviewed'&&!c.export_eligible));
  assert.equal((await post(page,endpoint+'/register',{comment:'Unreviewed images cannot export'})).status(),409);
  assert.equal((await post(page,endpoint+'/export')).status(),409);
  for(const planned of plan.images){
   await page.getByRole('button',{name:'Chọn camera '+planned.channel,exact:true}).click();
   await expect(page.getByTestId('camera-active-channel')).toHaveText(planned.channel);
   await expect(page.locator('[data-testid="camera-projection-svg"]')).toHaveAttribute('data-channel',planned.channel);
   await expect(page.locator('[data-testid="camera-projection-svg"]')).toHaveAttribute('data-projection-camera',planned.channel);
   await expect.poll(()=>page.locator('[data-testid="camera-projection-svg"] circle').count()).toBeGreaterThan(0);
   const camera=initial.channels.find(c=>c.channel===planned.channel);
   assert.equal(camera.source_image.sha256,planned.source_sha256);
   const result=await post(page,endpoint+'/'+planned.channel+'/privacy',{camera_fingerprint:camera.camera_fingerprint,
    previous_revision:null,reviewed:true,masks:planned.supplemental_masks,
    comment:plan.inspection+' Exact original source hash checked against manual inspection plan.'});
   assert.equal(result.status(),200,await result.text());
  }
  // Reload camera state only, then exercise pointer-to-original-pixel mapping.
  await page.getByRole('button',{name:'Đóng camera',exact:true}).click();
  await page.getByRole('button',{name:'Mở sáu camera',exact:true}).click();await imagesReady(page);
  await page.getByRole('button',{name:'Chọn camera CAM_FRONT_RIGHT',exact:true}).click();
  await expect(page.getByTestId('camera-active-channel')).toHaveText('CAM_FRONT_RIGHT');
  await page.setViewportSize({width:1000,height:1000});
  const zoom=page.getByLabel('Camera zoom',{exact:true});await zoom.focus();await zoom.press('Home');await zoom.press('ArrowRight');await zoom.press('ArrowRight');
  await page.getByText('Kiểm tra ảnh này & mask bổ sung',{exact:true}).click();
  await page.getByLabel('Draw supplemental mask',{exact:true}).check();
  const stage=page.getByTestId('camera-stage'),svg=page.getByTestId('camera-projection-svg');
  await stage.scrollIntoViewIfNeeded();
  const rect=await svg.boundingBox();assert(rect);
  const original=[78,428,144,494],start=[rect.x+original[0]*rect.width/1600,rect.y+original[1]*rect.height/900],
   end=[rect.x+original[2]*rect.width/1600,rect.y+original[3]*rect.height/900];
  await page.mouse.move(...start);await page.mouse.down();await page.mouse.move(...end,{steps:5});await page.mouse.up();
  await expect(page.getByTestId('camera-mask-count')).toHaveText('2 mask');
  await page.getByLabel('Acknowledge individual image inspection',{exact:true}).check();
  await page.getByLabel('Camera privacy comment',{exact:true}).fill('QA manual per-image inspection; supplemental face rectangle through responsive/zoom UI; not exhaustive PII certification.');
  const savedPrivacy=page.waitForResponse(r=>r.request().method()==='POST'&&r.url().endsWith('/CAM_FRONT_RIGHT/privacy'));
  await page.getByRole('button',{name:'Lưu kiểm tra & derivative ảnh này',exact:true}).click();
  assert.equal((await savedPrivacy).status(),200);
  await expect.poll(async()=>{
    const v=await(await page.request.get(base+endpoint)).json();
    return v.channels.find(c=>c.channel==='CAM_FRONT_RIGHT').privacy.masks.length;
  }).toBe(2);
  await imagesReady(page);
  const reviewed=await(await page.request.get(base+endpoint)).json(),right=reviewed.channels.find(c=>c.channel==='CAM_FRONT_RIGHT');
  const drawn=right.privacy.masks.at(-1);assert(drawn.every((v,i)=>Math.abs(v-original[i])<=1),JSON.stringify({drawn,original}));
  assert(reviewed.channels.every(c=>c.privacy.status==='engineer_reviewed'&&c.privacy.derivative));
  await page.setViewportSize({width:1520,height:1080});
  await page.getByTestId('camera-gallery').screenshot({path:path.join(out,'six-cameras-reviewed.png')});
  await page.getByRole('button',{name:'Chọn camera CAM_FRONT',exact:true}).click();
  await expect(page.getByTestId('camera-active-channel')).toHaveText('CAM_FRONT');
  await expect(svg).toHaveAttribute('data-projection-camera','CAM_FRONT');
  await expect.poll(()=>page.locator('[data-testid="camera-projection-svg"] line').count()).toBeGreaterThan(0);
  await page.getByLabel('Camera LiDAR projection',{exact:true}).uncheck();await expect(svg.locator('circle')).toHaveCount(0);
  await page.getByLabel('Camera LiDAR projection',{exact:true}).check();
  await page.getByLabel('Camera prediction boxes',{exact:true}).uncheck();await expect(svg.locator('line')).toHaveCount(0);
  await page.getByLabel('Camera prediction boxes',{exact:true}).check();
  await expect.poll(()=>svg.locator('circle').count()).toBeGreaterThan(0);
  await expect.poll(()=>svg.locator('line').count()).toBeGreaterThan(0);
  await page.getByRole('button',{name:'Phóng to camera',exact:true}).click();await expect(page.getByRole('dialog',{name:'Camera phóng to'})).toBeVisible();
  await page.getByRole('button',{name:'Đóng phóng to',exact:true}).click();
  await expect(svg).toHaveAttribute('data-projection-camera','CAM_FRONT');
  await expect.poll(()=>svg.locator('circle').count()).toBeGreaterThan(0);
  await expect.poll(()=>svg.locator('line').count()).toBeGreaterThan(0);
  await stage.evaluate(el=>el.scrollIntoView({block:'center'}));
  await stage.screenshot({path:path.join(out,'camera-front-projection.png')});
  // Register/submit old prediction independently; this does not approve camera derivatives.
  let parent=(await(await post(page,'/api/runs/detection/register',{comment:'QA parent access scope only'})).json());
  parent=(await(await post(page,'/api/runs/detection/review/submit',{comment:'QA parent submitted for Reviewer visibility',
   revision_id:parent.revision_id,fingerprint:parent.fingerprint})).json());
  assert.equal(parent.state,'submitted');
  const cr=page.getByTestId('camera-review');
  await cr.getByLabel('Nhận xét collection',{exact:true}).fill('QA independent six-camera derivative snapshot');
  await cr.getByRole('button',{name:'Đăng ký collection',exact:true}).click();
  await expect(cr.getByTestId('collection-state')).toHaveText('draft');
  let info=await(await page.request.get(base+endpoint+'/review')).json();
  assert.equal((await post(page,endpoint+'/review/decision',{comment:'Engineer denied',decision:'approved',revision_id:info.revision_id,fingerprint:info.fingerprint})).status(),403);
  await cr.getByLabel('Nhận xét collection',{exact:true}).fill('QA submit exact images/calibration/projection/source hashes; no PII completeness guarantee');
  await cr.getByRole('button',{name:'Gửi review collection',exact:true}).click();
  await expect(cr.getByTestId('collection-state')).toHaveText('submitted');
  const rc=await browser.newContext({viewport:{width:1520,height:1080},locale:'vi-VN'}),reviewer=await rc.newPage();
  reviewer.setDefaultTimeout(180000);reviewer.setDefaultNavigationTimeout(180000);
  await sign(reviewer,'qa_reviewer',process.env.CAMERA_REVIEW_PASSWORD);
  await reviewer.getByRole('button',{name:'Mở sáu camera',exact:true}).click();await imagesReady(reviewer);
  const rr=reviewer.getByTestId('camera-review');
  await expect(rr.getByTestId('collection-state')).toHaveText('submitted');
  await rr.getByLabel('Nhận xét collection',{exact:true}).fill('QA approve camera snapshot only, separate from prediction and benchmark approvals.');
  await rr.getByRole('button',{name:'Phê duyệt collection',exact:true}).click();await expect(rr.getByTestId('collection-state')).toHaveText('approved');
  await rr.screenshot({path:path.join(out,'camera-review-approved.png')});
  // Browser download uses existing CSRF path. Refresh only the camera listing.
  await page.getByRole('button',{name:'Đóng camera',exact:true}).click();await page.getByRole('button',{name:'Mở sáu camera',exact:true}).click();
  await expect(cr.getByTestId('collection-state')).toHaveText('approved');
  const download=page.waitForEvent('download');await cr.getByRole('button',{name:'Xuất collection đã duyệt',exact:true}).click();
  await(await download).saveAs(path.join(out,'camera-collection-approved.zip'));
  info=await(await page.request.get(base+endpoint+'/review')).json();
  // Derivative tamper only in isolated state, never source dataset pixels.
  const front=reviewed.channels.find(c=>c.channel==='CAM_FRONT');
  const file=path.join(process.env.VIEWER_STATE_ROOT,'cameras/derivatives',front.privacy.derivative.sha256+'.png'),saved=fs.readFileSync(file);
  try{
   fs.appendFileSync(file,' ');
   assert.equal((await post(page,endpoint+'/export')).status(),409);
   const bad=await(await page.request.get(base+endpoint)).json();
   assert.equal(bad.channels.find(c=>c.channel==='CAM_FRONT').status,'blocked');
   await page.getByRole('button',{name:'Đóng camera',exact:true}).click();await page.getByRole('button',{name:'Mở sáu camera',exact:true}).click();
   await expect(page.locator('.camera-blocked').filter({hasText:'Privacy derivative hash/dimensions mismatch'})).not.toHaveCount(0);
   await page.getByTestId('camera-viewer').screenshot({path:path.join(out,'camera-hash-blocked.png')});
  }finally{fs.writeFileSync(file,saved);}
  // Real saved benchmark frame navigation: a new frame has no privacy/source registration.
  await page.goto(base+'/benchmarks?benchmark=benchmark_detection_mini_val');
  await expect(page.getByTestId('detection-map')).toBeVisible();
  await page.getByLabel('Chọn frame benchmark detection',{exact:true}).selectOption(sample);
  await expect(page.getByTestId('detection-benchmark-token')).toHaveText(sample);
  await page.getByRole('button',{name:'Mở sáu camera',exact:true}).click();await imagesReady(page);
  await expect(page.getByTestId('camera-sample-token')).toHaveText(sample);
  await page.getByLabel('Chọn frame benchmark detection',{exact:true}).selectOption('3e8750f331d7499e9b5123e9eb70f2e2');
  await expect(page.getByTestId('detection-benchmark-token')).toHaveText('3e8750f331d7499e9b5123e9eb70f2e2');
  await expect(page.getByTestId('camera-gallery')).toHaveCount(0);
  await page.getByRole('button',{name:'Mở sáu camera',exact:true}).click();
  await expect(page.getByTestId('camera-sample-token')).toHaveText('3e8750f331d7499e9b5123e9eb70f2e2');
  await expect(page.locator('[data-testid="camera-gallery"] img')).toHaveCount(0);
  await expect(page.getByTestId('camera-projection-svg')).toHaveCount(0);
  assert.deepEqual(errors,[]);assert.deepEqual(jobs,[]);
  fs.writeFileSync(path.join(out,'browser-evidence.json'),JSON.stringify({status:'passed',sample_token:sample,camera_count:6,
   reviewed_images:reviewed.channels.map(c=>({channel:c.channel,sample_data_token:c.sample_data_token,source:c.source_image,derivative:c.privacy.derivative,privacy_revision:c.privacy.revision})),
   manual_inspection_plan:plan.inspection,mask_at_zoom:{zoom:1.5,viewport:[1000,1000],requested:original,actual:drawn},
   frame_change_clears_images:true,benchmark_source_matches:true,privacy_hash_blocks_export:true,
   roles_csrf_review_export:true,fingerprint:info.fingerprint,job_requests:jobs,page_errors:errors,model_modules_loaded:health.model_modules_loaded},null,2));
  await rc.close();await context.close();console.log('CAMERA_BROWSER_PASSED');
 }finally{await browser.close();}
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
"""


def verify_zip(path):
    with zipfile.ZipFile(path) as z:
        m=json.loads(z.read('manifest.json'));r=json.loads(z.read('review.json'))
        assert m['kind']=='rav25_camera_frame_collection' and r['state']=='approved'
        assert len(m['channels'])==6 and m['source_images_included'] is False
        assert not any(n.endswith('.jpg') for n in z.namelist())
        for name,value in m['artifacts'].items():
            content=z.read(name);assert hashlib.sha256(content).hexdigest()==value['sha256'] and len(content)==value['bytes']
        return dict(fingerprint=m['fingerprint_sha256'],files_verified=len(m['artifacts']),
            entries=len(z.namelist()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            review=r,channels=m['channels'],path=str(path))


def main(evidence_root=None):
    for port in (8007,3007):
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('QA port occupied: '+str(port))
    routes=json.loads((ROOT/'web/frontend/.next-camera-qa/routes-manifest.json').read_text())['rewrites']
    routes=routes if isinstance(routes,list) else sum(routes.values(),[])
    assert any(r['destination']=='http://127.0.0.1:8007/api/:path*' for r in routes)
    qa=Path(tempfile.mkdtemp(prefix='camera-web-',dir=ROOT/'work'));runs=qa/'runs';runs.mkdir()
    shutil.copytree(ROOT/'work/centerpoint-smoke-wyp1nl__/runs/infer_job_65e1b30833e343c3b434da8dc71423f9',runs/'detection')
    shutil.copytree(ROOT/'work_dirs/infer_mini_val_scene0103_f1',runs/'occupancy')
    shutil.copytree(ROOT/'work_dirs/benchmark_detection_mini_val',runs/'benchmark_detection_mini_val')
    env=dict(os.environ,VIEWER_RUNS_ROOT=str(runs),VIEWER_STATE_ROOT=str(qa/'state'),
        VIEWER_BENCHMARK_ROOTS=json.dumps([str(runs)]),VIEWER_EVALUATION_ROOTS=json.dumps([str(runs)]),
        VIEWER_ALLOWED_ORIGINS=json.dumps(['http://127.0.0.1:8007','http://127.0.0.1:3007']),
        VIEWER_API_ORIGIN='http://127.0.0.1:8007',VIEWER_NEXT_DIST_DIR='.next-camera-qa',
        NEXT_TELEMETRY_DISABLED='1',CAMERA_ROOT=str(ROOT),CAMERA_OUT=str(qa))
    env['PATH']=str(ROOT/'.cache/toolchains/node/bin')+os.pathsep+env.get('PATH','')
    env['PLAYWRIGHT_BROWSERS_PATH']=str(ROOT/'.cache/ms-playwright')
    env['LD_LIBRARY_PATH']=str(ROOT/'.cache/browser-libs/usr/lib/x86_64-linux-gnu')+os.pathsep+env.get('LD_LIBRARY_PATH','')
    for user,role,key in [('qa_engineer','engineer','CAMERA_PASSWORD'),('qa_reviewer','reviewer','CAMERA_REVIEW_PASSWORD')]:
        env[key]=secrets.token_urlsafe(28)
        subprocess.run([sys.executable,'tools/create_viewer_user.py','--username',user,'--role',role,'--password-stdin'],
            cwd=ROOT,env=env,input=env[key]+'\n',text=True,check=True)
    subprocess.run([sys.executable,'tools/register_camera_source.py','--run-id','detection','--state-root',str(qa/'state')],
        cwd=ROOT,env=env,check=True,stdout=(qa/'source-registration.json').open('w'))
    logs={name:(qa/(name+'.log')).open('ab') for name in ('backend','frontend')};backend=frontend=None
    command=[sys.executable,'-m','uvicorn','web.backend.app.main:app','--host','127.0.0.1','--port','8007']
    print('Camera QA:',qa,flush=True)
    try:
        backend=subprocess.Popen(command,cwd=ROOT,env=env,stdout=logs['backend'],stderr=subprocess.STDOUT)
        frontend=subprocess.Popen([shutil.which('node',path=env['PATH']),str(ROOT/'web/frontend/node_modules/next/dist/bin/next'),
            'start',str(ROOT/'web/frontend'),'--hostname','127.0.0.1','--port','3007'],cwd=ROOT,env=env,stdout=logs['frontend'],stderr=subprocess.STDOUT)
        wait_ready('http://127.0.0.1:8007/api/health',backend);wait_ready('http://127.0.0.1:3007/detections',frontend)
        script=qa/'browser.cjs';script.write_text(BROWSER)
        subprocess.run([shutil.which('node',path=env['PATH']),str(script)],cwd=ROOT,env=env,check=True)
        exported=verify_zip(qa/'camera-collection-approved.zip')
        stop(backend);backend=subprocess.Popen(command,cwd=ROOT,env=env,stdout=logs['backend'],stderr=subprocess.STDOUT)
        wait_ready('http://127.0.0.1:8007/api/health',backend)
        with httpx.Client(base_url='http://127.0.0.1:8007',timeout=180) as c:
            csrf=c.get('/api/auth/session').json()['csrf_token']
            r=c.post('/api/auth/login',json=dict(username='qa_engineer',password=env['CAMERA_PASSWORD']),
                headers={'X-CSRF-Token':csrf,'Origin':'http://127.0.0.1:8007'});assert r.status_code==200,r.text
            info=c.get('/api/runs/detection/cameras/review').json();assert info['approval_valid']
            assert info['fingerprint']==exported['fingerprint'] and any(e['action']=='export' for e in info['events'])
        result=json.loads((qa/'browser-evidence.json').read_text())
        result.update(qa_directory=str(qa),export=exported,restart='passed',worker_started=False,inference_or_evaluator_run=False)
        evidence_root=evidence_root or ROOT/'work_dirs/web_camera_qa'
        evidence_root.mkdir(parents=True,exist_ok=True)
        (evidence_root/'browser_evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(status='passed',qa=str(qa),export_files=exported['files_verified']),indent=2))
    finally:
        stop(frontend);stop(backend)
        for stream in logs.values():stream.close()


if __name__=='__main__':main()
