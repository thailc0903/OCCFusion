import { test, expect } from '@playwright/test';
import { readFileSync, writeFileSync, statSync, utimesSync } from 'node:fs';
import path from 'node:path';
import { apiPost, pageLogin } from './auth-helpers';

const root = path.resolve(__dirname, '../../..');
const runs = process.env.VIEWER_E2E_BENCHMARK_ROOT || '';
const bench = path.join(runs,'benchmark_mini_val');
const collection = '/api/collections/benchmark_mini_val';
const percentage = new Intl.NumberFormat('vi-VN',{style:'percent',maximumFractionDigits:4});
const artifact = (name: string) => JSON.parse(readFileSync(path.join(bench,name),'utf8'));
const evidence = (name: string) => path.join(root,'work_dirs',name);

test.beforeEach(async ({page}) => {
  page.on('pageerror',error=>console.error('BENCHMARK_PAGE_ERROR',error.message));
});

test('benchmark and collection require authentication', async ({page,request}) => {
  for (const url of ['/api/benchmarks','/api/benchmarks/benchmark_mini_val',collection+'/review']) expect((await request.get(url)).status()).toBe(401);
  expect((await request.post(collection+'/export')).status()).toBe(401);
  await page.goto('/benchmarks');await expect(page.getByRole('heading',{name:'Đăng nhập',exact:true})).toBeVisible();
  await expect(page.locator('canvas')).toHaveCount(0);
});

test('engineer views real pooled evidence, failed frames and both scene predictions without inference', async ({page}) => {
  const errors: string[]=[];const requests: string[]=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
  await pageLogin(page);await page.goto('/benchmarks');
  const pooled=artifact('metrics_pooled.json');const initial=artifact('initial_report.json');
  await expect(page.getByTestId('benchmark-coverage')).toHaveText(`${pooled.successful_frame_count}/${pooled.expected_frame_count} frame hợp lệ`);
  await expect(page.getByTestId('benchmark-status')).toHaveText('PARTIAL');
  await expect(page.getByTestId('pooled-iou')).toHaveText(percentage.format(pooled.metrics.occupancy_iou));
  await expect(page.getByTestId('pooled-miou')).toHaveText(percentage.format(pooled.metrics.semantic_miou));
  await expect(page.getByTestId('benchmark-median')).toHaveText(initial.latency_current_invocation.excluding_marked_warmup.median_s.toFixed(4)+' s');
  await expect(page.getByTestId('benchmark-latency-source')).toContainText('initial_report.json');
  await expect(page.getByTestId('benchmark-latency')).toContainText('78 mẫu');
  await expect(page.locator('[data-testid^="pooled-class-"]')).toHaveCount(16);
  await page.getByText('Confusion matrix ·',{exact:false}).click();await expect(page.getByRole('table',{name:'Confusion matrix pooled'})).toBeVisible();
  await expect(page.getByRole('table',{name:'Confusion matrix pooled'}).locator('tbody tr')).toHaveCount(17);
  await page.getByText('Confusion matrix ·',{exact:false}).click();
  await expect(page.getByTestId('benchmark-frame-token')).toHaveText('3950bd41f74548429c0f7700ff3d8269');
  await expect(page.locator('canvas')).toBeVisible();
  await page.getByRole('switch',{name:'LiDAR benchmark'}).check();
  await expect(page.locator('.benchmark-playback .viewport-count')).toContainText('điểm LiDAR');
  await page.getByLabel('Z min benchmark').fill('-4');await page.getByRole('button',{name:'Reset camera',exact:true}).click();
  await page.getByTestId('benchmark-playback').screenshot({path:evidence('web_benchmark_frame3d.png')});
  await page.evaluate(()=>window.scrollTo(0,0));
  await page.screenshot({path:evidence('web_benchmark_overview.png'),fullPage:true});
  await page.getByRole('combobox',{name:'Chọn frame benchmark'}).selectOption('3e8750f331d7499e9b5123e9eb70f2e2');
  await expect(page.getByTestId('failed-frame')).toContainText('Không khả dụng');await expect(page.locator('canvas')).toHaveCount(0);
  await expect(page.getByTestId('evaluation-panel')).toHaveCount(0);
  await page.screenshot({path:evidence('web_benchmark_failed_frame.png'),fullPage:true});
  await page.getByRole('combobox',{name:'Chọn scene benchmark'}).selectOption('scene-0916');
  const token='0bb62a68055249e381b039bf54b0ccf8';await expect(page.getByTestId('benchmark-frame-token')).toHaveText(token);
  const state=artifact('progress.json').frames[token];const evaluation=JSON.parse(readFileSync(path.join(bench,'evaluations',token,path.basename(state.evaluation_dir),'evaluation.json'),'utf8'));
  await expect(page.getByTestId('iou-value')).toHaveText(percentage.format(evaluation.metrics.occupancy_iou));
  await expect(page.getByTestId('evaluation-panel')).toHaveAttribute('data-status','matched');
  await page.getByRole('combobox',{name:'Chọn frame benchmark'}).selectOption('b5989651183643369174912bc5641d3b');
  await expect(page.getByTestId('failed-frame')).toContainText('RADAR_FRONT_LEFT');await expect(page.getByTestId('evaluation-panel')).toHaveCount(0);
  expect(requests.some(url=>/\/api\/(infer|inference|train|evaluate)(\/|\?)/.test(url))).toBe(false);
  const health=await (await page.request.get('/api/health')).json();expect(health.inference_enabled).toBe(false);expect(health.model_modules_loaded).toBe(false);
  expect(errors).toEqual([]);
});

test('engineer submits collection, reviewer approves and UI exports its own manifest', async ({page,browser}) => {
  test.setTimeout(240_000); // Multiple full 79-frame integrity checks and a real collection ZIP.
  await pageLogin(page);await page.goto('/benchmarks');await expect(page.getByTestId('collection-state')).toHaveText('unregistered');
  await page.getByLabel('Nhận xét collection',{exact:true}).fill('Review the saved 79/81 partial mini_val collection; radar startup omissions are explicit.');
  await page.getByRole('button',{name:'Đăng ký collection',exact:true}).click();await expect(page.getByTestId('collection-state')).toHaveText('draft');
  expect((await apiPost(page.request,collection+'/export')).status()).toBe(409);
  const info=await (await page.request.get(collection+'/review')).json();
  expect((await apiPost(page.request,collection+'/review/decision',{comment:'Engineer cannot approve',decision:'approved',revision_id:info.revision_id,fingerprint:info.fingerprint})).status()).toBe(403);
  await expect(page.getByRole('button',{name:'Phê duyệt collection',exact:true})).toHaveCount(0);
  await page.getByLabel('Nhận xét collection',{exact:true}).fill('Submit this exact collection fingerprint, including pooled evaluation and original latency evidence.');
  await page.getByRole('button',{name:'Gửi review collection',exact:true}).click();await expect(page.getByTestId('collection-state')).toHaveText('submitted');
  const reviewerContext=await browser.newContext();const reviewer=await reviewerContext.newPage();
  try {
    await pageLogin(reviewer,'reviewer');await reviewer.goto('/benchmarks');await expect(reviewer.getByTestId('collection-state')).toHaveText('submitted');
    await reviewer.getByLabel('Nhận xét collection',{exact:true}).fill('Approve the verified saved subset of 79/81 frames; this is not complete nuScenes validation or learned 4D flow.');
    await reviewer.getByRole('button',{name:'Phê duyệt collection',exact:true}).click();await expect(reviewer.getByTestId('collection-state')).toHaveText('approved');
    await reviewer.screenshot({path:evidence('web_benchmark_approved.png'),fullPage:true});
  } finally {await reviewerContext.close();}
  await page.reload();await expect(page.getByTestId('collection-state')).toHaveText('approved');
  const download=page.waitForEvent('download');await page.getByRole('button',{name:'Xuất collection đã duyệt',exact:true}).click();
  await (await download).saveAs(path.join(process.env.VIEWER_E2E_OUTPUT!,'collection-approved.zip'));
});

test('changed QA artifact blocks UI join and collection export; original sources are untouched', async ({page}) => {
  expect(path.resolve(runs)).toContain(path.sep+'work'+path.sep+'benchmark-qa-');
  await pageLogin(page);await page.goto('/benchmarks');await expect(page.getByTestId('collection-state')).toHaveText('approved');
  const token='3950bd41f74548429c0f7700ff3d8269';const state=artifact('progress.json').frames[token];
  const file=path.join(bench,'frames',token,path.basename(state.prediction_dir),'metadata.json');
  const original=readFileSync(file);const stat=statSync(file);const modified=Buffer.from(original);modified[modified.length-1]=32;
  try {
    writeFileSync(file,modified);utimesSync(file,stat.atime,stat.mtime);
    expect((await apiPost(page.request,collection+'/export')).status()).toBe(409);
    await page.getByRole('button',{name:'Đọc lại artifact',exact:true}).click();
    await expect(page.locator('.error-panel[role="alert"]')).toContainText('prediction receipt/token/hash mismatch');
    await expect(page.getByTestId('pooled-iou')).toHaveCount(0);await expect(page.locator('canvas')).toHaveCount(0);
    await page.screenshot({path:evidence('web_benchmark_hash_mismatch.png'),fullPage:true});
  } finally {writeFileSync(file,original);utimesSync(file,stat.atime,stat.mtime);}
  await page.reload();await expect(page.getByTestId('collection-state')).toHaveText('approved');await expect(page.getByTestId('collection-integrity')).toHaveText('Hash artifact & manifest khớp');
});
