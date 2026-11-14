import { expect, test } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { apiLogin, apiPost, pageLogin } from './auth-helpers';

const root = path.resolve(__dirname, '../../..');
const name = 'infer_mini_val_scene0103_f1';
const base = `/api/runs/${name}`;
const source = JSON.parse(fs.readFileSync(path.join(root, 'work_dirs/eval_mini_val_scene0103_f1/evaluation.json'), 'utf8'));
const percent = (value: number) => new Intl.NumberFormat('vi-VN', { style: 'percent', maximumFractionDigits: 4 }).format(value);

function qaEvaluation() {
  const runs = process.env.VIEWER_E2E_RUNS_ROOT;
  if (!runs || !path.resolve(runs).startsWith(path.join(root, 'work/review-qa-'))) throw new Error('Only isolated QA evaluation may be modified');
  return path.join(runs, 'eval_mini_val_scene0103_f1/evaluation.json');
}

test('saved evaluation ratios, per-class evidence and run switching never retain the other run metrics', async ({ page }) => {
  await pageLogin(page);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  const select = page.getByRole('combobox', { name: 'Chọn run' });
  for (let i = 0; i < 2; i++) {
    await select.selectOption(name);
    await expect(page.getByTestId('evaluation-panel')).toHaveAttribute('data-status', 'matched');
    await expect(page.getByTestId('iou-value')).toHaveText(percent(source.metrics.occupancy_iou));
    await expect(page.getByTestId('miou-value')).toHaveText(percent(source.metrics.semantic_miou));
    await expect(page.getByTestId('evaluation-scope')).toContainText('1 frame mini_val');
    await select.selectOption('infer_mini_cuda_v2');
    await expect(page.getByTestId('evaluation-panel')).toHaveAttribute('data-status', 'missing_gt');
    await expect(page.getByTestId('iou-value')).toHaveText('Chưa có GT');
    await expect(page.getByTestId('miou-value')).toHaveText('Chưa có GT');
  }
  await select.selectOption(name);
  await expect(page.getByTestId('evaluation-panel')).toHaveAttribute('data-status', 'matched');
  await page.getByText('IoU từng lớp semantic', { exact: true }).click();
  for (const entry of source.metrics.per_class_semantic) {
    await expect(page.getByTestId(`evaluation-class-${entry.label}`)).toContainText(percent(entry.iou));
  }
  await expect(page.locator('.evaluation-convention')).toContainText('Trung bình đủ 16 lớp semantic; union = 0');
  await expect(page.locator('.uncalibrated-label')).toHaveText('Chưa hiệu chuẩn');
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: path.join(root, 'work_dirs/web_evaluation_viewer.png'), fullPage: true });
  expect(errors).toEqual([]);
});

test('v1 approval stays scoped to four artifacts; Engineer attaches evaluation in v2 and Reviewer approves its export', async ({ page, browser, request, playwright, baseURL }) => {
  await apiLogin(request);
  let response = await apiPost(request, `${base}/register`, { comment: 'Legacy snapshot without evaluation' });
  expect(response.status()).toBe(200);
  let info = await response.json();
  response = await apiPost(request, `${base}/review/submit`, { comment: 'Submit four artifacts', revision_id: info.revision_id, fingerprint: info.fingerprint });
  expect(response.status()).toBe(200);
  const reviewerApi = await playwright.request.newContext({ baseURL });
  await apiLogin(reviewerApi, 'reviewer');
  response = await apiPost(reviewerApi, `${base}/review/decision`, { comment: 'Approve only the original four artifacts', decision: 'approved', revision_id: info.revision_id, fingerprint: info.fingerprint });
  expect(response.status()).toBe(200);
  const legacy = await response.json();
  expect(Object.keys(legacy.manifest.artifacts)).toHaveLength(4);

  await pageLogin(page);
  await page.getByRole('combobox', { name: 'Chọn run' }).selectOption(name);
  await expect(page.getByTestId('evaluation-review-scope')).toContainText('chưa thuộc revision hiện tại');
  const panel = page.getByTestId('review-panel');
  await panel.getByRole('textbox', { name: 'Nhận xét review' }).fill('Add the verified evaluation with a new approval scope');
  await panel.getByRole('button', { name: 'Thêm evaluation vào revision mới' }).click();
  await expect(panel.getByTestId('review-state')).toContainText('draft');
  await expect(panel.getByTestId('revision-history')).toContainText('r2 · draft · manifest v2');
  await expect(panel.getByTestId('revision-history')).toContainText('r1 · approved · manifest v1');
  await expect(page.getByTestId('evaluation-review-scope')).toContainText('revision 2 · draft');
  expect((await apiPost(request, `${base}/export`)).status()).toBe(409);
  await expect(panel.getByRole('button', { name: 'Xuất gói đã duyệt' })).toBeDisabled();
  await panel.getByRole('textbox', { name: 'Nhận xét review' }).fill('Submit evaluation and the four exact source artifacts');
  await panel.getByRole('button', { name: 'Gửi review' }).click();
  await expect(panel.getByTestId('review-state')).toContainText('submitted');

  const context = await browser.newContext({ baseURL });
  const reviewer = await context.newPage();
  await pageLogin(reviewer, 'reviewer');
  await reviewer.getByRole('combobox', { name: 'Chọn run' }).selectOption(name);
  const decision = reviewer.getByTestId('review-panel');
  await decision.getByRole('textbox', { name: 'Nhận xét review' }).fill('Approve v2 including the exact evaluation hash');
  await decision.getByRole('button', { name: 'Phê duyệt', exact: true }).click();
  await expect(decision.getByTestId('review-state')).toContainText('approved');
  await expect(reviewer.getByTestId('evaluation-review-scope')).toContainText('revision 2 · approved');
  const downloadPromise = reviewer.waitForEvent('download');
  await decision.getByRole('button', { name: 'Xuất gói đã duyệt' }).click();
  const download = await downloadPromise;
  await download.saveAs(path.join(process.env.VIEWER_E2E_OUTPUT!, 'evaluation-approved-export.zip'));
  await expect(decision).toHaveAttribute('aria-busy', 'false');
  await reviewer.evaluate(() => window.scrollTo(0, 0));
  await reviewer.screenshot({ path: path.join(root, 'work_dirs/web_evaluation_approved.png'), fullPage: true });
  info = await (await reviewer.request.get(`${base}/review`)).json();
  expect(info.fingerprint).not.toBe(legacy.fingerprint);
  expect(info.manifest.artifacts['evaluation.json']).toBeTruthy();
  await context.close();
  await reviewerApi.dispose();
});

test('changed evaluation blocks export and wrong token/hash or missing evaluation clear displayed metrics', async ({ page, request }) => {
  await apiLogin(request);
  await pageLogin(page);
  await page.getByRole('combobox', { name: 'Chọn run' }).selectOption(name);
  const target = qaEvaluation();
  const before = fs.readFileSync(target);
  const refresh = page.getByRole('button', { name: 'Làm mới danh sách run' });
  try {
    fs.writeFileSync(target, Buffer.concat([before, Buffer.from(' ')]));
    expect((await apiPost(request, `${base}/export`)).status()).toBe(409);
    await refresh.click();
    await expect(page.getByTestId('review-panel').getByRole('alert')).toContainText('đã thay đổi');
    for (const field of ['token', 'hash']) {
      const changed = JSON.parse(before.toString('utf8'));
      if (field === 'token') changed.frames[0].sample_token = 'wrong-token';
      else changed.frames[0].prediction_sha256 = '0'.repeat(64);
      fs.writeFileSync(target, JSON.stringify(changed));
      await refresh.click();
      await expect(page.getByTestId('evaluation-panel')).toHaveAttribute('data-status', field === 'token' ? 'token_mismatch' : 'hash_mismatch');
      await expect(page.getByTestId('iou-value')).toHaveText('Chưa có GT');
      await expect(page.getByTestId('miou-value')).toHaveText('Chưa có GT');
    }
    fs.renameSync(target, target + '.hidden');
    await refresh.click();
    await expect(page.getByTestId('evaluation-panel')).toHaveAttribute('data-status', 'unmatched');
    await expect(page.getByTestId('iou-value')).toHaveText('Chưa có GT');
    fs.renameSync(target + '.hidden', target);
  } finally {
    if (fs.existsSync(target + '.hidden')) fs.renameSync(target + '.hidden', target);
    fs.writeFileSync(target, before);
  }
  await refresh.click();
  await expect(page.getByTestId('evaluation-panel')).toHaveAttribute('data-status', 'matched');
  await expect(page.getByTestId('iou-value')).toHaveText(percent(source.metrics.occupancy_iou));
  expect((await (await request.get(`${base}/review`)).json()).approval_valid).toBe(true);
  expect((await (await request.get('/api/health')).json()).inference_enabled).toBe(false);
});
