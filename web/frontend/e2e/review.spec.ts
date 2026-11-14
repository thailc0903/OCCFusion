import { expect, test } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { apiLogin, apiPost, pageLogin } from './auth-helpers';

const root = path.resolve(__dirname, '../../..');
const base = '/api/runs/infer_mini_cuda_v2';
const files = ['occupancy.npz', 'metadata.json', 'report.json', 'bev.png'];

test('anonymous access is blocked and Engineer → Reviewer UI approves the exact artifact bundle', async ({ page, browser, request, baseURL }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  const originalHashes = Object.fromEntries(files.map(name => [name, createHash('sha256').update(fs.readFileSync(path.join(root, 'work_dirs/infer_mini_cuda_v2', name))).digest('hex')]));
  expect((await request.get('/api/runs')).status()).toBe(401);
  expect((await request.get(`${base}/occupancy`)).status()).toBe(401);
  expect((await request.post(`${base}/export`)).status()).toBe(401);
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Đăng nhập', exact: true })).toBeVisible();
  await expect(page.getByTestId('occupancy-canvas')).toHaveCount(0);
  await page.screenshot({ path: path.join(root, 'work_dirs/web_review_login.png'), fullPage: true });
  await pageLogin(page);
  await expect(page.getByTestId('occupancy-canvas')).toHaveAttribute('data-rendered-instances', '44856');
  await expect(page.getByRole('button', { name: 'Phê duyệt', exact: true })).toHaveCount(0);
  const panel = page.getByTestId('review-panel');
  await panel.getByRole('textbox', { name: 'Nhận xét review' }).fill('Register real saved inference artifacts for review');
  await panel.getByRole('button', { name: 'Đăng ký run', exact: true }).click();
  await expect(panel.getByTestId('review-state')).toContainText('draft');
  await expect(panel.getByRole('button', { name: 'Xuất gói đã duyệt' })).toBeDisabled();
  expect((await apiPost(page.request, `${base}/export`)).status()).toBe(409);
  await panel.getByRole('textbox', { name: 'Nhận xét review' }).fill('Coordinates and LiDAR sample checked by engineer');
  await panel.getByRole('button', { name: 'Lưu ghi chú' }).click();
  await expect(panel.locator('.review-history').getByText('Coordinates and LiDAR sample checked by engineer', { exact: true })).toBeVisible();
  await expect(panel.getByRole('textbox', { name: 'Nhận xét review' })).toHaveValue('');
  await panel.getByRole('textbox', { name: 'Nhận xét review' }).fill('Please independently review the saved prediction');
  await panel.getByRole('button', { name: 'Gửi review' }).click();
  await expect(panel.getByTestId('review-state')).toContainText('submitted');
  const submitted = await (await page.request.get(`${base}/review`)).json();
  expect((await apiPost(page.request, `${base}/review/decision`, {
    decision: 'approved', comment: 'Engineer must be forbidden', revision_id: submitted.revision_id, fingerprint: submitted.fingerprint,
  })).status()).toBe(403);

  const reviewerContext = await browser.newContext({ baseURL });
  const reviewer = await reviewerContext.newPage();
  await pageLogin(reviewer, 'reviewer');
  await expect(reviewer.getByRole('combobox', { name: 'Chọn run' })).toHaveValue('infer_mini_cuda_v2');
  await expect(reviewer.getByRole('button', { name: 'Gửi review', exact: true })).toHaveCount(0);
  const reviewerPanel = reviewer.getByTestId('review-panel');
  await reviewerPanel.getByRole('textbox', { name: 'Nhận xét review' }).fill('Reviewer approved the unchanged fingerprint; no GT IoU is claimed');
  await reviewerPanel.getByRole('button', { name: 'Phê duyệt', exact: true }).click();
  await expect(reviewerPanel.getByTestId('review-state')).toContainText('approved');
  await expect(reviewer.getByTestId('iou-value')).toHaveText('Chưa có GT');
  const downloadPromise = reviewer.waitForEvent('download');
  await reviewerPanel.getByRole('button', { name: 'Xuất gói đã duyệt' }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toContain(`${submitted.fingerprint.slice(0, 12)}.zip`);
  const output = process.env.VIEWER_E2E_OUTPUT;
  if (!output) throw new Error('QA output directory must be explicitly configured');
  await download.saveAs(path.join(output, 'approved-export.zip'));
  await reviewer.screenshot({ path: path.join(root, 'work_dirs/web_review_approved.png'), fullPage: true });
  await reviewer.getByRole('button', { name: 'Đăng xuất' }).click();
  await expect(reviewer.getByRole('button', { name: 'Đăng nhập', exact: true })).toBeVisible();
  expect((await reviewer.request.get('/api/runs')).status()).toBe(401);
  await reviewerContext.close();
  expect(Object.fromEntries(files.map(name => [name, createHash('sha256').update(fs.readFileSync(path.join(root, 'work_dirs/infer_mini_cuda_v2', name))).digest('hex')]))).toEqual(originalHashes);
  expect(errors).toEqual([]);
  fs.writeFileSync(path.join(output, 'browser-workflow.json'), JSON.stringify({
    status: 'passed', engineer_approve_http: 403, unapproved_export_http: 409,
    anonymous_http: 401, submitted_fingerprint: submitted.fingerprint,
    artifacts_unchanged: originalHashes, page_errors: errors,
  }, null, 2));
});

test('artifact changes invalidate approval/export; negative decisions require a new draft', async ({ request, playwright, baseURL }) => {
  await apiLogin(request);
  const reviewer = await playwright.request.newContext({ baseURL });
  await apiLogin(reviewer, 'reviewer');
  const run = '/api/runs/auth_tamper';
  let response = await apiPost(request, `${run}/register`, { comment: 'Tamper QA copy only' });
  expect(response.status()).toBe(200);
  let info = await response.json();
  response = await apiPost(request, `${run}/review/submit`, { comment: 'Submitted QA copy', revision_id: info.revision_id, fingerprint: info.fingerprint });
  expect(response.status()).toBe(200);
  info = await response.json();
  response = await apiPost(reviewer, `${run}/review/decision`, { decision: 'approved', comment: 'Approve QA copy before tamper', revision_id: info.revision_id, fingerprint: info.fingerprint });
  expect(response.status()).toBe(200);
  const qaRoot = process.env.VIEWER_E2E_RUNS_ROOT;
  if (!qaRoot || !path.resolve(qaRoot).startsWith(path.join(root, 'work', 'review-qa-'))) throw new Error('Refuse to modify anything except the isolated QA copy');
  const target = path.join(qaRoot, 'auth_tamper/bev.png');
  const before = fs.readFileSync(target), changed = Buffer.from(before);
  changed[changed.length - 1] ^= 1;
  try {
    fs.writeFileSync(target, changed);
    const status = await (await request.get(`${run}/review`)).json();
    expect(status.approval_valid).toBe(false);
    expect(status.changed_artifacts).toContain('bev.png');
    expect((await apiPost(request, `${run}/export`)).status()).toBe(409);
  } finally { fs.writeFileSync(target, before); }

  const negative = '/api/runs/auth_negative';
  info = await (await apiPost(request, `${negative}/register`, { comment: 'Negative review flow' })).json();
  for (const decision of ['needs_changes', 'rejected']) {
    response = await apiPost(request, `${negative}/review/submit`, { comment: 'Review required', revision_id: info.revision_id, fingerprint: info.fingerprint });
    expect(response.status()).toBe(200);
    response = await apiPost(reviewer, `${negative}/review/decision`, { decision, comment: `Decision ${decision}`, revision_id: info.revision_id, fingerprint: info.fingerprint });
    expect(response.status()).toBe(200);
    expect((await apiPost(request, `${negative}/export`)).status()).toBe(409);
    response = await apiPost(request, `${negative}/review/revise`, { comment: 'Address reviewer feedback in a fresh draft' });
    expect(response.status()).toBe(200);
    info = await response.json();
    expect((await reviewer.get(negative)).status()).toBe(403);
  }
  await reviewer.dispose();
});
