import { test, expect } from '@playwright/test';
import { spawn, type ChildProcess } from 'node:child_process';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { apiPost, pageLogin } from './auth-helpers';

const root = path.resolve(__dirname, '../../..');
const output = process.env.VIEWER_E2E_OUTPUT!;
const percentage = new Intl.NumberFormat('vi-VN', { style: 'percent', maximumFractionDigits: 4 });
const startWorker = (mode: string) => {
  if (process.env.VIEWER_JOB_TEST_MODE !== 'fixture') throw new Error('Fake worker tests require isolated fixture QA');
  return spawn(path.join(root, '.venv-web/bin/python'), ['-m', 'web.backend.tests.job_support', 'worker', '--target', process.env.VIEWER_JOB_TEST_SETTINGS!, '--mode', mode], { cwd: root, stdio: 'ignore' });
};
const finish = async (child: ChildProcess) => {
  if (child.exitCode !== null || child.signalCode !== null) return;
  child.kill('SIGTERM');
  await new Promise<void>(resolve => { child.once('exit', () => resolve()); setTimeout(() => { child.kill('SIGKILL'); resolve(); }, 3000); });
};

test('jobs require login and Engineer role', async ({ page, request }) => {
  expect((await request.get('/api/jobs')).status()).toBe(401);
  expect((await request.post('/api/jobs', { data: {} })).status()).toBe(401);
  await page.goto('/jobs'); await expect(page.getByRole('heading', { name: 'Đăng nhập', exact: true })).toBeVisible();
  await pageLogin(page, 'reviewer'); await page.goto('/jobs');
  await expect(page.getByRole('heading', { name: 'Dành cho Engineer' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Tạo job inference', exact: true })).toHaveCount(0);
  expect((await apiPost(page.request, '/api/jobs', { profile_id: 'occfusion_clr_fp32', scene_name: 'scene-0103', frame_index: 1, request_key: 'a'.repeat(32) })).status()).toBe(403);
});

test('Engineer creates, reloads and cancels queued job without launching CLI', async ({ page }) => {
  await pageLogin(page); await page.goto('/jobs');
  await expect(page.getByTestId('job-form-token')).toHaveText('3950bd41f74548429c0f7700ff3d8269');
  await page.getByRole('button', { name: 'Tạo job inference', exact: true }).click();
  await expect(page.getByTestId('job-status')).toHaveText('Chờ');
  const id = await page.getByTestId('job-id').innerText();
  await page.reload(); await expect(page.getByTestId('job-id')).toHaveText(id);
  await page.getByRole('button', { name: 'Hủy job', exact: true }).click();
  await expect(page.getByTestId('job-status')).toHaveText('Đã hủy');
  await expect(page.getByTestId('launch-count')).toHaveText('0');
  await page.screenshot({ path: path.join(output, 'queued-canceled.png'), fullPage: true });
});

test('fake CLI failure exposes stage/log and creates no viewer prediction', async ({ page }) => {
  await pageLogin(page); await page.goto('/jobs');
  await page.getByRole('button', { name: 'Tạo job inference', exact: true }).click();
  await expect(page.getByTestId('job-status')).toHaveText('Chờ');
  const id = await page.getByTestId('job-id').innerText();
  const worker = startWorker('fail');
  try {
    await expect(page.getByTestId('job-status')).toHaveText('Thất bại');
    await expect(page.locator('.job-detail .error-panel[role="alert"]')).toContainText('forward');
    await expect(page.getByTestId('inference-log')).toContainText('TEST FIXTURE');
    await expect(page.getByRole('link', { name: 'Mở kết quả trong viewer →' })).toHaveCount(0);
    const runs = await (await page.request.get('/api/runs')).json();
    expect(runs.runs.some((r: { run_id: string }) => r.run_id === `infer_job_${id}`)).toBe(false);
    await page.screenshot({ path: path.join(output, 'fake-failure.png'), fullPage: true });
  } finally { await finish(worker); }
});

test('two workers respect queue; cancellation releases lane; valid fixture opens BEV 3D LiDAR and evaluation', async ({ page }) => {
  test.setTimeout(150_000);
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await pageLogin(page); await page.goto('/jobs');
  await page.getByRole('button', { name: 'Tạo job inference', exact: true }).click();
  await expect(page.getByTestId('job-status')).toHaveText('Chờ');
  const first = await page.getByTestId('job-id').innerText();
  const slow = startWorker('slow'); let secondWorker: ChildProcess | null = null;
  try {
    await expect(page.getByTestId('job-status')).toHaveText('Đang chạy');
    await expect(page.getByTestId('inference-log')).toContainText('Waiting in fake CLI');
    await page.getByRole('button', { name: 'Tạo job inference', exact: true }).click();
    await expect(page.getByTestId('job-status')).toHaveText('Chờ');
    const second = await page.getByTestId('job-id').innerText(); expect(second).not.toBe(first);
    secondWorker = startWorker('success');
    await page.waitForTimeout(1000);
    let info = await (await page.request.get(`/api/jobs/${second}`)).json();
    expect(info.status).toBe('queued'); expect(info.inference_launch_count).toBe(0);
    await page.getByTestId(`job-row-${first}`).click();
    await page.getByRole('button', { name: 'Hủy job', exact: true }).click();
    await expect(page.getByTestId('job-status')).toHaveText('Đã hủy');
    await page.getByTestId(`job-row-${second}`).click();
    await expect(page.getByTestId('job-status')).toHaveText('Thành công');
    await expect(page.getByTestId('job-evaluation-status')).toHaveText('Có GT · đã đối chiếu token/hash');
    await expect(page.getByTestId('launch-count')).toHaveText('1');
    await expect(page.getByText('KIỂM THỬ WORKER', { exact: false })).toBeVisible();
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: path.join(output, 'fake-job-completed.png'), fullPage: true });
    await page.getByRole('link', { name: 'Mở kết quả trong viewer →' }).click();
    await expect(page.getByLabel('Chọn run')).toHaveValue(`infer_job_${second}`);
    await expect(page.locator('canvas')).toBeVisible();
    await expect(page.getByRole('img', { name: 'BEV prediction đã lưu của scene-0103, frame 1' })).toBeVisible();
    await expect(page.locator('.viewport-count')).toContainText('điểm keyframe');
    await expect(page.getByTestId('evaluation-panel')).toHaveAttribute('data-status', 'matched');
    const evaluation = JSON.parse(readFileSync(path.join(root, 'work_dirs/eval_mini_val_scene0103_f1/evaluation.json'), 'utf8'));
    await expect(page.getByTestId('iou-value')).toHaveText(percentage.format(evaluation.metrics.occupancy_iou));
    await expect(page.getByTestId('review-state')).toContainText('unregistered');
    expect((await apiPost(page.request, `/api/runs/infer_job_${second}/export`)).status()).toBe(409);
    await page.getByRole('switch', { name: 'Hiển thị LiDAR' }).uncheck();
    await page.getByRole('button', { name: 'Reset camera', exact: true }).click();
    info = await (await page.request.get(`/api/jobs/${second}`)).json(); expect(info.inference_launch_count).toBe(1);
    expect(errors).toEqual([]);
    await page.screenshot({ path: path.join(output, 'fake-viewer.png'), fullPage: true });
  } finally { await finish(slow); if (secondWorker) await finish(secondWorker); }
});
