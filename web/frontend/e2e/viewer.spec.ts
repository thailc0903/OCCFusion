import { expect, test } from '@playwright/test';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { apiLogin, pageLogin } from './auth-helpers';

const root = path.resolve(__dirname, '../../..');
const original = path.join(root, 'work_dirs/infer_mini_cuda_v2');
const digest = () => Object.fromEntries(['report.json', 'metadata.json', 'occupancy.npz', 'bev.png'].map(name => [name, createHash('sha256').update(fs.readFileSync(path.join(original, name))).digest('hex')]));

test('real replay, class/height filters, LiDAR and camera reset never invoke inference', async ({ page, request }) => {
  await apiLogin(request);
  await pageLogin(page);
  const before = digest();
  const errors: string[] = [];
  const apiCalls: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiCalls.push(`${request.method()} ${new URL(request.url()).pathname}`); });
  const response = await request.get('/api/runs/infer_mini_cuda_v2');
  expect(response.ok()).toBeTruthy();
  const detail = await response.json();
  expect(detail.sample_token).toBe('39586f9d59004284a7114a68825e8eec');
  expect(detail.lidar.sample_token).toBe(detail.sample_token);
  expect(detail.occupancy.point_count).toBe(44856);
  expect(detail.validation.sensor_sample_token).toBe(true);
  await page.goto('/');
  await expect(page.getByRole('combobox', { name: 'Chọn run' })).toHaveValue('infer_mini_cuda_v2');
  await expect(page.getByTestId('iou-value')).toHaveText('Chưa có GT');
  await expect(page.locator('.replay-badge')).toContainText('Phát lại prediction đã lưu');
  const bev = page.getByAltText(/BEV prediction đã lưu/);
  await expect(bev).toBeVisible();
  await expect.poll(() => bev.evaluate(image => (image as HTMLImageElement).naturalWidth)).toBe(200);
  const canvas = page.getByTestId('occupancy-canvas');
  await expect(canvas).toHaveAttribute('data-webgl-ready', 'true');
  await expect(canvas).toHaveAttribute('data-rendered-instances', String(detail.occupancy.point_count));
  await expect(canvas).toHaveAttribute('data-rendered-points', String(detail.lidar.point_count));
  await page.waitForLoadState('networkidle');
  await page.screenshot({ path: path.join(root, 'work_dirs/web_viewer_desktop.png'), fullPage: true });
  const callsBeforeFilters = apiCalls.length;
  await page.getByRole('button', { name: 'Bỏ chọn', exact: true }).click();
  await expect(canvas).toHaveAttribute('data-rendered-instances', '0');
  await page.getByRole('checkbox', { name: 'Lớp car', exact: true }).check();
  await expect(canvas).toHaveAttribute('data-rendered-instances', '611');
  await page.getByRole('slider', { name: 'Chiều cao tối đa' }).focus();
  await page.keyboard.press('Home');
  await expect(canvas).toHaveAttribute('data-rendered-instances', '0');
  await page.keyboard.press('End');
  await expect(canvas).toHaveAttribute('data-rendered-instances', '611');
  await page.getByRole('switch', { name: 'Hiển thị LiDAR' }).uncheck();
  await expect(canvas).toHaveAttribute('data-rendered-points', '0');
  await page.getByRole('switch', { name: 'Hiển thị LiDAR' }).check();
  await expect(canvas).toHaveAttribute('data-rendered-points', String(detail.lidar.point_count));
  const initialCamera = await canvas.getAttribute('data-camera-position');
  const box = (await canvas.boundingBox())!;
  await page.mouse.move(box.x + box.width * .4, box.y + box.height * .5);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * .6, box.y + box.height * .6, { steps: 12 });
  await page.mouse.up();
  await expect(canvas).not.toHaveAttribute('data-camera-position', initialCamera!);
  await page.getByRole('button', { name: 'Reset camera' }).click();
  await expect(canvas).toHaveAttribute('data-camera-position', initialCamera!);
  expect(apiCalls.slice(callsBeforeFilters)).toEqual([]);
  expect((await (await request.get('/api/health')).json()).inference_enabled).toBe(false);
  expect(digest()).toEqual(before);
  expect(errors).toEqual([]);
  fs.writeFileSync(path.join(root, 'work_dirs/web_viewer_browser_evidence.json'), JSON.stringify({
    status: 'passed', run_id: detail.run_id, sample_token: detail.sample_token,
    occupied_instances: detail.occupancy.point_count, lidar_points: detail.lidar.point_count,
    filters_triggered_api_requests: apiCalls.slice(callsBeforeFilters), inference_enabled: false,
    artifacts_unchanged_sha256: before, browser_page_errors: errors,
    rendering: 'Chromium headless with SwiftShader software WebGL',
  }, null, 2));
});

test('check-only and failed runs do not request occupancy or LiDAR', async ({ page }) => {
  await pageLogin(page);
  const calls: string[] = [];
  page.on('request', request => calls.push(new URL(request.url()).pathname));
  await page.goto('/');
  await expect(page.getByTestId('occupancy-canvas')).toHaveAttribute('data-webgl-ready', 'true');
  await page.getByRole('combobox', { name: 'Chọn run' }).selectOption('infer_mini_check');
  await expect(page.getByTestId('no-prediction')).toContainText('check-only');
  await expect(page.getByTestId('occupancy-canvas')).toHaveCount(0);
  await expect(page.getByTestId('forward-time')).toHaveText('—');
  expect(calls.filter(url => /infer_mini_check\/(occupancy|lidar|bev)/.test(url))).toEqual([]);
  await page.getByRole('combobox', { name: 'Chọn run' }).selectOption('infer_mini_cuda');
  await expect(page.getByTestId('no-prediction')).toContainText('chưa có prediction');
  expect(calls.filter(url => /infer_mini_cuda\/(occupancy|lidar|bev)/.test(url))).toEqual([]);
});

test('mobile layout remains usable with the real saved frame', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await pageLogin(page);
  await page.goto('/');
  await expect(page.getByTestId('occupancy-canvas')).toHaveAttribute('data-rendered-instances', '44856');
  await expect(page.getByTestId('iou-value')).toHaveText('Chưa có GT');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: path.join(root, 'work_dirs/web_viewer_mobile.png'), fullPage: true });
});
