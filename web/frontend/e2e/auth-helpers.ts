import { expect, type APIRequestContext, type Page } from '@playwright/test';

export function credentials(role: 'engineer' | 'reviewer' = 'engineer') {
  const prefix = `VIEWER_E2E_${role.toUpperCase()}`;
  const username = process.env[`${prefix}_USERNAME`];
  const password = process.env[`${prefix}_PASSWORD`];
  if (!username || !password) throw new Error('Run tools/check_web_review.py to provision isolated QA accounts; no default password is provided.');
  return { username, password };
}

export async function apiLogin(request: APIRequestContext, role: 'engineer' | 'reviewer' = 'engineer') {
  const session = await (await request.get('/api/auth/session')).json();
  const response = await request.post('/api/auth/login', { data: credentials(role), headers: { 'X-CSRF-Token': session.csrf_token } });
  expect(response.status()).toBe(200);
  return response.json();
}

export async function apiPost(request: APIRequestContext, url: string, data?: unknown) {
  const session = await (await request.get('/api/auth/session')).json();
  return request.post(url, { data, headers: { 'X-CSRF-Token': session.csrf_token } });
}

export async function pageLogin(page: Page, role: 'engineer' | 'reviewer' = 'engineer') {
  await page.goto('/');
  const user = credentials(role);
  await page.getByRole('textbox', { name: 'Tên tài khoản' }).fill(user.username);
  await page.getByLabel('Mật khẩu', { exact: true }).fill(user.password);
  await page.getByRole('button', { name: 'Đăng nhập', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Đăng xuất', exact: true })).toBeVisible();
}
