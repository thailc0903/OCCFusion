import { defineConfig } from '@playwright/test';
import { existsSync } from 'node:fs';
import path from 'node:path';

const root = path.resolve(__dirname, '../..');
const browserCache = path.join(root, '.cache/ms-playwright');
const localLibraries = path.join(root, '.cache/browser-libs/usr/lib/x86_64-linux-gnu');
if (!process.env.PLAYWRIGHT_BROWSERS_PATH && existsSync(browserCache)) process.env.PLAYWRIGHT_BROWSERS_PATH = browserCache;
if (existsSync(localLibraries)) process.env.LD_LIBRARY_PATH = [localLibraries, process.env.LD_LIBRARY_PATH].filter(Boolean).join(':');

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  timeout: 90_000,
  expect: { timeout: 30_000 },
  reporter: [['list'], ['json', { outputFile: 'test-results/browser-results.json' }]],
  use: {
    baseURL: process.env.VIEWER_URL || 'http://127.0.0.1:3000',
    viewport: { width: 1520, height: 1080 },
    locale: 'vi-VN',
    screenshot: 'only-on-failure',
    // Authentication traces record credentials and cookies; keep only screenshots
    // and the result report from QA accounts provisioned in the isolated database.
    trace: 'off',
    launchOptions: { args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] },
  },
});
