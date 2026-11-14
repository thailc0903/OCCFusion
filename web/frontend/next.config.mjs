/** @type {import('next').NextConfig} */
const config = {
  distDir: process.env.VIEWER_NEXT_DIST_DIR || '.next',
  poweredByHeader: false,
  // Verification and ZIP creation of 79 frames can exceed the 30s WSL/NTFS default.
  experimental: { cpus: 2, proxyTimeout: 300_000 },
  async rewrites() {
    const origin = process.env.VIEWER_API_ORIGIN || 'http://127.0.0.1:8000';
    return [{ source: '/api/:path*', destination: `${origin}/api/:path*` }];
  },
};
export default config;
