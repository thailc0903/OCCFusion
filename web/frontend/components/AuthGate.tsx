'use client';

import { FormEvent, useCallback, useEffect, useState } from 'react';
import { clearPayloadCache, getJSON, postJSON, rememberCsrf } from '../lib/api';
import type { SessionInfo } from '../lib/types';
import Viewer from './Viewer';
import BenchmarkViewer from './BenchmarkViewer';
import JobsViewer from './JobsViewer';
import DetectionExplorer from './DetectionExplorer';

export default function AuthGate({ mode = 'runs' }: { mode?: 'runs' | 'benchmarks' | 'jobs' | 'detections' }) {
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const load = useCallback(async () => {
    try {
      const value = await getJSON<SessionInfo>('/api/auth/session');
      rememberCsrf(value.csrf_token);
      if (!value.user) clearPayloadCache();
      setSession(value);
    } catch (error) { setError((error as Error).message); }
  }, []);
  useEffect(() => {
    load();
    window.addEventListener('rav25:session-expired', load);
    window.addEventListener('focus', load);
    return () => { window.removeEventListener('rav25:session-expired', load); window.removeEventListener('focus', load); };
  }, [load]);
  useEffect(() => {
    if (!session?.user || !session.expires_at) return;
    const timer = setTimeout(load, Math.max(0, session.expires_at * 1000 - Date.now() + 100));
    return () => clearTimeout(timer);
  }, [session, load]);

  async function login(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError('');
    try {
      const value = await postJSON<SessionInfo>('/api/auth/login', { username, password });
      rememberCsrf(value.csrf_token); clearPayloadCache(); setPassword(''); setSession(value);
    } catch (error) { setError((error as Error).message); }
    finally { setBusy(false); }
  }
  async function logout() {
    setError('');
    try {
      await postJSON('/api/auth/logout');
      clearPayloadCache(); rememberCsrf('');
      setSession(null);
      await load();
    } catch (error) { setError((error as Error).message); }
  }
  if (session?.user) return mode === 'detections' ? <DetectionExplorer key={session.user.id} user={session.user} onLogout={logout}/> : mode === 'jobs' ? <JobsViewer key={session.user.id} user={session.user} onLogout={logout} /> : mode === 'benchmarks' ? <BenchmarkViewer key={session.user.id} user={session.user} onLogout={logout} /> : <Viewer key={session.user.id} user={session.user} onLogout={logout} />;
  return <main className="auth-page">
    <section className="auth-intro"><span className="eyebrow">RAV25 / OCCUPANCY LAB</span><h1>Không gian phân tích.<br/><span>Kết quả có kiểm chứng.</span></h1><p>Đăng nhập để xem prediction đã lưu, trao đổi nhận xét và duyệt đúng bộ artifact cho bước tiếp theo.</p><div className="auth-steps"><span>01 · Kiểm tra dữ liệu</span><span>02 · Gửi review</span><span>03 · Duyệt & xuất kết quả</span></div></section>
    <section className="card login-card"><span className="micro-tag">LOCAL WORKSPACE</span><h2>Đăng nhập</h2><p>Tài khoản Engineer hoặc Reviewer của workspace.</p>
      {!session ? <div className="auth-loading"><span className="spinner"/> Đang kết nối…</div> : <form onSubmit={login}>
        <label>Tên tài khoản<input name="username" autoComplete="username" aria-label="Tên tài khoản" value={username} onChange={event => setUsername(event.target.value)} required maxLength={64}/></label>
        <label>Mật khẩu<input name="password" type="password" autoComplete="current-password" aria-label="Mật khẩu" value={password} onChange={event => setPassword(event.target.value)} required maxLength={1024}/></label>
        <button className="primary-button" type="submit" disabled={busy}>{busy ? 'Đang đăng nhập…' : 'Đăng nhập'}</button>
      </form>}
      {error && <p className="auth-error" role="alert">{error}</p>}
      {session?.needs_setup && <div className="setup-note"><strong>Workspace chưa có tài khoản</strong><p>Tạo tài khoản đầu tiên trong terminal WSL. Công cụ sẽ hỏi mật khẩu và không ghi mật khẩu vào mã nguồn.</p><code>.venv-web/bin/python tools/create_viewer_user.py --username ten-ban --role engineer</code></div>}
      <p className="login-footnote">Phiên có thời hạn · Đăng xuất để kết thúc phiên</p>
    </section>
  </main>;
}
