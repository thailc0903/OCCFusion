'use client';

import { FormEvent, useEffect, useRef, useState } from 'react';
import { getJSON, postJSON } from '../lib/api';
import type { User } from '../lib/types';
import type { InferenceJob, JobList, JobOptions, JobStatus } from '../lib/jobs';

const labels: Record<JobStatus, string> = { queued: 'Chờ', running: 'Đang chạy', succeeded: 'Thành công', failed: 'Thất bại', canceled: 'Đã hủy' };
const terminal = (job: InferenceJob) => ['succeeded', 'failed', 'canceled'].includes(job.status);
const date = (value: string | null) => value ? new Date(value).toLocaleString('vi-VN') : '—';

export default function JobsViewer({ user, onLogout }: { user: User; onLogout: () => Promise<void> }) {
  const [options, setOptions] = useState<JobOptions | null>(null);
  const [profile, setProfile] = useState('');
  const [scene, setScene] = useState('');
  const [frame, setFrame] = useState(1);
  const [list, setList] = useState<JobList>({ jobs: [], workers: [] });
  const [selected, setSelected] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [logs, setLogs] = useState<Record<string, string>>({});
  const request = useRef<{ parameters: string; key: string } | null>(null);
  const job = list.jobs.find(item => item.id === selected);
  const scenes = options?.scenes_by_profile[profile] || [];
  const currentScene = scenes.find(item => item.name === scene);
  const currentProfile = options?.profiles.find(item => item.id === profile);

  useEffect(() => {
    if (user.role !== 'engineer') return;
    const controller = new AbortController();
    getJSON<JobOptions>('/api/jobs/options', controller.signal).then(value => {
      setOptions(value);
      const p = value.profiles[0]?.id || '';
      setProfile(p);
      const s = value.scenes_by_profile[p]?.find(item => item.name === 'scene-0103') || value.scenes_by_profile[p]?.[0];
      setScene(s?.name || ''); setFrame(s && s.frames.length > 1 ? 1 : 0);
    }).catch(e => { if (e.name !== 'AbortError') setError(e.message); });
    return () => controller.abort();
  }, [user.role]);

  useEffect(() => {
    if (user.role !== 'engineer') return;
    const controller = new AbortController(); let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const value = await getJSON<JobList>('/api/jobs', controller.signal);
        if (controller.signal.aborted) return;
        setList(value); setSelected(previous => previous || value.jobs[0]?.id || '');
      } catch (e) { if ((e as Error).name !== 'AbortError') setError((e as Error).message); }
      if (!controller.signal.aborted) timer = setTimeout(poll, 2000);
    }
    poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [user.role]);

  useEffect(() => {
    setLogs({});
    if (!selected) return;
    const controller = new AbortController(); let timer: ReturnType<typeof setTimeout>;
    const offsets: Record<string, number> = { inference: 0, evaluation: 0 };
    async function poll() {
      try {
        for (const kind of ['inference', 'evaluation']) {
          const value = await getJSON<{ text: string; next_offset: number }>(`/api/jobs/${selected}/log?kind=${kind}&offset=${offsets[kind]}`, controller.signal);
          offsets[kind] = value.next_offset;
          if (!controller.signal.aborted && value.text) setLogs(previous => ({ ...previous, [kind]: ((previous[kind] || '') + value.text).slice(-100000) }));
        }
      } catch (e) { if ((e as Error).name !== 'AbortError') setError((e as Error).message); }
      if (!controller.signal.aborted) timer = setTimeout(poll, 2000);
    }
    poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [selected]);

  async function create(event: FormEvent) {
    event.preventDefault(); if (busy) return;
    setBusy(true); setError('');
    const parameters = JSON.stringify({ profile_id: profile, scene_name: scene, frame_index: frame });
    if (request.current?.parameters !== parameters) request.current = { parameters, key: crypto.randomUUID().replaceAll('-', '') };
    try {
      const value = await postJSON<InferenceJob>('/api/jobs', { ...JSON.parse(parameters), request_key: request.current!.key });
      setList(previous => ({ ...previous, jobs: [value, ...previous.jobs.filter(item => item.id !== value.id)] }));
      setSelected(value.id); request.current = null;
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }

  async function cancel() {
    if (!job) return;
    setBusy(true); setError('');
    try {
      const value = await postJSON<InferenceJob>(`/api/jobs/${job.id}/cancel`);
      setList(previous => ({ ...previous, jobs: previous.jobs.map(item => item.id === value.id ? value : item) }));
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }

  return <div className="app-shell">
    <header className="topbar"><a className="brand" href="/"><span className="brand-mark">R</span><span><strong>RAV25</strong><span className="brand-sub">OCCUPANCY LAB</span></span></a><div className="topbar-divider"/><div className="breadcrumb">Workspace <span>/</span> <b>Inference jobs</b></div><span className="local-badge"><i/> LOCAL WORKSPACE</span><div className="account-menu"><span><b>{user.username}</b><small>{user.role === 'engineer' ? 'Engineer' : 'Reviewer'}</small></span><button onClick={onLogout}>Đăng xuất</button></div></header>
    <main className="jobs-content">
      <nav className="jobs-nav"><a href="/">← Run explorer</a><a href="/benchmarks">Benchmark mini_val ↗</a></nav>
      <div className="page-heading"><div><div className="eyebrow">NUSCENES MINI / SINGLE FRAME</div><h1>Inference jobs<span className="title-period">.</span></h1><p>Chạy OCCFusion một frame, kiểm tra kết quả và đánh giá CPU khi có GT.</p></div></div>
      {user.role !== 'engineer' ? <section className="card no-prediction"><h2>Dành cho Engineer</h2><p>Reviewer xem và quyết định các run đã được gửi review trong Run explorer.</p></section> : <>
        {error && <div className="error-panel" role="alert"><strong>Không thể thực hiện thao tác</strong><p>{error}</p></div>}
        <div className="jobs-grid">
          <aside>
            <section className="card job-create"><div className="card-title"><h2>Tạo job</h2><span className="micro-tag">1 FRAME</span></div>
              <form onSubmit={create}>
                <label>Profile<select aria-label="Profile" value={profile} onChange={event => { const p = event.target.value; setProfile(p); setScene(options?.scenes_by_profile[p]?.[0]?.name || ''); setFrame(0); }} required>{options?.profiles.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
                <label>Scene<select aria-label="Scene" value={scene} onChange={event => { setScene(event.target.value); setFrame(0); }} required>{scenes.map(item => <option key={item.name} value={item.name}>{item.name} · {item.split}</option>)}</select></label>
                <label>Frame<select aria-label="Frame" value={frame} onChange={event => setFrame(Number(event.target.value))}>{currentScene?.frames.map(item => <option key={item.frame_index} value={item.frame_index}>{item.frame_index}</option>)}</select></label>
                <p className="job-sample"><small>Sample token · nuScenes metadata</small><code data-testid="job-form-token">{currentScene?.frames[frame]?.sample_token || '—'}</code></p>
                {currentProfile && <p className="footnote">{currentProfile.config}<br/>{currentProfile.checkpoint}<br/>FP32 · batch 1 · TF32 matmul {String(currentProfile.tf32.matmul)} / cuDNN {String(currentProfile.tf32.cudnn)}</p>}
                {currentProfile?.unavailable_reason && <p role="alert">{currentProfile.unavailable_reason}</p>}
                <button className="primary-button" type="submit" disabled={busy || !currentProfile?.available || !currentScene}> {busy ? 'Đang gửi…' : 'Tạo job inference'}</button>
              </form>
              <p className="footnote">Một job GPU tại một thời điểm. Hoàn thành job chưa phải approval; xuất kết quả vẫn cần review.</p>
              <p className="job-worker" data-testid="worker-status">{list.workers.length ? `Worker đang kết nối (${list.workers.map(item => item.execution_mode).join(', ')})` : 'Chưa có worker kết nối · job mới sẽ chờ trong hàng đợi.'}</p>
            </section>
            <section className="card job-history"><div className="card-title"><h2>Job của bạn</h2><span className="micro-tag">{list.jobs.length}</span></div>{!list.jobs.length && <p>Chưa có job.</p>}
              {list.jobs.map(item => <button className={`job-row ${selected === item.id ? 'selected' : ''}`} key={item.id} onClick={() => setSelected(item.id)} data-testid={`job-row-${item.id}`}><span><b>{item.scene_name} / {item.frame_index}</b><small>{date(item.created_at)}</small></span><span className={`job-status ${item.status}`}>{labels[item.status]}</span></button>)}
            </section>
          </aside>
          <section className="card job-detail">
            {!job ? <div className="no-prediction"><h2>Chọn hoặc tạo một job</h2><p>Log và trạng thái được lưu bền vững. Mở trang và đổi lựa chọn chỉ đọc tiến độ.</p></div> : <>
              <div className="card-title"><h2>{job.scene_name} / frame {job.frame_index}</h2><span className={`job-status ${job.status}`} data-testid="job-status">{labels[job.status]}</span></div>
              <p className="job-id"><small>JOB</small><code data-testid="job-id">{job.id}</code></p>
              <p className="job-sample"><small>SAMPLE TOKEN · {job.split}</small><code data-testid="job-token">{job.sample_token}</code></p>
              {job.execution_mode === 'fixture' && <p className="job-fixture" role="status">KIỂM THỬ WORKER · phát lại artifact fixture thật, không phải inference mới.</p>}
              <dl className="job-facts"><div><dt>Giai đoạn</dt><dd data-testid="job-phase">{job.phase}</dd></div><div><dt>Tạo</dt><dd>{date(job.created_at)}</dd></div><div><dt>Bắt đầu</dt><dd>{date(job.started_at)}</dd></div><div><dt>Kết thúc</dt><dd>{date(job.finished_at)}</dd></div><div><dt>Số lần gọi inference</dt><dd data-testid="launch-count">{job.inference_launch_count}</dd></div><div><dt>Evaluation</dt><dd data-testid="job-evaluation-status">{job.evaluation_status === 'matched' ? 'Có GT · đã đối chiếu token/hash' : job.evaluation_status === 'missing_gt' ? 'Chưa có GT · metric chưa khả dụng' : 'Chưa có evaluation hợp lệ'}</dd></div></dl>
              {job.worker_stale && !terminal(job) && <p role="status">Chưa nhận heartbeat worker. Job không được tự chạy lại; worker sẽ xác minh khi khởi động lại.</p>}
              {job.error && <div className="error-panel" role="alert"><strong>{job.error.stage} · {job.error.type}</strong><p>{job.error.message}</p><p>Output dở được giữ riêng, không thay baseline và không tự tạo prediction.</p></div>}
              <div className="job-actions">{!terminal(job) && <button className="reset-button" onClick={cancel} disabled={busy || job.cancel_requested}>{job.cancel_requested ? 'Đang yêu cầu hủy…' : 'Hủy job'}</button>}{job.result_url && <a className="primary-button" href={job.result_url}>Mở kết quả trong viewer →</a>}</div>
              {job.result_url && <p className="footnote">Viewer phát lại prediction đã lưu, có BEV/occupancy 3D/LiDAR và evaluation. Confidence chưa hiệu chuẩn. Job chưa đăng ký review, chưa được approved.</p>}
              <h3>Tiến độ</h3><ol className="job-events">{job.events.map(event => <li key={event.id}><time>{date(event.created_at)}</time><strong>{event.phase}</strong><p>{event.message}</p></li>)}</ol>
              <details className="job-log" open><summary>Log inference CLI</summary><pre data-testid="inference-log">{logs.inference || 'Chưa có log.'}</pre></details>
              <details className="job-log"><summary>Log evaluation CPU</summary><pre>{logs.evaluation || 'Chưa có log.'}</pre></details>
            </>}
          </section>
        </div>
      </>}
      <footer className="page-footer"><span>RAV25 / OFFLINE OCCUPANCY ANALYSIS</span><span>Worker một frame · playback chưa phải learned 4D flow</span></footer>
    </main>
  </div>;
}
