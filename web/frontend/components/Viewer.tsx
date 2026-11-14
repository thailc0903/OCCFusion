'use client';

import dynamic from 'next/dynamic';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { clearPayloadCache, getJSON, getPacked } from '../lib/api';
import type { PredictionRun, ReviewInfo, RunDetail, RunSummary, User } from '../lib/types';
import ReviewPanel from './ReviewPanel';
import EvaluationPanel from './EvaluationPanel';

const Scene = dynamic(() => import('./OccupancyScene'), { ssr: false, loading: () => <div className="scene-loading"><span className="spinner" />Đang khởi tạo không gian 3D…</div> });
const number = new Intl.NumberFormat('vi-VN');
const seconds = (value: number | null | undefined) => value == null ? '—' : `${value.toFixed(3)} s`;
const gib = (value: number | null | undefined) => value == null ? '—' : `${(value / 1024 ** 3).toFixed(3)} GiB`;

function Glyph({ name, size = 18 }: { name: 'cube' | 'reset' | 'layers' | 'refresh' | 'cloud' | 'check' | 'arrow'; size?: number }) {
  const paths = {
    cube: <><path d="m12 2 9 5v10l-9 5-9-5V7l9-5Z"/><path d="m3 7 9 5 9-5M12 12v10M7.5 4.5l9 5"/></>,
    reset: <><path d="M3 10a9 9 0 1 1 2.4 8.2M3 3v7h7"/></>,
    layers: <><path d="m12 3 10 5-10 5L2 8l10-5ZM2 12l10 5 10-5M2 16l10 5 10-5"/></>,
    refresh: <><path d="M20 7a8 8 0 0 0-14-2L3 8m0-5v5h5M4 17a8 8 0 0 0 14 2l3-3m0 5v-5h-5"/></>,
    cloud: <><path d="M7 18H6a4 4 0 0 1-1-7.9A7 7 0 0 1 18 8a5 5 0 0 1 0 10h-1"/><path d="M9 16v5m3-6v6m3-5v5"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    arrow: <><path d="M5 12h14m-6-6 6 6-6 6"/></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

function Timing({ run }: { run: RunDetail }) {
  return <section className="card timing-card">
    <div className="card-title"><h2>Thời gian đã ghi nhận</h2><span className="micro-tag">1 FRAME</span></div>
    <div className="timing-grid">
      <div className="time-stat"><span>Forward mô hình</span><strong data-testid="forward-time">{seconds(run.timings_s.forward)}</strong><small>Tensor forward</small></div>
      <div className="time-stat"><span>Pipeline frame</span><strong>{seconds(run.timings_s.pipeline_total)}</strong><small>Đọc cảm biến → xuất kết quả</small></div>
      <div className="time-stat"><span>Khởi tạo · suy ra</span><strong>{seconds(run.startup_derived_s)}</strong><small>Tiến trình trừ pipeline frame</small></div>
      <div className="time-stat"><span>Tiến trình</span><strong>{seconds(run.timings_s.process_until_report)}</strong><small>Đến trước ghi report cuối</small></div>
    </div>
    <div className="timing-breakdown"><span>Tiền xử lý <b>{seconds(run.timings_s.preprocessing)}</b></span><span>Xuất kết quả <b>{seconds(run.timings_s.postprocessing_export)}</b></span></div>
    <p className="footnote">Một frame, không warm-up. Chưa phải benchmark ổn định.</p>
  </section>;
}

export default function Viewer({ user, onLogout }: { user: User; onLogout: () => Promise<void> }) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selected, setSelected] = useState('');
  const [run, setRun] = useState<RunDetail | null>(null);
  const [occupancy, setOccupancy] = useState<Float32Array | null>(null);
  const [lidar, setLidar] = useState<Float32Array | null>(null);
  const [showLidar, setShowLidar] = useState(true);
  const [classIds, setClassIds] = useState<Set<number>>(new Set());
  const [height, setHeight] = useState<[number, number]>([-5, 3]);
  const [resetVersion, setResetVersion] = useState(0);
  const [revision, setRevision] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [lidarError, setLidarError] = useState('');
  const [sceneError, setSceneError] = useState('');
  const reportSceneError = useCallback((message: string) => setSceneError(message), []);

  useEffect(() => {
    const controller = new AbortController();
    getJSON<{ runs: RunSummary[] }>('/api/runs', controller.signal).then(value => {
      setRuns(value.runs);
      const requested = new URLSearchParams(window.location.search).get('run');
      setSelected(previous => value.runs.some(item => item.run_id === previous) ? previous : requested && value.runs.some(item => item.run_id === requested) ? requested :
        (value.runs.find(item => item.run_id === 'infer_mini_cuda_v2') || value.runs.find(item => item.prediction_available) || value.runs[0])?.run_id || '');
      if (!value.runs.length) { setLoading(false); setError(user.role === 'reviewer' ? 'Chưa có run được gửi review.' : 'Chưa tìm thấy run trong thư mục được cấu hình.'); }
    }).catch(error => { if (error.name !== 'AbortError') { setError(error.message); setLoading(false); } });
    return () => controller.abort();
  }, [revision, user.role]);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    setLoading(true); setError(''); setSceneError(''); setRun(null); setOccupancy(null); setLidar(null); setLidarError('');
    (async () => {
      const detail = await getJSON<RunDetail>(`/api/runs/${encodeURIComponent(selected)}`, controller.signal);
      let data: Float32Array | null = null;
      if (detail.prediction_available) {
        data = await getPacked(detail.occupancy.url, detail.occupancy.point_count, detail.occupancy.stride, controller.signal);
      }
      if (controller.signal.aborted) return;
      setRun(detail); setOccupancy(data);
      if (detail.prediction_available) {
        setClassIds(new Set(detail.classes.filter(item => !item.background).map(item => item.id)));
        setHeight([detail.bounds[2], detail.bounds[5]]);
      }
      setLoading(false);
    })().catch(error => { if (error.name !== 'AbortError') { setError(error.message); setLoading(false); } });
    return () => controller.abort();
  }, [selected, revision]);

  useEffect(() => {
    if (!run?.prediction_available || !showLidar) return;
    const controller = new AbortController();
    setLidarError('');
    getPacked(run.lidar.url, run.lidar.point_count, run.lidar.stride, controller.signal)
      .then(value => { if (!controller.signal.aborted) setLidar(value); })
      .catch(error => { if (error.name !== 'AbortError') setLidarError(error.message); });
    return () => controller.abort();
  }, [run, showLidar]);

  const visible = useMemo(() => {
    if (!occupancy) return new Float32Array();
    const values = new Float32Array(occupancy.length);
    let count = 0;
    for (let i = 0; i < occupancy.length; i += 5) {
      if (classIds.has(occupancy[i + 3]) && occupancy[i + 2] >= height[0] && occupancy[i + 2] <= height[1]) {
        values.set(occupancy.subarray(i, i + 5), count);
        count += 5;
      }
    }
    return values.subarray(0, count);
  }, [occupancy, classIds, height]);

  const prediction: PredictionRun | null = run?.prediction_available ? run : null;
  const visibleLidarCount = useMemo(() => {
    if (!lidar || !showLidar) return 0;
    let count = 0;
    for (let i = 0; i < lidar.length; i += 4) if (lidar[i + 2] >= height[0] && lidar[i + 2] <= height[1]) count++;
    return count;
  }, [lidar, showLidar, height]);
  const toggleClass = (id: number) => setClassIds(previous => {
    const next = new Set(previous);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });
  const refresh = () => { clearPayloadCache(); setRevision(value => value + 1); };
  const reviewChanged = (value: ReviewInfo) => setRun(previous => previous ? { ...previous, review: value } : previous);

  return <div className="app-shell">
    <header className="topbar">
      <a className="brand" href="/" aria-label="RAV25 Occupancy Lab"><span className="brand-mark"><Glyph name="cube" size={25}/></span><span><strong>RAV25</strong><span className="brand-sub">OCCUPANCY LAB</span></span></a>
      <div className="topbar-divider"/>
      <div className="breadcrumb">Workspace <span>/</span> <b>Run explorer</b></div>
      <span className="local-badge"><i/> LOCAL WORKSPACE</span>
      <div className="account-menu"><span><b>{user.username}</b><small>{user.role === 'engineer' ? 'Engineer' : 'Reviewer'}</small></span><button onClick={onLogout}>Đăng xuất</button></div>
    </header>

    <div className="workspace">
      <aside className="sidebar">
        <div className="workspace-label">PHÂN TÍCH DỮ LIỆU</div>
        <div className="nav-current"><Glyph name="layers"/> Occupancy explorer <span>01</span></div>
        <a className="benchmark-link" href="/benchmarks">Benchmark mini_val ↗</a>
        {user.role === 'engineer' && <a className="benchmark-link" href="/jobs">Chạy inference một frame ↗</a>}
        <section className="run-selector">
          <div className="section-caption"><label htmlFor="run-select">RUN ĐÃ LƯU</label><button className="icon-button" onClick={refresh} aria-label="Làm mới danh sách run" title="Đọc lại danh sách run"><Glyph name="refresh" size={15}/></button></div>
          <select id="run-select" value={selected} onChange={event => setSelected(event.target.value)} aria-label="Chọn run">
            {!runs.length && <option value="">Đang tải danh sách…</option>}
            {runs.map(item => <option key={item.run_id} value={item.run_id}>{item.run_id}{item.kind === 'check_only' ? ' · check-only' : item.kind === 'failed' ? ' · lỗi' : ''}</option>)}
          </select>
          <p className="selector-note"><i className={`status-dot ${run?.status === 'success' ? 'success' : 'muted'}`}/>{loading ? 'Đang đọc artifact…' : run?.kind === 'prediction' ? 'Prediction có sẵn' : run?.kind === 'check_only' ? 'Check-only · chưa forward' : 'Chưa có prediction hợp lệ'}</p>
        </section>

        {prediction && <>
          <section className="filter-section">
            <div className="section-caption"><span>LỚP HIỂN THỊ</span><span className="subtle-count">{classIds.size}/{prediction.classes.length - 1}</span></div>
            <div className="filter-actions"><button onClick={() => setClassIds(new Set(prediction.classes.filter(item => !item.background).map(item => item.id)))}>Tất cả</button><span>·</span><button onClick={() => setClassIds(new Set())}>Bỏ chọn</button></div>
            <div className="class-list">
              {prediction.classes.filter(item => !item.background).map(item => <label className={`class-item ${item.count ? '' : 'class-empty'}`} key={item.id}>
                <input type="checkbox" checked={classIds.has(item.id)} onChange={() => toggleClass(item.id)} aria-label={`Lớp ${item.name}`}/>
                <span className="class-swatch" style={{ background: `rgb(${item.color.join(',')})` }}/>
                <span className="class-name">{item.name.replaceAll('_', ' ')}</span><span className="class-count">{number.format(item.count)}</span>
              </label>)}
            </div>
            <div className="background-note"><span className="free-square"/><span>{prediction.classes[prediction.free_label].name} · nền</span><b>{number.format(prediction.classes[prediction.free_label].count)}</b></div>
          </section>
          <section className="filter-section height-section">
            <div className="section-caption"><span>KHOẢNG CHIỀU CAO</span><span className="subtle-count">Z · m</span></div>
            <div className="height-values"><span>{height[0].toFixed(2)} <small>m</small></span><span>→</span><span>{height[1].toFixed(2)} <small>m</small></span></div>
            <label className="range-control">Từ <input type="range" aria-label="Chiều cao tối thiểu" min={prediction.bounds[2]} max={prediction.bounds[5]} step={prediction.voxel_size[2] / 2} value={height[0]} onChange={event => setHeight([Math.min(Number(event.target.value), height[1]), height[1]])}/></label>
            <label className="range-control">Đến <input type="range" aria-label="Chiều cao tối đa" min={prediction.bounds[2]} max={prediction.bounds[5]} step={prediction.voxel_size[2] / 2} value={height[1]} onChange={event => setHeight([height[0], Math.max(Number(event.target.value), height[0])])}/></label>
          </section>
          <section className="filter-section sensor-section">
            <label className="lidar-toggle"><span><Glyph name="cloud"/> Point cloud LiDAR</span><input type="checkbox" role="switch" aria-label="Hiển thị LiDAR" checked={showLidar} onChange={event => setShowLidar(event.target.checked)}/><i/></label>
            <p>Keyframe đúng sample · trong bounds</p>
            {lidarError && <p className="inline-error" role="alert">{lidarError}</p>}
          </section>
        </>}
        <div className="sidebar-footer"><span className="status-dot success"/><span>Phát lại dữ liệu đã lưu<br/><small>Không kích hoạt inference</small></span></div>
      </aside>

      <main className="main-content">
        <div className="page-heading"><div><div className="eyebrow">SAVED INFERENCE / NUSCENES MINI</div><h1>Occupancy explorer<span className="title-period">.</span></h1><p>Khám phá không gian từ camera, LiDAR và radar.</p></div><span className="replay-badge"><span className="replay-icon">↻</span> {run?.kind === 'check_only' ? 'Đọc báo cáo check-only' : 'Phát lại prediction đã lưu'}</span></div>
        {error && <div className="error-panel" role="alert"><strong>Không thể mở run</strong><p>{error}</p><button onClick={refresh}>Thử đọc lại</button></div>}
        {loading && !error && <div className="loading-panel"><span className="spinner"/><div><strong>Đang đọc kết quả đã lưu</strong><p>Kiểm tra artifact và đối chiếu cảm biến với nuScenes.</p></div></div>}

        {run && !loading && <>
          <section className="run-strip">
            <div className="run-scene"><span className="strip-icon"><Glyph name="cube" size={20}/></span><div><span className="tiny-label">SCENE / FRAME</span><strong>{run.scene?.name || '—'} <span>/</span> {String(run.frame_index ?? '—').padStart(2, '0')}</strong></div></div>
            <div className="run-token"><span className="tiny-label">SAMPLE TOKEN</span><code title={run.sample_token}>{run.sample_token || '—'}</code></div>
            <span className="precision-chip">{run.precision === 'float32' ? 'FP32' : run.precision}</span>
            <span className={`run-status ${run.kind === 'prediction' ? 'ready' : 'checked'}`}><Glyph name="check" size={14}/>{run.kind === 'prediction' ? 'SUCCESS' : run.kind === 'check_only' ? 'CHECK-ONLY' : 'FAILED'}</span>
          </section>

          {prediction && occupancy ? <>
            <div className="visual-grid">
              <section className="card viewport-card">
                <div className="viewport-heading"><div><h2>Semantic occupancy <span>3D</span></h2><p>LIDAR_TOP · mét · Z hướng lên</p></div><button className="reset-button" onClick={() => setResetVersion(value => value + 1)}><Glyph name="reset" size={15}/> Reset camera</button></div>
                <div className="viewport">
                  <Scene run={prediction} voxels={visible} lidar={lidar} showLidar={showLidar} height={height} resetVersion={resetVersion} onError={reportSceneError}/>
                  <div className="viewport-count"><span className="live-square"/><b data-testid="visible-count">{number.format(visible.length / 5)}</b> voxel{showLidar && lidar && <><span className="count-divider"/> {number.format(visibleLidarCount)} điểm keyframe</>}</div>
                  <div className="axis-legend" aria-label="Trục tọa độ"><span className="axis-x">X</span><span className="axis-y">Y</span><span className="axis-z">Z ↑</span><small>LiDAR reference</small></div>
                  <div className="viewport-hint">Kéo để xoay <span>·</span> Cuộn để zoom <span>·</span> Chuột phải để dịch</div>
                  {sceneError && <div className="scene-error" role="alert">{sceneError}</div>}
                  {!visible.length && <div className="empty-filter">Không có voxel trong bộ lọc hiện tại.</div>}
                </div>
                <div className="viewport-footer"><span><i className="status-dot success"/> {prediction.grid_shape.join(' × ')} grid</span><span>{prediction.voxel_size.join(' × ')} m / voxel</span><span>Trục mảng: {prediction.grid_axis_order.join(' → ')}</span></div>
              </section>

              <aside className="right-column">
                <section className="card bev-card"><div className="card-title"><h2>Bird’s-eye view</h2><span className="micro-tag">BEV</span></div><div className="bev-image"><img src={prediction.bev_url} alt={`BEV prediction đã lưu của ${prediction.scene.name}, frame ${prediction.frame_index}`} width={prediction.bev.width} height={prediction.bev.height}/><span className="bev-axis-y">{prediction.bev.rows === 'y decreasing' ? 'y ↑' : prediction.bev.rows}</span><span className="bev-axis-x">{prediction.bev.columns === 'x increasing' ? 'x →' : prediction.bev.columns}</span></div><p className="bev-caption">BEV gốc · toàn bộ lớp<span>{prediction.bev.width} × {prediction.bev.height}</span></p></section>
                <EvaluationPanel run={prediction}/>
              </aside>
            </div>
          </> : <section className="card no-prediction" data-testid="no-prediction"><Glyph name="layers" size={36}/><h2>{run.kind === 'check_only' ? 'Run check-only — chưa có prediction' : 'Run chưa có prediction thành công'}</h2><p>{run.kind === 'check_only' ? 'Run này đã kiểm tra dữ liệu và checkpoint trên CPU, chưa chạy forward.' : run.error?.message || 'Không có artifact prediction hợp lệ để hiển thị.'}</p><p>Chọn một run success có occupancy đã lưu để xem không gian 3D.</p></section>}

          <div className="details-grid"><Timing run={run}/><section className="card allocator-card"><div className="card-title"><h2>Bộ cấp phát PyTorch</h2><span className="micro-tag">ALLOCATOR</span></div><div className="allocator-values"><div><span>Peak allocated</span><strong>{gib(run.allocator.peak_allocated_bytes)}</strong></div><div><span>Peak reserved</span><strong>{gib(run.allocator.peak_reserved_bytes)}</strong></div></div><p className="footnote">Thống kê allocator, không phải VRAM vật lý được đo độc lập.</p></section></div>
          <section className="card provenance-card"><div className="provenance-label"><Glyph name="check" size={16}/> Nguồn kết quả</div><div className="checkpoint"><span>CHECKPOINT</span><code title={run.checkpoint?.path}>{run.checkpoint?.path.split('/').pop() || '—'}</code></div><div className="checkpoint-hash"><span>SHA-256</span><code title={run.checkpoint?.sha256}>{run.checkpoint?.sha256 ? `${run.checkpoint.sha256.slice(0, 16)}…` : '—'}</code></div>{prediction && <span className="verified-pill" title={prediction.validation.npz_token}><Glyph name="check" size={13}/> Sample & cảm biến đã đối chiếu</span>}</section>
          {prediction && <p className="provenance-note">Đối chiếu metadata, report và nuScenes. NPZ không chứa sample token riêng; shape và số voxel từng lớp khớp report.</p>}
          <ReviewPanel key={run.run_id} run={run} user={user} onChange={reviewChanged}/>
        </>}
        <footer className="page-footer"><span>RAV25 / OFFLINE OCCUPANCY ANALYSIS</span><span>{run?.kind === 'check_only' ? 'Check-only · chưa chạy forward' : run?.evaluation?.status === 'matched' ? 'Prediction đã lưu · GT đã ghép' : 'Prediction đã lưu · chưa có GT đã ghép'}</span></footer>
      </main>
    </div>
  </div>;
}
