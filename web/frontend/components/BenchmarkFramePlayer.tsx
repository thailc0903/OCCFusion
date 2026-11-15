'use client';

import dynamic from 'next/dynamic';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { getJSON, getPacked } from '../lib/api';
import type { BenchmarkFrame, PredictionRun, RadarAuditFrame, User } from '../lib/types';
import CameraViewer from './CameraViewer';
import EvaluationPanel from './EvaluationPanel';

const Scene = dynamic(() => import('./OccupancyScene'), { ssr: false });
const number = new Intl.NumberFormat('vi-VN');

export default function BenchmarkFramePlayer({ id, frame, audit, user }: { id: string; frame: BenchmarkFrame; audit?: RadarAuditFrame; user: User }) {
  const [run, setRun] = useState<PredictionRun | null>(null);
  const [voxels, setVoxels] = useState<Float32Array | null>(null);
  const [lidar, setLidar] = useState<Float32Array | null>(null);
  const [showLidar, setShowLidar] = useState(false);
  const [height, setHeight] = useState<[number,number]>([-5,3]);
  const [classes, setClasses] = useState<Set<number>>(new Set());
  const [reset, setReset] = useState(0);
  const [error, setError] = useState('');
  const reportError = useCallback((message: string) => setError(message), []);
  useEffect(() => {
    if (!frame.prediction_available) return;
    const controller = new AbortController();
    setRun(null); setVoxels(null); setLidar(null); setError('');
    (async () => {
      const value = await getJSON<PredictionRun>(`/api/benchmarks/${encodeURIComponent(id)}/frames/${frame.sample_token}`, controller.signal);
      const data = await getPacked(value.occupancy.url, value.occupancy.point_count, value.occupancy.stride, controller.signal);
      if (controller.signal.aborted) return;
      setRun(value); setVoxels(data); setHeight([value.bounds[2],value.bounds[5]]);
      setClasses(new Set(value.classes.filter(c => !c.background).map(c => c.id)));
    })().catch(e => { if (e.name !== 'AbortError') setError(e.message); });
    return () => controller.abort();
  }, [id, frame.sample_token, frame.prediction_available]);
  useEffect(() => {
    if (!showLidar || !run) return;
    const controller = new AbortController();
    getPacked(run.lidar.url,run.lidar.point_count,run.lidar.stride,controller.signal)
      .then(value => { if (!controller.signal.aborted) setLidar(value); })
      .catch(e => { if (e.name !== 'AbortError') setError(e.message); });
    return () => controller.abort();
  }, [run,showLidar]);
  const visible = useMemo(() => {
    if (!voxels) return new Float32Array();
    const values: number[] = [];
    for (let i=0;i<voxels.length;i+=5) if (classes.has(voxels[i+3]) && voxels[i+2]>=height[0] && voxels[i+2]<=height[1]) values.push(...voxels.subarray(i,i+5));
    return new Float32Array(values);
  }, [voxels,classes,height]);

  if (!frame.prediction_available) return <section className="card benchmark-failed" data-testid="failed-frame">
    <span className="micro-tag">FRAME THẤT BẠI</span><h2>Không có prediction hoặc evaluation</h2>
    <p>{frame.scene} / {frame.frame_index} · <code>{frame.sample_token}</code></p>
    <p>Giai đoạn: <b>{frame.error?.stage}</b></p><p>{audit?.reason || frame.error?.message}</p>
    <p>IoU/mIoU: <b>Không khả dụng</b>. Viewer không sinh prediction hoặc chạy inference để thay thế.</p>
    {audit && <p>LiDAR: {audit.reference_timestamp_us} µs · radar bị loại: {audit.rejected_channels.join(', ')}.</p>}
  </section>;
  if (error) return <section className="error-panel" role="alert"><strong>Không thể ghép frame</strong><p>{error}</p></section>;
  if (!run || !voxels) return <section className="loading-panel"><span className="spinner"/><p>Đang xác minh và mở prediction đã lưu của {frame.scene} / {frame.frame_index}…</p></section>;
  return <div className="benchmark-playback" data-testid="benchmark-playback">
    <section className="run-strip"><strong>{run.scene.name} / {run.frame_index}</strong><code data-testid="benchmark-frame-token">{run.sample_token}</code><span className="precision-chip">{run.precision}</span><span className="run-status ready">SAVED PREDICTION</span></section>
    <div className="benchmark-player-grid">
      <section className="card viewport-card"><div className="viewport-heading"><div><h2>Semantic occupancy <span>3D</span></h2><p>LIDAR_TOP · X/Y/Z · mét · tâm voxel theo metadata</p></div><button className="reset-button" onClick={() => setReset(n => n+1)}>Reset camera</button></div>
        <div className="benchmark-frame-controls">
          <label><input type="checkbox" role="switch" aria-label="LiDAR benchmark" checked={showLidar} onChange={e => setShowLidar(e.target.checked)}/> LiDAR đúng sample</label>
          <label>Z min <input aria-label="Z min benchmark" type="number" step={run.voxel_size[2]} min={run.bounds[2]} max={height[1]} value={height[0]} onChange={e => setHeight([Math.max(run.bounds[2],Math.min(Number(e.target.value),height[1])),height[1]])}/></label>
          <label>Z max <input aria-label="Z max benchmark" type="number" step={run.voxel_size[2]} min={height[0]} max={run.bounds[5]} value={height[1]} onChange={e => setHeight([height[0],Math.min(run.bounds[5],Math.max(Number(e.target.value),height[0]))])}/></label>
          <details className="benchmark-class-filter"><summary>Lọc lớp ({classes.size})</summary><div>{run.classes.filter(c => !c.background).map(c => <label key={c.id}><input type="checkbox" aria-label={`Benchmark lớp ${c.name}`} checked={classes.has(c.id)} onChange={() => setClasses(previous => { const next=new Set(previous);if(next.has(c.id))next.delete(c.id);else next.add(c.id);return next; })}/><i style={{background:`rgb(${c.color.join(',')})`}}/>{c.name}</label>)}</div></details>
        </div>
        <div className="viewport"><Scene run={run} voxels={visible} lidar={lidar} showLidar={showLidar} height={height} resetVersion={reset} onError={reportError}/><div className="viewport-count"><b data-testid="benchmark-voxel-count">{number.format(visible.length/5)}</b> voxel · nền đã bỏ{showLidar && lidar && ` · ${number.format(lidar.length/4)} điểm LiDAR`}</div><div className="axis-legend"><span className="axis-x">X</span><span className="axis-y">Y</span><span className="axis-z">Z ↑</span><small>LiDAR reference</small></div></div>
        <div className="viewport-footer"><span>{run.grid_shape.join(' × ')} · {run.grid_axis_order.join(' → ')}</span><span>{run.voxel_size.join(' × ')} m</span></div>
        <p className="benchmark-frame-timing">Forward frame: {run.timings_s.forward?.toFixed(4)} s · Pipeline frame: {run.timings_s.pipeline_total?.toFixed(4)} s. Khởi tạo chung được hiển thị riêng ở trên.</p>
      </section>
      <aside className="right-column"><section className="card bev-card"><div className="card-title"><h2>BEV đã xuất</h2></div><div className="bev-image"><img src={run.bev_url} alt={`BEV benchmark ${run.scene.name} frame ${run.frame_index}`} width={run.bev.width} height={run.bev.height}/></div></section><EvaluationPanel run={run} collection/></aside>
    </div>
    <CameraViewer key={id+':'+frame.sample_token} endpoint={'/api/benchmarks/'+encodeURIComponent(id)+'/frames/'+frame.sample_token+'/cameras'} user={user}/>
  </div>;
}
