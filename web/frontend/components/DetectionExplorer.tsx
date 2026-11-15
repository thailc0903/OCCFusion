'use client';
import dynamic from 'next/dynamic';
import {useCallback,useEffect,useState} from 'react';
import {getJSON,getPacked} from '../lib/api';
import type {ReviewInfo,User,RunSummary} from '../lib/types';
import type {DetectionRun,DetectionBox} from '../lib/detections';
import DetectionOverlay from './DetectionOverlay';
import ReviewPanel from './ReviewPanel';
import CameraViewer from './CameraViewer';
const Scene=dynamic(()=>import('./OccupancyScene'),{ssr:false});const EMPTY=new Float32Array();

export default function DetectionExplorer({user,onLogout}:{user:User;onLogout:()=>Promise<void>}){
 const [runs,setRuns]=useState<RunSummary[]>([]),[selected,setSelected]=useState(''),[run,setRun]=useState<DetectionRun|null>(null),[lidar,setLidar]=useState<Float32Array|null>(null),[boxes,setBoxes]=useState<DetectionBox[]>([]),[showLidar,setShowLidar]=useState(true),[reset,setReset]=useState(0),[error,setError]=useState(''),[occupancy,setOccupancy]=useState<RunSummary[]>([]);
 const receiveBoxes=useCallback((value:DetectionBox[])=>setBoxes(value),[]);const sceneError=useCallback((message:string)=>setError(message),[]);
 useEffect(()=>{const c=new AbortController();getJSON<{runs:RunSummary[]}>('/api/detections',c.signal).then(v=>{setRuns(v.runs);const requested=new URLSearchParams(window.location.search).get('run');setSelected(v.runs.find(r=>r.run_id===requested)?.run_id||v.runs[0]?.run_id||'');}).catch(e=>{if(e.name!=='AbortError')setError(e.message);});getJSON<{runs:RunSummary[]}>('/api/runs',c.signal).then(v=>setOccupancy(v.runs)).catch(()=>{});return()=>c.abort();},[]);
 useEffect(()=>{setRun(null);setLidar(null);setBoxes([]);setError('');if(!selected)return;const c=new AbortController();getJSON<DetectionRun>(`/api/detections/${encodeURIComponent(selected)}`,c.signal).then(async v=>{const data=await getPacked(v.lidar.url,v.lidar.point_count,v.lidar.stride,c.signal);if(!c.signal.aborted){setRun(v);setLidar(data);}}).catch(e=>{if(e.name!=='AbortError')setError(e.message);});return()=>c.abort();},[selected]);
 const reviewChanged=(review:ReviewInfo)=>setRun(previous=>previous?{...previous,review}:previous);
 return <div className="app-shell"><header className="topbar"><a className="brand" href="/">RAV25</a><nav className="benchmark-nav"><a href="/">Occupancy</a><a href="/jobs">Jobs</a><a href="/benchmarks">Benchmark</a></nav><div className="account-menu"><span>{user.username} · {user.role}</span><button onClick={onLogout}>Đăng xuất</button></div></header><main className="jobs-content">
  <div className="page-heading"><div><div className="eyebrow">SAVED DETECTION / LIDAR ONLY</div><h1>CenterPoint 3D boxes</h1><p>Phát lại detection đã lưu · chưa phải learned 4D flow.</p></div></div>
  <label>Detection run <select aria-label="Chọn detection run" value={selected} onChange={e=>setSelected(e.target.value)}>{!runs.length&&<option value="">Chưa có detection đã gửi/đã lưu</option>}{runs.map(r=><option key={r.run_id} value={r.run_id}>{r.run_id} · {r.scene_name}/{r.frame_index}</option>)}</select></label>
  {error&&<div role="alert" className="error-panel">{error}</div>}{run&&<>
   {run.fixture&&<p className="job-fixture">Fixture tổng hợp kiểm thử · không phải prediction mô hình.</p>}
   <section className="run-strip"><b>{run.scene.name} / {run.frame_index}</b><code data-testid="detection-token">{run.sample_token}</code><span>{run.precision} · {run.box_count} box gốc</span></section>
   <div className="visual-grid"><section className="card viewport-card"><div className="viewport-heading"><h2>Box + LiDAR · LIDAR_TOP, Z ↑, m</h2><label><input type="checkbox" checked={showLidar} onChange={e=>setShowLidar(e.target.checked)}/>LiDAR</label><button onClick={()=>setReset(v=>v+1)}>Reset camera</button></div><div className="viewport"><Scene run={run} voxels={EMPTY} lidar={lidar} showLidar={showLidar} height={[-5,3]} resetVersion={reset} onError={sceneError} boxes={boxes}/></div></section><aside><DetectionOverlay key={run.run_id} runId={run.run_id} onBoxes={receiveBoxes}/></aside></div>
   <section className="card detection-panel"><h2>Nguồn và phép đo</h2><p>{run.source}</p><code>{run.checkpoint.path}</code><p>SHA-256 {run.checkpoint.sha256}</p><p>mAP / NDS: <b>null · pending_dataset_evaluation</b>. Detection GT annotations: {run.ground_truth.detection_annotations_available?'có':'không có cho sample'}. Một frame chưa đánh giá toàn mini_val.</p><p>Forward {run.timings_s.forward?.toFixed(3)} s · pipeline {run.timings_s.pipeline_total?.toFixed(3)} s · tiến trình {run.timings_s.process_until_report?.toFixed(3)} s.</p><p>PyTorch allocator peak allocated {run.allocator.peak_allocated_bytes} / reserved {run.allocator.peak_reserved_bytes} byte; không phải VRAM vật lý đo độc lập.</p>{occupancy.filter(r=>r.sample_token===run.sample_token&&r.prediction_available).map(r=><p key={r.run_id}><a href={`/?run=${r.run_id}`}>Mở overlay cùng sample trên occupancy: {r.run_id} ↗</a></p>)}</section>
   <CameraViewer key={'camera-'+run.run_id} endpoint={'/api/runs/'+encodeURIComponent(run.run_id)+'/cameras'} user={user}/>
   <ReviewPanel run={run} user={user} onChange={reviewChanged}/>
  </>}
 </main></div>;
}
