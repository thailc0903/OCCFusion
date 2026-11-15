'use client';
import dynamic from 'next/dynamic';
import {useCallback,useEffect,useMemo,useState} from 'react';
import {getJSON,getPacked} from '../lib/api';
import type {BenchmarkFrame,PredictionRun} from '../lib/types';
import type {DetectionBenchmarkFrame} from '../lib/detectionBenchmarks';
const Scene=dynamic(()=>import('./OccupancyScene'),{ssr:false});
const EMPTY=new Float32Array();
const number=new Intl.NumberFormat('vi-VN');

export default function DetectionBenchmarkFramePlayer({id,frame}:{id:string;frame:BenchmarkFrame}) {
  const [run,setRun]=useState<DetectionBenchmarkFrame|null>(null),[lidar,setLidar]=useState<Float32Array|null>(null);
  const [showLidar,setShowLidar]=useState(true),[showBoxes,setShowBoxes]=useState(true),[score,setScore]=useState(.35);
  const [classes,setClasses]=useState<Set<number>>(new Set()),[reset,setReset]=useState(0),[error,setError]=useState('');
  const [overlays,setOverlays]=useState<{run_id?:string;benchmark_id?:string}[]>([]),[overlayId,setOverlayId]=useState('');
  const [overlay,setOverlay]=useState<PredictionRun|null>(null),[voxels,setVoxels]=useState<Float32Array>(EMPTY);
  const [overlayError,setOverlayError]=useState('');
  const reportError=useCallback((value:string)=>setError(value),[]);
  const base='/api/benchmarks/'+encodeURIComponent(id)+'/frames/'+frame.sample_token;
  useEffect(()=>{
    const c=new AbortController();setRun(null);setLidar(null);setError('');setOverlayError('');setOverlays([]);setOverlayId('');setOverlay(null);setVoxels(EMPTY);
    getJSON<DetectionBenchmarkFrame>(base,c.signal).then(async value=>{
      const points=await getPacked(value.lidar.url,value.lidar.point_count,value.lidar.stride,c.signal);
      if(c.signal.aborted)return;setRun(value);setLidar(points);setClasses(new Set(value.classes.map(c=>c.id)));
    }).catch(e=>{if(e.name!=='AbortError')setError(e.message);});
    getJSON<{matches:{run_id?:string;benchmark_id?:string}[]}>(base+'/overlays',c.signal).then(value=>{if(!c.signal.aborted)setOverlays(value.matches);})
      .catch(e=>{if(e.name!=='AbortError')setOverlayError(e.message);});
    return()=>c.abort();
  },[base]);
  useEffect(()=>{
    setOverlay(null);setVoxels(EMPTY);setOverlayError('');if(!overlayId)return;
    const c=new AbortController();
    const query=overlayId.startsWith('benchmark:')?'occupancy_benchmark_id='+encodeURIComponent(overlayId.slice(10)):'occupancy_run_id='+encodeURIComponent(overlayId);
    getJSON<DetectionBenchmarkFrame>(base+'?'+query,c.signal).then(async value=>{
      if(!value.occupancy_overlay)throw Error('Không có occupancy đã xác minh');
      const data=await getPacked(value.occupancy_overlay.occupancy.url,value.occupancy_overlay.occupancy.point_count,value.occupancy_overlay.occupancy.stride,c.signal);
      if(!c.signal.aborted){setOverlay(value.occupancy_overlay);setVoxels(data);}
    }).catch(e=>{if(e.name!=='AbortError')setOverlayError(e.message);});
    return()=>c.abort();
  },[base,overlayId]);
  const boxes=useMemo(()=>run&&showBoxes?run.boxes.filter(b=>b.score>=score&&classes.has(b.label)):[],[run,showBoxes,score,classes]);
  if(error)return <section className="error-panel" role="alert"><h2>Không thể ghép frame detection</h2><p>{error}</p></section>;
  if(!run)return <section className="loading-panel"><span className="spinner"/><p>Đang mở box và LiDAR đã lưu…</p></section>;
  const geometry=overlay||run;
  return <section className="benchmark-playback detection-benchmark-playback" data-testid="detection-benchmark-playback">
    <div className="run-strip"><strong>{run.scene.name} / {run.frame_index}</strong><code data-testid="detection-benchmark-token">{run.sample_token}</code><span>{run.precision} · {run.origin}</span></div>
    <div className="card viewport-card">
      <div className="viewport-heading"><div><h2>CenterPoint boxes + LiDAR</h2><p>LIDAR_TOP · mét · Z hướng lên · phát lại artifact</p></div><button className="reset-button" onClick={()=>setReset(x=>x+1)}>Reset camera</button></div>
      <div className="benchmark-frame-controls">
        <label><input type="checkbox" aria-label="Detection benchmark boxes" checked={showBoxes} onChange={e=>setShowBoxes(e.target.checked)}/> Box</label>
        <label><input type="checkbox" aria-label="Detection benchmark LiDAR" checked={showLidar} onChange={e=>setShowLidar(e.target.checked)}/> LiDAR đúng sample</label>
        <label>Score hiển thị ≥ {score.toFixed(2)}<input type="range" aria-label="Detection benchmark minimum score" min="0" max="1" step=".01" value={score} onChange={e=>setScore(Number(e.target.value))}/></label>
        <details className="benchmark-class-filter"><summary>Lọc lớp detection</summary><div>{run.classes.map(c=><label key={c.id}><input type="checkbox" aria-label={'Benchmark detection class '+c.name} checked={classes.has(c.id)} onChange={()=>setClasses(p=>{const n=new Set(p);if(n.has(c.id))n.delete(c.id);else n.add(c.id);return n;})}/>{c.name}</label>)}</div></details>
        <label>Occupancy overlay<select aria-label="Detection benchmark occupancy overlay" value={overlayId} onChange={e=>setOverlayId(e.target.value)}><option value="">Không overlay</option>{overlays.map(o=>{const value=o.benchmark_id?'benchmark:'+o.benchmark_id:o.run_id;return <option key={value} value={value}>{o.benchmark_id?'Benchmark '+o.benchmark_id:'Run '+o.run_id}</option>;})}</select></label>
      </div>
      {overlayError&&<p role="alert" className="inline-error">{overlayError}</p>}
      <div className="viewport"><Scene run={geometry} voxels={voxels} lidar={lidar} showLidar={showLidar}
        height={[geometry.bounds[2],geometry.bounds[5]]} resetVersion={reset} onError={reportError} boxes={boxes}/>
        <div className="viewport-count"><b data-testid="detection-visible-boxes">{boxes.length}</b> / {run.box_count} box đã lưu{showLidar&&lidar?' · '+number.format(lidar.length/4)+' điểm LiDAR':''}{overlay?' · '+number.format(voxels.length/5)+' voxel occupancy':''}</div>
        <div className="axis-legend"><span className="axis-x">X</span><span className="axis-y">Y</span><span className="axis-z">Z ↑</span><small>LiDAR reference</small></div>
      </div>
      <p className="benchmark-frame-timing">Score/lớp chỉ lọc hiển thị; prediction đánh giá giữ nguyên. Score chưa hiệu chuẩn. Occupancy overlay ghép đúng token/timestamp/calibration, không nằm trong approval/export collection detection.</p>
    </div>
  </section>;
}
