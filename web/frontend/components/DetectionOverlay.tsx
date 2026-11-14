'use client';
import {useEffect,useState} from 'react';
import {getJSON} from '../lib/api';
import type {DetectionBox,DetectionRun} from '../lib/detections';

export default function DetectionOverlay({runId,onBoxes}:{runId:string;onBoxes:(boxes:DetectionBox[])=>void}){
 const [runs,setRuns]=useState<DetectionRun[]>([]),[selected,setSelected]=useState(''),[enabled,setEnabled]=useState(true),[score,setScore]=useState(.35),[classes,setClasses]=useState<Set<number>>(new Set()),[error,setError]=useState('');
 const active=runs.find(r=>r.run_id===selected);
 useEffect(()=>{
  const controller=new AbortController();setRuns([]);setSelected('');setError('');onBoxes([]);
  getJSON<{matches:DetectionRun[];rejected:{run_id:string;reason:string}[]}>(`/api/runs/${encodeURIComponent(runId)}/detections`,controller.signal).then(value=>{
   setRuns(value.matches);setSelected(value.matches[0]?.run_id||'');setClasses(new Set(Array.from({length:10},(_,i)=>i)));setError(value.rejected.map(v=>v.reason).join('; '));
  }).catch(e=>{if(e.name!=='AbortError')setError(e.message);});return()=>controller.abort();
 },[runId,onBoxes]);
 useEffect(()=>{onBoxes(active&&enabled?active.boxes.filter(b=>b.score>=score&&classes.has(b.label)):[]);},[active,enabled,score,classes,onBoxes]);
 return <section className="card detection-panel" data-testid="detection-overlay"><div className="card-title"><h2>3D detection</h2><span className="micro-tag">CENTERPOINT · LIDAR ONLY</span></div>
  {error&&<p role="alert">{error}</p>}{!runs.length?<p>Chưa có detection cùng token/timestamp và hệ LiDAR đã kiểm chứng.</p>:<>
  <label><input type="checkbox" aria-label="Hiển thị detection box" checked={enabled} onChange={e=>setEnabled(e.target.checked)}/> Hiển thị box</label>
  <select aria-label="Detection overlay run" value={selected} onChange={e=>setSelected(e.target.value)}>{runs.map(r=><option key={r.run_id} value={r.run_id}>{r.run_id}</option>)}</select>
  <label>Score hiển thị ≥ {score.toFixed(2)}<input aria-label="Detection minimum score" type="range" min="0" max="1" step=".05" value={score} onChange={e=>setScore(Number(e.target.value))}/></label>
  <details><summary>Lọc lớp detection</summary>{active?.classes.map(c=><label key={c.id}><input aria-label={`Detection class ${c.name}`} type="checkbox" checked={classes.has(c.id)} onChange={()=>setClasses(prev=>{const v=new Set(prev);if(v.has(c.id))v.delete(c.id);else v.add(c.id);return v;})}/>{c.name}</label>)}</details>
  <p data-testid="detection-box-count">{active?.box_count} box đã lưu · chỉ lọc hiển thị</p><p>mAP/NDS: pending_dataset_evaluation. GT annotations có thể có; chưa đánh giá toàn tập.</p>
  {active?.fixture&&<p className="job-fixture">Box fixture tổng hợp, không phải prediction thật.</p>}
  <a href={`/detections?run=${encodeURIComponent(selected)}`}>Xem nguồn và review detection ↗</a><p className="footnote">Snapshot detection riêng; approval/export occupancy không bao gồm overlay này. Score chưa hiệu chuẩn.</p>
  </>}
 </section>;
}
