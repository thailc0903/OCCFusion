'use client';

import { useEffect, useState } from 'react';
import { clearPayloadCache, getJSON } from '../lib/api';
import type { BenchmarkDetail, CollectionReviewInfo, User } from '../lib/types';
import BenchmarkFramePlayer from './BenchmarkFramePlayer';
import CollectionReviewPanel from './CollectionReviewPanel';

const percent = new Intl.NumberFormat('vi-VN',{style:'percent',maximumFractionDigits:4});
const number = new Intl.NumberFormat('vi-VN');
const seconds = (n: number | null) => n === null ? 'Không khả dụng' : `${n.toFixed(4)} s`;

export default function BenchmarkViewer({user,onLogout}: {user: User;onLogout:()=>void}) {
  const [items,setItems] = useState<{benchmark_id:string;status:string}[]>([]);
  const [id,setId] = useState('');
  const [data,setData] = useState<BenchmarkDetail | null>(null);
  const [scene,setScene] = useState('');
  const [token,setToken] = useState('');
  const [error,setError] = useState('');
  const [reload,setReload] = useState(0);
  useEffect(() => {
    const controller=new AbortController();
    getJSON<{benchmarks:typeof items}>('/api/benchmarks',controller.signal).then(value => {
      setItems(value.benchmarks);setId(previous => value.benchmarks.some(b => b.benchmark_id===previous) ? previous : value.benchmarks[0]?.benchmark_id || '');
      if (!value.benchmarks.length) setError(user.role==='reviewer'?'Chưa có collection được gửi review.':'Chưa có benchmark mini_val trong thư mục cấu hình.');
    }).catch(e => { if(e.name!=='AbortError')setError(e.message); });
    return () => controller.abort();
  }, [reload,user.role]);
  useEffect(() => {
    if(!id)return;
    const controller=new AbortController();setData(null);setToken('');setError('');
    getJSON<BenchmarkDetail>(`/api/benchmarks/${encodeURIComponent(id)}`,controller.signal).then(value => {
      if(controller.signal.aborted)return;
      setData(value);const first=value.frames.find(f=>f.prediction_available)||value.frames[0];setScene(first.scene);setToken(first.sample_token);
    }).catch(e => { if(e.name!=='AbortError')setError(e.message); });
    return () => controller.abort();
  }, [id,reload]);
  const selected=data?.frames.find(f=>f.sample_token===token);
  function changeScene(name:string) {
    setScene(name);const frames=data!.frames.filter(f=>f.scene===name);setToken((frames.find(f=>f.prediction_available)||frames[0]).sample_token);
  }
  function reviewChanged(review:CollectionReviewInfo) {setData(previous => previous?{...previous,review}:previous);}
  return <div className="benchmark-shell">
    <header className="topbar"><a className="brand" href="/"><span className="brand-mark">R</span><span><strong>RAV25</strong><span className="brand-sub">OCCUPANCY LAB</span></span></a><nav className="benchmark-nav"><a href="/">Run explorer</a><a href="/benchmarks" aria-current="page">Benchmark</a></nav><span className="local-badge"><i/> LOCAL / SAVED ARTIFACTS</span><div className="account-menu"><span><b>{user.username}</b><small>{user.role}</small></span><button onClick={onLogout}>Đăng xuất</button></div></header>
    <main className="benchmark-main">
      <div className="page-heading"><div><div className="eyebrow">SET-LEVEL EVALUATION / NUSCENES MINI_VAL</div><h1>Benchmark collection<span className="title-period">.</span></h1><p>Phát lại prediction đã lưu; chưa phải learned 4D flow.</p></div><span className="replay-badge">↻ Không chạy inference / GPU</span></div>
      <div className="benchmark-picker"><label>Benchmark đã lưu<select aria-label="Chọn benchmark" value={id} onChange={e=>setId(e.target.value)}>{items.map(b=><option key={b.benchmark_id} value={b.benchmark_id}>{b.benchmark_id} · {b.status}</option>)}</select></label><button onClick={()=>{clearPayloadCache();setReload(n=>n+1);}}>Đọc lại artifact</button></div>
      {error&&<div className="error-panel" role="alert"><h2>Không thể xác minh collection</h2><p>{error}</p><p>Metric không được ghép khi token, hình học, protocol hoặc hash không hợp lệ.</p></div>}
      {!data&&!error&&<div className="loading-panel"><span className="spinner"/><div><strong>Đang xác minh artifact benchmark</strong><p>Đối chiếu các frame, pooled confusion matrix, protocol, audit và hash. Lượt đầu có thể mất khoảng vài chục giây.</p></div></div>}
      {data&&<>
        <section className="benchmark-status card"><div><span className={`benchmark-badge ${data.status}`} data-testid="benchmark-status">{data.status.toUpperCase()}</span><h2 data-testid="benchmark-coverage">{data.successful_frame_count}/{data.expected_frame_count} frame hợp lệ</h2><p>{data.failed_frame_count} frame thất bại · {data.split}, không phải toàn nuScenes validation</p></div><div className="benchmark-scenes">{Object.entries(data.scene_counts).map(([name,count])=><p key={name}><b>{name}</b><span>{count} frame có evaluation</span></p>)}</div><div><span className="precision-chip">{data.precision} · batch 1</span><p>TF32 matmul: {String(data.tf32.matmul)}<br/>TF32 cuDNN: {String(data.tf32.cudnn)}</p></div></section>
        <div className="benchmark-summary-grid"><section className="card benchmark-metrics"><div className="card-title"><h2>Pooled accuracy</h2><span className="micro-tag">TỔNG COUNTS → TỶ LỆ</span></div><div className="benchmark-metric-pair"><div><span>Occupancy IoU</span><b data-testid="pooled-iou">{percent.format(data.metrics.occupancy_iou)}</b></div><div><span>Semantic mIoU</span><b data-testid="pooled-miou">{percent.format(data.metrics.semantic_miou)}</b></div></div><p>Trung bình đủ <b>{data.protocol.semantic_mean_class_count} lớp</b>; union = 0 nhận IoU 0. Warm-up vẫn tính accuracy.</p><p>{number.format(data.metrics.valid_voxel_count)} voxel hợp lệ · chỉ {data.successful_frame_count} frame thành công. Không trung bình IoU từng frame.</p></section>
          <section className="card benchmark-latency" data-testid="benchmark-latency"><div className="card-title"><h2>Forward từ lượt GPU gốc</h2><span className="micro-tag">CUDA SYNCHRONIZED</span></div><div className="benchmark-metric-pair"><div><span>Median</span><b data-testid="benchmark-median">{seconds(data.latency.median_s)}</b></div><div><span>p95</span><b data-testid="benchmark-p95">{seconds(data.latency.p95_s)}</b></div></div><p><b>{data.latency.sample_count} mẫu</b>, bỏ {data.latency.warmup_count} warm-up khỏi {data.latency.all_forward_sample_count} forward thật. Không dùng thời gian đọc cache/evaluate-only.</p><p data-testid="benchmark-latency-source">Nguồn: <code>{data.latency.source_report}</code> · session <code>{data.latency.source_session_id}</code></p><small>Một lượt sequential trên laptop; chưa phải latency ổn định qua nhiều trial.</small></section></div>
        <details className="card benchmark-startup"><summary>Khởi tạo, tiến trình và PyTorch allocator</summary><div className="benchmark-startup-grid"><p>Bootstrap <b>{seconds(data.latency.startup.bootstrap_s)}</b></p><p>Setup chung <b>{seconds(data.latency.startup.shared_setup_s)}</b></p><p>Khởi tạo model/checkpoint ở lượt GPU gốc <b>{seconds(data.latency.startup.model_initialization_s)}</b></p><p>Tiến trình vòng benchmark gốc <b>{seconds(data.latency.startup.invocation_wall_s)}</b></p><p>Peak allocated <b>{(data.allocator.peak_allocated_bytes/2**30).toFixed(3)} GiB</b></p><p>Peak reserved <b>{(data.allocator.peak_reserved_bytes/2**30).toFixed(3)} GiB</b></p></div><p>Allocator của lượt đã lưu; không phải VRAM vật lý thực đo. Khởi tạo chung được tách khỏi thời gian từng frame.</p></details>
        <section className="card benchmark-failures" data-testid="benchmark-failures"><div className="card-title"><h2>Frame thất bại & audit radar</h2><span className="micro-tag">{data.radar_audit.status}</span></div>{data.radar_audit.frames.map(frame=><details key={frame.sample_token}><summary>{frame.scene} / frame {frame.frame_index} · {frame.rejected_channels.length} radar không có sweep hợp lệ</summary><code>{frame.sample_token}</code><p>{frame.reason}</p><p>LiDAR tham chiếu: {frame.reference_timestamp_us} µs. Δ = radar − LiDAR; dương là tương lai.</p><div className="benchmark-table-scroll"><table><thead><tr><th>Radar</th><th>Δ (ms)</th><th>prev</th><th>File</th><th>Record ≤ LiDAR</th></tr></thead><tbody>{frame.radars.map(r=><tr key={r.channel}><td>{r.channel}</td><td>{r.delta_ms>0?'+':''}{r.delta_ms.toFixed(3)}</td><td>{r.prev_token||'rỗng'}</td><td>{r.file_exists?'Có':'Thiếu'}</td><td>{r.eligible_count}</td></tr>)}</tbody></table></div><p>{frame.adapter_bug_detected?'Audit phát hiện lỗi cần kiểm tra.':'Không phát hiện lỗi duyệt/lọc trong frame này.'}</p></details>)}<p>Audit: <code>{data.radar_audit.source}</code> · token, timestamp và file hash đã đối chiếu.</p></section>
        <section className="card benchmark-classes"><div className="card-title"><h2>IoU đủ 16 lớp semantic</h2><span className="micro-tag">POOLED / {data.successful_frame_count} FRAMES</span></div><div className="benchmark-table-scroll"><table><thead><tr><th>ID</th><th>Lớp</th><th>IoU</th><th>Intersection</th><th>Union</th><th>GT count</th><th>Prediction count</th></tr></thead><tbody>{data.metrics.per_class_semantic.map(c=><tr key={c.label} data-testid={`pooled-class-${c.label}`}><td>{c.label}</td><td>{c.name}</td><td>{percent.format(c.iou)}</td><td>{number.format(c.intersection)}</td><td>{number.format(c.union)}</td><td>{number.format(c.gt_count)}</td><td>{number.format(c.prediction_count)}</td></tr>)}</tbody></table></div></section>
        <details className="card benchmark-confusion"><summary>Confusion matrix · {data.protocol.class_names.length} × {data.protocol.class_names.length} (gồm empty/free)</summary><p>Hàng = GT, cột = prediction. Các số là voxel counts sau bỏ GT ignore.</p><div className="benchmark-table-scroll"><table aria-label="Confusion matrix pooled"><thead><tr><th>GT ↓ / Pred →</th>{data.protocol.class_names.map((c,i)=><th key={c} title={c}>{i}</th>)}</tr></thead><tbody>{data.metrics.confusion_matrix.map((row,i)=><tr key={i}><th>{i} · {data.protocol.class_names[i]}</th>{row.map((count,j)=><td key={j} title={`${data.protocol.class_names[i]} → ${data.protocol.class_names[j]}: ${count}`} style={{background:`rgba(32,116,168,${count?0.05+Math.log1p(count)/25:0})`}}>{number.format(count)}</td>)}</tr>)}</tbody></table></div></details>
        <details className="card benchmark-protocol"><summary>Protocol, hình học & bằng chứng hash</summary><p><code>{data.protocol.name}</code> · đơn vị metric: tỷ lệ [0,1], chỉ format phần trăm một lần.</p><p>Bounds {data.protocol.bounds.join(', ')} · Grid {data.protocol.grid_shape.join(' × ')} · trục {data.protocol.grid_axis_order.join(' → ')} · voxel {data.protocol.voxel_size.join(' × ')} m.</p><p>GT sparse label 0 → ignore 255; ô không liệt kê → empty 0. Sample token lấy từ metadata/report, không giả định NPZ chứa token.</p><p>Max softmax probability là confidence chưa hiệu chuẩn.</p>{Object.entries(data.evidence_hashes).map(([name,r])=><p className="benchmark-hash" key={name}><b>{name}</b><code>{r.sha256}</code></p>)}<p>Audit SHA-256: <code>{data.radar_audit.sha256}</code></p><pre>{JSON.stringify(data.protocol,null,2)}</pre></details>
        <section className="benchmark-frame-picker card"><div><h2>Phát lại frame đã lưu</h2><p>Chọn scene/frame; không chạy lại inference. Playback chưa phải learned 4D flow.</p></div><label>Scene<select aria-label="Chọn scene benchmark" value={scene} onChange={e=>changeScene(e.target.value)}>{Array.from(new Set(data.frames.map(f=>f.scene))).map(s=><option key={s}>{s}</option>)}</select></label><label>Frame<select aria-label="Chọn frame benchmark" value={token} onChange={e=>setToken(e.target.value)}>{data.frames.filter(f=>f.scene===scene).map(f=><option value={f.sample_token} key={f.sample_token}>Frame {f.frame_index} · {f.prediction_available?'prediction + evaluation':'LỖI / không prediction'}</option>)}</select></label></section>
        {selected&&<BenchmarkFramePlayer key={`${id}:${token}:${reload}`} id={id} frame={selected} audit={data.radar_audit.frames.find(f=>f.sample_token===token)}/>}
        <CollectionReviewPanel id={id} info={data.review} user={user} onChange={reviewChanged}/>
      </>}
      <footer className="page-footer"><span>RAV25 / MINI_VAL SAVED BENCHMARK</span><span>Playback chưa phải learned 4D flow · Không chạy model</span></footer>
    </main>
  </div>;
}
