'use client';
import {useEffect,useRef,useState,type PointerEvent} from 'react';
import {getJSON,postJSON} from '../lib/api';
import type {User} from '../lib/types';
import type {CameraListing,CameraProjection,Mask} from '../lib/cameras';
import CollectionReviewPanel from './CollectionReviewPanel';

export default function CameraViewer({endpoint,user}:{endpoint:string;user:User}) {
  const [opened,setOpened]=useState(false),[data,setData]=useState<CameraListing|null>(null);
  const [channel,setChannel]=useState('CAM_FRONT'),[projection,setProjection]=useState<CameraProjection|null>(null);
  const [error,setError]=useState(''),[imageError,setImageError]=useState(''),[projectionError,setProjectionError]=useState('');
  const [imageReady,setImageReady]=useState('');
  const [reload,setReload]=useState(0),[zoom,setZoom]=useState(1),[large,setLarge]=useState(false);
  const [showPoints,setShowPoints]=useState(true),[showBoxes,setShowBoxes]=useState(true),[score,setScore]=useState(.35);
  const [maskMode,setMaskMode]=useState(false),[masks,setMasks]=useState<Mask[]>([]),[draft,setDraft]=useState<Mask|null>(null);
  const [ack,setAck]=useState(false),[comment,setComment]=useState(''),[saving,setSaving]=useState(false);
  const drag=useRef<[number,number]|null>(null);
  useEffect(()=>{setOpened(false);setData(null);setProjection(null);setError('');setLarge(false);},[endpoint]);
  useEffect(()=>{
    if(!opened)return;
    const c=new AbortController();setData(null);setProjection(null);setError('');
    getJSON<CameraListing>(endpoint,c.signal).then(v=>{if(!c.signal.aborted)setData(v);})
      .catch(e=>{if(e.name!=='AbortError')setError(e.message);});
    return()=>c.abort();
  },[opened,endpoint,reload]);
  const camera=data?.channels.find(c=>c.channel===channel);
  useEffect(()=>{
    const c=new AbortController();setProjection(null);setProjectionError('');setImageError('');
    setMasks(camera?.privacy.masks||[]);setDraft(null);drag.current=null;setAck(false);setComment('');setMaskMode(false);setZoom(1);
    if(camera?.status==='available')getJSON<CameraProjection>(camera.projection_url,c.signal).then(v=>{
      if(c.signal.aborted)return;
      if(v.sample_data_token!==camera.sample_data_token||v.sample_token!==data?.sample_token||v.width!==camera.width||v.height!==camera.height)throw Error('Projection/image identity mismatch');
      setProjection(v);
    }).catch(e=>{if(e.name!=='AbortError')setProjectionError(e.message);});
    return()=>c.abort();
  },[camera,data?.sample_token]);
  useEffect(()=>{const close=(event:KeyboardEvent)=>{if(event.key==='Escape')setLarge(false);};window.addEventListener('keydown',close);return()=>window.removeEventListener('keydown',close);},[]);
  function coordinate(event:PointerEvent<SVGSVGElement>):[number,number] {
    const r=event.currentTarget.getBoundingClientRect();
    return [Math.max(0,Math.min(camera!.width,(event.clientX-r.left)*camera!.width/r.width)),
      Math.max(0,Math.min(camera!.height,(event.clientY-r.top)*camera!.height/r.height))];
  }
  function rectangle(a:[number,number],b:[number,number]):Mask {
    return [Math.floor(Math.min(a[0],b[0])),Math.floor(Math.min(a[1],b[1])),Math.ceil(Math.max(a[0],b[0])),Math.ceil(Math.max(a[1],b[1]))];
  }
  async function save() {
    if(!camera)return;setSaving(true);setError('');
    try{
      await postJSON(endpoint+'/'+channel+'/privacy',{camera_fingerprint:camera.camera_fingerprint,
        previous_revision:camera.privacy.revision,reviewed:ack,masks,comment});
      setReload(n=>n+1);
    }catch(e){setError((e as Error).message);}finally{setSaving(false);}
  }
  // Render-time identity guard also covers the render before the effect clears
  // a previous channel's asynchronous response.
  const activeProjection=projection?.sample_data_token===camera?.sample_data_token
    && projection?.sample_token===data?.sample_token && projection?.channel===channel
    && imageReady===camera?.image_fingerprint && !imageError ? projection : null;
  const boxes=activeProjection?.boxes.filter(b=>b.score>=score)||[];
  return <section className="card camera-viewer" data-testid="camera-viewer">
    <div className="card-title"><h2>Camera nuScenes · ảnh gốc</h2><button onClick={()=>setOpened(v=>!v)}>{opened?'Đóng camera':'Mở sáu camera'}</button></div>
    {!opened&&<p>Ảnh đúng sample, projection box prediction/LiDAR và provenance ẩn danh riêng. Bấm để xác minh dữ liệu đã lưu.</p>}
    {opened&&<>
      {error&&<p role="alert" className="inline-error">{error}</p>}
      {!data&&!error&&<p>Đang kiểm tra provenance, file và calibration của sáu camera…</p>}
      {data&&<>
        <div className="camera-identity"><b>{data.scene.name} / {data.frame_index}</b> · <code data-testid="camera-sample-token">{data.sample_token}</code><span>Phát lại dữ liệu đã lưu</span></div>
        <div className="camera-gallery" data-testid="camera-gallery">{data.channels.map(c=><button className={channel===c.channel?'selected':''} key={c.channel} onClick={()=>setChannel(c.channel)} aria-label={'Chọn camera '+c.channel}>
          {c.status==='available'&&c.image_url?<img src={c.image_url} width={c.width} height={c.height} alt={'nuScenes '+c.channel+' '+data.sample_token}/>:<span className="camera-blocked">{c.reason}</span>}
          <strong>{c.channel}</strong><small>{c.status==='available'?'Δ camera − LiDAR '+c.camera_minus_lidar_ms.toFixed(3)+' ms':'BLOCKED'}</small>
          <small>{c.privacy.status==='engineer_reviewed'?'Đã kiểm tra ảnh này':'Chưa được Engineer kiểm tra'}</small>
        </button>)}</div>
        <p className="camera-publisher-note">Nguồn: ảnh nuScenes release được nhà phát hành xử lý theo <a href="https://www.nuscenes.org/data-format" target="_blank" rel="noreferrer">Privacy protection</a>. Ứng dụng không tự phát hiện/che toàn bộ PII. Trạng thái Engineer áp dụng từng ảnh cụ thể.</p>
        {camera?.status==='blocked'&&<p role="alert" className="camera-blocked">Không phục vụ {channel}: {camera.reason}</p>}
        {camera?.status==='available'&&<div className={large?'camera-focus camera-enlarged':'camera-focus'} role={large?'dialog':undefined} aria-label={large?'Camera phóng to':undefined}>
          <div className="camera-controls"><h3 data-testid="camera-active-channel">{channel}</h3><button onClick={()=>setLarge(v=>!v)}>{large?'Đóng phóng to':'Phóng to camera'}</button>
            <label>Zoom <input aria-label="Camera zoom" type="range" min="1" max="3" step=".25" value={zoom} onChange={e=>setZoom(Number(e.target.value))}/><b>{zoom.toFixed(2)}×</b></label>
            <label><input type="checkbox" aria-label="Camera LiDAR projection" checked={showPoints} onChange={e=>setShowPoints(e.target.checked)}/> LiDAR keyframe</label>
            <label><input type="checkbox" aria-label="Camera prediction boxes" checked={showBoxes} onChange={e=>setShowBoxes(e.target.checked)}/> Box prediction</label>
            <label>Score hiển thị ≥ {score.toFixed(2)}<input type="range" aria-label="Camera box score" min="0" max="1" step=".01" value={score} onChange={e=>setScore(Number(e.target.value))}/></label>
          </div>
          <p data-testid="camera-timestamps">Camera {camera.camera_timestamp_us} µs · LiDAR {camera.lidar_timestamp_us} µs · Δ {camera.camera_minus_lidar_ms.toFixed(3)} ms · {camera.width} × {camera.height} pixel gốc</p>
          {imageError&&<p role="alert">{imageError}</p>}{projectionError&&<p role="alert">{projectionError}</p>}
          <div className="camera-scroll"><div className="camera-stage" style={{width:(zoom*100)+'%',aspectRatio:camera.width+'/'+camera.height}} data-testid="camera-stage">
            <img key={camera.image_fingerprint} src={camera.image_url!} alt={'Camera gốc '+channel} width={camera.width} height={camera.height}
              onLoad={()=>setImageReady(camera.image_fingerprint)}
              onError={()=>{setImageError('Ảnh không còn khớp provenance/privacy; đọc lại frame.');setProjection(null);}}/>
            {!imageError&&<svg viewBox={'0 0 '+camera.width+' '+camera.height} preserveAspectRatio="xMidYMid meet" data-testid="camera-projection-svg"
              data-sample-token={data.sample_token} data-channel={channel} data-projection-camera={activeProjection?.channel||''} className={maskMode?'camera-mask-mode':''}
              onPointerDown={e=>{if(!maskMode)return;e.preventDefault();drag.current=coordinate(e);e.currentTarget.setPointerCapture(e.pointerId);}}
              onPointerMove={e=>{if(drag.current)setDraft(rectangle(drag.current,coordinate(e)));}}
              onPointerCancel={()=>{drag.current=null;setDraft(null);}}
              onPointerUp={e=>{if(!drag.current)return;const rect=rectangle(drag.current,coordinate(e));if(rect[2]>rect[0]&&rect[3]>rect[1])setMasks(m=>[...m,rect]);drag.current=null;setDraft(null);}}>
              {showPoints&&activeProjection?.points.map((p,i)=><circle key={i} cx={p[1]} cy={p[2]} r="1.5" fill={'hsl('+Math.min(240,p[3]*5)+' 95% 60%)'} opacity=".8"/>)}
              {showBoxes&&boxes.map(b=><g key={b.index} data-label={activeProjection!.class_names[b.label]}>{b.segments.map((s,i)=><line key={i} x1={s[0]} y1={s[1]} x2={s[2]} y2={s[3]} stroke="#ffd23c" strokeWidth="2" vectorEffect="non-scaling-stroke"/>)}</g>)}
              {[...masks,...(draft?[draft]:[])].map((m,i)=><rect key={i} x={m[0]} y={m[1]} width={m[2]-m[0]} height={m[3]-m[1]} fill="black" fillOpacity={draft && i===masks.length ? 0.7 : 1} data-testid="camera-mask"/>)}
            </svg>}
          </div></div>
          {!activeProjection&&!projectionError&&!imageError&&<p>Đang ghép projection đúng camera đã chọn…</p>}
          <p className="camera-projection-count" data-testid="camera-projection-count">{showPoints&&activeProjection?activeProjection.point_count:0} điểm chiếu · {showBoxes?boxes.filter(b=>b.segments.length).length:0} box cắt ảnh · {data.projection_protocol.precision}; lớp vàng = prediction đã lưu.</p>
          <p>Ego-motion compensation dùng pose riêng từng timestamp; chưa tự bù chuyển động vật thể hoặc trong một vòng LiDAR. Projection hình học chưa chứng minh visibility/occlusion.</p>
          {user.role==='engineer'&&<details className="camera-privacy-editor"><summary>Kiểm tra ảnh này &amp; mask bổ sung</summary>
            <p>Kéo hình chữ nhật trên ảnh theo pixel gốc. Lưu tạo derivative PNG riêng, không thay ảnh inference. Kiểm tra thủ công không bảo đảm đã loại toàn bộ PII.</p>
            <label><input type="checkbox" aria-label="Draw supplemental mask" checked={maskMode} onChange={e=>setMaskMode(e.target.checked)}/>Vẽ mask bổ sung</label>
            <button disabled={!masks.length} onClick={()=>setMasks(m=>m.slice(0,-1))}>Bỏ mask cuối</button><span data-testid="camera-mask-count">{masks.length} mask</span>
            <label><input type="checkbox" aria-label="Acknowledge individual image inspection" checked={ack} onChange={e=>setAck(e.target.checked)}/>Tôi đã kiểm tra đúng ảnh này ({channel})</label>
            <textarea aria-label="Camera privacy comment" maxLength={4000} value={comment} onChange={e=>setComment(e.target.value)} placeholder="Phạm vi kiểm tra, vùng cần che và giới hạn"/>
            <button className="primary-button" disabled={!ack||!comment.trim()||saving} onClick={save}>Lưu kiểm tra &amp; derivative ảnh này</button>
          </details>}
          <details className="camera-provenance"><summary>Hash, calibration &amp; privacy provenance</summary>
            <p>Source SHA-256 <code>{camera.source_image.sha256}</code></p><p>Source token <code>{camera.sample_data_token}</code></p>
            <p data-testid="camera-privacy-state">{camera.privacy.status}{camera.privacy.reviewed_by?' · '+camera.privacy.reviewed_by.username:''}</p>
            <p>Derivative SHA-256 <code>{camera.privacy.derivative?.sha256||'Chưa có derivative / chưa kiểm tra'}</code></p>
            <pre>{JSON.stringify({intrinsic:camera.intrinsic,camera_from_lidar:camera.camera_from_lidar,privacy:camera.privacy},null,2)}</pre>
          </details>
        </div>}
        <div data-testid="camera-review"><CollectionReviewPanel id={data.scope_id} endpoint={endpoint} task="camera" info={data.review} user={user}
          onChange={review=>setData(previous=>previous?{...previous,review}:previous)}/></div>
      </>}
    </>}
  </section>;
}
