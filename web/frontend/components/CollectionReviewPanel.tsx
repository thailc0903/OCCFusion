'use client';

import { useState } from 'react';
import { downloadCollection, postJSON } from '../lib/api';
import type { CollectionReviewInfo, User } from '../lib/types';

export default function CollectionReviewPanel({ id, info, user, onChange, task='occupancy' }: {
  id: string; info: CollectionReviewInfo; user: User; onChange: (value: CollectionReviewInfo) => void; task?:'occupancy'|'detection';
}) {
  const [comment, setComment] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  async function action(name: string, decision?: string) {
    setError(''); setBusy(true);
    try {
      if (name === 'export') await downloadCollection(id);
      else {
        const body = { comment, ...(name === 'register' || name === 'notes' ? {} : { revision_id: info.revision_id, fingerprint: info.fingerprint }), ...(decision ? { decision } : {}) };
        const value = await postJSON<CollectionReviewInfo>(`/api/collections/${encodeURIComponent(id)}/${name === 'register' ? name : `review/${name}`}`, body);
        onChange(value); setComment('');
      }
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  const disabled = busy || !comment.trim();
  return <section className="card collection-review" data-testid="collection-review">
    <div className="card-title"><h2>Review collection</h2><span className={`review-state state-${info.state}`} data-testid="collection-state">{info.state}</span></div>
    <p>{task==='detection'?'Snapshot riêng của manifest, detection từng frame, prediction nuScenes, protocol, evaluation bất biến và report GPU gốc. Approval occupancy hoặc detection một frame không bao gồm collection này.':'Snapshot riêng của tập frame và pooled evaluation. Approval của run v1/v2 giữ nguyên. Phê duyệt tập con không biến coverage thành complete.'}</p>
    {info.registered && <><p>Revision <b>{info.revision_number}</b> · Owner <b>{info.owner?.username}</b></p><code className="collection-fingerprint">{info.fingerprint}</code>
      <p data-testid="collection-integrity">{info.artifact_matches && info.manifest_matches ? 'Hash artifact & manifest khớp' : `Hash không khớp: ${info.error || 'sidecar đã thay đổi'}`}</p></>}
    {info.permissions.note || info.permissions.register || info.permissions.submit || info.permissions.decide || info.permissions.revise ? <label className="collection-comment">Nhận xét collection<textarea aria-label="Nhận xét collection" value={comment} onChange={e => setComment(e.target.value)} disabled={busy} maxLength={4000}/></label> : null}
    <div className="collection-actions">
      {info.permissions.register && <button className="primary-button" disabled={disabled} onClick={() => action('register')}>Đăng ký collection</button>}
      {info.permissions.note && <button disabled={disabled} onClick={() => action('notes')}>Lưu nhận xét collection</button>}
      {info.permissions.submit && <button className="primary-button" disabled={disabled} onClick={() => action('submit')}>Gửi review collection</button>}
      {info.permissions.revise && <button disabled={disabled} onClick={() => action('revise')}>Tạo revision collection mới</button>}
      {user.role === 'reviewer' && info.permissions.decide && <><button className="primary-button" disabled={disabled} onClick={() => action('decision', 'approved')}>Phê duyệt collection</button><button disabled={disabled} onClick={() => action('decision', 'needs_changes')}>Yêu cầu sửa collection</button><button disabled={disabled} onClick={() => action('decision', 'rejected')}>Từ chối collection</button></>}
      <button disabled={busy || !info.permissions.export} onClick={() => action('export')}>Xuất collection đã duyệt</button>
    </div>
    {error && <p className="inline-error" role="alert">{error}</p>}
    {!!info.revisions?.length && <details><summary>Các revision collection</summary>{info.revisions.map(revision => <p key={revision.id}>r{revision.number} · {revision.state} · <code>{revision.fingerprint.slice(0,16)}…</code></p>)}</details>}
    <details open={!!info.events.length}><summary>Lịch sử collection ({info.events.length})</summary><div data-testid="collection-history">{info.events.map(event => <div className="collection-event" key={event.id}><b>{event.actor_username}</b> · {event.actor_role} · r{event.revision_number} · {event.action} → {event.to_state}<small>{new Date(event.created_at).toLocaleString('vi-VN')}</small><p>{event.comment}</p></div>)}</div></details>
  </section>;
}
