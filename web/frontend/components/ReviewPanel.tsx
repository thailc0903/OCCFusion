'use client';

import { useEffect, useState } from 'react';
import { downloadApproved, getJSON, postJSON } from '../lib/api';
import type { ReviewInfo, RunDetail, User } from '../lib/types';

const stateLabels: Record<string, string> = {
  unregistered: 'Chưa đăng ký', draft: 'Bản nháp', submitted: 'Đã gửi review',
  approved: 'Đã phê duyệt', rejected: 'Từ chối', needs_changes: 'Cần chỉnh sửa',
};

export default function ReviewPanel({ run, user, onChange }: { run: Pick<RunDetail,'run_id'|'review'|'prediction_available'|'evaluation'> & {task?:string}; user: User; onChange: (review: ReviewInfo) => void }) {
  const [review, setReview] = useState<ReviewInfo>(run.review);
  const [comment, setComment] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  useEffect(() => { setReview(run.review); }, [run.review]);
  function update(value: ReviewInfo) { setReview(value); onChange(value); }
  const base = `/api/runs/${encodeURIComponent(run.run_id)}`;
  async function action(name: string, decision?: string) {
    setBusy(true); setError('');
    try {
      const body = name === 'submit' || name === 'decision' || name === 'evaluation'
        ? { comment, revision_id: review.revision_id, fingerprint: review.fingerprint, ...(decision ? { decision } : {}), ...(name === 'evaluation' ? { evaluation_id: run.evaluation.matched?.id } : {}) }
        : { comment };
      const url = name === 'register' ? `${base}/register` : `${base}/review/${name}`;
      update(await postJSON<ReviewInfo>(url, body)); setComment('');
    } catch (error) {
      setError((error as Error).message);
      update(await getJSON<ReviewInfo>(`${base}/review`).catch(() => review));
    } finally { setBusy(false); }
  }
  async function exportBundle() {
    setBusy(true); setError('');
    try { await downloadApproved(run.run_id); update(await getJSON<ReviewInfo>(`${base}/review`)); }
    catch (error) { setError((error as Error).message); update(await getJSON<ReviewInfo>(`${base}/review`).catch(() => review)); }
    finally { setBusy(false); }
  }
  const invalid = review.registered && (!review.artifact_matches || !review.manifest_matches);
  const evaluation = run.evaluation.matched;
  const canAttach = review.permissions.revise && evaluation && (review.manifest?.evaluation?.id !== evaluation.id || review.manifest?.artifacts['evaluation.json']?.sha256 !== evaluation.sha256);
  const detection=run.task==='detection';
  return <section className="card review-panel" data-testid="review-panel" aria-busy={busy}>
    <div className="review-title"><div><span className="eyebrow">REVIEW WORKFLOW</span><h2>Nhận xét & phê duyệt</h2></div><div className={`review-state state-${review.state}`} data-testid="review-state">{review.state}<span>{stateLabels[review.state]}</span></div></div>
    <div className="review-summary"><span>Vai trò: <b>{user.role === 'engineer' ? 'Engineer' : 'Reviewer'}</b></span>{review.owner && <span>Engineer: <b>{review.owner.username}</b></span>}{review.revision_number && <span>Revision <b>{review.revision_number}</b></span>}{review.fingerprint && <code title={review.fingerprint}>SHA-256 {review.fingerprint.slice(0, 20)}…</code>}</div>
    {invalid && <div className="integrity-warning" role="alert">Bộ artifact hoặc sidecar manifest đã thay đổi. Approval trước đó không áp dụng; export bị chặn. {review.changed_artifacts?.join(', ')}</div>}
    {!run.prediction_available && <p className="review-empty">Run check-only/lỗi không có đủ bộ prediction thành công để đăng ký review.</p>}
    {(review.permissions.note || review.permissions.register && run.prediction_available) && <label className="review-comment">Nhận xét<textarea aria-label="Nhận xét review" value={comment} onChange={event => setComment(event.target.value)} disabled={busy} placeholder="Ghi nội dung kiểm tra, điểm cần sửa hoặc lý do quyết định…" maxLength={4000} rows={3}/></label>}
    <div className="review-actions">
      {review.permissions.register && run.prediction_available && <button className="primary-button" disabled={busy || !comment.trim()} onClick={() => action('register')}>Đăng ký run</button>}
      {review.permissions.note && <button disabled={busy || !comment.trim()} onClick={() => action('notes')}>Lưu ghi chú</button>}
      {review.permissions.submit && <button className="primary-button" disabled={busy || !comment.trim()} onClick={() => action('submit')}>Gửi review</button>}
      {review.permissions.decide && <><button className="primary-button" disabled={busy || !comment.trim()} onClick={() => action('decision', 'approved')}>Phê duyệt</button><button disabled={busy || !comment.trim()} onClick={() => action('decision', 'needs_changes')}>Cần chỉnh sửa</button><button className="reject-button" disabled={busy || !comment.trim()} onClick={() => action('decision', 'rejected')}>Từ chối</button></>}
      {review.permissions.revise && <button disabled={busy || !comment.trim()} onClick={() => action('revise')}>Tạo revision nháp mới</button>}
      {canAttach && <button className="primary-button" disabled={busy || !comment.trim()} onClick={() => action('evaluation')}>Thêm evaluation vào revision mới</button>}
      {review.registered && <button className="export-button" disabled={busy || !review.permissions.export} onClick={exportBundle}>Xuất gói đã duyệt</button>}
    </div>
    {error && <p className="auth-error" role="alert">{error}</p>}
    {review.registered && <p className="footnote" data-testid="revision-export-scope">Snapshot hiện tại: {detection?'Detection JSON + metadata/report/resolved config; mAP pending':review.evaluation_included ? '4 artifact inference + evaluation.json' : '4 artifact inference; chưa bao gồm evaluation'}. Chỉ xuất khi revision này approved và toàn bộ hash còn khớp.</p>}
    {review.revisions && <div className="revision-history" data-testid="revision-history">{review.revisions.map(item => <p key={item.id}>r{item.number} · {item.state} · manifest v{item.manifest_version} · {item.evaluation_included ? 'có evaluation' : '4 artifact'}<code title={item.fingerprint}>{item.fingerprint.slice(0, 16)}…</code></p>)}</div>}
    {review.manifest && <details className="manifest-details"><summary>Manifest của bộ artifact · token từ metadata/report</summary><p>Sample: <code>{review.manifest.sample_token}</code>. {detection?'Token đối chiếu detections.json/metadata/report.':'NPZ không chứa sample token riêng.'}</p>{Object.entries(review.manifest.artifacts).map(([name, value]) => <div key={name}><span>{name}</span><code title={value.sha256}>{value.sha256}</code><small>{value.bytes.toLocaleString('vi-VN')} byte</small></div>)}</details>}
    <div className="review-history"><h3>Lịch sử review</h3>{!review.events.length && <p>Chưa có sự kiện review.</p>}{review.events.map(event => <article key={event.id}><div className="history-dot"/><div><div className="history-meta"><b>{event.actor_username}</b><span>{event.actor_role}</span><span>r{event.revision_number} · {event.action}</span><time>{new Date(event.created_at).toLocaleString('vi-VN', { timeZone: 'Asia/Ho_Chi_Minh', hour12: false })}</time></div><div className="history-transition">{event.from_state || 'unregistered'} → {event.to_state}</div><p>{event.comment}</p></div></article>)}</div>
  </section>;
}
