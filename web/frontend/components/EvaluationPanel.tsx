'use client';

import type { PredictionRun } from '../lib/types';

// Artifact metrics are ratios. Intl percent formatting scales exactly once.
const percentage = new Intl.NumberFormat('vi-VN', { style: 'percent', maximumFractionDigits: 4 });
const percent = (value: number) => percentage.format(value);

export default function EvaluationPanel({ run, collection = false }: { run: PredictionRun; collection?: boolean }) {
  const evidence = run.evaluation;
  const evaluation = evidence.status === 'matched' ? evidence.matched : null;
  const covered = !!evaluation && run.review.manifest?.evaluation?.id === evaluation.id && run.review.manifest?.artifacts['evaluation.json']?.sha256 === evaluation.sha256;
  return <section className="card evaluation-card" data-testid="evaluation-panel" data-status={evidence.status}>
    <div className="card-title"><h2>Đánh giá & confidence</h2>{evaluation && <span className="micro-tag">GT ĐÃ GHÉP</span>}</div>
    <div className="metric-row"><span>Occupancy IoU</span><b data-testid="iou-value">{evaluation ? percent(evaluation.metrics.occupancy_iou) : 'Chưa có GT'}</b></div>
    <div className="metric-row"><span>Semantic mIoU</span><b data-testid="miou-value">{evaluation ? percent(evaluation.metrics.semantic_miou) : 'Chưa có GT'}</b></div>
    <div className="evaluation-evidence">
      <p className={`evaluation-binding binding-${evidence.status}`} data-testid="evaluation-binding">{evidence.status} · {evidence.reason}</p>
      {evaluation && <>
        <p data-testid="evaluation-scope"><b>{evaluation.frame_count_evaluated} frame {evaluation.split}</b> · {evaluation.scene} / {evaluation.frame_index}. Kết quả hiện tại chưa phải benchmark toàn split.</p>
        <p className="evaluation-convention">Trung bình đủ {evaluation.semantic_mean_class_count} lớp semantic; union = {evaluation.union_zero_iou} nhận IoU {percent(evaluation.union_zero_iou)}.</p>
        <p className={covered ? 'evaluation-covered' : 'evaluation-pending'} data-testid="evaluation-review-scope">{collection ? 'Evaluation của frame này được gắn với snapshot collection riêng; không mở rộng approval v1/v2 của run cũ.' : covered
          ? `Evaluation thuộc revision ${run.review.revision_number} · ${run.review.state}.`
          : 'Evaluation đã ghép, chưa thuộc revision hiện tại. Approval snapshot cũ không bao gồm evaluation này.'}</p>
        <details className="evaluation-classes"><summary>IoU từng lớp semantic</summary><table><thead><tr><th>Lớp</th><th>IoU</th><th>I / U</th></tr></thead><tbody>{evaluation.metrics.per_class_semantic.map(entry => <tr key={entry.label} data-testid={`evaluation-class-${entry.label}`}><td>{entry.name}</td><td>{percent(entry.iou)}</td><td>{entry.intersection} / {entry.union}</td></tr>)}</tbody></table></details>
        <details className="evaluation-hashes"><summary>Protocol & bằng chứng hash</summary><p>Protocol: <code>{evaluation.protocol_name}</code></p><p>Đơn vị metric: tỷ lệ [0,1], format phần trăm trên giao diện.</p><p>Sample: <code>{evaluation.sample_token}</code></p><p>Prediction SHA-256: <code>{evaluation.prediction_sha256}</code></p><p>GT SHA-256: <code>{evaluation.ground_truth_sha256}</code></p><p>Evaluation SHA-256: <code>{evaluation.sha256}</code></p>{evaluation.reference_check && <p>Đối chiếu evaluator gốc: {evaluation.reference_check.passed ? 'đạt' : 'chưa đạt'}, sai số {evaluation.reference_check.max_abs_difference}.</p>}</details>
      </>}
      {!!evidence.candidates.length && <details className="evaluation-candidates"><summary>Trạng thái các evaluation đã tìm thấy</summary>{evidence.candidates.map(item => <p key={item.id}><code>{item.label}</code> · {item.status}: {item.reason}</p>)}</details>}
    </div>
    <div className="confidence-panel"><div><span>Max softmax probability</span><b className="uncalibrated-label">Chưa hiệu chuẩn</b></div><div className="confidence-scale"/><div className="confidence-range"><span>{run.confidence.min.toFixed(3)}</span><span>{run.confidence.max.toFixed(3)}</span></div><p>Confidence chưa hiệu chuẩn; chưa phải độ bất định.</p></div>
  </section>;
}
