export interface RunSummary {
  run_id: string;
  status: string;
  kind: 'prediction' | 'check_only' | 'failed';
  prediction_available: boolean;
  sample_token: string | null;
  scene_name: string | null;
  frame_index: number | null;
  review_state?: string;
}

export interface User { id: number; username: string; role: 'engineer' | 'reviewer' }
export interface SessionInfo { user: User | null; csrf_token: string; expires_at: number | null; needs_setup: boolean }
export interface ReviewEvent {
  id: number; revision_number: number; actor_username: string; actor_role: string;
  action: string; from_state: string | null; to_state: string; comment: string; created_at: string; fingerprint: string;
}
export interface ReviewInfo {
  registered: boolean; state: string; revision_id?: number; revision_number?: number;
  fingerprint?: string; owner?: { id: number; username: string };
  artifact_matches?: boolean; manifest_matches?: boolean; approval_valid?: boolean; changed_artifacts?: string[];
  permissions: Record<string, boolean>; events: ReviewEvent[];
  evaluation_included?: boolean;
  revisions?: { id: number; number: number; state: string; fingerprint: string; manifest_version: number; evaluation_included: boolean; approved_at: string | null }[];
  manifest?: { schema_version: number; sample_token: string; npz_contains_sample_token: false; evaluation?: { id: string }; artifacts: Record<string, { sha256: string; bytes: number }> };
}

export interface EvaluationMetrics {
  occupancy_iou: number; semantic_miou: number; semantic_mean_class_count: number;
  per_class_semantic: { label: number; name: string; intersection: number; union: number; gt_count: number; prediction_count: number; iou: number }[];
  confusion_matrix: number[][];
  valid_voxel_count: number;
}
export interface MatchedEvaluation {
  id: string; label: string; sha256: string; bytes: number; metric_unit: 'ratio_0_1';
  split: string; scene: string; frame_index: number; sample_token: string; prediction_sha256: string;
  frame_count_requested: number; frame_count_evaluated: number; protocol_name: string; metrics_scope: string;
  semantic_mean_class_count: number; union_zero_iou: number; absent_classes_included_in_mean: boolean;
  ground_truth_sha256: string; metrics: EvaluationMetrics;
  reference_check?: { passed: boolean; max_abs_difference: number };
}
export interface EvaluationEvidence {
  status: string; reason: string; matched: MatchedEvaluation | null;
  candidates: { id: string; label: string; sha256: string; status: string; reason: string }[];
}

export interface SemanticClass {
  id: number;
  name: string;
  color: [number, number, number];
  count: number;
  background: boolean;
}

interface BaseRun {
  run_id: string;
  status: string;
  kind: RunSummary['kind'];
  replay: true;
  inference_enabled: false;
  scene: { name: string; token?: string; index?: number };
  frame_index: number;
  sample_token: string;
  precision: string;
  checkpoint: { path: string; sha256: string; bytes: number } | null;
  timings_s: Record<string, number | null>;
  startup_derived_s: number | null;
  timing_note: string;
  allocator: { peak_allocated_bytes: number | null; peak_reserved_bytes: number | null; scope: string };
  allocator_note: string;
  metrics: { occupancy_iou: number | null; occupancy_miou: number | null; reason: string };
  error: { stage: string; message: string } | null;
  warnings: string[];
  review: ReviewInfo;
  evaluation: EvaluationEvidence;
}

interface PackedData {
  url: string;
  point_count: number;
  fields: string[];
  stride: number;
  dtype: 'float32-le';
}

export interface PredictionRun extends BaseRun {
  prediction_available: true;
  bounds: number[];
  grid_shape: number[];
  grid_axis_order: string[];
  voxel_size: [number, number, number];
  coordinate_system: { name: string; unit: string; axes: string };
  classes: SemanticClass[];
  free_label: number;
  occupancy: PackedData;
  lidar: PackedData & { sample_token: string; original_point_count: number; sample_data_token: string; source: string };
  bev_url: string;
  bev: { rows: string; columns: string; width: number; height: number; projection: string };
  confidence: { min: number; max: number; note: string };
  validation: { npz_token: string; sensor_sample_token: boolean };
}

export type RunDetail = PredictionRun | (BaseRun & { prediction_available: false });

export interface BenchmarkFrame {
  sample_token: string; scene: string; frame_index: number; timestamp_us: number;
  status: string; prediction_available: boolean; url?: string;
  prediction_sha256?: string; evaluation_sha256?: string;
  error?: { stage: string; message: string };
}
export interface RadarAuditFrame {
  sample_token: string; scene: string; frame_index: number; reference_timestamp_us: number;
  rejected_channels: string[]; reason: string; adapter_bug_detected: boolean;
  radars: { channel: string; sample_data_token: string; timestamp_us: number; delta_ms: number;
    prev_token: string; file_exists: boolean; eligible_count: number }[];
}
export interface CollectionReviewInfo extends Omit<ReviewInfo, 'manifest' | 'revisions'> {
  manifest?: { kind: 'occfusion_benchmark_collection'; schema_version: number; successful_frame_count: number;
    expected_frame_count: number; artifacts: Record<string,{sha256: string; bytes: number}> };
  revisions?: { id: number; number: number; state: string; fingerprint: string; approved_at: string | null }[];
  error?: string | null;
}
export interface BenchmarkDetail {
  benchmark_id: string; status: string; split: string; scope: string;
  expected_frame_count: number; successful_frame_count: number; failed_frame_count: number;
  scene_counts: Record<string, number>; metrics: EvaluationMetrics; metric_unit: 'ratio_0_1';
  protocol: { name: string; class_names: string[]; semantic_mean_class_count: number; union_zero_iou: number;
    bounds: number[]; grid_shape: number[]; voxel_size: number[]; grid_axis_order: string[]; [key: string]: unknown };
  frames: BenchmarkFrame[]; radar_audit: { status: string; source: string; sha256: string; frames: RadarAuditFrame[] };
  latency: { sample_count: number; median_s: number | null; p95_s: number | null; source_report: string;
    source_report_sha256: string; source_session_id: string; warmup_count: number; all_forward_sample_count: number;
    min_forward_s: number; max_forward_s: number; scope: string;
    startup: { bootstrap_s: number; shared_setup_s: number; model_initialization_s: number; invocation_wall_s: number } };
  precision: string; tf32: {matmul: boolean; cudnn: boolean};
  allocator: {peak_allocated_bytes: number; peak_reserved_bytes: number; source: string; scope: string};
  evidence_hashes: Record<string,{sha256: string; bytes: number}>;
  playback_note: string; replay: true; inference_enabled: false; review: CollectionReviewInfo;
}
