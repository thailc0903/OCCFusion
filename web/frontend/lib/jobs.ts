export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'canceled';
export interface InferenceJob {
  id: string; profile_id: string; profile_label: string; scene_name: string;
  frame_index: number; sample_token: string; split: string; status: JobStatus;
  phase: string; created_at: string; started_at: string | null; finished_at: string | null;
  cancel_requested: boolean; worker_stale: boolean; inference_launch_count: number;
  execution_mode: 'real' | 'fixture'; evaluation_status: string | null;
  result_url: string | null; precision: string; tf32: { matmul: boolean; cudnn: boolean };
  config: string; checkpoint: string; checkpoint_sha256: string;
  events: { id: number; phase: string; message: string; created_at: string }[];
  error: { stage: string; type: string; message: string } | null;
  result: { timings_s: Record<string, number>; evaluation_status: string; approved: boolean } | null;
}
export interface JobOptions {
  profiles: { id: string; label: string; available: boolean; unavailable_reason: string | null;
    config: string; checkpoint: string; precision: string; fingerprint: string;
    tf32: { matmul: boolean; cudnn: boolean } }[];
  scenes_by_profile: Record<string, { name: string; split: string; frames: {
    frame_index: number; sample_token: string; timestamp_us: number }[] }[]>;
}
export interface JobList {
  jobs: InferenceJob[];
  workers: { id: string; status: string; execution_mode: string; heartbeat_age_s: number }[];
}
