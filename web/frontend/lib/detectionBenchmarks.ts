import type {BenchmarkFrame, CollectionReviewInfo, PredictionRun, SemanticClass} from './types';
import type {DetectionBox} from './detections';
export type TPError = Record<string, number | null>;
export interface Distribution {sample_count:number;median_s:number|null;p95_s:number|null}
export interface DetectionBenchmarkDetail {
  task:'detection';benchmark_id:string;status:string;split:string;scope:string;
  expected_frame_count:number;successful_frame_count:number;failed_frame_count:number;scene_counts:Record<string,number>;
  model_source:string;model:{checkpoint:{path:string;sha256:string};config:{path:string;sha256:string}};
  sensor_configuration:{use_lidar:boolean;use_camera:boolean;use_radar:boolean};
  metrics:{mAP:number;NDS:number;tp_errors:TPError};
  classes:{name:string;ap:number;gt_count:number;gt_count_before_filters:number;has_gt:boolean;
    predicted_count_after_filters:number;tp_errors:TPError}[];
  support:{saved_prediction_box_count:number;official_filtered_box_count:number;display_score_filter_applied:false};
  metric_unit:'ratio_0_1';protocol:{name:string;evaluator:string;devkit_version:string;[key:string]:unknown};
  frames:(BenchmarkFrame&{box_count:number;origin:string})[];
  latency:{all_completed_forwards:Distribution;excluding_marked_warmup:Distribution;
    end_to_end_excluding_marked_warmup:Distribution;sample_count:number;all_forward_sample_count:number;
    warmup_count:number;new_prediction_count:number;smoke_cache_count:number;source_report:string;
    source_report_sha256:string;source_session_id:string;scope:string;end_to_end_scope:string;
    startup:{common_cpu_s:number;model_checkpoint_cpu_s:number;device_setup_s:number;invocation_wall_s:number}};
  loader_audit:{sweep_count:number;loader_vs_se3_translation_m:{max:number;median:number;p95:number};
    worst_frame:{sample_token:string;scene:string;frame_index:number;maximum_m:number};limitation:string};
  precision:string;tf32:{matmul:boolean;cudnn:boolean};
  allocator:{peak_allocated_bytes:number;peak_reserved_bytes:number;source:string;scope:string};
  evaluation_source:{snapshot:string;immutable:true;evaluator:string;devkit_version:string};
  evidence_hashes:Record<string,{sha256:string;bytes:number}>;review:CollectionReviewInfo;
  replay:true;inference_enabled:false;playback_note:string;
}
export interface DetectionBenchmarkFrame {
  task:'detection';collection_id:string;sample_token:string;frame_index:number;scene:{name:string};
  timestamp_us:number;reference_timestamp_us:number;prediction_available:true;replay:true;inference_enabled:false;
  bounds:number[];voxel_size:[number,number,number];classes:SemanticClass[];
  boxes:DetectionBox[];box_count:number;origin:string;source:string;precision:string;
  lidar:{url:string;point_count:number;stride:number};timings_s:Record<string,number>;
  artifacts:Record<string,{sha256:string;bytes:number}>;bundle_fingerprint:string;occupancy_overlay?:PredictionRun;
}
