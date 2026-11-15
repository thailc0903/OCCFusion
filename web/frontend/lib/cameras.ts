import type {CollectionReviewInfo} from './types';
export type Mask=[number,number,number,number];
export interface Camera {
  channel:string;status:'available'|'blocked';reason:string|null;
  sample_token:string;sample_data_token:string;width:number;height:number;
  camera_timestamp_us:number;lidar_timestamp_us:number;camera_minus_lidar_ms:number;
  source_image:{sha256:string;bytes:number};intrinsic:number[][];camera_from_lidar:number[][];
  camera_fingerprint:string;image_fingerprint:string;image_url:string|null;projection_url:string;
  export_eligible:boolean;
  privacy:{status:string;revision:string|null;masks:Mask[];reviewed_by:{username:string}|null;reviewed_at?:string;
    derivative:{sha256:string;bytes:number}|null;source_policy:{publisher_documentation:string;publisher_processing:string}};
}
export interface CameraListing {
  scope_id:string;sample_token:string;scene:{name:string};frame_index:number;channels:Camera[];
  projection_protocol:{precision:string;motion_limit:string;visibility_limit:string;[key:string]:unknown};
  prediction_box_count:number;review:CollectionReviewInfo;
}
export interface CameraProjection {
  sample_token:string;sample_data_token:string;channel:string;width:number;height:number;
  points:number[][];point_count:number;boxes:{index:number;label:number;score:number;segments:number[][]}[];
  box_count:number;class_names:string[];box_source:string;fingerprint_sha256:string;
}
