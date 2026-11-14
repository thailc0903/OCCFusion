import type {ReviewInfo,SemanticClass,EvaluationEvidence} from './types';
export interface DetectionBox { center: number[]; dimensions: number[]; yaw: number; velocity: number[]; score: number; label: number }
export interface DetectionRun {
 run_id: string; task: 'detection'; fixture?:boolean; prediction_available: true; scene: {name:string}; frame_index:number; sample_token:string;
 timestamp_us:number; reference_timestamp_us:number; bounds:number[]; voxel_size:[number,number,number]; classes:SemanticClass[];
 boxes:DetectionBox[]; box_count:number; source:string; precision:string; checkpoint:{path:string;sha256:string};
 timings_s:Record<string,number>; allocator:{peak_allocated_bytes:number;peak_reserved_bytes:number};
 metrics:{mAP:null;NDS:null;status:string}; ground_truth:{detection_annotations_available:boolean};
 lidar:{url:string;point_count:number;stride:number}; review:ReviewInfo; evaluation:EvaluationEvidence;
}
export function boxCorners(box:DetectionBox):number[][] {
 const c=Math.cos(box.yaw),s=Math.sin(box.yaw);const [dx,dy,dz]=box.dimensions;
 return [[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1],[1,-1,-1],[1,-1,1],[1,1,1],[1,1,-1]].map(([x,y,z])=>[
  box.center[0]+c*x*dx/2-s*y*dy/2,box.center[1]+s*x*dx/2+c*y*dy/2,box.center[2]+z*dz/2]);
}
