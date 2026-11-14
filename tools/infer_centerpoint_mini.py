#!/usr/bin/env python3
"""One CenterPoint frame, official v1.4 test pipeline, no annotation/GT input."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.infer_nuscenes_mini import file_hash,select_sample,resolve_scene_index,strict_load_checkpoint,write_json


def initialize_cuda(torch,device):
    """Initialize the selected device before touching its allocator statistics."""
    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)


def frame_info(nusc,scene_name,frame_index):
    import numpy as np
    from nuscenes.utils.geometry_utils import transform_matrix
    from nuscenes.utils.splits import create_splits_scenes
    from pyquaternion import Quaternion
    scene,sample=select_sample(nusc,resolve_scene_index(nusc,scene_name=scene_name),frame_index)
    ref=nusc.get('sample_data',sample['data']['LIDAR_TOP'])
    def record(sd):
        if nusc.get('sample',sd['sample_token'])['scene_token']!=scene['token']:raise ValueError('Sweep crossed scene')
        cal=nusc.get('calibrated_sensor',sd['calibrated_sensor_token']);ego=nusc.get('ego_pose',sd['ego_pose_token'])
        if nusc.get('sensor',cal['sensor_token'])['channel']!='LIDAR_TOP':raise ValueError('Not LIDAR_TOP')
        path=Path(nusc.get_sample_data_path(sd['token'])).resolve()
        if not path.is_file() or path.stat().st_size==0 or path.stat().st_size%20:raise FileNotFoundError(f'Missing/invalid LiDAR file: {path}')
        a=transform_matrix(cal['translation'],Quaternion(cal['rotation']));b=transform_matrix(ego['translation'],Quaternion(ego['rotation']))
        return dict(channel='LIDAR_TOP',sample_data_token=sd['token'],path=str(path),timestamp_us=sd['timestamp'],sha256=file_hash(path),bytes=path.stat().st_size,
            calibrated_sensor_token=cal['token'],ego_pose_token=ego['token'],sensor_to_ego=a.tolist(),ego_to_global=b.tolist()),b@a
    key,world=record(ref);key['sensor_to_reference_lidar']=np.eye(4).tolist()
    sweeps=[];records=[];token=ref['prev'];seen={ref['token']};last=ref['timestamp']
    while token and len(sweeps)<9:
        if token in seen:raise ValueError('Cycle in sweep prev chain')
        seen.add(token);sd=nusc.get('sample_data',token)
        if sd['timestamp']>=last or sd['timestamp']>ref['timestamp']:raise ValueError('Sweep is future/nondecreasing')
        last=sd['timestamp'];info,pose=record(sd);physical=np.linalg.inv(world)@pose
        info['sensor_to_reference_lidar']=physical.tolist();records.append(info)
        # Match update_infos_to_v2.py and LoadPointsFromMultiSweeps exactly.
        # The installed loader applies row_xyz @ R - t, recorded explicitly.
        inverse=np.linalg.inv(physical)
        info['official_loader_translation_difference_m']=(-inverse[:3,3]-physical[:3,3]).tolist()
        sweeps.append(dict(timestamp=sd['timestamp']/1e6,lidar_points=dict(lidar_path=info['path'],lidar2sensor=inverse.tolist())))
        token=sd['prev']
    split=next(k for k in ('mini_train','mini_val') if scene_name in create_splits_scenes()[k])
    metadata=dict(task='detection',schema_version=1,sample_token=sample['token'],timestamp_us=sample['timestamp'],reference_timestamp_us=ref['timestamp'],
        scene=dict(name=scene_name,token=scene['token'],index=resolve_scene_index(nusc,scene_name=scene_name)),frame_index=frame_index,
        dataset=dict(root=str(Path(nusc.dataroot).resolve()),version=nusc.version,split=split),
        coordinate_system=dict(name='LIDAR_TOP at reference_timestamp_us',unit='metres',axes='native right-handed LIDAR_TOP xyz; Z up',
            lidar_to_ego=key['sensor_to_ego'],lidar_to_global=world.tolist()),sensors=dict(keyframes=dict(LIDAR_TOP=key),lidar_sweeps=records),
        sensor_configuration=dict(use_camera=False,use_lidar=True,use_radar=False),ground_truth=dict(detection_annotations_available=bool(sample.get('anns')),input_used=False),
        sweep_protocol=dict(requested_past=9,actual_past=len(records),future_used=False,missing_sweeps='official pad_empty_sweeps=True only when history absent',
            transform='official v1.4 inverse lidar2sensor metadata + installed loader row_xyz @ R - t; unchanged',
            limitation='Installed v1.4 sweep loader subtraction differs from complete inverse SE(3); physical transforms are retained separately. No silent baseline correction.'))
    inputs=dict(lidar_points=dict(lidar_path=key['path']),timestamp=ref['timestamp']/1e6,
        box_type_3d=__import__('mmdet3d.structures',fromlist=['LiDARInstance3DBoxes']).LiDARInstance3DBoxes,
        box_mode_3d=__import__('mmdet3d.structures',fromlist=['Box3DMode']).Box3DMode.LIDAR,sample_idx=sample['token'])
    if sweeps:inputs['lidar_sweeps']=sweeps
    return inputs,metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data-root',default=str(ROOT/'data/nuscenes'));parser.add_argument('--scene-name',required=True)
    parser.add_argument('--frame-index',type=int,required=True);parser.add_argument('--out-dir',required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();out=Path(args.out_dir).resolve()
    if out.exists():raise FileExistsError('Use a new output directory; overwrite forbidden')
    out.mkdir(parents=True);start=time.perf_counter();phase='imports';metadata=None
    report=dict(task='detection',status='failed',check_only=args.check_only,command=[sys.executable,*sys.argv],started_at_utc=datetime.now(timezone.utc).isoformat(),
        precision='float32',autocast=False,forward_calls=0,forward_completed=False,warmup_forward_calls=0,timings_s={},vram={},metrics=dict(mAP=None,NDS=None,status='pending_dataset_evaluation',reason='One frame; full detection dataset evaluation has not run.'))
    try:
        import numpy as np
        import torch
        from mmengine.config import Config
        from mmengine.dataset import Compose,pseudo_collate
        from mmdet3d.registry import MODELS
        from mmdet3d.utils import register_all_modules
        from nuscenes.nuscenes import NuScenes
        register_all_modules(init_default_scope=True)
        cfg=Config.fromfile(args.config)
        assert cfg.model.type=='CenterPoint' and cfg.model.test_cfg.pts.nms_type=='circle'
        assert list(cfg.voxel_size)==[.2,.2,8] and cfg.model.pts_bbox_head.type=='CenterHead'
        phase='input_validation';t=time.perf_counter()
        nusc=NuScenes(version='v1.0-mini',dataroot=args.data_root,verbose=False)
        inputs,metadata=frame_info(nusc,args.scene_name,args.frame_index)
        cfg.dump(str(out/'resolved_config.py'))
        provenance=dict(config=dict(path=str(Path(args.config).resolve()),sha256=file_hash(Path(args.config)),resolved_sha256=file_hash(out/'resolved_config.py')),
            checkpoint=dict(path=str(Path(args.checkpoint).resolve()),sha256=file_hash(Path(args.checkpoint)),bytes=Path(args.checkpoint).stat().st_size),
            code=dict(inference_script_sha256=file_hash(Path(__file__))),framework=dict(mmdet3d='1.4.0',torch=torch.__version__))
        metadata.update(**provenance,bounds=list(cfg.point_cloud_range),voxel_size=list(cfg.voxel_size),class_names=list(cfg.class_names),
            box_convention=dict(center='geometric',dimensions=['dx','dy','dz'],yaw='radians, CCW about +Z, zero along +X',velocity=['vx','vy'],velocity_unit='m/s'),
            test_pipeline=cfg.test_pipeline,model_source='CenterPoint PointPillars 0.2m circle NMS / LiDAR-only',result_source='prediction',metrics=report['metrics'])
        report.update(**provenance,sample_token=metadata['sample_token'],scene_name=args.scene_name,scene_index=metadata['scene']['index'],frame_index=args.frame_index)
        report['timings_s']['input_validation']=time.perf_counter()-t
        phase='model_checkpoint_cpu';t=time.perf_counter();model=MODELS.build(cfg.model)
        strict_load_checkpoint(model,Path(args.checkpoint),report);model.eval()
        report['timings_s'][phase]=time.perf_counter()-t
        if not args.check_only:
            phase='device_setup';t=time.perf_counter();device=torch.device(args.device)
            if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('CUDA required; no fallback')
            initialize_cuda(torch,device);model.to(device).eval();torch.cuda.synchronize(device)
            report['timings_s']['device_setup']=time.perf_counter()-t
            report['tf32']=dict(matmul=torch.backends.cuda.matmul.allow_tf32,cudnn=torch.backends.cudnn.allow_tf32)
        pipeline_start=time.perf_counter();phase='preprocessing';t=time.perf_counter()
        packed=Compose(cfg.test_pipeline)(inputs);batch=pseudo_collate([packed])
        points=packed['inputs']['points'];assert points.numel()>0 and torch.isfinite(points).all()
        report['point_count']=len(points);report['timings_s']['preprocessing']=time.perf_counter()-t
        if args.check_only:
            report['status']='check_only_passed'
        else:
            phase='device_preprocessing';t=time.perf_counter()
            with torch.inference_mode():
                processed=model.data_preprocessor(batch,training=False)
                torch.cuda.synchronize(device);report['timings_s']['preprocessing']+=time.perf_counter()-t
                phase='forward';t=time.perf_counter()
                report['forward_calls']+=1
                print('CenterPoint forward invocation 1; no warm-up, no retry.',flush=True)
                output=model(**processed,mode='predict')
                torch.cuda.synchronize(device);report['timings_s']['forward']=time.perf_counter()-t
                report['forward_completed']=True
            phase='export';t=time.perf_counter();prediction=output[0].pred_instances_3d
            b=prediction.bboxes_3d;centers=b.gravity_center.cpu().numpy();dims=b.dims.cpu().numpy();tensor=b.tensor.cpu().numpy()
            scores=prediction.scores_3d.cpu().numpy();labels=prediction.labels_3d.cpu().numpy()
            if not all(np.isfinite(a).all() for a in [centers,dims,tensor,scores]):raise ValueError('Non-finite detection')
            boxes=[dict(center=c.tolist(),dimensions=s.tolist(),yaw=float(row[6]),velocity=row[7:9].tolist(),score=float(score),label=int(label)) for c,s,row,score,label in zip(centers,dims,tensor,scores,labels)]
            doc=dict(schema_version=1,task='detection',sample_token=metadata['sample_token'],timestamp_us=metadata['timestamp_us'],reference_timestamp_us=metadata['reference_timestamp_us'],
                coordinate_system=metadata['coordinate_system'],class_names=metadata['class_names'],box_convention=metadata['box_convention'],boxes=boxes,provenance=provenance,postprocessing=dict(nms_type='circle',config_score_threshold=.1,display_filter_applied=False))
            write_json(out/'detections.json',doc);report['box_count']=len(boxes);report['status']='success'
            report['timings_s']['postprocessing_export']=time.perf_counter()-t
            report['timings_s']['pipeline_total']=time.perf_counter()-pipeline_start
            report['vram']=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),scope='model setup and one-frame detection pipeline; allocator, not physical residency')
            del output,prediction,b,processed,model
    except Exception as error:
        report['error']=dict(stage=phase,type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        if 'device' in locals() and torch.cuda.is_initialized():
            report['vram']=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        report['status']='failed';print(report['error']['message'],flush=True)
    finally:
        report['timings_s']['process_until_report']=time.perf_counter()-start
        if metadata is not None:
            metadata.update(status=report['status'],precision='float32',autocast=False);write_json(out/'metadata.json',metadata)
        write_json(out/'report.json',report)
    print('Detection report:',out/'report.json',report['status'],flush=True)
    return 0 if report['status'] in ('success','check_only_passed') else 1

if __name__=='__main__':raise SystemExit(main())
