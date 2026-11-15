#!/usr/bin/env python3
"""CPU-only independent nuScenes-devkit reference for six saved sample cameras."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def stats(values):
    values=np.asarray(values,dtype=np.float64)
    return dict(count=len(values),median_px=float(np.median(values)) if len(values) else None,
        p95_px=float(np.percentile(values,95)) if len(values) else None,max_px=float(np.max(values)) if len(values) else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-state-root',type=Path,default=ROOT/'work/camera-real-source-qa/state')
    parser.add_argument('--out-dir',type=Path,default=ROOT/'work_dirs/web_camera_qa')
    args=parser.parse_args()
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud,Box
    from nuscenes.utils.geometry_utils import view_points
    from pyquaternion import Quaternion
    from web.backend.app.camera_projection import CHANNELS,camera_from_lidar,project_points,box_corners,transform,PROTOCOL
    token='3950bd41f74548429c0f7700ff3d8269'
    data=ROOT/'data/nuscenes'
    nusc=NuScenes(version='v1.0-mini',dataroot=str(data),verbose=False)
    sample=nusc.get('sample',token);lidar=nusc.get('sample_data',sample['data']['LIDAR_TOP'])
    lc=nusc.get('calibrated_sensor',lidar['calibrated_sensor_token']);le=nusc.get('ego_pose',lidar['ego_pose_token'])
    raw=LidarPointCloud.from_file(str(data/lidar['filename'])).points
    predictions=json.loads((ROOT/'work/centerpoint-smoke-wyp1nl__/runs/infer_job_65e1b30833e343c3b434da8dc71423f9/detections.json').read_text())
    results=[]
    sdk_order=[7,5,4,6,3,1,0,2]
    for channel in CHANNELS:
        sd=nusc.get('sample_data',sample['data'][channel])
        cc=nusc.get('calibrated_sensor',sd['calibrated_sensor_token']);ce=nusc.get('ego_pose',sd['ego_pose_token'])
        pin=json.loads((args.source_state_root/'cameras/sources'/(sd['token']+'.json')).read_text())
        source=(data/sd['filename']).read_bytes()
        assert hashlib.sha256(source).hexdigest()==pin['evidence']['source_image']['sha256']
        matrix=camera_from_lidar(lc,le,cc,ce)
        ours=project_points(raw[:3].T,matrix,cc['camera_intrinsic'],sd['width'],sd['height'])
        errors={};memberships={}
        for precision in ('float32','float64'):
            pc=LidarPointCloud(raw.astype(precision,copy=True))
            pc.rotate(Quaternion(lc['rotation']).rotation_matrix);pc.translate(np.array(lc['translation']))
            pc.rotate(Quaternion(le['rotation']).rotation_matrix);pc.translate(np.array(le['translation']))
            pc.translate(-np.array(ce['translation']));pc.rotate(Quaternion(ce['rotation']).rotation_matrix.T)
            pc.translate(-np.array(cc['translation']));pc.rotate(Quaternion(cc['rotation']).rotation_matrix.T)
            uv=view_points(pc.points[:3],np.asarray(cc['camera_intrinsic']),normalize=True)
            mask=(pc.points[2]>1)&(uv[0]>1)&(uv[0]<sd['width']-1)&(uv[1]>1)&(uv[1]<sd['height']-1)
            ids=np.flatnonzero(mask);common,a,b=np.intersect1d(ours[:,0].astype(int),ids,return_indices=True)
            pixel_error=np.linalg.norm(ours[a,1:3]-uv[:2,ids[b]].T,axis=1)
            errors[precision]=stats(pixel_error)
            web_ids=set(ours[:,0].astype(int).tolist());reference_ids=set(ids.tolist())
            boundary=sorted(web_ids^reference_ids)
            all_web=transform(raw[:3].T,matrix)
            memberships[precision]=dict(web_count=len(ours),reference_count=len(ids),common=len(common),
                symmetric_difference=len(boundary),web_only=sorted(web_ids-reference_ids),reference_only=sorted(reference_ids-web_ids),
                boundary_points=[dict(index=i,web_uv=((np.asarray(cc['camera_intrinsic'])@all_web[i])[:2]/all_web[i,2]).tolist(),
                    web_depth=float(all_web[i,2]),reference_uv=uv[:2,i].tolist()) for i in boundary])
            if precision=='float64':assert not boundary
            if precision=='float32':
                official,_,_=nusc.explorer.map_pointcloud_to_image(lidar['token'],sd['token'])
                np.testing.assert_allclose(official[:2],uv[:2,mask],rtol=0,atol=1e-12)
        box_errors=[]
        for saved in predictions['boxes']:
            b=Box(saved['center'],[saved['dimensions'][1],saved['dimensions'][0],saved['dimensions'][2]],Quaternion(axis=[0,0,1],angle=saved['yaw']))
            b.rotate(Quaternion(lc['rotation']));b.translate(np.array(lc['translation']))
            b.rotate(Quaternion(le['rotation']));b.translate(np.array(le['translation']))
            b.translate(-np.array(ce['translation']));b.rotate(Quaternion(ce['rotation']).inverse)
            b.translate(-np.array(cc['translation']));b.rotate(Quaternion(cc['rotation']).inverse)
            ref=b.corners();actual=transform(box_corners(saved),matrix)[sdk_order].T
            front=(ref[2]>.1)&(actual[2]>.1)
            if front.any():
                a=view_points(actual[:,front],np.asarray(cc['camera_intrinsic']),True)
                r=view_points(ref[:,front],np.asarray(cc['camera_intrinsic']),True)
                box_errors.extend(np.linalg.norm(a[:2]-r[:2],axis=0).tolist())
        assert errors['float64']['max_px']<1e-6
        assert errors['float32']['max_px']<.5
        assert max(box_errors,default=0)<1e-5
        results.append(dict(channel=channel,sample_data_token=sd['token'],width=sd['width'],height=sd['height'],
            camera_timestamp_us=sd['timestamp'],lidar_timestamp_us=lidar['timestamp'],
            camera_minus_lidar_ms=(sd['timestamp']-lidar['timestamp'])/1000,source_image=pin['evidence']['source_image'],
            point_errors=errors,frustum_membership=memberships,box_corner_float64=stats(box_errors),
            boxes_compared=len(predictions['boxes'])))
    forbidden=[n for n in ('torch','mmcv','mmdet','mmdet3d','occfusion','tools.evaluate_detection') if n in sys.modules]
    assert not forbidden
    report=dict(status='passed',sample_token=token,scene='scene-0103',frame_index=1,protocol=PROTOCOL,
        reference='Installed nuscenes-devkit 1.1.11 map_pointcloud_to_image, LidarPointCloud, Box, Quaternion and view_points',
        precision_note='Web combines SE(3) in float64. Official PointCloud uses in-place float32; a second SDK sequence uses float64. Pixel errors are measured, not assumed zero.',
        images=results,model_modules_loaded=forbidden,evaluator_called=False,inference_called=False)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/'projection_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='passed',images=len(results),float32_max_pixel_error=max(r['point_errors']['float32']['max_px'] for r in results),
        float64_max_pixel_error=max(r['point_errors']['float64']['max_px'] for r in results)),indent=2))


if __name__=='__main__':main()
