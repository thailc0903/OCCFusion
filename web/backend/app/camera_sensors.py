"""Verify original camera pixels and timestamp-specific poses from allowed roots."""
from io import BytesIO
import json
from pathlib import PurePosixPath
import numpy as np
from PIL import Image, UnidentifiedImageError

from .benchmarks import record, fingerprint
from .camera_projection import CHANNELS, camera_from_lidar
from .store import contained, InvalidArtifact, quaternion_pose

TABLES=('sample','sample_data','scene','sensor','calibrated_sensor','ego_pose')


def require(value,message):
    if not value:raise InvalidArtifact(message)


def image_size(content):
    try:
        with Image.open(BytesIO(content)) as image:
            require(image.format in ('JPEG','PNG'),'Unsupported source image format')
            require(image.width*image.height <= 16_000_000,'Camera image too large')
            size=image.size;image.verify()
        with Image.open(BytesIO(content)) as image:
            require(image.getexif().get(274,1)==1,'Rotated EXIF image is incompatible with raw intrinsics')
            image.load()
        return size
    except (OSError,UnidentifiedImageError) as error:
        raise InvalidArtifact('Camera image cannot be decoded: '+str(error)) from error


def camera_frame(catalog,metadata):
    # Reuse the reference LiDAR's existing scene/frame/pose verification.
    catalog.lidar(metadata,metadata['bounds'])
    version=metadata['dataset']['version'];token=metadata['sample_token']
    roots=[]
    for root in catalog.settings.dataset_roots:
        if not (root/version/'sample.json').is_file():continue
        raw={name:contained(root/version/(name+'.json'),root).read_bytes() for name in TABLES}
        hashes={name:record(data) for name,data in raw.items()}
        key=(root,version,fingerprint(hashes))
        cache=getattr(catalog,'camera_indexes',{})
        if key not in cache:
            cache={key:{name:{r['token']:r for r in json.loads(content)} for name,content in raw.items()}}
            catalog.camera_indexes=cache
        tables=cache[key]
        if token in tables['sample']:roots.append((root,tables,hashes))
    require(len(roots)==1,'Camera sample missing/ambiguous across configured dataset roots')
    root,tables,hashes=roots[0]
    lidar=tables['sample_data'][metadata['sensors']['keyframes']['LIDAR_TOP']['sample_data_token']]
    cal=tables['calibrated_sensor'][lidar['calibrated_sensor_token']]
    ego=tables['ego_pose'][lidar['ego_pose_token']]
    require(ego['timestamp']==lidar['timestamp'],'LiDAR ego_pose timestamp differs from reference scan')
    lidar_path=contained(root/lidar['filename'],root)
    lidar_bytes=lidar_path.read_bytes()
    require(lidar_bytes and len(lidar_bytes)%20==0,'Invalid reference LiDAR binary')
    points=np.frombuffer(lidar_bytes,dtype='<f4').reshape(-1,5)[:,:3]
    require(np.isfinite(points).all(),'Nonfinite LiDAR projection input')
    return dict(root=root,tables=tables,table_hashes=hashes,metadata=metadata,lidar=lidar,
        lidar_cal=cal,lidar_ego=ego,lidar_file=record(lidar_bytes),points=points)


def camera_record(frame,channel):
    require(channel in CHANNELS,'Unsupported camera channel')
    root,tables,meta=(frame[k] for k in ('root','tables','metadata'))
    token=meta['sample_token']
    matches=[]
    for sd in tables['sample_data'].values():
        if sd['sample_token']!=token or not sd['is_key_frame']:continue
        cal=tables['calibrated_sensor'][sd['calibrated_sensor_token']]
        sensor=tables['sensor'][cal['sensor_token']]
        if sensor['channel']==channel and sensor['modality']=='camera':matches.append((sd,cal))
    require(len(matches)==1,channel+': camera keyframe missing or ambiguous for sample')
    sd,cal=matches[0];ego=tables['ego_pose'][sd['ego_pose_token']]
    require(sd['fileformat']=='jpg' and sd['width']>0 and sd['height']>0,'Invalid camera sample_data image schema')
    require(ego['timestamp']==sd['timestamp'], 'Camera ego_pose timestamp differs from exposure')
    intrinsic=np.asarray(cal['camera_intrinsic'],dtype=np.float64)
    require(intrinsic.shape==(3,3) and np.isfinite(intrinsic).all() and intrinsic[0,0]>0 and intrinsic[1,1]>0
        and np.allclose(intrinsic[2],[0,0,1],rtol=0,atol=1e-12),'Invalid camera intrinsic')
    matrix=camera_from_lidar(frame['lidar_cal'],frame['lidar_ego'],cal,ego)
    declared=meta['sensors']['keyframes'].get(channel)
    if declared is not None:
        require(declared['channel']==channel and declared['sample_data_token']==sd['token']
            and declared['timestamp_us']==sd['timestamp'] and declared['calibrated_sensor_token']==cal['token']
            and declared['ego_pose_token']==ego['token'] and declared['width']==sd['width'] and declared['height']==sd['height'],
            channel+': inference/dataset camera token/timestamp/calibration/size mismatch')
        relative=PurePosixPath(declared['path']).relative_to(PurePosixPath(meta['dataset']['root']))
        require(str(relative)==sd['filename'],'Camera filename differs from inference provenance')
        for actual,expected in ((declared['intrinsic'],intrinsic),
                (declared['sensor_to_ego'],quaternion_pose(cal['rotation'],cal['translation'])),
                (declared['ego_to_global'],quaternion_pose(ego['rotation'],ego['translation'])),
                (declared['sensor_to_reference_lidar'],np.linalg.inv(matrix))):
            require(np.asarray(actual).shape==expected.shape and np.allclose(actual,expected,rtol=0,atol=1e-6),
                channel+': inference camera calibration differs from dataset')
    path=contained(root/sd['filename'],root);content=path.read_bytes()
    require(image_size(content)==(sd['width'],sd['height']), 'Decoded original image dimensions differ from sample_data')
    evidence=dict(schema_version=1,sample_token=token,scene=meta['scene']['name'],frame_index=meta['frame_index'],
        dataset_version=meta['dataset']['version'],channel=channel,sample_data_token=sd['token'],
        filename=sd['filename'],source_image=record(content),width=sd['width'],height=sd['height'],
        camera_timestamp_us=sd['timestamp'],lidar_timestamp_us=frame['lidar']['timestamp'],
        camera_minus_lidar_ms=(sd['timestamp']-frame['lidar']['timestamp'])/1000,
        intrinsic=intrinsic.tolist(),calibrated_sensor=cal,ego_pose=ego,
        lidar_sample_data_token=frame['lidar']['token'],lidar_calibrated_sensor=frame['lidar_cal'],
        lidar_ego_pose=frame['lidar_ego'],lidar_source=frame['lidar_file'],
        camera_from_lidar=matrix.tolist(),dataset_table_hashes=frame['table_hashes'],
        calibration_source='nuScenes configured dataset root; separate sensor timestamps and ego poses',
        image_geometry='original release pixels; independent of inference padding/resize')
    return dict(evidence=evidence,source_path=path,content=content)
