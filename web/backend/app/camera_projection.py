"""Float64 native LiDAR -> timestamped camera projection, without ML imports."""
import itertools
import numpy as np

from .store import InvalidArtifact, quaternion_pose

CHANNELS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK',
            'CAM_BACK_LEFT', 'CAM_FRONT_LEFT')
PROTOCOL = dict(name='rav25_timestamped_camera_projection_v1', precision='float64',
    coordinates='camera optical: X right, Y down, Z forward; original-image pixels',
    transform='inv(global_from_camera_at_camera_time) @ global_from_lidar_at_lidar_time',
    box_center='geometric', box_dimensions=['dx','dy','dz'], box_yaw='CCW about native LiDAR +Z',
    point_min_depth_m=1.0, box_near_plane_m=0.1, image_margin_px=1.0,
    edge_clipping='camera-space near-plane, then Liang-Barsky in original-image pixels',
    motion_limit='Ego-motion compensation uses both timestamps; object motion and intra-scan LiDAR motion are not compensated.',
    visibility_limit='Geometric projections, not an occlusion/visibility guarantee.',
    point_source='Unfiltered LIDAR_TOP keyframe only; no sweeps or GT boxes',
    image_geometry='Original release image dimensions/intrinsics; no model resize or padding')


def camera_from_lidar(lidar_cal, lidar_ego, camera_cal, camera_ego):
    global_lidar = quaternion_pose(lidar_ego['rotation'], lidar_ego['translation']) @ quaternion_pose(lidar_cal['rotation'], lidar_cal['translation'])
    global_camera = quaternion_pose(camera_ego['rotation'], camera_ego['translation']) @ quaternion_pose(camera_cal['rotation'], camera_cal['translation'])
    return np.linalg.inv(global_camera) @ global_lidar


def transform(points, matrix):
    points, matrix = np.asarray(points,dtype=np.float64), np.asarray(matrix,dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or matrix.shape != (4,4) or not np.isfinite(points).all() or not np.isfinite(matrix).all():
        raise InvalidArtifact('Projection requires finite [N,3] points and a 4x4 transform')
    return points @ matrix[:3,:3].T + matrix[:3,3]


def pixels(camera_points, intrinsic):
    values = np.asarray(camera_points,dtype=np.float64) @ np.asarray(intrinsic,dtype=np.float64).T
    return values[:,:2] / values[:,2:3]


def project_points(points, matrix, intrinsic, width, height):
    camera = transform(points,matrix)
    ids = np.flatnonzero(camera[:,2] > PROTOCOL['point_min_depth_m'])
    uv = pixels(camera[ids],intrinsic)
    margin = PROTOCOL['image_margin_px']
    inside = (uv[:,0]>margin)&(uv[:,0]<width-margin)&(uv[:,1]>margin)&(uv[:,1]<height-margin)
    ids, uv = ids[inside], uv[inside]
    return np.column_stack((ids,uv,camera[ids,2]))


def clip_image(a, b, width, height):
    """Liang-Barsky; intersections retain edges whose two endpoints are outside."""
    delta = b-a
    low,high = 0.,1.
    for p,q in ((-delta[0],a[0]),(delta[0],width-a[0]),(-delta[1],a[1]),(delta[1],height-a[1])):
        if abs(p)<1e-15:
            if q<0:return None
        elif p<0:low=max(low,q/p)
        else:high=min(high,q/p)
        if low>high:return None
    return np.clip(a+low*delta,[0,0],[width,height]),np.clip(a+high*delta,[0,0],[width,height])


def box_corners(box):
    signs=np.asarray(list(itertools.product((-1,1),repeat=3)),dtype=np.float64)
    corners=signs*np.asarray(box['dimensions'],dtype=np.float64)/2
    c,s=np.cos(box['yaw']),np.sin(box['yaw'])
    rotation=np.asarray([[c,-s,0],[s,c,0],[0,0,1]])
    return corners@rotation.T+box['center']


EDGES = tuple((i,j) for i in range(8) for j in range(i+1,8) if (i^j) in (1,2,4))


def project_box(box, matrix, intrinsic, width, height):
    corners = transform(box_corners(box),matrix)
    near = PROTOCOL['box_near_plane_m']
    segments=[]
    for i,j in EDGES:
        a,b=corners[i].copy(),corners[j].copy()
        if a[2]<near and b[2]<near:continue
        if a[2]<near:a=a+(b-a)*((near-a[2])/(b[2]-a[2]))
        elif b[2]<near:b=b+(a-b)*((near-b[2])/(a[2]-b[2]))
        uv=pixels(np.stack((a,b)),intrinsic)
        clipped=clip_image(uv[0],uv[1],width,height)
        if clipped is not None:segments.append(np.concatenate(clipped).tolist())
    return dict(label=box['label'],score=box['score'],segments=segments)
