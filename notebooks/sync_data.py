import os
import json
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from collections import defaultdict

def load_pairs(file_path):
    """Load pairs from filtered file"""
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            pair = line.strip().split(', ')
            pairs.append(tuple(pair))
    return pairs


def load_images(file_path):
    """Load images metadata"""
    with open(file_path, 'r', encoding='utf-8') as f:
        images = defaultdict(dict)
        for line in f:
            if line.startswith("#"): 
                continue
            timestamp, sensor_id, image_path = line.strip().split(", ")
            images[image_path] = {
                "timestamp": timestamp,
                "sensor_id": sensor_id
            }
    return images


def load_intrinsics(file_path):
    """Load camera intrinsics"""
    sensors = defaultdict(dict)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#"): 
                continue
            line = line.strip().split(", ")
            if len(line) < 6: 
                continue
            
            sensor_id = line[0]
            width, height = line[4:6]
            fx, fy, cx, cy = line[6:]
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ], dtype=float)
            
            sensors[sensor_id] = {
                'K': K,
                'width': int(width),
                'height': int(height),
            }
    return sensors


def load_rigs(file_path):
    """Load rig transformations"""
    if file_path is None:
        q = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
        t = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        q_xyzw = np.array([q['x'], q['y'], q['z'], q['w']])
        Q = Rotation.from_quat(q_xyzw).as_matrix()
        T = np.array([t['x'], t['y'], t['z']])
        
        cam2rig = np.eye(4)
        cam2rig[:3, :3] = Q
        cam2rig[:3, 3] = T
        
        return {
            'rig_sensors': {
                'cam2rig': cam2rig,
            }
        }
        
    rigs = defaultdict(dict)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#"): 
                continue
            line = line.strip().split(", ")
            
            rig_id = line[0]
            sensor_id = line[1]
            q = {
                'w': float(line[2]),
                'x': float(line[3]),
                'y': float(line[4]),
                'z': float(line[5]),
            }
            t = {
                'x': float(line[6]),
                'y': float(line[7]),
                'z': float(line[8]),
            }
            
            q_xyzw = np.array([q['x'], q['y'], q['z'], q['w']])
            Q = Rotation.from_quat(q_xyzw).as_matrix()
            T = np.array([t['x'], t['y'], t['z']])
            
            cam2rig = np.eye(4)
            cam2rig[:3, :3] = Q
            cam2rig[:3, 3] = T
            
            rigs[rig_id][sensor_id] = {
                'cam2rig': cam2rig,
            }
    return rigs


def load_poses(file_path):
    """Load pose trajectories"""
    poses = defaultdict(dict)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#"): 
                continue
            line = line.strip().split(", ")
            timestamp = line[0]
            device_id = line[1]
            
            q = {
                'w': float(line[2]),
                'x': float(line[3]),
                'y': float(line[4]),
                'z': float(line[5]),
            }
            t = {
                'x': float(line[6]),
                'y': float(line[7]),
                'z': float(line[8]),
            }
            
            q_xyzw = np.array([q['x'], q['y'], q['z'], q['w']])
            Q = Rotation.from_quat(q_xyzw).as_matrix()
            T = np.array([t['x'], t['y'], t['z']])
            pose = np.eye(4)
            pose[:3, :3] = Q
            pose[:3, 3] = T
            poses[timestamp] = {
                'pose': pose,
                'device_id': device_id
            }
    return poses


def get_groundtruth(poses, rigs, timestamp, cam):
    """Get groundtruth camera pose in world coordinates"""
    rig2world = poses[timestamp]['pose']
    rig_id = poses[timestamp]['device_id']
    
    if "rig_sensors" in rigs.keys():
        cam2rig = rigs["rig_sensors"]['cam2rig']
    else:
        cam2rig = rigs[rig_id][cam]['cam2rig']
    
    return rig2world @ cam2rig


def pose_to_dict(pose):
    """Convert 4x4 pose matrix to serializable dict"""
    R = pose[:3, :3]
    t = pose[:3, 3]
    r = Rotation.from_matrix(R)
    q_xyzw = r.as_quat()
    
    return {
        'qw': float(q_xyzw[3]),
        'qx': float(q_xyzw[0]),
        'qy': float(q_xyzw[1]),
        'qz': float(q_xyzw[2]),
        'tx': float(t[0]),
        'ty': float(t[1]),
        'tz': float(t[2]),
        'matrix': pose.tolist()
    }


def sync_data(capture_dir, query_device, map_device, group_name, output_dir):
    """
    Sync filtered pairs to organized dataset structure
    
    Args:
        capture_dir: Path to capture directory
        query_device: Query device name (ios/hl/spot)
        map_device: Map device name (ios/hl/spot)
        group_name: Group name (e.g., 'keep_trimesh_depth10.0_thresh0.3')
        output_dir: Output directory for synced dataset
    """
    
    # Load filtered pairs
    pairs_path = f"estimate_pose/{query_device}_query/{map_device}_map/{group_name}_filtered/filtered_keep.txt"
    pairs = load_pairs(pairs_path)
    
    # Load metadata
    query_images_path = f"{capture_dir}/ARCHE_D2/sessions/{query_device}_query/images.txt"
    map_images_path = f"{capture_dir}/ARCHE_D2/sessions/{map_device}_map/images.txt"
    query_images = load_images(query_images_path)
    map_images = load_images(map_images_path)
    
    # Load poses
    query_poses_path = f"{capture_dir}/ARCHE_D2/sessions/{query_device}_query/proc/alignment_trajectories.txt"
    map_poses_path = f"{capture_dir}/ARCHE_D2/sessions/{map_device}_map/trajectories.txt"
    query_poses = load_poses(query_poses_path)
    map_poses = load_poses(map_poses_path)
    
    # Load rigs
    query_rigs_path = f"{capture_dir}/ARCHE_D2/sessions/{query_device}_query/rigs.txt" if query_device != "ios" else None
    map_rigs_path = f"{capture_dir}/ARCHE_D2/sessions/{map_device}_map/rigs.txt" if map_device != "ios" else None
    query_rigs = load_rigs(query_rigs_path)
    map_rigs = load_rigs(map_rigs_path)
    
    # Load estimated poses
    est_poses_path = f"estimate_pose/{query_device}_query/{map_device}_map/est_poses.txt"
    est_poses = load_poses(est_poses_path)
    
    # Create group folder
    group_dir = os.path.join(output_dir, f"{query_device}_query_{map_device}_map", group_name)
    os.makedirs(group_dir, exist_ok=True)
    
    print(f"Syncing {len(pairs)} pairs to {group_dir}...")
    
    for idx, pair in enumerate(pairs):
        query_img, map_img, err, matching_score, overlap = pair
        
        # Create pair folder
        pair_dir = os.path.join(group_dir, f"pair_{idx:05d}")
        os.makedirs(pair_dir, exist_ok=True)
        
        # Get timestamps and sensors
        query_timestamp = query_images[query_img]['timestamp']
        query_sensor = query_images[query_img]['sensor_id']
        map_timestamp = map_images[map_img]['timestamp']
        map_sensor = map_images[map_img]['sensor_id']
        
        # Copy images
        query_img_src = f"{capture_dir}/ARCHE_D2/sessions/{query_device}_query/raw_data/{query_img}"
        map_img_src = f"{capture_dir}/ARCHE_D2/sessions/{map_device}_map/raw_data/{map_img}"
        
        query_img_ext = os.path.splitext(query_img)[1]
        map_img_ext = os.path.splitext(map_img)[1]
        
        query_img_dst = os.path.join(pair_dir, f"query{query_img_ext}")
        map_img_dst = os.path.join(pair_dir, f"map{map_img_ext}")
        
        shutil.copy2(query_img_src, query_img_dst)
        shutil.copy2(map_img_src, map_img_dst)
        
        # Get groundtruth poses
        query_gt_pose = get_groundtruth(query_poses, query_rigs, query_timestamp, query_sensor)
        map_gt_pose = get_groundtruth(map_poses, map_rigs, map_timestamp, map_sensor)
        
        # Get estimated pose
        est_key = f"{query_timestamp}-{map_timestamp}"
        if est_key in est_poses:
            est_pose = est_poses[est_key]['pose']
            # print(est_pose_data)
            # # Convert (R, t, mask) tuple to 4x4 matrix
            # R, t, _ = est_pose_data
            # est_pose = np.eye(4)
            # est_pose[:3, :3] = R
            # est_pose[:3, 3] = t
        else:
            est_pose = None
        
        # Create metadata JSON
        metadata = {
            'query_img_path': query_img,
            'map_img_path': map_img,
            'error': float(err),
            'overlap': float(overlap),
            'query_groundtruth_pose': pose_to_dict(query_gt_pose),
            'map_groundtruth_pose': pose_to_dict(map_gt_pose),
            'estimated_pose': pose_to_dict(est_pose) if est_pose is not None else None,
            'query_timestamp': query_timestamp,
            'map_timestamp': map_timestamp,
            'query_sensor': query_sensor,
            'map_sensor': map_sensor
        }
        
        # Save JSON
        json_path = os.path.join(pair_dir, "metadata.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Synced {len(pairs)} pairs successfully!")


if __name__ == "__main__":
    # Configuration
    capture = "/home/long/Workspace/crocodl-benchmark/capture"
    query_device = 'ios'
    map_device = 'ios'
    depth = 10.0
    thresh = 0.3
    group = f'keep_trimesh_depth{depth}_thresh{thresh}'
    output_dir = "./synced_dataset"
    
    sync_data(capture, query_device, map_device, group, output_dir)