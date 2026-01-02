import os
import sys
import numpy as np
from collections import defaultdict

# Add project root to path
project_root = os.path.abspath(os.path.join(os.getcwd(), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from visualize.cam_pose_visualizer import CamPoseVisualizer, read_rigs, load_poses


def read_poses(poses_path):
    poses = []
    with open(poses_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#"): continue
            line = line.strip().split(" ")
            
            timestamp = line[0].split(".")[0]
            device = f"cam_phone_{timestamp}"
            q = {
                'w': float(line[4]),
                'x': float(line[5]),
                'y': float(line[6]),
                'z': float(line[7]),
            }
            t = {
                'x': float(line[1]),
                'y': float(line[2]),
                'z': float(line[3]),
            }
            
            poses.append({
                'timestamp': timestamp,
                'device_id': device,
                'q': q,
                't': t,
                'covar': []
            })
    return poses    

def read_sensors(sensors_path):
    sensors = defaultdict(dict)
    with open(sensors_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#"): continue
            line = line.strip().split(", ")
            if len(line) < 6: continue
            
            sensor_id = f"cam_phone_{line[0]}"
            width, height = line[9:11]
            fx, fy, cx, cy, *_ = line[11:]
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

def load_generated_poses(poses_path, sensors_path, rigs_path=None, len=None, color=[255, 255, 255]):
    poses = read_poses(poses_path)
    sensors = read_sensors(sensors_path)
    rigs = read_rigs(rigs_path)
    
    if len is not None:
        poses = poses[:len]
    
    for pose in poses:
        pose['color'] = color
        
        if "rig_sensors" in rigs: # ios
            pose['rig_sensors'] = rigs
            pose['rig_sensors']['rig_sensors'] = {
                    **pose['rig_sensors']['rig_sensors'],
                    **sensors[pose['device_id']]
                }
        else:
            pose['rig_sensors'] = rigs[pose['device_id']]
            for sensor_id in pose['rig_sensors']:
                pose['rig_sensors'][sensor_id] = {
                    **pose['rig_sensors'][sensor_id],
                    **sensors[sensor_id]
                }
    
    return poses


if __name__ == "__main__":
    poses_path = "/home/long/Workspace/VNU-AR/capture/UET-G2/mapping_poses.txt"
    sensors_path = "/home/long/Workspace/crocodl-benchmark/capture/UET_G2/raw/phone/2025-12-22_16.43.15/sensors.txt"
    poses = load_generated_poses(poses_path, sensors_path, rigs_path=None, len=None, color=[255, 0, 0])
    
    gt_poses_path0 = "/home/long/Workspace/crocodl-benchmark/capture/UET_G2/sessions/ios_2025-12-22_16.43.15_000/trajectories.txt"
    gt_sensors_path0 = "/home/long/Workspace/crocodl-benchmark/capture/UET_G2/sessions/ios_2025-12-22_16.43.15_000/sensors.txt"
    gt_poses0 = load_poses(gt_poses_path0, gt_sensors_path0, rigs_path=None, len=None, color=[0, 255, 0])
    
    gt_poses_path1 = "/home/long/Workspace/crocodl-benchmark/capture/UET_G2/sessions/ios_2025-12-22_16.43.15_001/trajectories.txt"
    gt_sensors_path1 = "/home/long/Workspace/crocodl-benchmark/capture/UET_G2/sessions/ios_2025-12-22_16.43.15_001/sensors.txt"
    gt_poses1 = load_poses(gt_poses_path1, gt_sensors_path1, rigs_path=None, len=None, color=[0, 255, 0])
    
    visualizer = CamPoseVisualizer()
    visualizer.visualize(poses + gt_poses0)