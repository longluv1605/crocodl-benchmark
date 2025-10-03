#!/usr/bin/env python3
"""
Cross-device pose estimation evaluation with success rate matrix analysis.
Simple script without complex class structure.
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Configuration
DEVICES = ['ios', 'hl', 'spot']
LOCATIONS = ['HYDRO', 'SUCCULENT']
BENCHMARK_DIR = 'benchmarking_all_ml_sp_lg'  # Fixed benchmark directory

def load_poses_file(file_path):
    """Load poses from trajectories.txt or poses.txt file."""
    poses = {}
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return poses
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split(', ')
            if len(parts) < 9:  # At least timestamp, device_id, qw, qx, qy, qz, tx, ty, tz
                continue
            
            try:
                timestamp = int(parts[0])
                device_id = parts[1]
                qw, qx, qy, qz = map(float, parts[2:6])
                tx, ty, tz = map(float, parts[6:9])
                
                poses[timestamp] = {
                    'device_id': device_id,
                    'position': np.array([tx, ty, tz]),
                    'quaternion': np.array([qw, qx, qy, qz]),
                    'timestamp': timestamp
                }
            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse line: {line} - {e}")
                continue
    
    return poses

def quaternion_to_rotation_matrix(q):
    """Convert quaternion [qw, qx, qy, qz] to rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])

def compute_pose_error(gt_pose, est_pose):
    """Compute position and rotation error between ground truth and estimated poses."""
    # Position error (Euclidean distance)
    pos_error = np.linalg.norm(gt_pose['position'] - est_pose['position'])
    
    # Rotation error (angle between quaternions)
    R1 = quaternion_to_rotation_matrix(gt_pose['quaternion'])
    R2 = quaternion_to_rotation_matrix(est_pose['quaternion'])
    R_rel = np.dot(R1.T, R2)
    
    # Extract angle from rotation matrix
    trace = np.trace(R_rel)
    cos_angle = (trace - 1) / 2
    cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return pos_error, angle_deg

def evaluate_device_pair(capture_dir, location, query_device, map_device, pos_threshold, rot_threshold):
    """Evaluate a single query-map device pair for a location."""
    print(f"Evaluating {location}: {query_device} query vs {map_device} map")
    
    # Load ground truth trajectories
    gt_file = f"{capture_dir}/{location}/sessions/{query_device}_query/trajectories.txt"
    gt_poses = load_poses_file(gt_file)
    
    # Load estimated poses
    pose_dir = f"{capture_dir}/{location}/{BENCHMARK_DIR}/pose_estimation/{query_device}_query/{map_device}_map"
    estimated_poses = {}
    
    # Search for poses.txt in all subdirectories
    for root, dirs, files in os.walk(pose_dir):
        if 'poses.txt' in files:
            poses_file = os.path.join(root, 'poses.txt')
            poses = load_poses_file(poses_file)
            estimated_poses.update(poses)  # Merge all found poses
    
    # Compare poses
    total_gt = len(gt_poses)
    successful_poses = 0
    total_matched = 0
    
    # Simply find common timestamps (same timestamps, just different order)
    gt_timestamps = set(gt_poses.keys())
    est_timestamps = set(estimated_poses.keys())
    matched_timestamps = gt_timestamps & est_timestamps
    total_matched = len(matched_timestamps)
    
    for timestamp in matched_timestamps:
        gt_pose = gt_poses[timestamp]
        est_pose = estimated_poses[timestamp]
        
        pos_error, rot_error = compute_pose_error(gt_pose, est_pose)
        
        # Check if pose estimation is successful
        if pos_error < pos_threshold and rot_error < rot_threshold:
            successful_poses += 1
    
    # Calculate success rate
    success_rate = (successful_poses / total_gt * 100) if total_gt > 0 else 0.0
    
    return {
        'query_device': query_device,
        'map_device': map_device,
        'total_gt_poses': total_gt,
        'successful_poses': successful_poses,
        'total_matched': total_matched,
        'success_rate': success_rate
    }

def compute_success_rate_matrices(capture_dir, pos_threshold=1.0, rot_threshold=5.0):
    """Compute success rate matrices for all locations and overall."""
    location_results = {}
    
    # Process each location
    for location in LOCATIONS:
        print(f"\nProcessing location: {location}")
        location_matrix = {}
        location_details = {}
        
        for query_device in DEVICES:
            location_matrix[query_device] = {}
            for map_device in DEVICES:
                result = evaluate_device_pair(
                    capture_dir, location, query_device, map_device, 
                    pos_threshold, rot_threshold
                )
                location_matrix[query_device][map_device] = result['success_rate']
                location_details[f"{query_device}_vs_{map_device}"] = result
        
        location_results[location] = {
            'matrix': location_matrix,
            'details': location_details
        }
    
    # Compute overall matrix
    print(f"\nComputing overall success rate matrix")
    overall_matrix = {}
    overall_details = {}
    
    for query_device in DEVICES:
        overall_matrix[query_device] = {}
        for map_device in DEVICES:
            # Aggregate across locations
            total_gt = 0
            total_successful = 0
            
            for location in LOCATIONS:
                detail_key = f"{query_device}_vs_{map_device}"
                if detail_key in location_results[location]['details']:
                    detail = location_results[location]['details'][detail_key]
                    total_gt += detail['total_gt_poses']
                    total_successful += detail['successful_poses']
            
            overall_success_rate = (total_successful / total_gt * 100) if total_gt > 0 else 0.0
            overall_matrix[query_device][map_device] = overall_success_rate
            overall_details[f"{query_device}_vs_{map_device}"] = {
                'total_gt_poses': total_gt,
                'successful_poses': total_successful,
                'success_rate': overall_success_rate
            }
    
    overall_results = {
        'matrix': overall_matrix,
        'details': overall_details
    }
    
    return location_results, overall_results

def print_success_matrix(matrix, title):
    """Print success rate matrix in a formatted way."""
    print(f"\n{title}")
    print("-" * len(title))
    
    # Header
    header_text = "Query\\Map"
    print(f"{header_text:<12}", end="")
    for map_device in DEVICES:
        print(f"{map_device.upper():<12}", end="")
    print()
    print("-" * (12 + 12 * len(DEVICES)))
    
    # Rows
    for query_device in DEVICES:
        print(f"{query_device.upper():<12}", end="")
        for map_device in DEVICES:
            rate = matrix[query_device][map_device]
            print(f"{rate:>8.2f}%   ", end="")
        print()

def save_results(location_results, overall_results, output_dir):
    """Save evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON results
    with open(f"{output_dir}/location_results.json", 'w') as f:
        json.dump(location_results, f, indent=2)
    
    with open(f"{output_dir}/overall_results.json", 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    # Save CSV matrices
    for location, data in location_results.items():
        matrix_data = []
        for query_device in DEVICES:
            row = {'Query_Device': query_device.upper()}
            for map_device in DEVICES:
                row[f'{map_device.upper()}_Map'] = f"{data['matrix'][query_device][map_device]:.2f}%"
            matrix_data.append(row)
        
        df = pd.DataFrame(matrix_data)
        df.to_csv(f"{output_dir}/{location.lower()}_success_matrix.csv", index=False)
    
    # Overall success matrix CSV
    overall_matrix_data = []
    for query_device in DEVICES:
        row = {'Query_Device': query_device.upper()}
        for map_device in DEVICES:
            row[f'{map_device.upper()}_Map'] = f"{overall_results['matrix'][query_device][map_device]:.2f}%"
        overall_matrix_data.append(row)
    
    df = pd.DataFrame(overall_matrix_data)
    df.to_csv(f"{output_dir}/overall_success_matrix.csv", index=False)

def main():
    parser = argparse.ArgumentParser(description='Evaluate cross-device pose estimation with success rate matrix')
    parser.add_argument('--capture_dir', type=str, required=True,
                        help='Path to capture directory (contains HYDRO, SUCCULENT)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--position_threshold', type=float, default=1.0,
                        help='Position error threshold in meters (default: 1.0)')
    parser.add_argument('--rotation_threshold', type=float, default=5.0,
                        help='Rotation error threshold in degrees (default: 5.0)')
    
    args = parser.parse_args()
    
    print("Starting cross-device pose estimation evaluation")
    
    # Compute success rate matrices
    location_results, overall_results = compute_success_rate_matrices(
        args.capture_dir, args.position_threshold, args.rotation_threshold
    )
    
    # Print results
    for location, data in location_results.items():
        print_success_matrix(data['matrix'], f"{location} Success Rate Matrix")
    
    print_success_matrix(overall_results['matrix'], "OVERALL Success Rate Matrix")
    
    # Save results with benchmark directory in path
    output_dir = os.path.join(args.output_dir, BENCHMARK_DIR)
    save_results(location_results, overall_results, output_dir)
    
    print(f"\nResults saved to {output_dir}")
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()