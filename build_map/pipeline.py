#!/usr/bin/env python3
"""
Pipeline script to build a metric SfM map with per-frame intrinsics and depth-based scale.
Steps are split into subcommands so you can run them end-to-end or individually:

1) select-keyframes   : Downsample frames and gather per-frame intrinsics.
2) create-db          : Create a COLMAP database with one camera per frame.
3) run-sfm            : Run COLMAP feature extraction, sequential matching, and mapper.
4) scale-from-depth   : Estimate and apply a global scale using depth maps.
5) export-poses       : Export poses for localization.

Example (for capture/UET_G2/raw/phone/2025-12-22_16.43.15):

python build_map/pipeline.py \
    select-keyframes \
    --session-root capture/UET_G2/raw/phone/2025-12-22_16.43.15 \
    --output-root build_map/output_keyframes

python build_map/pipeline.py \
    create-db \
    --keyframes-root build_map/output_keyframes \
    --database build_map/output_keyframes/colmap.db

python build_map/pipeline.py \
    run-sfm \
    --database build_map/output_keyframes/colmap.db \
    --image-path build_map/output_keyframes/images \
    --output-model build_map/output_sfm \
    --colmap-bin colmap

python build_map/pipeline.py \
    scale-from-depth \
    --model-path build_map/output_sfm/0 \
    --intrinsics build_map/output_keyframes/intrinsics.json \
    --depth-dir build_map/output_keyframes/depth \
    --output-model build_map/output_sfm_scaled

python build_map/pipeline.py \
    export-poses \
    --model-path build_map/output_sfm_scaled \
    --output poses.txt
"""
import argparse
import json
import math
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pycolmap
from PIL import Image


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class SensorRecord:
    timestamp: int
    status: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    exposure: int


@dataclass
class FrameSample:
    timestamp: int
    image_path: Path
    depth_path: Optional[Path]
    conf_path: Optional[Path]
    sensor: SensorRecord


# ----------------------------
# Parsing helpers
# ----------------------------

def read_sensors(sensors_path: Path) -> Dict[int, SensorRecord]:
    records: Dict[int, SensorRecord] = {}
    with sensors_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 16:
                continue
            ts = int(parts[0])
            status = parts[1]
            # tx, ty, tz = map(float, parts[2:5])
            # qx, qy, qz, qw = map(float, parts[5:9])
            width, height = map(int, parts[9:11])
            fx, fy, cx, cy = map(float, parts[11:15])
            exposure = int(parts[15])
            records[ts] = SensorRecord(
                timestamp=ts,
                status=status,
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                exposure=exposure,
            )
    return records


def build_frame_samples(session_root: Path) -> List[FrameSample]:
    frames_dir = session_root / "frames"
    depth_dir = session_root / "depth"
    sensors_path = session_root / "sensors.txt"
    sensors = read_sensors(sensors_path)

    image_files = {int(p.stem): p for p in frames_dir.glob("*.jpg")}
    depth_files = {int(p.stem): p for p in depth_dir.glob("*.bin")}
    conf_files = {int(p.stem.split(".")[0]): p for p in depth_dir.glob("*.confidence.png")}

    samples: List[FrameSample] = []
    for ts, img_path in image_files.items():
        if ts not in sensors:
            continue
        samples.append(
            FrameSample(
                timestamp=ts,
                image_path=img_path,
                depth_path=depth_files.get(ts),
                conf_path=conf_files.get(ts),
                sensor=sensors[ts],
            )
        )
    samples.sort(key=lambda x: x.timestamp)
    return samples


# ----------------------------
# Keyframe selection (stride only, no sensor poses)
# ----------------------------

def select_keyframes(
    samples: List[FrameSample],
    stride: int = 5,
) -> List[FrameSample]:
    if not samples:
        return []
    return [s for idx, s in enumerate(samples) if idx % stride == 0]


def materialize_keyframes(samples: List[FrameSample], output_root: Path, copy_mode: str = "symlink") -> None:
    images_out = output_root / "images"
    depth_out = output_root / "depth"
    output_root.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    def place(src: Path, dst: Path) -> None:
        if dst.exists():
            dst.unlink()
        if copy_mode == "copy":
            shutil.copy2(src, dst)
        elif copy_mode == "hardlink":
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
        else:
            os.symlink(src.resolve(), dst)

    for s in samples:
        img_dst = images_out / f"{s.timestamp}{s.image_path.suffix}"
        place(s.image_path, img_dst)
        if s.depth_path is not None:
            depth_dst = depth_out / f"{s.timestamp}.bin"
            place(s.depth_path, depth_dst)
        if s.conf_path is not None:
            conf_dst = depth_out / f"{s.timestamp}.confidence.png"
            place(s.conf_path, conf_dst)

    intrinsics = {
        "frames": [
            {
                "timestamp": s.timestamp,
                "image": f"{s.timestamp}{s.image_path.suffix}",
                "depth": f"{s.timestamp}.bin" if s.depth_path else None,
                "confidence": f"{s.timestamp}.confidence.png" if s.conf_path else None,
                "width": s.sensor.width,
                "height": s.sensor.height,
                "fx": s.sensor.fx,
                "fy": s.sensor.fy,
                "cx": s.sensor.cx,
                "cy": s.sensor.cy,
            }
            for s in samples
        ]
    }
    with (output_root / "intrinsics.json").open("w") as f:
        json.dump(intrinsics, f, indent=2)


# ----------------------------
# COLMAP helpers
# ----------------------------

def load_intrinsics(intrinsics_path: Path) -> Dict[str, Dict]:
    with intrinsics_path.open("r") as f:
        data = json.load(f)
    mapping = {str(it["timestamp"]): it for it in data.get("frames", [])}
    return mapping


def create_colmap_db(keyframes_root: Path, db_path: Path) -> None:
    """Tạo COLMAP database với sqlite3 trực tiếp để tương thích COLMAP v3.8."""
    import sqlite3
    
    intr_path = keyframes_root / "intrinsics.json"
    intr = load_intrinsics(intr_path)
    if db_path.exists():
        db_path.unlink()
    
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    
    # Schema COLMAP v3.8
    c.execute('''CREATE TABLE IF NOT EXISTS cameras (
        camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        model INTEGER NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        params BLOB,
        prior_focal_length INTEGER NOT NULL)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS images (
        image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        name TEXT NOT NULL UNIQUE,
        camera_id INTEGER NOT NULL,
        prior_qw REAL,
        prior_qx REAL,
        prior_qy REAL,
        prior_qz REAL,
        prior_tx REAL,
        prior_ty REAL,
        prior_tz REAL,
        CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < 2147483647),
        FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))''')
    
    c.execute('CREATE TABLE IF NOT EXISTS keypoints (image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB, FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)')
    c.execute('CREATE TABLE IF NOT EXISTS descriptors (image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB, FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)')
    c.execute('CREATE TABLE IF NOT EXISTS matches (pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB)')
    c.execute('CREATE TABLE IF NOT EXISTS two_view_geometries (pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB, config INTEGER NOT NULL, F BLOB, E BLOB, H BLOB)')
    c.execute('CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)')
    conn.commit()
    
    # PINHOLE = 1 trong COLMAP
    camera_ids: Dict[str, int] = {}
    for ts_str, meta in intr.items():
        params = np.array([meta["fx"], meta["fy"], meta["cx"], meta["cy"]], dtype=np.float64)
        c.execute(
            'INSERT INTO cameras(model, width, height, params, prior_focal_length) VALUES(?, ?, ?, ?, ?)',
            (1, int(meta["width"]), int(meta["height"]), params.tobytes(), 1)
        )
        camera_ids[ts_str] = c.lastrowid
    
    for ts_str, meta in intr.items():
        name = meta["image"]
        cam_id = camera_ids[ts_str]
        c.execute('INSERT INTO images(name, camera_id) VALUES(?, ?)', (name, cam_id))
    
    conn.commit()
    conn.close()
    print(f"[create-db] Wrote database with {len(camera_ids)} cameras and {len(intr)} images to {db_path}")


def run_colmap_cli(
    database_path: Path,
    image_path: Path,
    output_model: Path,
    colmap_bin: str = "colmap",
    use_gpu: bool = True,
) -> None:
    output_model.mkdir(parents=True, exist_ok=True)
    feature_cmd = [
        colmap_bin,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--ImageReader.camera_model",
        "PINHOLE",
        "--ImageReader.single_camera",
        "0",
        "--SiftExtraction.use_gpu",
        "1" if use_gpu else "0",
    ]
    match_cmd = [
        colmap_bin,
        "sequential_matcher",
        "--database_path",
        str(database_path),
        "--SequentialMatching.overlap",
        "4",
        "--SiftMatching.use_gpu",
        "1" if use_gpu else "0",
    ]
    mapper_cmd = [
        colmap_bin,
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--output_path",
        str(output_model),
        "--Mapper.ba_refine_focal_length",
        "0",
        "--Mapper.ba_refine_principal_point",
        "0",
        "--Mapper.ba_refine_extra_params",
        "0",
    ]
    print("[run-sfm] Running feature extraction...")
    subprocess.run(feature_cmd, check=True)
    print("[run-sfm] Running sequential matcher...")
    subprocess.run(match_cmd, check=True)
    print("[run-sfm] Running mapper...")
    subprocess.run(mapper_cmd, check=True)
    print(f"[run-sfm] Model written to {output_model}")


# ----------------------------
# Depth helpers
# ----------------------------

def load_depth(depth_path: Path, width: int, height: int) -> np.ndarray:
    file_size = depth_path.stat().st_size
    numel = width * height
    candidates = [np.float32, np.float16, np.uint16]
    
    # Try exact match first
    for dtype in candidates:
        expected = numel * np.dtype(dtype).itemsize
        if expected == file_size:
            arr = np.fromfile(depth_path, dtype=dtype, count=numel)
            depth = arr.reshape((height, width))
            if dtype == np.uint16:
                depth = depth.astype(np.float32) / 1000.0
            return depth.astype(np.float32)
    
    # If no exact match, try to infer dimensions from file size
    for dtype in candidates:
        itemsize = np.dtype(dtype).itemsize
        if file_size % itemsize == 0:
            total_pixels = file_size // itemsize
            target_aspect = width / height
            
            # Find all divisors of total_pixels
            sqrt_pixels = int(np.sqrt(total_pixels))
            for h_candidate in range(1, sqrt_pixels + 1):
                if total_pixels % h_candidate == 0:
                    w_candidate = total_pixels // h_candidate
                    candidate_aspect = w_candidate / h_candidate
                    # Check if aspect ratio is similar (within 20% tolerance)
                    if abs(candidate_aspect - target_aspect) / target_aspect < 0.2:
                        print(f"[load_depth] Auto-detected size {w_candidate}x{h_candidate} (dtype={dtype.__name__}) for {depth_path.name}")
                        arr = np.fromfile(depth_path, dtype=dtype, count=total_pixels)
                        depth = arr.reshape((h_candidate, w_candidate))
                        if dtype == np.uint16:
                            depth = depth.astype(np.float32) / 1000.0
                        depth = depth.astype(np.float32)
                        # Resize to expected dimensions if needed
                        if (h_candidate, w_candidate) != (height, width):
                            from PIL import Image as PILImage
                            img = PILImage.fromarray(depth)
                            img_resized = img.resize((width, height), PILImage.BILINEAR)
                            depth = np.array(img_resized, dtype=np.float32)
                        return depth
    
    raise ValueError(
        f"Cannot infer depth dtype for {depth_path} (size {file_size}, expected {width}x{height}). "
        f"Expected sizes: {[width*height*np.dtype(d).itemsize for d in candidates]}"
    )


def load_confidence(conf_path: Path, width: int, height: int, threshold: int = 64) -> np.ndarray:
    img = Image.open(conf_path)
    # Convert to grayscale if RGB
    if img.mode == 'RGB' or img.mode == 'RGBA':
        img = img.convert('L')
    # Resize to match depth dimensions if needed
    if img.size != (width, height):
        img = img.resize((width, height), Image.BILINEAR)
    mask = np.array(img)
    return (mask >= threshold).astype(np.uint8)


# ----------------------------
# Scale estimation and export
# ----------------------------

# In pycolmap 3.13.0, invalid point3D_id is uint64 max instead of -1
INVALID_POINT3D_ID = 2**64 - 1

def image_depth_ratios(
    rec: pycolmap.Reconstruction,
    image: pycolmap.Image,
    intr_meta: Dict,
    depth_dir: Path,
    max_samples: int = 2000,
    conf_threshold: int = 64,
) -> List[float]:
    timestamp = Path(image.name).stem
    depth_path = depth_dir / f"{timestamp}.bin"
    if not depth_path.exists():
        return []
    w = int(intr_meta["width"])
    h = int(intr_meta["height"])
    depth = load_depth(depth_path, w, h)
    conf_mask = None
    conf_path = depth_dir / f"{timestamp}.confidence.png"
    if conf_path.exists():
        conf_mask = load_confidence(conf_path, w, h, threshold=conf_threshold)

    # In pycolmap 3.13.0, image.points2D is a list of Point2D objects with .point3D_id attribute
    valid_indices = [i for i, pt in enumerate(image.points2D) if pt.point3D_id != INVALID_POINT3D_ID]
    if not valid_indices:
        return []
    random.shuffle(valid_indices)
    valid_indices = valid_indices[:max_samples]

    # In pycolmap 3.13.0, use cam_from_world() method to get pose
    pose = image.cam_from_world()
    R = pose.rotation.matrix()
    t = pose.translation
    ratios: List[float] = []
    w = int(intr_meta["width"])
    h = int(intr_meta["height"])
    
    for idx in valid_indices:
        # In pycolmap 3.13.0, use .xy instead of .xys
        pt2d = image.points2D[idx]
        u, v = pt2d.xy
        pid = pt2d.point3D_id
        if pid == INVALID_POINT3D_ID:
            continue
        ui = int(round(u))
        vi = int(round(v))
        if ui < 0 or vi < 0 or ui >= w or vi >= h:
            continue
        if conf_mask is not None:
            conf_val = conf_mask[vi, ui]
            if hasattr(conf_val, 'item'):
                conf_val = conf_val.item()
            if conf_val == 0:
                continue
        z_meas = float(depth[vi, ui])
        if not math.isfinite(z_meas) or z_meas <= 0:
            continue
        Xw = rec.points3D[pid].xyz
        depth_sfm = float((R @ Xw + t)[2])
        if depth_sfm <= 0 or not math.isfinite(depth_sfm):
            continue
        ratios.append(z_meas / depth_sfm)
    
    return ratios


def estimate_scale_from_depth(
    model_path: Path,
    intrinsics_path: Path,
    depth_dir: Path,
    conf_threshold: int = 64,
    min_samples: int = 100,
) -> Tuple[float, List[float]]:
    rec = pycolmap.Reconstruction(model_path)
    intr = load_intrinsics(intrinsics_path)
    all_ratios: List[float] = []
    for image in rec.images.values():
        ts = Path(image.name).stem
        if ts not in intr:
            continue
        ratios = image_depth_ratios(rec, image, intr[ts], depth_dir, conf_threshold=conf_threshold)
        if ratios:
            print(f"[scale] Image {ts}: {len(ratios)} valid samples")
        all_ratios.extend(ratios)
    print(f"[scale] Total samples: {len(all_ratios)}, min required: {min_samples}")
    if len(all_ratios) < min_samples:
        raise RuntimeError(f"Not enough depth samples ({len(all_ratios)}) to estimate scale")
    scale = float(np.median(all_ratios))
    return scale, all_ratios


def apply_scale_and_write(model_path: Path, output_path: Path, scale: float) -> None:
    rec = pycolmap.Reconstruction(model_path)
    # In pycolmap 3.13.0, use Sim3d transform to scale the reconstruction
    sim = pycolmap.Sim3d(scale=scale)
    rec.transform(sim)
    output_path.mkdir(parents=True, exist_ok=True)
    rec.write(output_path)
    print(f"[scale] Applied scale {scale:.6f} and wrote model to {output_path}")


def export_poses(model_path: Path, output_file: Path) -> None:
    rec = pycolmap.Reconstruction(model_path)
    images = sorted(rec.images.values(), key=lambda x: x.name)
    with output_file.open("w") as f:
        f.write("# image_name tx ty tz qx qy qz qw (camera-to-world)\n")
        for img in images:
            # In pycolmap 3.13.0, get pose via cam_from_world() which returns world-to-cam
            pose_wc = img.cam_from_world()
            R_wc = pose_wc.rotation.matrix()  # World to camera rotation
            t_wc = pose_wc.translation  # Camera position in camera frame (negative of cam center in world)
            
            # Convert to camera-to-world
            cam_center = -R_wc.T @ t_wc  # Camera center in world frame
            R_cw = R_wc.T  # Camera to world rotation
            
            # Convert rotation matrix to quaternion (w, x, y, z)
            # Using same formula as pycolmap internally
            trace = R_cw.trace()
            if trace > 0:
                s = 0.5 / np.sqrt(trace + 1.0)
                qw = 0.25 / s
                qx = (R_cw[2, 1] - R_cw[1, 2]) * s
                qy = (R_cw[0, 2] - R_cw[2, 0]) * s
                qz = (R_cw[1, 0] - R_cw[0, 1]) * s
            elif R_cw[0, 0] > R_cw[1, 1] and R_cw[0, 0] > R_cw[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R_cw[0, 0] - R_cw[1, 1] - R_cw[2, 2])
                qw = (R_cw[2, 1] - R_cw[1, 2]) / s
                qx = 0.25 * s
                qy = (R_cw[0, 1] + R_cw[1, 0]) / s
                qz = (R_cw[0, 2] + R_cw[2, 0]) / s
            elif R_cw[1, 1] > R_cw[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R_cw[1, 1] - R_cw[0, 0] - R_cw[2, 2])
                qw = (R_cw[0, 2] - R_cw[2, 0]) / s
                qx = (R_cw[0, 1] + R_cw[1, 0]) / s
                qy = 0.25 * s
                qz = (R_cw[1, 2] + R_cw[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R_cw[2, 2] - R_cw[0, 0] - R_cw[1, 1])
                qw = (R_cw[1, 0] - R_cw[0, 1]) / s
                qx = (R_cw[0, 2] + R_cw[2, 0]) / s
                qy = (R_cw[1, 2] + R_cw[2, 1]) / s
                qz = 0.25 * s
            
            f.write(
                f"{img.name} {cam_center[0]:.6f} {cam_center[1]:.6f} {cam_center[2]:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )
    print(f"[export] Wrote poses to {output_file}")


# ----------------------------
# CLI
# ----------------------------

def cmd_select_keyframes(args: argparse.Namespace) -> None:
    samples = build_frame_samples(Path(args.session_root))
    selected = select_keyframes(samples, stride=args.stride)
    materialize_keyframes(selected, Path(args.output_root), copy_mode=args.copy_mode)
    print(f"[select] Selected {len(selected)} / {len(samples)} frames -> {args.output_root}")


def cmd_create_db(args: argparse.Namespace) -> None:
    create_colmap_db(Path(args.keyframes_root), Path(args.database))


def cmd_run_sfm(args: argparse.Namespace) -> None:
    run_colmap_cli(
        database_path=Path(args.database),
        image_path=Path(args.image_path),
        output_model=Path(args.output_model),
        colmap_bin=args.colmap_bin,
        use_gpu=not args.cpu,
    )


def cmd_scale_from_depth(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path)
    if model_path.is_dir():
        model_in = model_path
    else:
        model_in = model_path.parent
    scale, ratios = estimate_scale_from_depth(
        model_in,
        Path(args.intrinsics),
        Path(args.depth_dir),
        conf_threshold=args.conf_threshold,
        min_samples=args.min_samples,
    )
    print(
        f"[scale] Scale={scale:.6f} (median of {len(ratios)} samples). "
        f"p5={np.percentile(ratios,5):.4f}, p95={np.percentile(ratios,95):.4f}"
    )
    apply_scale_and_write(model_in, Path(args.output_model), scale)


def cmd_export_poses(args: argparse.Namespace) -> None:
    export_poses(Path(args.model_path), Path(args.output))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build metric map pipeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sel = sub.add_parser("select-keyframes", help="Downsample frames and write intrinsics")
    p_sel.add_argument("--session-root", type=Path, required=True, help="Path to raw session (containing frames/ depth/ sensors.txt)")
    p_sel.add_argument("--output-root", type=Path, required=True, help="Where to write keyframes, depth, intrinsics.json")
    p_sel.add_argument("--stride", type=int, default=5, help="Keep 1 every N frames (pose-free)")
    p_sel.add_argument("--copy-mode", type=str, default="symlink", choices=["symlink", "copy", "hardlink"], help="How to materialize frames/depth")
    p_sel.set_defaults(func=cmd_select_keyframes)

    p_db = sub.add_parser("create-db", help="Create COLMAP database with per-frame intrinsics")
    p_db.add_argument("--keyframes-root", type=Path, required=True, help="Folder created by select-keyframes")
    p_db.add_argument("--database", type=Path, required=True, help="Output COLMAP database path")
    p_db.set_defaults(func=cmd_create_db)

    p_sfm = sub.add_parser("run-sfm", help="Run COLMAP feature extraction + sequential matcher + mapper")
    p_sfm.add_argument("--database", type=Path, required=True)
    p_sfm.add_argument("--image-path", type=Path, required=True)
    p_sfm.add_argument("--output-model", type=Path, required=True, help="Mapper output directory")
    p_sfm.add_argument("--colmap-bin", type=str, default="colmap", help="Path to COLMAP binary")
    p_sfm.add_argument("--cpu", action="store_true", help="Disable GPU for SIFT")
    p_sfm.set_defaults(func=cmd_run_sfm)

    p_scale = sub.add_parser("scale-from-depth", help="Estimate scale from depth maps and write scaled model")
    p_scale.add_argument("--model-path", type=Path, required=True, help="Mapper model directory (e.g., output_sfm/0)")
    p_scale.add_argument("--intrinsics", type=Path, required=True, help="intrinsics.json from keyframes")
    p_scale.add_argument("--depth-dir", type=Path, required=True, help="keyframes/depth directory")
    p_scale.add_argument("--output-model", type=Path, required=True, help="Directory to write scaled model")
    p_scale.add_argument("--conf-threshold", type=int, default=1, help="Confidence mask threshold (typically 1-2 for iPhone depth)")
    p_scale.add_argument("--min-samples", type=int, default=100, help="Minimum ratios needed")
    p_scale.set_defaults(func=cmd_scale_from_depth)

    p_exp = sub.add_parser("export-poses", help="Export poses from a COLMAP model")
    p_exp.add_argument("--model-path", type=Path, required=True, help="Model directory (scaled)")
    p_exp.add_argument("--output", type=Path, required=True, help="Output pose file")
    p_exp.set_defaults(func=cmd_export_poses)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
