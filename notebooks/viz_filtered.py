import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - activates 3D plots

from scantools.capture import Capture, Pose
from scantools.utils.frustum import pyramid_from_camera
from scantools.proc.overlap import OverlapTracer, overlay
from scantools.proc.rendering import Renderer, compute_rays


def parse_pairs(path: Path):
	pairs = []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			if line.startswith("#") or not line.strip():
				continue
			items = [x.strip() for x in line.split(",")]
			if len(items) < 5:
				raise ValueError(f"Expect 5 fields per line (got {len(items)}): {line}")
			q_rel, r_rel, r_err, match_score, frustum_overlap = items[:5]
			pairs.append(
				dict(
					q_rel=q_rel,
					r_rel=r_rel,
					r_err=float(r_err),
					match_score=float(match_score),
					frustum_overlap=float(frustum_overlap),
				)
			)
	return pairs


def index_images(session):
	path_to_key = {}
	for ts, cam_id in session.images.key_pairs():
		rel_path = session.images[ts, cam_id]
		path_to_key[rel_path] = (ts, cam_id)
	return path_to_key


def get_pose_aligned(session, ts, cam_id):
	traj = session.proc.alignment_trajectories if session.proc else None
	traj = traj or session.trajectories
	if session.proc and session.proc.alignment_global:
		T_session2w = session.proc.alignment_global.get_abs_pose("pose_graph_optimized")
		if T_session2w is not None:
			traj = T_session2w * traj
	return session.get_pose(ts, cam_id, traj)


def frustum_lines(pyramid: np.ndarray):
	edges = [
		(0, 1), (0, 2), (0, 3), (0, 4),
		(1, 2), (2, 3), (3, 4), (4, 1),
	]
	lines = []
	for a, b in edges:
		lines.append((pyramid[a], pyramid[b]))
	return lines


def add_frustum(ax, pyramid, color, label, lw: float = 1.5):
	for a, b in frustum_lines(pyramid):
		ax.plot(
			[a[0], b[0]],
			[a[1], b[1]],
			[a[2], b[2]],
			color=color,
			linewidth=lw,
		)
	ax.scatter(pyramid[0, 0], pyramid[0, 1], pyramid[0, 2], color=color, label=label, s=30)


def add_axes(ax, pose: Pose, length: float = 0.3, lw: float = 2.0):
	origin = pose.t
	R = pose.R
	axes = np.stack([R[:, 0], R[:, 1], R[:, 2]], axis=1) * length
	colors = ["r", "g", "b"]  # x, y, z
	for i in range(3):
		vec = axes[:, i]
		ax.quiver(
			origin[0], origin[1], origin[2],
			vec[0], vec[1], vec[2],
			color=colors[i], linewidth=lw, arrow_length_ratio=0.15
		)


def plane_from_camera(cam, pose: Pose, depth: float):
	fx, fy, cx, cy = cam.projection_params
	w, h = cam.width, cam.height
	pts_cam = np.array([
		[(0 - cx) / fx * depth, (0 - cy) / fy * depth, depth],
		[(0 - cx) / fx * depth, (h - cy) / fy * depth, depth],
		[(w - cx) / fx * depth, (h - cy) / fy * depth, depth],
		[(w - cx) / fx * depth, (0 - cy) / fy * depth, depth],
	])
	pts_world = (pose.R @ pts_cam.T).T + pose.t
	return pts_world


def add_image_plane(ax, img_bgr: np.ndarray, verts: np.ndarray, alpha: float = 0.85,
					edge_color: str = "k", edge_lw: float = 1.5):
	# verts: (4,3) in order TL, BL, BR, TR
	tl, bl, br, tr = verts
	m, n = 40, 40
	us = np.linspace(0, 1, n)
	vs = np.linspace(0, 1, m)
	U, V = np.meshgrid(us, vs)
	# bilinear interpolation of 3D coords
	U3 = U[..., None]
	V3 = V[..., None]
	pts = (
		tl * (1 - U3) * (1 - V3)
		+ tr * U3 * (1 - V3)
		+ bl * (1 - U3) * V3
		+ br * U3 * V3
	)
	X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]

	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
	tex = cv2.resize(img_rgb, (n, m)) / 255.0
	ax.plot_surface(
		X, Y, Z, rstride=1, cstride=1, facecolors=tex, shade=False,
		linewidth=0, alpha=alpha, antialiased=False, zorder=0
	)
	# draw border
	borders = [(tl, bl), (bl, br), (br, tr), (tr, tl)]
	for a, b in borders:
		ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=edge_color, linewidth=edge_lw, alpha=0.9)


def draw_bbox(ax, xyz_min: np.ndarray, xyz_max: np.ndarray, color: str = "0.5", lw: float = 1.0, alpha: float = 0.5):
	x0, y0, z0 = xyz_min
	x1, y1, z1 = xyz_max
	corners = np.array([
		[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
		[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
	])
	edges = [
		(0, 1), (1, 2), (2, 3), (3, 0),
		(4, 5), (5, 6), (6, 7), (7, 4),
		(0, 4), (1, 5), (2, 6), (3, 7),
	]
	for a, b in edges:
		pa, pb = corners[a], corners[b]
		ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], color=color, linewidth=lw, alpha=alpha)


def compute_overlap_masks(tracer, T_q: Pose, T_r: Pose, cam_q, cam_r):
	ov_q = tracer.compute_overlap_pair(T_q, T_r, cam_q, cam_r)
	ov_r = tracer.compute_overlap_pair(T_r, T_q, cam_r, cam_q)
	mask_q = tracer.upsample_overlap(ov_q, cam_q) if ov_q is not None else None
	mask_r = tracer.upsample_overlap(ov_r, cam_r) if ov_r is not None else None
	return mask_q, mask_r


def image_cloud_from_renderer(renderer: Renderer, pose: Pose, cam, img_bgr: np.ndarray, stride: int = 4):
	# Project image colors onto mesh intersections for a sparse colored point cloud
	rays = compute_rays(pose, cam, stride=stride)
	p3d, valid = renderer.compute_intersections(rays)
	if p3d is None or valid is None:
		return None
	valid_idx = np.flatnonzero(valid.reshape(-1))
	if len(valid_idx) == 0:
		return None
	colors = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)[::stride, ::stride] / 255.0
	flat_colors = colors.reshape(-1, 3)
	# p3d is already compacted to valid intersections in the same scan order as valid_idx
	if len(p3d) != len(valid_idx):
		k = min(len(p3d), len(valid_idx))
		valid_idx = valid_idx[:k]
		p3d = p3d[:k]
	cols = flat_colors[valid_idx]
	return p3d, cols


def auto_plane_depth(renderer: Renderer, pose: Pose, cam, fallback: float) -> float:
	# Use the center ray to estimate a plausible plane depth that sits on the first hit
	rays = compute_rays(pose, cam, stride=max(cam.width, cam.height))
	center_idx = (len(rays[0]) // 2)
	origins = rays[0][center_idx:center_idx+1]
	directions = rays[1][center_idx:center_idx+1]
	p3d, valid = renderer.compute_intersections((origins, directions))
	if p3d is None or valid is None or valid.size == 0 or not valid[0]:
		return fallback
	depth = np.linalg.norm(p3d[0] - pose.t)
	return float(depth)


def load_image(capture: Capture, session_id: str, rel_path: str):
	img_path = capture.data_path(session_id) / rel_path
	img = cv2.imread(str(img_path))
	if img is None:
		raise FileNotFoundError(f"Cannot read image at {img_path}")
	return img


def visualize_pair(args):
	pair_file = Path(args.pairs)
	pairs = parse_pairs(pair_file)
	pair = pairs[args.pair_idx]

	capture_root = Path(args.capture_root)
	capture = Capture.load(capture_root)
	session_q = capture.sessions[args.query_session]
	session_r = capture.sessions[args.ref_session]

	q_index = index_images(session_q)
	r_index = index_images(session_r)
	if pair["q_rel"] not in q_index:
		raise KeyError(f"{pair['q_rel']} not found in {args.query_session} images")
	if pair["r_rel"] not in r_index:
		raise KeyError(f"{pair['r_rel']} not found in {args.ref_session} images")

	ts_q, cam_id_q = q_index[pair["q_rel"]]
	ts_r, cam_id_r = r_index[pair["r_rel"]]

	pose_q = get_pose_aligned(session_q, ts_q, cam_id_q)
	pose_r = get_pose_aligned(session_r, ts_r, cam_id_r)
	cam_q = session_q.sensors[cam_id_q]
	cam_r = session_r.sensors[cam_id_r]

	pyr_q = pyramid_from_camera(
		pose_q.R, pose_q.t, cam_q.width, cam_q.height, *cam_q.projection_params, args.max_depth
	)
	pyr_r = pyramid_from_camera(
		pose_r.R, pose_r.t, cam_r.width, cam_r.height, *cam_r.projection_params, args.max_depth
	)

	mesh_rel = session_r.proc.meshes.get(args.mesh_id)
	if mesh_rel is None:
		available = ", ".join(session_r.proc.meshes.keys())
		raise KeyError(f"Mesh id {args.mesh_id} not found. Available: {available}")
	mesh_path = capture.proc_path(args.ref_session) / mesh_rel
	renderer = Renderer(mesh_path)
	tracer = OverlapTracer(renderer, num_rays=args.num_rays)
	mask_q, mask_r = compute_overlap_masks(tracer, pose_q, pose_r, cam_q, cam_r)

	img_q = load_image(capture, args.query_session, pair["q_rel"])
	img_r = load_image(capture, args.ref_session, pair["r_rel"])
	vis_q = overlay(mask_q, img_q) if mask_q is not None else img_q
	vis_r = overlay(mask_r, img_r) if mask_r is not None else img_r

	show_2d = not args.hide_2d
	cols = 3 if show_2d else 1
	fig = plt.figure(figsize=(16, 6) if show_2d else (7, 7))
	ax_3d = fig.add_subplot(1, cols, cols, projection="3d")

	if show_2d:
		aq = fig.add_subplot(1, cols, 1)
		aq.imshow(cv2.cvtColor(vis_q, cv2.COLOR_BGR2RGB))
		aq.set_title(f"Query overlap")
		aq.axis("off")

		ar = fig.add_subplot(1, cols, 2)
		ar.imshow(cv2.cvtColor(vis_r, cv2.COLOR_BGR2RGB))
		ar.set_title("Reference overlap")
		ar.axis("off")

	depth_q = args.plane_depth_q if args.plane_depth_q is not None else args.plane_depth
	depth_r = args.plane_depth_r if args.plane_depth_r is not None else args.plane_depth
	if depth_q is None or depth_q <= 0:
		depth_q = 0.2 * args.max_depth
	if depth_r is None or depth_r <= 0:
		depth_r = 0.2 * args.max_depth
	if args.auto_plane_depth:
		depth_q = auto_plane_depth(renderer, pose_q, cam_q, fallback=depth_q)
		depth_r = auto_plane_depth(renderer, pose_r, cam_r, fallback=depth_r)
	plane_q = plane_from_camera(cam_q, pose_q, depth=depth_q)
	plane_r = plane_from_camera(cam_r, pose_r, depth=depth_r)
	# draw planes first (lower zorder), then frustums/axes on top for visibility
	if not args.hide_planes:
		add_image_plane(ax_3d, img_q, plane_q, alpha=args.plane_alpha, edge_color="tab:blue", edge_lw=args.plane_edge_lw)
		add_image_plane(ax_3d, img_r, plane_r, alpha=args.plane_alpha, edge_color="tab:orange", edge_lw=args.plane_edge_lw)

	add_frustum(ax_3d, pyr_q, color="tab:blue", label="query", lw=args.frustum_lw)
	add_frustum(ax_3d, pyr_r, color="tab:orange", label="ref", lw=args.frustum_lw)
	add_axes(ax_3d, pose_q)
	add_axes(ax_3d, pose_r)

	cloud_q = None
	cloud_r = None
	if args.image3d:
		cloud_q = image_cloud_from_renderer(renderer, pose_q, cam_q, img_q, stride=args.pc_stride)
		cloud_r = image_cloud_from_renderer(renderer, pose_r, cam_r, img_r, stride=args.pc_stride)
		if cloud_q is not None:
			pts, cols = cloud_q
			ax_3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=args.pc_size, alpha=0.9, depthshade=False)
		if cloud_r is not None:
			pts, cols = cloud_r
			ax_3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=args.pc_size, alpha=0.6, depthshade=False)

	pts_bounds = [pyr_q, pyr_r]
	if not args.hide_planes:
		pts_bounds.extend([plane_q, plane_r])
	if args.image3d and not args.tight_frustum:
		if cloud_q is not None:
			pts_bounds.append(cloud_q[0])
		if cloud_r is not None:
			pts_bounds.append(cloud_r[0])

	ax_3d.legend(loc="upper right")
	ax_3d.set_xlabel("X [m]")
	ax_3d.set_ylabel("Y [m]")
	ax_3d.set_zlabel("Z [m]")
	ax_3d.set_title("Camera frustums")
	ax_3d.view_init(elev=20.0, azim=-60.0)
	ax_3d.grid(True, alpha=0.2)

	plt.suptitle(
		f"err={pair['r_err']:.2f} | frustum_overlap={pair['frustum_overlap']:.3f}",
		fontsize=11,
	)
	pts_all = np.vstack(pts_bounds)
	xyz_min = pts_all.min(axis=0)
	xyz_max = pts_all.max(axis=0)
	center = 0.5 * (xyz_min + xyz_max)
	extent = (xyz_max - xyz_min).max() * 0.6
	ax_3d.set_xlim(center[0] - extent, center[0] + extent)
	ax_3d.set_ylim(center[1] - extent, center[1] + extent)
	ax_3d.set_zlim(center[2] - extent, center[2] + extent)
	ax_3d.set_box_aspect([1, 1, 1])
	if args.frame_cube:
		draw_bbox(ax_3d, xyz_min, xyz_max, color="0.4", lw=1.2, alpha=0.6)
	plt.tight_layout()
	plt.show()


def build_argparser():
	parser = argparse.ArgumentParser(description="Visualize camera frustum overlaps for a pair")
	parser.add_argument("--pairs", required=True, help="Path to *_pairs.txt file")
	parser.add_argument("--pair-idx", type=int, default=0, help="Zero-based pair index to visualize")
	parser.add_argument("--capture-root", default="/home/long/Workspace/crocodl-benchmark/capture/ARCHE_D2",
						help="Capture root that contains the sessions folder")
	parser.add_argument("--query-session", default="ios_query", help="Query session id")
	parser.add_argument("--ref-session", default="spot_map", help="Reference session id")
	parser.add_argument("--mesh-id", default="mesh_simplified", help="Mesh id inside proc/meshes")
	parser.add_argument("--max-depth", type=float, default=10.0, help="Frustum depth for visualization")
	parser.add_argument("--num-rays", type=int, default=80, help="Rays per image edge for overlap tracing")
	parser.add_argument("--plane-depth", type=float, default=None, help="Depth (m) for both image planes (<=0 to auto default)")
	parser.add_argument("--plane-depth-q", type=float, default=None, help="Depth (m) for query plane (overrides plane-depth)")
	parser.add_argument("--plane-depth-r", type=float, default=None, help="Depth (m) for ref plane (overrides plane-depth)")
	parser.add_argument("--auto-plane-depth", action="store_true", help="Estimate plane depth from center ray hit on mesh (per image)")
	parser.add_argument("--image3d", action="store_true", help="Project image into 3D colored point cloud on the mesh")
	parser.add_argument("--pc-stride", type=int, default=4, help="Stride when sampling pixels for 3D projection")
	parser.add_argument("--pc-size", type=float, default=3.0, help="Marker size for projected points")
	parser.add_argument("--frustum-lw", type=float, default=2.0, help="Line width for frustum edges")
	parser.add_argument("--plane-edge-lw", type=float, default=1.8, help="Line width for plane borders")
	parser.add_argument("--plane-alpha", type=float, default=0.85, help="Alpha for textured planes in 3D")
	parser.add_argument("--tight-frustum", action="store_true", help="Axis limits only from frustums + planes (ignore projected points)")
	parser.add_argument("--hide-2d", action="store_true", help="Hide 2D image subplots; show only 3D view")
	parser.add_argument("--hide-planes", action="store_true", help="Hide textured image planes inside the frustum")
	parser.add_argument("--frame-cube", action="store_true", help="Draw bounding cube around scene for orientation")
	return parser


if __name__ == "__main__":
	visualize_pair(build_argparser().parse_args())
