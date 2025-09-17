#!/usr/bin/env python3
"""
ROS2 Obstacle Radial UI
=======================

A simple Tkinter-based UI that shows an 8-sector radial display around the robot
(two distance rings × four quadrants: front/right/back/left). Colors:

- Human (camera, 3D bbox): NEAR=red, MEDIUM=orange (priority over LiDAR)
- LiDAR (point cloud, size-filtered): CLOSE=orange, MEDIUM=yellow
- Empty: dark gray background for sector

Assumptions & Notes
-------------------
- Camera publishes 3D human detections as vision_msgs/Detection3DArray (center in robot base frame).
- LiDAR publishes sensor_msgs/PointCloud2 in the robot base frame.
- "Front" is +X, "Left" is +Y, consistent with ROS REP-103.
- No TF transforms are applied. If your topics are not in base_frame, adapt `transform_points(...)` and
  `transform_pose(...)` accordingly.
- Minimal deps: rclpy, numpy, Tkinter (stdlib), sensor_msgs_py.point_cloud2.

Run
---
  python3 ros2_obstacle_radial_ui.py \
    --camera_topic /detections \
    --lidar_topic /points \
    --base_frame base_link \
    --near_radius 1.5 \
    --medium_radius 3.0

Press `q` in the UI window or Ctrl+C in the terminal to quit.

Replaceable LiDAR Obstacle Filter
---------------------------------
`detect_obstacles_from_pointcloud(...)` is a self-contained function. Swap it out with your
preferred clustering/size filter later without touching the rest of the code.

"""

from __future__ import annotations
import math
import sys
import threading
import time
import argparse
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

import numpy as np
import os
import csv
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import MarkerArray
import tf2_ros
from rclpy.time import Time
# vision_msgs is standard but optional depending on your image pipeline
try:
    from vision_msgs.msg import Detection3DArray
    HAVE_VISION_MSGS = True
except Exception:
    HAVE_VISION_MSGS = False

# ------------------------------ Config & Types ------------------------------ #

QUADRANTS = ("front", "right", "back", "left")

@dataclass
class SectorFlags:
    human_near: bool = False
    human_medium: bool = False
    lidar_close: bool = False
    lidar_medium: bool = False

    def effective_color(self) -> str:
        """Return hex color for UI based on priority rules."""
        # Priority: Human near (red) > Human medium (orange) > LiDAR close (orange) > LiDAR medium (yellow)
        if self.human_near:
            return "#e53935"  # red
        if self.human_medium:
            return "#fb8c00"  # orange
        if self.lidar_close:
            return "#fb8c00"  # orange
        if self.lidar_medium:
            return "#fdd835"  # yellow
        return "#444444"      # empty

@dataclass
class WorldConfig:
    base_frame: str = "base_link"
    near_radius: float = 1.5
    medium_radius: float = 3.0
    lidar_min_points_close: int = 40
    lidar_min_points_medium: int = 60
    lidar_min_extent_m: float = 0.20  # legacy param (kept for compatibility)
    # New LiDAR size filter & clustering params
    lidar_voxel_m: float = 0.15
    lidar_min_cluster_points: int = 30
    lidar_min_width_m: float = 0.25
    lidar_min_length_m: float = 0.25
    lidar_sector_min_points: int = 5

@dataclass
class WorldState:
    # sector -> flags
    sectors: Dict[str, SectorFlags] = field(default_factory=lambda: {q: SectorFlags() for q in QUADRANTS})
    last_update_time: float = field(default_factory=time.time)

    def clear_dynamic(self):
        for s in self.sectors.values():
            s.human_near = s.human_medium = False
            s.lidar_close = s.lidar_medium = False

# ------------------------------ Metrics ------------------------------------ #

@dataclass
class SectorStats:
    time_human_near: float = 0.0
    time_human_medium: float = 0.0
    time_lidar_close: float = 0.0
    time_lidar_medium: float = 0.0
    count_human_near: int = 0
    count_human_medium: int = 0
    count_lidar_close: int = 0
    count_lidar_medium: int = 0
    prev_human_near: bool = False
    prev_human_medium: bool = False
    prev_lidar_close: bool = False
    prev_lidar_medium: bool = False

class Metrics:
    def __init__(self, sectors=QUADRANTS, report_csv: Optional[str] = None):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.stats: Dict[str, SectorStats] = {q: SectorStats() for q in sectors}
        self.report_csv = report_csv
        self._wrote_header = False

    def reset(self):
        self.__init__(sectors=list(self.stats.keys()), report_csv=self.report_csv)

    def update(self, current: Dict[str, SectorFlags], now: Optional[float] = None):
        t = now if now is not None else time.time()
        dt = max(0.0, t - self.last_time)
        self.last_time = t
        if dt == 0.0:
            return
        for q, flags in current.items():
            st = self.stats[q]
            # accumulate time per active flag
            if flags.human_near:
                st.time_human_near += dt
            if flags.human_medium:
                st.time_human_medium += dt
            if flags.lidar_close:
                st.time_lidar_close += dt
            if flags.lidar_medium:
                st.time_lidar_medium += dt
            # rising edges -> counts
            if flags.human_near and not st.prev_human_near:
                st.count_human_near += 1
            if flags.human_medium and not st.prev_human_medium:
                st.count_human_medium += 1
            if flags.lidar_close and not st.prev_lidar_close:
                st.count_lidar_close += 1
            if flags.lidar_medium and not st.prev_lidar_medium:
                st.count_lidar_medium += 1
            # store prev
            st.prev_human_near = flags.human_near
            st.prev_human_medium = flags.human_medium
            st.prev_lidar_close = flags.lidar_close
            st.prev_lidar_medium = flags.lidar_medium

    def elapsed(self) -> float:
        return max(1e-6, self.last_time - self.start_time)

    def percentages(self) -> Dict[str, Dict[str, float]]:
        total = self.elapsed()
        out: Dict[str, Dict[str, float]] = {}
        for q, st in self.stats.items():
            out[q] = {
                'H_near_pct': 100.0 * st.time_human_near / total,
                'H_med_pct':  100.0 * st.time_human_medium / total,
                'L_near_pct': 100.0 * st.time_lidar_close / total,
                'L_med_pct':  100.0 * st.time_lidar_medium / total,
            }
        return out

    def save_snapshot_csv(self):
        if not self.report_csv:
            return None
        os.makedirs(os.path.dirname(self.report_csv) or '.', exist_ok=True)
        write_header = not self._wrote_header or not os.path.exists(self.report_csv)
        with open(self.report_csv, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                header = [
                    'timestamp_iso', 'elapsed_s',
                ]
                for q in QUADRANTS:
                    header += [
                        f'{q}_Hnear_pct', f'{q}_Hmed_pct', f'{q}_Lnear_pct', f'{q}_Lmed_pct',
                        f'{q}_Hnear_cnt', f'{q}_Hmed_cnt', f'{q}_Lnear_cnt', f'{q}_Lmed_cnt',
                    ]
                w.writerow(header)
                self._wrote_header = True
            pct = self.percentages()
            row = [datetime.utcnow().isoformat(), f"{self.elapsed():.3f}"]
            for q in QUADRANTS:
                st = self.stats[q]
                row += [
                    f"{pct[q]['H_near_pct']:.2f}", f"{pct[q]['H_med_pct']:.2f}", f"{pct[q]['L_near_pct']:.2f}", f"{pct[q]['L_med_pct']:.2f}",
                    st.count_human_near, st.count_human_medium, st.count_lidar_close, st.count_lidar_medium,
                ]
            w.writerow(row)
        return self.report_csv

# ------------------------------ Geometry Utils ----------------------------- #

def angle_to_quadrant(theta: float) -> str:
    """Map angle (rad), x-forward, y-left, to one of front/right/back/left.
    theta from atan2(y, x), range [-pi, pi].
    """
    deg = math.degrees(theta) % 360.0
    # Define 4 quadrants centered on 0°(front), 90°(left), 180°(back), 270°(right)
    # We'll use half-open intervals to avoid ambiguity on boundaries.
    if 315.0 <= deg or deg < 45.0:
        return "front"
    elif 45.0 <= deg < 135.0:
        return "left"
    elif 135.0 <= deg < 225.0:
        return "back"
    else:
        return "right"

# ------------------------------ LiDAR Filter ------------------------------- #

def detect_obstacles_from_pointcloud(
    xyz: np.ndarray,
    near_radius: float,
    medium_radius: float,
    voxel_m: float = 0.15,
    min_cluster_points: int = 30,
    min_width_m: float = 0.25,
    min_length_m: float = 0.25,
    sector_min_points: int = 5,
) -> Dict[str, Tuple[bool, bool]]:
    """Cluster LiDAR points in 2D and flag sectors by ring.

    Returns dict quadrant -> (near_close, medium).
    Small clusters are rejected via voxel-grid connected components and bbox size checks.
    """
    result = {q: (False, False) for q in QUADRANTS}
    if xyz.size == 0:
        return result

    # Clean
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if xyz.size == 0:
        return result

    x = xyz[:, 0]
    y = xyz[:, 1]

    # --- Build voxel grid (2D) ---
    ix = np.floor(x / voxel_m).astype(np.int32)
    iy = np.floor(y / voxel_m).astype(np.int32)
    grid: Dict[Tuple[int, int], list] = {}
    for i, (gx, gy) in enumerate(zip(ix, iy)):
        grid.setdefault((gx, gy), []).append(i)

    visited = set()
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    def quadrant_mask(thetas: np.ndarray, q: str) -> np.ndarray:
        if q == "front":
            return (thetas < np.deg2rad(45)) & (thetas >= -np.deg2rad(45))
        if q == "left":
            return (thetas >= np.deg2rad(45)) & (thetas < np.deg2rad(135))
        if q == "back":
            return (thetas >= np.deg2rad(135)) | (thetas < -np.deg2rad(135))
        # right
        return (thetas >= -np.deg2rad(135)) & (thetas < -np.deg2rad(45))

    # --- Connected components over voxels ---
    for cell in list(grid.keys()):
        if cell in visited:
            continue
        # BFS over cells
        comp_cells = []
        stack = [cell]
        visited.add(cell)
        while stack:
            c = stack.pop()
            comp_cells.append(c)
            cx, cy = c
            for dx, dy in neighbors:
                n = (cx + dx, cy + dy)
                if n in grid and n not in visited:
                    visited.add(n)
                    stack.append(n)
        # Collect points in this component
        idx = [i for c in comp_cells for i in grid[c]]
        if len(idx) < min_cluster_points:
            continue

        qx = x[idx]
        qy = y[idx]
        # Bounding box size
        width = float(np.max(qx) - np.min(qx))
        length = float(np.max(qy) - np.min(qy))
        if width < min_width_m and length < min_length_m:
            continue

        thetas = np.arctan2(qy, qx)
        radii = np.hypot(qx, qy)

        # For each quadrant, decide near/medium based on #points in that ring within this cluster
        for q in QUADRANTS:
            qm = quadrant_mask(thetas, q)
            if not np.any(qm):
                continue
            rq = radii[qm]
            # Count points per ring
            near_pts = int(np.sum(rq <= near_radius))
            med_pts = int(np.sum((rq > near_radius) & (rq <= medium_radius)))

            threshold = max(1, min(sector_min_points, len(idx)))
            nclose = near_pts >= threshold
            nmed = med_pts >= threshold

            old_close, old_med = result[q]
            result[q] = (old_close or nclose, old_med or nmed)

    return result

# ------------------------------ ROS Node ----------------------------------- #

class ObstacleUI(Node):
    def __init__(self, cfg: WorldConfig, camera_topic: str, lidar_topic: str, camera_msg: str = "marker", use_camera: bool = True):
        super().__init__("obstacle_radial_ui")
        self.cfg = cfg
        self.state = WorldState()
        self._lock = threading.Lock()

        # TF buffer/listener for frame transforms (e.g., front_lidar_link -> base_link)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers
        if use_camera:
            if camera_msg == "vision" and HAVE_VISION_MSGS:
                self.create_subscription(Detection3DArray, camera_topic, self.on_detections, qos_profile=qos_profile_sensor_data)
                self.get_logger().info(f"Subscribed to camera (Detection3DArray): {camera_topic}")
            elif camera_msg == "marker":
                self.create_subscription(MarkerArray, camera_topic, self.on_marker_array, qos_profile=qos_profile_sensor_data)
                self.get_logger().info(f"Subscribed to camera (MarkerArray): {camera_topic}")
            else:
                self.get_logger().warn("Camera subscription disabled or message type not available.")

        self.create_subscription(PointCloud2, lidar_topic, self.on_pointcloud, qos_profile=qos_profile_sensor_data)
        self.get_logger().info(f"Subscribed to LiDAR pointcloud: {lidar_topic}")

    # -------------------------- Callbacks ---------------------------------- #

    def _apply_quat(self, qx, qy, qz, qw, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Rotate vector v by quaternion q (unit)."""
        vx, vy, vz = v
        # t = 2 * cross(q.xyz, v)
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)
        # v' = v + qw * t + cross(q.xyz, t)
        vpx = vx + qw * tx + (qy * tz - qz * ty)
        vpy = vy + qw * ty + (qz * tx - qx * tz)
        vpz = vz + qw * tz + (qx * ty - qy * tx)
        return (vpx, vpy, vpz)

    def transform_xyz(self, x: float, y: float, z: float, from_frame: str) -> Optional[Tuple[float, float, float]]:
        if not from_frame or from_frame == self.cfg.base_frame:
            return (x, y, z)
        try:
            ts = self.tf_buffer.lookup_transform(self.cfg.base_frame, from_frame, Time())
            tx = ts.transform.translation.x
            ty = ts.transform.translation.y
            tz = ts.transform.translation.z
            qx = ts.transform.rotation.x
            qy = ts.transform.rotation.y
            qz = ts.transform.rotation.z
            qw = ts.transform.rotation.w
            rx, ry, rz = self._apply_quat(qx, qy, qz, qw, (x, y, z))
            return (rx + tx, ry + ty, rz + tz)
        except Exception as e:
            self.get_logger().debug(f"TF transform failed {from_frame}->{self.cfg.base_frame}: {e}")
            return None

    def on_detections(self, msg: 'Detection3DArray'):
        try:
            with self._lock:
                for s in self.state.sectors.values():
                    s.human_near = s.human_medium = False

                for det in msg.detections:
                    cx = det.bbox.center.position.x
                    cy = det.bbox.center.position.y
                    cz = det.bbox.center.position.z
                    frame = msg.header.frame_id if hasattr(msg, 'header') else self.cfg.base_frame
                    xyz_t = self.transform_xyz(cx, cy, cz, frame)
                    if xyz_t is None:
                        continue
                    x_t, y_t, _ = xyz_t
                    r = math.hypot(x_t, y_t)
                    th = math.atan2(y_t, x_t)
                    quad = angle_to_quadrant(th)
                    flags = self.state.sectors[quad]
                    if r <= self.cfg.near_radius:
                        flags.human_near = True
                    elif r <= self.cfg.medium_radius:
                        flags.human_medium = True

                self.state.last_update_time = time.time()
        except Exception as e:
            self.get_logger().error(f"Error in on_detections: {e}")

    def on_marker_array(self, msg: MarkerArray):
        try:
            with self._lock:
                for s in self.state.sectors.values():
                    s.human_near = s.human_medium = False

                count = 0
                for mk in msg.markers:
                    px = mk.pose.position.x
                    py = mk.pose.position.y
                    pz = mk.pose.position.z
                    frame = mk.header.frame_id if mk.header.frame_id else self.cfg.base_frame
                    xyz_t = self.transform_xyz(px, py, pz, frame)
                    if xyz_t is None:
                        continue
                    x_t, y_t, _ = xyz_t
                    r = math.hypot(x_t, y_t)
                    th = math.atan2(y_t, x_t)
                    quad = angle_to_quadrant(th)
                    flags = self.state.sectors[quad]
                    if r <= self.cfg.near_radius:
                        flags.human_near = True
                    elif r <= self.cfg.medium_radius:
                        flags.human_medium = True
                    count += 1
                self.state.last_update_time = time.time()
        except Exception as e:
            self.get_logger().error(f"Error in on_marker_array: {e}")

    def on_pointcloud(self, msg: PointCloud2):
        try:
            # Extract XYZ (ignore NaNs)
            pts = []
            for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                pts.append((p[0], p[1], p[2]))
            xyz = np.asarray(pts, dtype=np.float32) if len(pts) > 0 else np.empty((0, 3), dtype=np.float32)

            detections = detect_obstacles_from_pointcloud(
                xyz,
                near_radius=self.cfg.near_radius,
                medium_radius=self.cfg.medium_radius,
                voxel_m=self.cfg.lidar_voxel_m,
                min_cluster_points=self.cfg.lidar_min_cluster_points,
                min_width_m=self.cfg.lidar_min_width_m,
                min_length_m=self.cfg.lidar_min_length_m,
                sector_min_points=self.cfg.lidar_sector_min_points,
            )

            with self._lock:
                # Reset only lidar flags; preserve human flags until next camera update
                for q, (is_close, is_medium) in detections.items():
                    self.state.sectors[q].lidar_close = bool(is_close)
                    self.state.sectors[q].lidar_medium = bool(is_medium)
                self.state.last_update_time = time.time()
        except Exception as e:
            self.get_logger().error(f"Error in on_pointcloud: {e}")

# ------------------------------ UI (Tkinter) -------------------------------- #

def ring_color(flags: SectorFlags, ring: str) -> str:
    """Return hex color for a specific ring based on priority rules.
    Inner (near):  Human near (red) > LiDAR near (orange)
    Outer (medium): Human medium (dark orange) > LiDAR medium (yellow)
    """
    if ring == "near":
        if flags.human_near:
            return "#e53935"  # red
        if flags.lidar_close:
            return "#fb8c00"  # orange
        return "#444444"
    else:  # medium
        if flags.human_medium:
            return "#fb4c00"  # dark orange
        if flags.lidar_medium:
            return "#fdd835"  # yellow
        return "#444444"

try:
    import tkinter as tk
except Exception as e:
    print("Error: Tkinter is required for the UI (usually included with Python).", file=sys.stderr)
    raise

class RadialUI:
    def __init__(self, node: ObstacleUI, width: int = 420, height: int = 500, report_csv: Optional[str] = None, autosave_sec: float = 0.0, overlay: bool = False):
        self.node = node
        self.w = width
        self.h = height
        self.r_outer = min(self.w, self.h) * 0.40
        self.r_inner = self.r_outer * (node.cfg.near_radius / node.cfg.medium_radius)

        self.root = tk.Tk()
        self.root.title("Obstacle Radial UI")
        self.canvas = tk.Canvas(self.root, width=self.w, height=self.h, bg="#202124", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Metrics/overlay config
        self.metrics = Metrics(report_csv=report_csv)
        self.autosave_sec = autosave_sec
        self._last_autosave = time.time()
        self.overlay = overlay

        self.root.bind("q", lambda e: self.root.destroy())
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

        # Schedule periodic redraw
        self._tick()

    def _tick(self):
        try:
            # Update metrics snapshot
            with self.node._lock:
                snap = {q: SectorFlags(**vars(s)) for q, s in self.node.state.sectors.items()}
            self.metrics.update(snap)
            # Autosave CSV if requested
            if self.autosave_sec and (time.time() - self._last_autosave) >= self.autosave_sec:
                path = self.metrics.save_snapshot_csv()
                if path:
                    print(f"[metrics] autosaved to {path}")
                self._last_autosave = time.time()
            # Redraw UI
            self.redraw()
        except Exception as e:
            print(f"UI redraw error: {e}")
        # ~10 FPS
        self.root.after(100, self._tick)

    def run(self):
        self.root.mainloop()

    def redraw(self):
        self.canvas.delete("all")
        cx, cy = self.w // 2, self.h // 2

        # Draw rings (outlines)
        self._draw_circle(cx, cy, self.r_outer, outline="#666666")
        self._draw_circle(cx, cy, self.r_inner, outline="#666666")

        # Draw cross lines (front/back/left/right)
        self.canvas.create_line(cx - self.r_outer, cy, cx + self.r_outer, cy, fill="#666666")
        self.canvas.create_line(cx, cy - self.r_outer, cx, cy + self.r_outer, fill="#666666")

        # Labels
        self.canvas.create_text(cx, cy - self.r_outer - 12, text="FRONT", fill="#cccccc")
        self.canvas.create_text(cx + self.r_outer + 24, cy, text="RIGHT", fill="#cccccc", angle=270)
        self.canvas.create_text(cx, cy + self.r_outer + 12, text="BACK", fill="#cccccc")
        self.canvas.create_text(cx - self.r_outer - 24, cy, text="LEFT", fill="#cccccc", angle=90)

        # Fetch state snapshot
        with self.node._lock:
            sectors = {q: SectorFlags(**vars(s)) for q, s in self.node.state.sectors.items()}

        # Map quadrants to start angles in Tk (0° at 3 o'clock, CCW positive)
        arcs = {
            "front":  (90 - 45, 90),   # 45°..135° (upwards)
            "left":   (180 - 45, 90),  # 135°..225°
            "back":   (270 - 45, 90),  # 225°..315°
            "right":  (360 - 45, 90),  # 315°..45°
        }

        bg = "#202124"

        for q, (start_deg, extent_deg) in arcs.items():
            # Compute colors per ring independently
            near_color = ring_color(sectors[q], ring="near")
            med_color = ring_color(sectors[q], ring="medium")
            # MEDIUM ring segment (outer annulus sector)
            self._draw_annular_sector(cx, cy, self.r_inner, self.r_outer, start_deg, extent_deg, fill=med_color, bg=bg)
            # NEAR ring segment (inner solid sector)
            self._draw_sector(cx, cy, 0, self.r_inner, start_deg, extent_deg, fill=near_color)

        # Legend
        legend = [
            ("Human near", "#e53935"),
            ("Human medium (dark)", "#fb4c00"),
            ("LiDAR near", "#fb8c00"),
            ("LiDAR medium", "#fdd835"),
        ]
        lx, ly = 12, 16
        for label, col in legend:
            self.canvas.create_rectangle(lx, ly - 8, lx + 16, ly + 8, fill=col, outline="")
            self.canvas.create_text(lx + 24, ly, text=label, fill="#dddddd", anchor=tk.W)
            ly += 20

        # Robot center
        self._draw_circle(cx, cy, 6, fill="#bbbbbb", outline="")

        # Metrics overlay (percent active time and counts)
        if self.overlay:
            try:
                pct = self.metrics.percentages()
                lines = [f"elapsed: {self.metrics.elapsed():.1f}s"]
                for q in ("front", "right", "back", "left"):
                    st = self.metrics.stats[q]
                    p = pct[q]
                    lines.append(
                        f"{q[:1].upper()} | "
                        f"Hn {p['H_near_pct']:.0f}% ({st.count_human_near})  "
                        f"Hm {p['H_med_pct']:.0f}% ({st.count_human_medium})  "
                        f"Ln {p['L_near_pct']:.0f}% ({st.count_lidar_close})  "
                        f"Lm {p['L_med_pct']:.0f}% ({st.count_lidar_medium})"
                    )
                block = "\n".join(lines)
                # Place below the circle if possible; otherwise pin above bottom margin
                line_h = 14
                block_h = line_h * len(lines)
                y0 = int(cy + self.r_outer + 12)
                if y0 + block_h > self.h - 8:
                    y0 = max(8, self.h - 8 - block_h)
                x0 = 8
                self.canvas.create_text(x0, y0, text=block, fill="#9e9e9e", anchor=tk.NW, font=("TkDefaultFont", 9))
            except Exception as ex:
                # keep overlay non-fatal
                self.canvas.create_text(8, self.h - 8, text=f"overlay error: {ex}", fill="#f28b82", anchor=tk.SW, font=("TkDefaultFont", 9))

    # -------------------------- Drawing helpers ---------------------------- #
    def _draw_circle(self, cx, cy, r, fill="", outline="#888888"):
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=fill, outline=outline)

    def _draw_sector(self, cx, cy, r_inner, r_outer, start_deg, extent_deg, fill):
        # For a solid sector (r_inner=0), just an arc with style=PIESLICE
        bbox = (cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer)
        self.canvas.create_arc(*bbox, start=start_deg, extent=extent_deg, fill=fill, outline="", style=tk.PIESLICE)

    def _draw_annular_sector(self, cx, cy, r_inner, r_outer, start_deg, extent_deg, fill, bg):
        # Draw outer sector then carve inner with a bg-colored sector
        bbox_outer = (cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer)
        self.canvas.create_arc(*bbox_outer, start=start_deg, extent=extent_deg, fill=fill, outline="", style=tk.PIESLICE)
        bbox_inner = (cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner)
        self.canvas.create_arc(*bbox_inner, start=start_deg, extent=extent_deg, fill=bg, outline=bg, style=tk.PIESLICE)

# ------------------------------ Main --------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="ROS2 Obstacle Radial UI")
    parser.add_argument("--camera_topic", type=str, default="/detections", help="Camera topic (Detection3DArray or MarkerArray)")
    parser.add_argument("--camera_msg", type=str, choices=["marker", "vision"], default="marker", help="Camera message type")
    parser.add_argument("--cam_allow_no_tf", action="store_true", help="If TF missing, treat camera positions as already in base frame (fallback)")
    parser.add_argument("--lidar_topic", type=str, default="/points", help="sensor_msgs/PointCloud2 topic")
    parser.add_argument("--base_frame", type=str, default="base_link")
    parser.add_argument("--near_radius", type=float, default=1.5)
    parser.add_argument("--medium_radius", type=float, default=3.0)
    # legacy heuristic knobs (kept for compat)
    parser.add_argument("--lidar_min_points_close", type=int, default=40)
    parser.add_argument("--lidar_min_points_medium", type=int, default=60)
    parser.add_argument("--lidar_min_extent_m", type=float, default=0.20)
    # new clustering + size filter
    parser.add_argument("--lidar_voxel_m", type=float, default=0.15, help="Voxel size (m) for 2D clustering")
    parser.add_argument("--lidar_min_cluster_points", type=int, default=30, help="Minimum points per LiDAR cluster")
    parser.add_argument("--lidar_min_width_m", type=float, default=0.25, help="Minimum bbox width (m) for a cluster")
    parser.add_argument("--lidar_min_length_m", type=float, default=0.25, help="Minimum bbox length (m) for a cluster")
    parser.add_argument("--lidar_sector_min_points", type=int, default=5, help="Minimum points of that cluster in a specific sector+ring")
    # reporting
    parser.add_argument("--report_csv", type=str, default="", help="Path to CSV for periodic snapshots (empty=disabled)")
    parser.add_argument("--report_autosave_sec", type=float, default=0.0, help="Autosave interval in seconds (0=off)")
    parser.add_argument("--report_overlay", action="store_true", help="Show on-screen percentages/counts overlay")
    parser.add_argument("--no_camera", action="store_true", help="Disable camera subscriber")

    args = parser.parse_args()

    cfg = WorldConfig(
        base_frame=args.base_frame,
        near_radius=args.near_radius,
        medium_radius=args.medium_radius,
        lidar_min_points_close=args.lidar_min_points_close,
        lidar_min_points_medium=args.lidar_min_points_medium,
        lidar_min_extent_m=args.lidar_min_extent_m,
        lidar_voxel_m=args.lidar_voxel_m,
        lidar_min_cluster_points=args.lidar_min_cluster_points,
        lidar_min_width_m=args.lidar_min_width_m,
        lidar_min_length_m=args.lidar_min_length_m,
        lidar_sector_min_points=args.lidar_sector_min_points,
    )

    rclpy.init()
    node = ObstacleUI(cfg, args.camera_topic, args.lidar_topic, camera_msg=args.camera_msg, use_camera=not args.no_camera)

    # ROS2 executor on background thread
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    try:
        ui = RadialUI(node,
                report_csv=(args.report_csv or None),
                autosave_sec=args.report_autosave_sec,
                overlay=args.report_overlay)
        ui.run()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
