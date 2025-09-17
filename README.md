# ROS2 Obstacle Radial UI

A lightweight Tkinter UI for ROS2 that visualises obstacles around the robot in 8 sectors
(4 directions × 2 distance rings). It fuses **camera human bboxes** and **LiDAR point clouds**
with clear priority rules and includes a built-in **metrics overlay** and optional **CSV reporting**.

## Features
- 8-sector radial map (front/right/back/left × near/medium)
- Priority & colors per ring:
  - **Inner (near):** Human red > LiDAR orange
  - **Outer (medium):** Human dark-orange > LiDAR yellow
- LiDAR obstacle filter: voxel-grid + connected-components with size gating
- Camera inputs: `visualization_msgs/MarkerArray` (default) or `vision_msgs/Detection3DArray`
- TF transform of camera markers into `--base_frame`
- Metrics overlay (time-on % and counts) + CSV autosave

## Install
Ensure your ROS2 environment is sourced. Requires Python3 with Tkinter, `numpy`, and ROS2:
```bash
sudo apt-get install python3-tk
pip3 install numpy --user
```

## Run
```bash
python3 obstacle_ui.py   --camera_topic /test_bbox_front   --camera_msg marker   --lidar_topic /front_lidar/points   --base_frame front_lidar_link   --near_radius 1.5   --medium_radius 3.0   --report_csv obstacle_report.csv   --report_autosave_sec 10   --report_overlay
```
> Tip: **mind the backslashes** for line continuation in bash.

## Controls
- `q` — quit
- `o` — toggle metrics overlay on/off (if `--report_overlay` was passed)
- `s` — save a CSV snapshot immediately (requires `--report_csv`)
- `r` — reset metrics (keeps CSV path)

## CLI Reference
- `--camera_topic` (str): MarkerArray or Detection3DArray topic
- `--camera_msg` (`marker`|`vision`): message type
- `--lidar_topic` (str): PointCloud2
- `--base_frame` (str): target frame for binning (e.g., `base_link` or `front_lidar_link`)
- `--near_radius` (m), `--medium_radius` (m)
- LiDAR clustering:
  - `--lidar_voxel_m` (default 0.15)
  - `--lidar_min_cluster_points` (default 30)
  - `--lidar_min_width_m`, `--lidar_min_length_m` (defaults 0.25)
  - `--lidar_sector_min_points` (default 5)
- Reporting:
  - `--report_overlay` (flag)
  - `--report_csv` (path)
  - `--report_autosave_sec` (seconds; 0=off)
  - `--overlay_margin_px` (pixels; extra margin below the circle for overlay text)

## Adjust overlay position (avoid covering the circle)
Use the new flag to move the overlay lower:
```bash
--overlay_margin_px 32   # try 32–48 for "slightly lower"
```
Or change it permanently by editing the call site in `main()`:
```python
ui = RadialUI(node, ..., overlay_margin_px=32)
```
The draw code computes:
```
y0 = cy + r_outer + overlay_margin_px
```
and clamps near the bottom if it would go off-screen.

## CSV format (example columns)
```
timestamp_iso, elapsed_s,
front_Hnear_pct, front_Hmed_pct, front_Lnear_pct, front_Lmed_pct, front_Hnear_cnt, front_Hmed_cnt, front_Lnear_cnt, front_Lmed_cnt,
right_..., back_..., left_...
```

## Troubleshooting
- **Nothing from camera in UI but visible in RViz:** check TF from camera frame to `--base_frame`, or run with `--base_frame` set to the camera’s frame. You can also add a temporary `--cam_allow_no_tf` fallback (if present in your version).
- **Best Effort publishers:** the node subscribes with `qos_profile_sensor_data` (BestEffort), compatible with Reliable publishers too.
- **KeyboardInterrupt on Ctrl+C:** Prefer `q` to quit. The code now guards against double shutdown, but closing the window is cleanest.

## License
MIT 
