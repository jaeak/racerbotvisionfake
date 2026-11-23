# Car 1 (rb1) webcam calibration record

**Date:** 2026-11-23
**Camera:** Logitech C922 Pro Stream Webcam, `/dev/video0` (OpenCV index 0)
**Resolution:** 1280x720 (all calibration is only valid at this resolution)

If the camera is moved, bumped, refocused or replaced, redo all of this.

## 1. Intrinsics (`calibrate_intrinsics.py`)

Saved in `camera_params.yaml`. Converted for ROS (`usb_cam`) into
`camera_info_rb1.yaml` with `make_camera_info.py`.

| | Value |
|---|---|
| fx, fy | 957.35, 956.32 px |
| cx, cy | 652.73, 350.19 px |
| Distortion (k1, k2, p1, p2, k3) | 0.0367, -0.1031, -0.0014, 0.0026, -0.1645 |
| Reprojection error (reported) | 0.024 px |
| Checkerboard | 9x6 inner corners, 25 mm squares (script defaults) |

Camera controls locked first with `./set_camera_controls.sh /dev/video0`
(manual exposure, white balance and focus).

## 2. AprilTag detection check (`apriltag_test.py`)

- Tag: tag36h11, **id 2**, black square **0.167 m**
- Detected reliably. Distances checked against a tape measure and were accurate
  (example: 0.45 m straight ahead read 0.45 m).
- Command: `python3 apriltag_test.py --camera 0 --size 0.167 --params camera_params.yaml`

## 3. Extrinsics: camera position on the car (`calibrate_extrinsics_tag.py`)

Measured from `base_link` = centre of the rear axle (x forward, y left, z up).

**Inputs (tag position, measured with a tape from the rear axle):**

| | Value |
|---|---|
| tag-x (forward) | 0.725 m (tag about 0.44 m in front of the camera lens) |
| tag-y (left) | 0.0 m |
| tag-z (up) | 0.141 m (tag centre 0.195 m off floor - rear axle 0.054 m off floor) |

**Result (camera lens relative to rear axle centre):**

| | Value |
|---|---|
| Forward (x) | 0.284 m |
| Left (y) | -0.058 m (5.8 cm right of centre line) |
| Up (z) | 0.149 m above rear axle centre (~0.203 m off the floor) |
| Yaw | +8.3 deg (turned left) |
| Pitch | +3.9 deg (tilted down) |
| Roll | +0.1 deg |

Checked against a tape measure: matches.

**Static TF (base_link -> camera_optical_frame):**

```bash
ros2 run tf2_ros static_transform_publisher --x 0.2838 --y -0.0580 --z 0.1493 --qx -0.55184 --qy 0.47862 --qz -0.44617 --qw 0.51703 --frame-id base_link --child-frame-id camera_optical_frame
```

Verified with `ros2 run tf2_ros tf2_echo base_link camera_optical_frame`.
(Its RPY shows about -93.9, 0.1, -81.7 deg; that is normal for an optical
frame, which uses z forward / y down. It is the same rotation as above.)

## Notes / lessons

- **Printed tags are rotated 180 deg vs OpenCV.** Our printed tag36h11 tags
  (official images, label underneath) are drawn 180 deg rotated compared to
  OpenCV's `DICT_APRILTAG_36h11`. OpenCV still reads the right ID, but the
  pose comes out upside down (roll ~180) unless corrected.
  `calibrate_extrinsics_tag.py` corrects this by default. Anything else using
  OpenCV's AprilTag detector for pose (e.g. `opponent_perception`) needs the
  same care.
- **Keep the tag roughly 0.4-0.8 m from the camera for extrinsics.** Far tags
  (1.5 m+) give unreliable angles. The first run (before the 180 deg fix) gave a
  wrong camera position (0.56 m forward); the final run, checked by tape, is
  the one recorded above.
- `/dev/video1` is the same webcam's metadata node, not pictures. Use
  `/dev/video0` / index 0.

## Starting the calibrated camera in ROS

```bash
ros2 run usb_cam usb_cam_node_exe --ros-args \
  -p video_device:=/dev/video0 -p image_width:=1280 -p image_height:=720 \
  -p pixel_format:=mjpeg2rgb -p frame_id:=camera_optical_frame \
  -p camera_name:=webcam \
  -p camera_info_url:=file://$HOME/racerbot-vision/src/camera_info_rb1.yaml
```

## Running AprilTag tracking (one command)

```bash
source /opt/ros/jazzy/setup.bash
source ~/racerbot-ws/install/setup.bash
ros2 launch ~/racerbot-vision/src/apriltag_rb1_launch.py            # parked car (bench test)
ros2 launch ~/racerbot-vision/src/apriltag_rb1_launch.py bench:=false  # car bringup running
```

Starts the camera TF, `webcam_publisher.py`, and `opponent_perception` with
`apriltag_bench_rb1.yaml`. Check with
`ros2 topic echo --once --full-length /perception/opponents`.
Bench test 2026-11-23: status ok, tag at 0.725 m read `position_m [0.727, 0.001]`,
yaw 0.086 rad, speed ~0, reprojection 0.89 px.

## AprilTag tracking accuracy (fill in)

Tag 2 upright, facing the car. Tape measured from the floor point under the
rear axle centre (forward along the centre line, left = +, right = -).
`position_m` from `/perception/opponents`.

| # | Tape forward (m) | Tape sideways (m) | position_m forward | position_m left | Error forward (cm) | Error sideways (cm) |
|---|---|---|---|---|---|---|
| 1 | 0.725 | 0.000 | 0.731 | +0.011 | +0.6 | +1.1 |
| 2 | 0.860 | 0.000 | 0.865 | +0.029 | +0.5 | +2.9 |
| 3 | 0.660 | +0.085 | 0.666 | +0.114 | +0.6 | +2.9 |
| 4 | 0.660 | -0.105 | 0.690 | -0.083 | +3.0 | +2.2 |

Result (2026-11-23): every reading within 3 cm of the tape. Forward error is
about +0.5 cm except row 4 (+3 cm). Sideways readings sit consistently about
2 cm to the left of the tape, a small fixed offset (likely the camera's
sideways position or yaw from the extrinsic calibration).
Row 1 is the calibration spot, so it isn't an independent check.
Tested only from 0.66 to 0.86 m (limited floor space); longer range
(1-3 m, typical racing distance) still to be tested.
Sign check: `position_m` uses left = +. The sheet column was labelled
"right = +" but its signs match the readings, so confirm which side the tag
was physically on for row 3 (left of the car's centre line expected).
