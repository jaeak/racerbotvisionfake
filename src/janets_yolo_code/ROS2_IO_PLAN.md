# Proposed ROS 2 input/output & deployment plan

**Status: DESIGN PROPOSAL — NOT DEPLOYED.** Existing architecture drawing shows LiDAR/odometry/localization nodes but not a confirmed webcam perception node or its topic names. Validate all names against the running system. Webcam group and RealSense group are treated as separate pipelines pending team decisions.

## 1. Proposed graph

```text
Car 1 UVC webcam (/dev/video0) [if Car 1 is confirmed target]
   -> usb_cam or other V4L2 ROS 2 camera driver (ONE owner of device)
   -> /camera/image_raw         [sensor_msgs/msg/Image]
   -> /webcam_perception        [cv_bridge -> BGR -> perception_core.process_frame]
   -> /webcam/detections       [vision_msgs/msg/Detection2DArray]
   -> future tracking / planning subscribers

Camera intrinsics after physical calibration:
   /camera/camera_info         [sensor_msgs/msg/CameraInfo]
```

**Important:** names are proposed, not discovered. If ROS camera driver already exists, subscribe to its real topic and avoid running another `cv2.VideoCapture` instance on the same device.

## 2. I/O contract

| Interface | Producer -> Consumer | Message/data | Status |
| --- | --- | --- | --- |
| Webcam video | Camera driver -> webcam perception | `sensor_msgs/msg/Image` BGR8/RGB8 (convert via cv_bridge), header stamp, frame_id | Proposed |
| Camera calibration | Calibration publisher/driver -> geometry module | `sensor_msgs/msg/CameraInfo` (`k`, `d` etc.) | Later |
| Detections | webcam perception -> tracker/consumers | `vision_msgs/msg/Detection2DArray`, one detection with `bbox` and `results` per object | Proposed |
| Annotated video | perception -> debugging only | optional `sensor_msgs/msg/Image` topic `/webcam/debug_image` | Optional |
| Ground pose | IPM/geometry module -> fusion | frame-referenced metric coordinates, uncertainty, timestamp | TBD with team |
| Tracked speed | tracker -> fusion/planner | track ID, velocity, uncertainties, ego-motion-compensated, timestamps | TBD with team |

For each `Detection2D` use a 2D box center and size in **pixels**, with classification in `results`. Leave `id` unset until tracking provides stable IDs. Preserve the incoming camera image's stamp and frame_id for synchronization; do not substitute inference completion time. Do **not** fill 3D pose or velocity with guessed values.

## 3. Questions requiring team agreement

1. Is webcam deployed on Car 1, or will a webcam be mounted on Car 2?
2. What machine runs inference? CPU/GPU specifications, OS, Python and ROS 2 version?
3. Is the webcam already published via ROS 2? Exact topic and camera driver?
4. Required image resolution/FPS, allowed latency and target lighting?
5. Which downstream node consumes detections? Pixel boxes or metric ground positions? Required coordinate frame?
6. Will another team run object tracking, position fusion, and vehicle-motion compensation?
7. What are AprilTag target dimensions, installation height/tilt and calibration ownership?

## 4. On-car verification (Linux + installed tools)

```bash
ls -l /dev/video*
# If v4l-utils is installed:
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext
# If ROS 2 is available and sourced:
ros2 topic list
ros2 topic list -t
ros2 topic info /camera/image_raw
ros2 topic hz /camera/image_raw
```

`/camera/image_raw` is a **placeholder**, not necessarily present; replace it with the discovered topic. If remote over SSH, use `--no-display` for the standalone test and print/log results. When ready, create a dedicated ROS 2 Python package/adapter in the team workspace; that part is not included as tested production code here.

## 5. Acceptance criteria for phase 1

- [ ] Webcam device path and owner confirmed on onboard computer
- [ ] Stable frame delivery demonstrated at agreed resolution/FPS
- [ ] HSV + YOLO outputs verified on representative camera footage
- [ ] Detection boxes/class/confidence exposed as machine-readable output
- [ ] ROS input/output topic names and message types approved by integration team
- [ ] End-to-end ROS messages tested with subscriber (future)

## 6. Documentation
- ROS USB camera driver: https://github.com/ros-drivers/usb_cam
- ROS vision detection messages: https://github.com/ros-perception/vision_msgs
- ROS sensor camera calibration metadata: https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/CameraInfo.msg
