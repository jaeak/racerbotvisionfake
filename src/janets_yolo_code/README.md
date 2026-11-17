# Racerbots — Webcam Perception Prototype (Phase 1)

## Purpose
Standalone **monocular USB webcam** perception prototype for the Webcam team. It consumes video frames, runs orange HSV segmentation plus pretrained YOLOv8n, and emits structured 2D detections. It is **not** the Intel RealSense team's pipeline. It is not yet integrated with the vehicle's ROS 2 stack.

## Current state (November 2026)
- **Implemented:** local OpenCV capture; orange HSV mask; pretrained YOLOv8n inference; configurable capture source; structured detection records; optional JSONL output; standalone camera diagnostic.
- **Not yet validated:** vehicle webcam connection; target detection performance; real-time throughput on onboard computer; ROS 2 deployment; calibration; AprilTag; IPM; tracking; speed; LiDAR fusion; visual SLAM.
- Documentation lists **Car 1 UVC webcam at `/dev/video0`** and Car 2 Intel RealSense D435i. **Confirm intended deployment car** with the team leader.

## Files
- `camera_test.py` — independent camera connection/resolution/loop FPS check
- `perception_core.py` — receives BGR NumPy frame and returns annotated image, orange mask and structured detections
- `webcam_perception.py` — command-line runner, shows live windows and optionally saves detections
- `requirements.txt` — Python dependencies (ROS 2 packages are separate)
- `ROS2_IO_PLAN.md` — proposed message interfaces, deployment and verification steps

## Local install / run
Use a Python environment compatible with your platform and Ultralytics/PyTorch. On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python camera_test.py --source 0
python webcam_perception.py --source 0 --jsonl logs/detections.jsonl
```

`yolov8n.pt` is downloaded automatically on first use if not already cached (internet required for first download). Press **q** in the OpenCV window to quit. Your OS may ask for camera permission.

For Car 1's documented Linux UVC device, *after verifying it on the actual onboard computer*:

```bash
python camera_test.py --source /dev/video0
python webcam_perception.py --source /dev/video0 --no-display --jsonl logs/detections.jsonl
```

Run `python webcam_perception.py --help` for additional flags. Headless mode requires Ctrl+C to stop. OpenCV `--source` can also point to a video file, useful when car access is unavailable.

## Input (implemented)
- **Source:** webcam through `cv2.VideoCapture` (default camera index `0`).
- **Per-frame Python input:** BGR image as `numpy.ndarray`, shape `(height,width,3)`, `uint8`.
- **Metadata:** capture-time approximation recorded as UTC timestamp after `cap.read()`. Hardware/ROS synchronized timestamps are a future task.

## Output (implemented)
- **Visual:** annotated YOLO window plus orange binary mask window (optional).
- **Machine readable:** one JSON object per frame in optional `.jsonl`, e.g.

```json
{"frame_number":42,"captured_at_utc":"2026-11-09T22:00:00+00:00","image_width":640,"image_height":480,"detections":[{"class_id":2,"class_name":"car","confidence":0.91,"bbox_xyxy_pixels":[120.0,80.0,300.0,240.0],"center_xy_pixels":[210.0,160.0]}]}
```

The values above are **illustrative, not measurements**. Coordinates are pixels in the original frame. No depth, meters, velocity or stable track ID are produced. JSONL is for testing/inspection and is **not** a live ROS 2 topic.

## Limitations
- COCO-pretrained YOLOv8n recognizes general objects; **Racerbot-sized cars may not be detected reliably**. Custom data and training may be needed.
- HSV segmentation selects orange pixels; it does **not** identify valid track barriers by itself.
- A 2D bounding box does not determine metric distance with an uncalibrated monocular camera.
- Loop FPS readings depend on the computer and are not a substitute for latency benchmarking on the car.
- To use frames over ROS 2, create a subscriber adapter around `process_frame`; do not also open `/dev/video0` from a competing process.

## Proposed next steps
1. Confirm car, onboard computer, ROS 2 distro, camera device, camera topic and target resolution with the leader.
2. Test device on the car with `camera_test.py` or the team's existing camera driver.
3. Decide ROS ownership: `usb_cam` publishes images and perception subscribes **or** perception owns camera and publishes images. Prefer a single camera owner.
4. Build ROS 2 adapter with topic names/message types in `ROS2_IO_PLAN.md`; verify end-to-end messages.
5. Calibrate actual mounted camera (AprilTag grid/ChArUco or checkerboard), then estimate IPM ground coordinates.
6. Validate car/barrier detection, tracking, ego-motion compensation and LiDAR fusion.

## Hand-off summary
**Delivered:** reproducible standalone local prototype + explicit input/output format + device configuration + ROS 2 interface proposal. **Pending:** real vehicle access and full ROS 2 implementation/validation.
