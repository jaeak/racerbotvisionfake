# yolo

ROS 2 node that runs YOLO on the car's webcam images and publishes what it finds.
It's the ROS version of [Janet's webcam script](../janets_yolo_code/README.md):
same orange mask and same detection format, but it gets images from a ROS topic
instead of opening the webcam itself.

```
webcam → usb_cam driver → /image_raw → yolo_node → /perception/yolo/detections
                                    ↘ dashboard            ↘ /perception/yolo/debug_image
```

Support code only: it never publishes `/drive`, so it can't move the car.

## Topics

| | Topic | Type | What |
|---|---|---|---|
| In | `/image_raw` | `sensor_msgs/Image` | Camera frames from `usb_cam` |
| Out | `/perception/yolo/detections` | `std_msgs/String` (JSON) | Boxes, class, confidence, plus the image's timestamp |
| Out | `/perception/yolo/debug_image` | `sensor_msgs/Image` | Frame with boxes drawn (orange areas tinted) |

Example detections message:

```json
{"stamp": {"sec": 1763520000, "nanosec": 0}, "frame_id": "camera",
 "image_size": [640, 480], "inference_ms": 85.2,
 "barrier_color": "orange", "barrier_pixels": 1520,
 "detections": [{"class_id": 2, "class_name": "car", "confidence": 0.81,
                 "bbox_xyxy_pixels": [120.0, 200.0, 260.0, 300.0],
                 "center_xy_pixels": [190.0, 250.0]}]}
```

## GPU (TensorRT) and CPU (OpenCV)

The car's Jetson has no PyTorch, so Ultralytics can't run there. The model is
exported once to `.onnx` on a laptop, then run on the car one of two ways:

| Backend | Runs on | Needs |
|---|---|---|
| **TensorRT** (default) | GPU | a `.engine` file, built from the `.onnx` **on the car** |
| OpenCV | CPU (slower) | just the `.onnx` |

`backend: auto` (in `config/yolo.yaml`) tries TensorRT first. If anything is
missing it prints why and falls back to OpenCV, so the node still runs. The
node's startup line and FPS lines say which one it's using.

The engine is built automatically the first time (a few minutes, once) with
TensorRT's `trtexec` in FP16, and saved next to the `.onnx`
(`~/models/yolov8n.engine`). An engine only works on the GPU and TensorRT
version it was built on: never copy it between machines or commit it.
After a JetPack/TensorRT update, delete it and it rebuilds.

GPU memory is handled through CUDA's own runtime library (libcudart), so no
PyCUDA/CuPy is needed; only TensorRT's Python bindings (`import tensorrt`).

**Check everything (and build the engine) before using the node:**

```bash
ros2 run yolo yolo_check --model ~/models/yolov8n.onnx
```

It checks TensorRT, CUDA and `trtexec`, builds the engine, confirms TensorRT
gives the same output as OpenCV, and prints the speed of both. If a step
fails it says what to fix. The most likely one is `import tensorrt` failing:
then TensorRT's Python package needs installing (ask whoever manages the car).

## Run it (step by step)

**1. Export the model (once, on a laptop that has `ultralytics`):**

```bash
yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12
```

This makes `yolov8n.onnx`. Copy it to the car, e.g. into `~/models/`:

```bash
scp yolov8n.onnx <user>@<car-ip>:~/models/
```

**2. Build (on the car, from the `racerbot-vision` folder, which is its own ROS workspace):**

```bash
source /opt/ros/jazzy/setup.bash
cd ~/vision-test            # wherever the repo is cloned
colcon build --packages-select yolo
```

**3. Car 1 only: make sure nothing else is using the webcam.** Only one program
can open it. Don't run the dashboard webcam stream unless it's set to
`image_topic: /image_raw`. (Car 2's dashboard already reads the RealSense topic.)

**4. Terminal 1, start the camera.**

Car 1 (webcam), opens the webcam and publishes `/image_raw`:

```bash
source /opt/ros/jazzy/setup.bash
ros2 run usb_cam usb_cam_node_exe --ros-args -p video_device:=/dev/video0
```

If it says `Package 'usb_cam' not found`: `sudo apt install ros-jazzy-usb-cam`
(ask whoever manages the car first). Or, with nothing to install, use the
simple publisher in this repo (car 1 webcam, with its calibration):

```bash
python3 ~/racerbot-vision/src/webcam_publisher.py --info ~/racerbot-vision/src/camera_info_rb1.yaml
```

Car 2 (RealSense), publishes `/camera/camera/color/image_raw`:

```bash
source /opt/ros/jazzy/setup.bash && source ~/racerbot-ws/install/setup.bash
ros2 launch racerbot_launch realsense_camera_launch.py
```

**5. Terminal 2, start the YOLO node:**

```bash
source /opt/ros/jazzy/setup.bash
source ~/racerbot-ws/install/setup.bash        # so it can tell which car it's on
source ~/racerbot-vision/install/setup.bash
ros2 launch yolo yolo_launch.py model_path:=$HOME/models/yolov8n.onnx
```

The camera topic is picked automatically from the car's profile
(`racerbot_car`, `cars/<car>/car.yaml` in the car workspace), so this is the
same command on both cars. The first line it prints says which one it chose:

```
YOLO camera topic: /image_raw (car rb1 has a webcam)
YOLO camera topic: /camera/camera/color/image_raw (car rb2 has a RealSense)
```

If the car workspace isn't sourced, or it's a laptop, it falls back to
`/image_raw`. To force a topic: add `image_topic:=/some/topic` to the command.

Working when it prints `YOLO node ready, listening on <topic>`, then a line
like `8.3 FPS, last inference 110 ms, 1 detections` every 5 seconds.

**6. Terminal 3, check the output:**

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic hz /image_raw                      # camera running? (car 2: /camera/camera/color/image_raw)
ros2 topic hz /perception/yolo/detections     # YOLO running?
ros2 topic echo /perception/yolo/detections   # see the detections
```

To see the boxes, on the car's screen (remote desktop) run
`ros2 run rqt_image_view rqt_image_view` and pick `/perception/yolo/debug_image`.
(Lichtblick/Foxglove on the dashboard hides raw images, so it won't show there.)

**If it's too slow:** first make sure it says `tensorrt`, not `opencv`, in the
FPS line. If it's already on TensorRT, set `input_size: 320`, re-export the
`.onnx` with `imgsz=320`, delete the old `.engine` and rebuild.

**Troubleshooting**

| Problem | Fix |
|---|---|
| Node says `Set the model_path parameter` | You forgot `model_path:=...` in step 5 |
| No `FPS` lines appear | No images arriving: check step 4 and `ros2 topic hz /image_raw` |
| Camera driver can't open `/dev/video0` | Something else has the webcam (usually the dashboard stream). Stop it. |
| `ros2 launch yolo ...` says package not found | You didn't `source ~/racerbot-vision/install/setup.bash` |
| Car 2 picks `/image_raw` | The car workspace isn't sourced in that terminal: `source ~/racerbot-ws/install/setup.bash` |

## Settings (`config/yolo.yaml`)

| Parameter | Default | Meaning |
|---|---|---|
| `model_path` | `''` | Path to the `.onnx` file (required) |
| `image_topic` | picked by car | Set by `yolo_launch.py` from the car profile (car 1: `/image_raw`, car 2: `/camera/camera/color/image_raw`). Override with `image_topic:=...` |
| `confidence` | `0.35` | Ignore boxes below this score |
| `iou` | `0.45` | Overlap threshold for removing duplicate boxes (NMS) |
| `input_size` | `640` | Must match `imgsz` used when exporting |
| `publish_debug_image` | `true` | Publish the image with boxes drawn |
| `backend` | `auto` | `auto` (TensorRT, else OpenCV), `tensorrt` or `opencv` |
| `engine_path` | `''` | TensorRT engine; empty = `model_path` with `.engine` |
| `auto_build_engine` | `true` | Build the engine from the `.onnx` if it doesn't exist |
| `classes` | `['car']` | COCO class names to keep, e.g. `['car', 'truck']`. `['all']` keeps everything |
| `barrier_color` | `orange` | Barrier colour filter: `orange` or `yellow` (one at a time) |

To switch, e.g. to yellow barriers: change `barrier_color: yellow` in
`config/yolo.yaml`, then `colcon build --packages-select yolo` and relaunch.
The node prints its classes and barrier colour when it starts.

**Colour ranges** (`COLOR_RANGES` in `yolo/detector.py`, OpenCV HSV):

| Colour | Hue | Saturation | Value |
|---|---|---|---|
| orange | 5-18 | 150-255 | 100-255 |
| yellow | 20-35 | 150-255 | 100-255 |

Janet's original single range was hue 5-25, saturation 100+, which also caught
skin. These are starting values: tune them with pictures of the real barriers.

## Tests

```bash
python3 -m pytest src/yolo/test/test_detector.py
```

These test the detection logic with fake YOLO output, so they need no model or camera.

## Limits (for now)

- Uses the stock COCO model, so "car" means a real car, not necessarily a Racerbot car.
  A model trained on our own images comes later.
- Boxes are in pixels only. A webcam can't measure distance on its own.
- The barrier colour filter is a rough colour filter, not a verified barrier detector.
- YOLO's pretrained model has no "barrier" class, and its "car" class is real
  cars, so it may miss Racerbot cars. Reliable car/barrier detection needs a
  model trained on our own pictures.
