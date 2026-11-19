"""YOLO + orange-mask detection logic, with no ROS in it.

Same idea as Janet's perception_core.py (HSV orange mask + YOLO boxes,
same output format), but runs the model with OpenCV instead of
Ultralytics, because the car's Jetson has no PyTorch.

The model is a YOLOv8 .onnx file exported on a laptop with:
    yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12
"""

import cv2
import numpy as np

# The 80 COCO classes yolov8n.pt was trained on, in order (class_id = index).
COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]

# Same orange range as Janet's code (OpenCV HSV: H 0-179, S and V 0-255).
LOWER_ORANGE = np.array([5, 100, 100], dtype=np.uint8)
UPPER_ORANGE = np.array([25, 255, 255], dtype=np.uint8)


def orange_mask(frame_bgr):
    """Return a black/white image: white where the pixel is orange."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)


def parse_yolo_output(output, frame_w, frame_h, input_size=640,
                      confidence=0.35, iou=0.45, names=COCO_NAMES):
    """Turn the raw YOLOv8 output into a list of detection dicts.

    output: array shaped (1, 4 + num_classes, num_boxes), e.g. (1, 84, 8400).
    Each of the 8400 columns is one guess: [cx, cy, w, h, score_class0, ...],
    with the box in pixels of the 640x640 model input.
    """
    rows = np.squeeze(output, axis=0).T            # -> (8400, 84), one guess per row
    class_scores = rows[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)    # best class for each guess
    scores = class_scores[np.arange(len(rows)), class_ids]

    keep = scores >= confidence                    # drop low-confidence guesses
    rows, class_ids, scores = rows[keep], class_ids[keep], scores[keep]
    if len(rows) == 0:
        return []

    # Scale boxes from the 640x640 input back to the real frame size.
    sx, sy = frame_w / input_size, frame_h / input_size
    cx, cy, w, h = rows[:, 0] * sx, rows[:, 1] * sy, rows[:, 2] * sx, rows[:, 3] * sy
    boxes_xywh = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)

    # Non-max suppression: when several boxes cover the same object, keep the best.
    kept = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), confidence, iou)

    detections = []
    for i in np.array(kept).flatten():
        x, y, bw, bh = boxes_xywh[i]
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(frame_w), x + bw), min(float(frame_h), y + bh)
        class_id = int(class_ids[i])
        detections.append({
            'class_id': class_id,
            'class_name': names[class_id] if class_id < len(names) else str(class_id),
            'confidence': round(float(scores[i]), 4),
            'bbox_xyxy_pixels': [round(float(v), 1) for v in (x1, y1, x2, y2)],
            'center_xy_pixels': [round(float((x1 + x2) / 2), 1),
                                 round(float((y1 + y2) / 2), 1)],
        })
    return detections


class YoloDetector:
    """Loads a YOLOv8 .onnx model once, then finds objects in frames."""

    def __init__(self, model_path, input_size=640, confidence=0.35, iou=0.45):
        # Runs on the CPU. The car's OpenCV has no CUDA support
        # (docs/gpu-acceleration.md), so the GPU path will be TensorRT later.
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.input_size = input_size
        self.confidence = confidence
        self.iou = iou

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        # Resize to 640x640, scale pixels to 0-1, and swap BGR -> RGB.
        blob = cv2.dnn.blobFromImage(frame_bgr, 1 / 255.0,
                                     (self.input_size, self.input_size),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        output = self.net.forward()
        return parse_yolo_output(output, w, h, self.input_size,
                                 self.confidence, self.iou)


def draw_detections(frame_bgr, detections, mask=None):
    """Return a copy of the frame with boxes drawn (and orange areas tinted)."""
    out = frame_bgr.copy()
    if mask is not None:
        out[mask > 0] = (0.5 * out[mask > 0] + [0, 64, 127]).astype(np.uint8)
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d['bbox_xyxy_pixels'])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d['class_name']} {d['confidence']:.2f}"
        cv2.putText(out, label, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out
