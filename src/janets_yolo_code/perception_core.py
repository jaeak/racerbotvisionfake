"""Camera-independent HSV + YOLO perception logic.

Note: COCO-pretrained YOLO detects common object categories, not guaranteed
Racerbot cars or track barriers. Orange mask is not a verified barrier detector.
"""
import cv2
import numpy as np


def process_frame(frame, model, confidence=0.35):
    """Return annotated BGR frame, orange binary mask, and serializable detections."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 100, 100], dtype=np.uint8)
    upper_orange = np.array([25, 255, 255], dtype=np.uint8)
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

    result = model(frame, conf=confidence, verbose=False)[0]
    annotated = result.plot()
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            x1, y1, x2, y2 = (float(x) for x in box.xyxy[0].tolist())
            detections.append({
                "class_id": class_id,
                "class_name": str(model.names[class_id]),
                "confidence": round(float(box.conf[0].item()), 4),
                "bbox_xyxy_pixels": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "center_xy_pixels": [round((x1+x2)/2, 1), round((y1+y2)/2, 1)],
            })
    return annotated, orange_mask, detections
