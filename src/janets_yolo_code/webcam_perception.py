"""Standalone webcam perception prototype. Run --help for options."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
from ultralytics import YOLO

from perception_core import process_frame


def camera_source(value):
    return int(value) if value.isdecimal() else value


def main():
    parser = argparse.ArgumentParser(description="HSV + YOLOv8n standalone webcam perception")
    parser.add_argument("--source", default="0", help="Camera index, device, or video path")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--jsonl", type=Path, help="Optional path for machine-readable JSON lines output")
    parser.add_argument("--no-display", action="store_true", help="For headless onboard use")
    args = parser.parse_args()

    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    model = YOLO(args.model)
    source = camera_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {source!r}")

    output = None
    try:
        if args.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if args.jsonl:
            args.jsonl.parent.mkdir(parents=True, exist_ok=True)
            output = args.jsonl.open("w", encoding="utf-8")
        frame_number = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stream ended or frame could not be read")
                break
            # This local timestamp is assigned immediately after frame capture.
            # It is NOT a hardware camera timestamp or synchronized ROS stamp.
            captured_at = datetime.now(timezone.utc).isoformat()
            start = time.perf_counter()
            annotated, orange_mask, detections = process_frame(frame, model, args.confidence)
            inference_loop_fps = 1.0 / max(time.perf_counter() - start, 1e-6)
            h, w = frame.shape[:2]
            data = {
                "frame_number": frame_number,
                "captured_at_utc": captured_at,
                "image_width": w,
                "image_height": h,
                "detections": detections,
            }
            if output:
                output.write(json.dumps(data) + "\n")
                output.flush()
            if frame_number % 30 == 0:
                print(f"Frame {frame_number}: {len(detections)} detections; processing rate ~{inference_loop_fps:.1f} FPS")
            if not args.no_display:
                cv2.putText(annotated, f"Processing ~{inference_loop_fps:.1f} FPS", (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.imshow("YOLO Detection", annotated)
                cv2.imshow("Orange Mask", orange_mask)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_number += 1
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if output:
            output.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
