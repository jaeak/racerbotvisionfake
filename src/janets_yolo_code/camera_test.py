"""Test an OpenCV-compatible camera before running perception."""
import argparse
import time
import cv2


def camera_source(value):
    # Accept camera indexes (0, 1, ...) or /dev/video0, file, RTSP URL.
    return int(value) if value.isdecimal() else value


def main():
    parser = argparse.ArgumentParser(description="Verify camera input and frame delivery")
    parser.add_argument("--source", default="0", help="Camera index or path, e.g. 0 or /dev/video0")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--no-display", action="store_true", help="Headless mode: print periodic measurements")
    args = parser.parse_args()
    source = camera_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source: {source!r}")
    try:
        if args.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        print(f"Source: {source!r}")
        print(f"Reported capture FPS: {cap.get(cv2.CAP_PROP_FPS):.1f} (may be unreliable)")
        count = 0
        started = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No further frames received")
                break
            count += 1
            elapsed = time.perf_counter() - started
            measured_fps = count / max(elapsed, 1e-6)
            h, w = frame.shape[:2]
            if count == 1 or (args.no_display and count % 60 == 0):
                print(f"Frame: {w}x{h} | average capture/display loop FPS: {measured_fps:.1f}")
            if not args.no_display:
                cv2.putText(frame, f"{w}x{h} | {measured_fps:.1f} FPS", (10, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Camera Connection Test (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
