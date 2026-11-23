"""Quick AprilTag check: webcam -> detect tag36h11 tags -> show ID and distance.

No ROS, no TF, no config. Just to see that tags are detected and the
calibration gives sensible distances before using opponent_perception.

    python3 apriltag_test.py --size 0.12
    python3 apriltag_test.py --camera /dev/video0 --size 0.12 --params camera_params.yaml

--size is the outer black square of the tag in metres (measure it, don't
include the white border). Press q to quit.

The webcam must be free: stop usb_cam, the dashboard stream and the YOLO node.
"""

import argparse

import cv2
import numpy as np
import yaml


def make_detector():
    """AprilTag 36h11 detector that works on old (4.6) and new OpenCV."""
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    if hasattr(aruco, 'ArucoDetector'):          # OpenCV 4.7+
        params = aruco.DetectorParameters()
    else:                                        # OpenCV 4.6 (Ubuntu 24.04)
        params = aruco.DetectorParameters_create()
    # Sub-pixel corners: much steadier angles, which matters for extrinsics.
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    if hasattr(aruco, 'ArucoDetector'):
        return aruco.ArucoDetector(dictionary, params).detectMarkers
    return lambda gray: aruco.detectMarkers(gray, dictionary, parameters=params)


def load_calibration(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    K = np.array(data['camera_matrix'], dtype=np.float64)
    dist = np.array(data['distortion_coefficients'], dtype=np.float64)
    size = (data['image_width'], data['image_height'])
    return K, dist, size


def tag_pose(corners, size_m, K, dist):
    """Return (x, y, z) of the tag centre in metres, in the camera frame.

    x = right, y = down, z = forward (straight out of the lens).
    """
    s = size_m / 2
    # Tag corners in the tag's own frame, same order OpenCV returns them:
    # top-left, top-right, bottom-right, bottom-left.
    object_points = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]],
                             dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(object_points, corners.reshape(4, 2).astype(np.float64),
                                  K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return tvec.ravel() if ok else None


def find_tags(frame, detect, size_m=None, K=None, dist=None):
    """Return a list of (tag_id, corners, xyz-or-None)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detect(gray)
    tags = []
    if ids is None:
        return tags
    for tag_corners, tag_id in zip(corners, ids.ravel()):
        xyz = None
        if K is not None and size_m:
            xyz = tag_pose(tag_corners, size_m, K, dist)
        tags.append((int(tag_id), tag_corners, xyz))
    return tags


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera', default='/dev/video0', help='device path or index')
    parser.add_argument('--size', type=float, help='tag black-square size in metres')
    parser.add_argument('--params', default='camera_params.yaml',
                        help='calibration from calibrate_intrinsics.py')
    args = parser.parse_args()

    K = dist = None
    width, height = 1280, 720
    try:
        K, dist, (width, height) = load_calibration(args.params)
        print(f'Loaded calibration {args.params} ({width}x{height})')
    except OSError:
        print(f'No {args.params}: will show tag IDs only, no distance')
    if K is not None and not args.size:
        print('No --size given: will show tag IDs only, no distance')

    source = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(source)
    # Must match the calibration resolution, or the distances are wrong.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(f'Could not open camera {args.camera} (is something else using it?)')

    detect = make_detector()
    while True:
        ok, frame = cap.read()
        if not ok:
            print('Lost the camera')
            break
        if K is not None and frame.shape[1::-1] != (width, height):
            print(f'Warning: camera gives {frame.shape[1]}x{frame.shape[0]}, '
                  f'calibration is {width}x{height}; distances will be wrong')
            K = None

        for tag_id, corners, xyz in find_tags(frame, detect, args.size, K, dist):
            pts = corners.reshape(4, 2).astype(int)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            label = f'id {tag_id}'
            if xyz is not None:
                x, _, z = xyz
                label += f'  {np.linalg.norm(xyz):.2f} m (fwd {z:.2f}, right {x:.2f})'
            cv2.putText(frame, label, (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(label)

        cv2.imshow('AprilTag test (q to quit)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
