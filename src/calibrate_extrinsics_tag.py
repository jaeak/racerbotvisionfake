"""Find where the camera sits on the car (extrinsics) using one AprilTag.

Put a tag upright, facing the car squarely, about 0.6-0.8 m in front of the
CAMERA (closer = more accurate; at 1.5 m the result can be ~15 cm off). Measure the
tag's CENTRE from the car's base_link (centre of the rear axle):
    --tag-x  metres forward of the rear axle
    --tag-y  metres to the LEFT (right = negative)
    --tag-z  metres above the rear axle centre (height above floor minus
             the axle's height above floor)

    python3 calibrate_extrinsics_tag.py --camera 0 --params camera_params.yaml \\
        --size 0.167 --tag-x 1.50 --tag-y 0.0 --tag-z 0.10

It averages many frames, then prints the camera's position/angles on the car
and a ready-to-use static TF command (base_link -> camera_optical_frame).
Press q to stop early (uses the frames collected so far).

Tag orientation: the team's printed tags (official AprilTag images, label
underneath) are rotated 180 degrees compared to how OpenCV draws the same ID.
OpenCV still reads the right ID, but would think the tag is upside down
(roll ~180). This script corrects for that by default. If you print tags with
OpenCV's generateImageMarker instead, add --opencv-tags.
If the result says roll is about +/-180, flip that setting.

Needs apriltag_test.py in the same folder.
"""

import argparse
import math

import cv2
import numpy as np

from apriltag_test import find_tags, load_calibration, make_detector

# Tag frame (x right, y up, z out of its face) -> base_link (x fwd, y left,
# z up) when the tag stands upright and faces straight back at the car.
R_BASE_TAG = np.array([[0, 0, -1],
                       [-1, 0, 0],
                       [0, 1, 0]], dtype=np.float64)
# camera_link (x fwd, y left, z up) expressed in the optical frame
# (x right, y down, z fwd).
R_OPT_LINK = np.array([[0, -1, 0],
                       [0, 0, -1],
                       [1, 0, 0]], dtype=np.float64)


def tag_in_camera(corners, size_m, K, dist):
    """Rotation and translation of the tag in the camera optical frame."""
    s = size_m / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, corners.reshape(4, 2).astype(np.float64),
                                  K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.ravel()


def camera_in_base(R_cam_tag, t_cam_tag, tag_xyz):
    """T_base_cam = T_base_tag * inverse(T_cam_tag)."""
    R_tag_cam = R_cam_tag.T
    t_tag_cam = -R_tag_cam @ t_cam_tag
    R_base_cam = R_BASE_TAG @ R_tag_cam
    t_base_cam = R_BASE_TAG @ t_tag_cam + np.asarray(tag_xyz, dtype=np.float64)
    return R_base_cam, t_base_cam


def average_rotation(rotations):
    """Average rotation matrices (project the mean back onto a rotation)."""
    U, _, Vt = np.linalg.svd(np.mean(rotations, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def yaw_pitch_roll(R):
    """ZYX angles in degrees. Pitch is positive when tilted DOWN (REP-103)."""
    yaw = math.atan2(R[1, 0], R[0, 0])
    pitch = math.asin(-max(-1.0, min(1.0, R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    return tuple(math.degrees(a) for a in (yaw, pitch, roll))


def quaternion(R):
    """(x, y, z, w) quaternion from a rotation matrix."""
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return x, y, z, w


def report(R_base_cam, t_base_cam, samples, spread_m, tag_dist_m=None):
    yaw, pitch, roll = yaw_pitch_roll(R_base_cam @ R_OPT_LINK)
    qx, qy, qz, qw = quaternion(R_base_cam)
    x, y, z = t_base_cam
    print()
    print(f'Averaged {samples} frames (position spread {spread_m * 100:.1f} cm)')
    print('Camera on the car, measured from base_link (rear axle centre):')
    print(f'  forward {x:+.3f} m   left {y:+.3f} m   up {z:+.3f} m')
    print(f'  yaw {yaw:+.1f} deg (left +)   pitch {pitch:+.1f} deg (down +)   roll {roll:+.1f} deg')
    print()
    print('Static TF (base_link -> camera_optical_frame):')
    print(f'  ros2 run tf2_ros static_transform_publisher '
          f'--x {x:.4f} --y {y:.4f} --z {z:.4f} '
          f'--qx {qx:.5f} --qy {qy:.5f} --qz {qz:.5f} --qw {qw:.5f} '
          f'--frame-id base_link --child-frame-id camera_optical_frame')
    if tag_dist_m is not None and tag_dist_m > 1.0:
        print(f'\nWarning: the tag was {tag_dist_m:.2f} m from the camera. Move it to '
              f'0.6-0.8 m and run again; far tags give wrong angles.')
    if spread_m > 0.02:
        print('\nWarning: results jumped around by more than 2 cm. Hold everything '
              'still, check lighting, and run again.')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--camera', default='0')
    p.add_argument('--params', default='camera_params.yaml')
    p.add_argument('--size', type=float, required=True, help='tag black square, metres')
    p.add_argument('--tag-x', type=float, required=True, help='tag centre, m forward of rear axle')
    p.add_argument('--tag-y', type=float, required=True, help='tag centre, m left (+) / right (-)')
    p.add_argument('--tag-z', type=float, required=True, help='tag centre, m above rear axle centre')
    p.add_argument('--id', type=int, help='only use this tag id')
    p.add_argument('--frames', type=int, default=60, help='frames to average')
    p.add_argument('--opencv-tags', action='store_true',
                   help='tag was printed by OpenCV (no 180 degree correction)')
    args = p.parse_args()

    K, dist, (width, height) = load_calibration(args.params)
    source = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(f'Could not open camera {args.camera} (is something else using it?)')

    detect = make_detector()
    tag_xyz = (args.tag_x, args.tag_y, args.tag_z)
    rotations, positions, distances = [], [], []
    print(f'Collecting {args.frames} frames. Keep the car and tag still...')
    while len(positions) < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1::-1] != (width, height):
            raise SystemExit(f'Camera gives {frame.shape[1]}x{frame.shape[0]} but the '
                             f'calibration is {width}x{height}')
        tags = [t for t in find_tags(frame, detect) if args.id is None or t[0] == args.id]
        if len(tags) == 1:
            corners = tags[0][1].reshape(4, 2)
            if not args.opencv_tags:
                corners = corners[[2, 3, 0, 1]]   # official print = OpenCV turned 180
            pose = tag_in_camera(corners, args.size, K, dist)
            if pose is not None:
                distances.append(float(np.linalg.norm(pose[1])))
                R, t = camera_in_base(*pose, tag_xyz)
                rotations.append(R)
                positions.append(t)
            cv2.polylines(frame, [tags[0][1].reshape(4, 2).astype(int)], True, (0, 255, 0), 2)
        msg = f'{len(positions)}/{args.frames} frames'
        if len(tags) > 1:
            msg += ' - more than one tag visible, use --id'
        cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Extrinsic calibration (q to stop)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if not positions:
        raise SystemExit('No tag seen. Check the tag is in view and well lit.')
    positions = np.array(positions)
    t_mean = positions.mean(axis=0)
    spread = float(np.max(np.linalg.norm(positions - t_mean, axis=1)))
    report(average_rotation(rotations), t_mean, len(positions), spread,
           float(np.mean(distances)))


if __name__ == '__main__':
    main()
