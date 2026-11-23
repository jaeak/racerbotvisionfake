"""Convert camera_params.yaml (from calibrate_intrinsics.py) into the ROS
camera_info format that usb_cam loads with camera_info_url.

    python3 make_camera_info.py camera_params.yaml camera_info_rb1.yaml

Then start the camera with:
    ros2 run usb_cam usb_cam_node_exe --ros-args \\
      -p video_device:=/dev/video0 -p image_width:=1280 -p image_height:=720 \\
      -p pixel_format:=mjpeg2rgb -p frame_id:=camera_optical_frame \\
      -p camera_name:=webcam \\
      -p camera_info_url:=file://$PWD/camera_info_rb1.yaml
"""

import sys

import yaml


def to_camera_info(params, camera_name='webcam'):
    K = params['camera_matrix']
    D = list(params['distortion_coefficients'])
    if len(D) != 5:
        raise ValueError(f'expected 5 distortion coefficients (plumb_bob), got {len(D)}')
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    return {
        'image_width': int(params['image_width']),
        'image_height': int(params['image_height']),
        'camera_name': camera_name,
        'camera_matrix': {'rows': 3, 'cols': 3,
                          'data': [float(v) for row in K for v in row]},
        'distortion_model': 'plumb_bob',
        'distortion_coefficients': {'rows': 1, 'cols': 5, 'data': [float(v) for v in D]},
        'rectification_matrix': {'rows': 3, 'cols': 3,
                                 'data': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]},
        # Raw (unrectified) camera: projection = [K | 0].
        'projection_matrix': {'rows': 3, 'cols': 4,
                              'data': [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0,
                                       0.0, 0.0, 1.0, 0.0]},
    }


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    with open(sys.argv[1]) as f:
        params = yaml.safe_load(f)
    info = to_camera_info(params)
    with open(sys.argv[2], 'w') as f:
        yaml.safe_dump(info, f, sort_keys=False)
    print(f'Wrote {sys.argv[2]} ({info["image_width"]}x{info["image_height"]}, '
          f'fx={info["camera_matrix"]["data"][0]:.1f})')


if __name__ == '__main__':
    main()
