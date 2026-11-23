"""Minimal webcam -> ROS 2 publisher, for when usb_cam isn't installed.

Publishes /image_raw (bgr8) and /camera_info (from camera_info_rb1.yaml) with
matching timestamps and frame_id, which is what opponent_perception needs.
Uses only what's already on the car: rclpy, OpenCV, cv_bridge.

    source /opt/ros/jazzy/setup.bash
    python3 webcam_publisher.py --info camera_info_rb1.yaml

Stop with Ctrl+C. Nothing else can use the webcam while this runs.
"""

import argparse

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
import yaml


def load_camera_info(path, frame_id):
    with open(path) as f:
        data = yaml.safe_load(f)
    info = CameraInfo()
    info.header.frame_id = frame_id
    info.width = int(data['image_width'])
    info.height = int(data['image_height'])
    info.distortion_model = data['distortion_model']
    info.d = [float(v) for v in data['distortion_coefficients']['data']]
    info.k = [float(v) for v in data['camera_matrix']['data']]
    info.r = [float(v) for v in data['rectification_matrix']['data']]
    info.p = [float(v) for v in data['projection_matrix']['data']]
    return info


class WebcamPublisher(Node):

    def __init__(self, args):
        super().__init__('webcam_publisher')
        self.info = load_camera_info(args.info, args.frame_id)
        self.frame_id = args.frame_id
        device = int(args.device) if args.device.isdigit() else args.device
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        # MJPG lets the C922 do 1280x720 at 30 fps (raw YUYV is much slower).
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.info.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.info.height)
        self.cap.set(cv2.CAP_PROP_FPS, args.fps)
        if not self.cap.isOpened():
            raise RuntimeError(f'Could not open {args.device} (is something else using it?)')

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, '/image_raw', qos_profile_sensor_data)
        self.info_pub = self.create_publisher(CameraInfo, '/camera_info', qos_profile_sensor_data)
        self.create_timer(1.0 / args.fps, self.tick)
        self.frames = 0
        self.warned = False
        self.get_logger().info(
            f'Publishing {args.device} as /image_raw + /camera_info, frame {self.frame_id}')

    def tick(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn('No frame from the camera')
            return
        h, w = frame.shape[:2]
        if (w, h) != (self.info.width, self.info.height):
            if not self.warned:
                self.get_logger().error(
                    f'Camera gives {w}x{h} but calibration is {self.info.width}x'
                    f'{self.info.height}; not publishing (distances would be wrong)')
                self.warned = True
            return

        stamp = self.get_clock().now().to_msg()
        image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        image.header.stamp = stamp
        image.header.frame_id = self.frame_id
        self.info.header.stamp = stamp          # same stamp = matched pair
        self.image_pub.publish(image)
        self.info_pub.publish(self.info)

        self.frames += 1
        if self.frames == 1:
            self.get_logger().info(f'First frame published ({w}x{h})')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--device', default='/dev/video0')
    p.add_argument('--info', required=True, help='camera_info yaml (camera_info_rb1.yaml)')
    p.add_argument('--frame-id', default='camera_optical_frame')
    p.add_argument('--fps', type=float, default=30.0)
    args = p.parse_args()

    rclpy.init()
    node = None
    try:
        node = WebcamPublisher(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.cap.release()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
