"""Save camera pictures from a ROS topic, for training your own YOLO model.

    source /opt/ros/jazzy/setup.bash
    python3 save_frames.py --out ~/yolo_dataset/raw --every 0.5

Saves one JPEG every --every seconds from /image_raw (change with --topic),
so you don't get hundreds of near-identical pictures. Works while the car
drives and while the YOLO node runs (it only listens). Ctrl+C to stop.
"""

import argparse
import os
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class FrameSaver(Node):

    def __init__(self, args):
        super().__init__('frame_saver')
        self.out = os.path.expanduser(args.out)
        os.makedirs(self.out, exist_ok=True)
        self.every = args.every
        self.limit = args.max
        self.last = 0.0
        self.count = 0
        self.bridge = CvBridge()
        self.create_subscription(Image, args.topic, self.on_image, qos_profile_sensor_data)
        self.get_logger().info(f'Saving from {args.topic} to {self.out} every {self.every} s')

    def on_image(self, msg):
        now = time.monotonic()
        if now - self.last < self.every:
            return
        self.last = now
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        name = f'{time.strftime("%Y%m%d_%H%M%S")}_{self.count:05d}.jpg'
        cv2.imwrite(os.path.join(self.out, name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.count += 1
        if self.count % 20 == 0:
            self.get_logger().info(f'{self.count} pictures saved')
        if self.limit and self.count >= self.limit:
            self.get_logger().info(f'Reached {self.limit} pictures, stopping')
            raise SystemExit


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--topic', default='/image_raw')
    p.add_argument('--out', default='~/yolo_dataset/raw')
    p.add_argument('--every', type=float, default=0.5, help='seconds between pictures')
    p.add_argument('--max', type=int, default=0, help='stop after this many (0 = no limit)')
    args = p.parse_args()
    rclpy.init()
    node = FrameSaver(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print(f'Saved {node.count} pictures to {node.out}')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
