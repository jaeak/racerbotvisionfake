"""Show a ROS image topic in a window (stand-in for rqt_image_view).

    source /opt/ros/jazzy/setup.bash
    python3 view_image.py                                  # YOLO boxes
    python3 view_image.py --topic /image_raw               # plain camera

Press q in the window (or Ctrl+C) to quit. Only listens; changes nothing.
"""

import argparse

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class Viewer(Node):

    def __init__(self, topic, scale):
        super().__init__('image_viewer')
        self.bridge = CvBridge()
        self.scale = scale
        self.frame = None
        self.title = f'{topic} (q to quit)'
        self.create_subscription(Image, topic, self.on_image, qos_profile_sensor_data)
        self.get_logger().info(f'Waiting for pictures on {topic}...')

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.scale != 1.0:
            frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale)
        self.frame = frame


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--topic', default='/perception/yolo/debug_image')
    p.add_argument('--scale', type=float, default=0.6, help='window size (1.0 = full)')
    args = p.parse_args()
    rclpy.init()
    node = Viewer(args.topic, args.scale)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)
            if node.frame is not None:
                cv2.imshow(node.title, node.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
