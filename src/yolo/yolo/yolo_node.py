"""ROS 2 node: camera image in -> YOLO -> detections out.

Subscribes:  /image_raw (sensor_msgs/Image), from the camera driver (usb_cam)
Publishes:   /perception/yolo/detections   (std_msgs/String, JSON)
             /perception/yolo/debug_image  (sensor_msgs/Image, boxes drawn on)

Support code only: it never publishes /drive, so it can't move the car.
"""

import json
import time

import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .detector import color_mask, draw_detections, YoloDetector


class YoloNode(Node):

    def __init__(self):
        super().__init__('yolo_node')
        # Settings (overridable from config/yolo.yaml or the command line).
        self.declare_parameter('model_path', '')
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('confidence', 0.35)
        self.declare_parameter('iou', 0.45)
        self.declare_parameter('input_size', 640)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('classes', ['car'])        # ['all'] = every class
        self.declare_parameter('barrier_color', 'orange')  # 'orange' or 'yellow'

        model_path = self.get_parameter('model_path').value
        if not model_path:
            raise RuntimeError('Set the model_path parameter to your yolov8n.onnx file')

        self.detector = YoloDetector(
            model_path,
            input_size=self.get_parameter('input_size').value,
            confidence=self.get_parameter('confidence').value,
            iou=self.get_parameter('iou').value,
            classes=self.get_parameter('classes').value,
        )
        self.barrier_color = self.get_parameter('barrier_color').value
        color_mask(np.zeros((1, 1, 3), dtype=np.uint8), self.barrier_color)  # fail early if wrong
        self.publish_debug = self.get_parameter('publish_debug_image').value
        self.bridge = CvBridge()

        self.detections_pub = self.create_publisher(
            String, '/perception/yolo/detections', 10)
        self.debug_pub = self.create_publisher(
            Image, '/perception/yolo/debug_image', qos_profile_sensor_data)

        image_topic = self.get_parameter('image_topic').value
        # Sensor QoS + small queue: if YOLO is slow, old frames are dropped
        # instead of piling up, so we always work on a recent image.
        self.create_subscription(Image, image_topic, self.on_image,
                                 qos_profile_sensor_data)

        self.frames = 0
        self.last_log = time.monotonic()
        self.get_logger().info(
            f'YOLO node ready, listening on {image_topic}, classes '
            f'{list(self.get_parameter("classes").value)}, barrier colour {self.barrier_color}')

    def on_image(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().warn(f'Could not convert image: {e}')
            return

        start = time.monotonic()
        detections = self.detector.detect(frame)
        mask = color_mask(frame, self.barrier_color)
        infer_ms = (time.monotonic() - start) * 1000

        # Keep the camera's timestamp + frame_id so other nodes can match
        # these detections to the exact image they came from.
        out = String()
        out.data = json.dumps({
            'stamp': {'sec': msg.header.stamp.sec,
                      'nanosec': msg.header.stamp.nanosec},
            'frame_id': msg.header.frame_id,
            'image_size': [msg.width, msg.height],
            'inference_ms': round(infer_ms, 1),
            'barrier_color': self.barrier_color,
            'barrier_pixels': int((mask > 0).sum()),
            'detections': detections,
        })
        self.detections_pub.publish(out)

        if self.publish_debug and self.debug_pub.get_subscription_count() > 0:
            debug = self.bridge.cv2_to_imgmsg(
                draw_detections(frame, detections, mask), encoding='bgr8')
            debug.header = msg.header
            self.debug_pub.publish(debug)

        # Print the frame rate every 5 seconds so you can see how fast it runs.
        self.frames += 1
        now = time.monotonic()
        if now - self.last_log >= 5.0:
            fps = self.frames / (now - self.last_log)
            self.get_logger().info(
                f'{fps:.1f} FPS, last inference {infer_ms:.0f} ms, '
                f'{len(detections)} detections')
            self.frames, self.last_log = 0, now


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = YoloNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
