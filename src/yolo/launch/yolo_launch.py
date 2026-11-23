"""Start the YOLO node with config/yolo.yaml.

    ros2 launch yolo yolo_launch.py model_path:=/path/to/yolov8n.onnx

The camera topic is picked automatically from the car's profile in the car
workspace (racerbot_car, cars/<car>/car.yaml):
    car 1 (webcam)    -> /image_raw
    car 2 (RealSense) -> the topic in its car.yaml (/camera/camera/color/image_raw)
To force a topic: add image_topic:=/some/topic to the command.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WEBCAM_TOPIC = '/image_raw'


def car_camera_topic():
    """Return (topic, reason) for the car this is running on."""
    try:
        from racerbot_car.identity import current_car_id
        from racerbot_car.profile import CameraRealSense, load_profile
    except ImportError:
        return WEBCAM_TOPIC, 'car workspace not sourced, using the webcam default'
    try:
        car_id = current_car_id()
        camera = load_profile(car_id).camera
    except Exception as e:  # e.g. a laptop that isn't one of the cars
        return WEBCAM_TOPIC, f'could not tell which car this is ({e}), using the webcam default'
    if isinstance(camera, CameraRealSense):
        return camera.topic, f'car {car_id} has a RealSense'
    return WEBCAM_TOPIC, f'car {car_id} has a webcam'


def launch_setup(context):
    config = os.path.join(get_package_share_directory('yolo'), 'config', 'yolo.yaml')
    image_topic = LaunchConfiguration('image_topic').perform(context)
    if image_topic:
        reason = 'set on the command line'
    else:
        image_topic, reason = car_camera_topic()

    return [
        LogInfo(msg=f'YOLO camera topic: {image_topic} ({reason})'),
        Node(
            package='yolo',
            executable='yolo_node',
            name='yolo_node',
            output='screen',
            parameters=[config, {
                'model_path': LaunchConfiguration('model_path'),
                'image_topic': image_topic,
            }],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model_path', description='Path to the YOLOv8 .onnx file'),
        DeclareLaunchArgument('image_topic', default_value='',
                              description='Camera topic; empty = pick from the car profile'),
        OpaqueFunction(function=launch_setup),
    ])
