"""Start the YOLO node with config/yolo.yaml.

    ros2 launch yolo yolo_launch.py model_path:=/path/to/yolov8n.onnx
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(get_package_share_directory('yolo'), 'config', 'yolo.yaml')
    model_path = LaunchConfiguration('model_path')

    return LaunchDescription([
        DeclareLaunchArgument('model_path', description='Path to the YOLOv8 .onnx file'),
        Node(
            package='yolo',
            executable='yolo_node',
            name='yolo_node',
            output='screen',
            parameters=[config, {'model_path': model_path}],
        ),
    ])
