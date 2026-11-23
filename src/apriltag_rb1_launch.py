"""Car 1 webcam AprilTag tracking in one command.

Starts:
  1. camera TF        base_link -> camera_optical_frame (from the 2026-11-23
                      extrinsic calibration, see calibration_record_rb1.md)
  2. webcam publisher /image_raw + /camera_info (webcam_publisher.py)
  3. opponent_perception with the tag config
  4. (bench only)     odom -> base_link fixed at zero, for a parked car

Run (from anywhere, workspace sourced so opponent_perception is found):
    source /opt/ros/jazzy/setup.bash
    source ~/racerbot-ws/install/setup.bash
    ros2 launch ~/racerbot-vision/src/apriltag_rb1_launch.py

Options:
    bench:=false       the car's bringup is running and publishes odom itself
    config:=<file>     tag config (default apriltag_bench_rb1.yaml next to this file)

Nothing here publishes /drive; it cannot move the car.
The webcam must be free (no dashboard stream, no other camera program).
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))

# Camera pose on car 1 (calibrate_extrinsics_tag.py, 2026-11-23).
# Redo the calibration and update these if the camera is moved.
CAMERA_TF = ['--x', '0.2838', '--y', '-0.0580', '--z', '0.1493',
             '--qx', '-0.55184', '--qy', '0.47862', '--qz', '-0.44617', '--qw', '0.51703',
             '--frame-id', 'base_link', '--child-frame-id', 'camera_optical_frame']


def generate_launch_description():
    bench = LaunchConfiguration('bench')
    config = LaunchConfiguration('config')

    return LaunchDescription([
        DeclareLaunchArgument('bench', default_value='true',
                              description='true = car parked, publish a fixed odom -> base_link'),
        DeclareLaunchArgument('config',
                              default_value=os.path.join(HERE, 'apriltag_bench_rb1.yaml'),
                              description='opponent_perception tag config'),

        Node(package='tf2_ros', executable='static_transform_publisher',
             name='camera_tf', arguments=CAMERA_TF, output='log'),

        Node(package='tf2_ros', executable='static_transform_publisher',
             name='bench_odom_tf', condition=IfCondition(bench),
             arguments=['--frame-id', 'odom', '--child-frame-id', 'base_link'],
             output='log'),

        ExecuteProcess(
            name='webcam_publisher',
            cmd=['python3', os.path.join(HERE, 'webcam_publisher.py'),
                 '--info', os.path.join(HERE, 'camera_info_rb1.yaml')],
            output='screen'),

        Node(package='opponent_perception', executable='opponent_perception_node',
             name='opponent_perception_node', output='screen',
             parameters=[{'config_file': config,
                          'image_topic': '/image_raw',
                          'camera_info_topic': '/camera_info',
                          'use_sim_time': False}]),
    ])
