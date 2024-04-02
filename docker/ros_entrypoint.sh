#!/bin/bash
set -e

source /opt/ros/$ROS_DISTRO/setup.bash
source /workspace/ros2_ws/install/setup.bash

exec "$@"