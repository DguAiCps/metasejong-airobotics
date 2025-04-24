#!/bin/bash
set -e

# Source ROS2 environment
source /opt/ros/humble/setup.bash

# Activate virtual environment
if [ -f "/venv/metacom_client_env/bin/activate" ]; then
    source /venv/metacom_client_env/bin/activate
fi

# Source workspace if it exists
if [ -f "/workspace/install/setup.bash" ]; then
    source /workspace/install/setup.bash
fi

# Set environment variables
export ROS_DISTRO=humble
export ROS_PYTHON_VERSION=3
export PYTHONPATH="/workspace/src/airobotics_app:${PYTHONPATH}"
export PATH="/venv/metacom_client_env/bin:${PATH}"
export VIRTUAL_ENV="/venv/metacom_client_env"

# Execute the command passed to the container
if [ "$1" = 'bash' ]; then
    exec "$@"
else
    # Run ROS2 node
    exec ros2 run airobotics_app airobotics_node
fi 