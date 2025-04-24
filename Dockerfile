FROM ros:humble

# Install basic tools and Python virtual environment
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    python3-rosdep \
    python3-colcon-common-extensions \
    python3-rosinstall-generator \
    python3-rosinstall \
    python3-wstool \
    python3-numpy \
    python3-opencv \
    python3-tk \
    cython3 \
    python3-dev \
    build-essential \
    python3-yaml \
    python3-scipy \
    && rm -rf /var/lib/apt/lists/*

# Set up workspace
WORKDIR /workspace

# Copy requirements first
COPY metasejong_competitor_ws/requirements.txt /workspace/

# Create virtual environment and install packages
RUN python3 -m venv /venv/metacom_client_env && \
    . /venv/metacom_client_env/bin/activate && \
    pip install --upgrade pip && \
    pip install --no-cache-dir wheel && \
    pip install --no-cache-dir -r /workspace/requirements.txt

# Copy ROS2 package files
COPY metasejong_competitor_ws/src/airobotics_app /workspace/src/airobotics_app

# Install ROS2 dependencies
RUN . /opt/ros/humble/setup.sh && \
    rosdep update && \
    rosdep install --from-paths /workspace/src --ignore-src -y

# Build the workspace
RUN . /opt/ros/humble/setup.sh && \
    cd /workspace && \
    colcon build --symlink-install --packages-select airobotics_app

# Set up entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set environment variables
ENV ROS_DISTRO=humble
ENV ROS_PYTHON_VERSION=3
ENV PYTHONPATH="/workspace/src/airobotics_app:${PYTHONPATH}"
ENV PATH="/venv/metacom_client_env/bin:${PATH}"
ENV VIRTUAL_ENV="/venv/metacom_client_env"


ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]