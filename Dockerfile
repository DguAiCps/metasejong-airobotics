FROM ros:humble

RUN rm -f /etc/apt/sources.list.d/ros*.list
RUN apt-get update && apt-get install -y curl gnupg2 lsb-release

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list

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
    rosdep install --from-paths /workspace/src --ignore-src -y && \
    apt update && \
    apt install -y ros-humble-rmw-cyclonedds-cpp

# Build the workspace
RUN . /opt/ros/humble/setup.sh && \
    cd /workspace && \
    colcon build --symlink-install --packages-select airobotics_app

RUN pip install scikit-learn
RUN pip install cv_bridge
RUN pip install ultralytics
RUN pip install numpy==1.24.3
RUN pip install scipy==1.11.4

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
