# ------------------------------------------------------------------------------
#                              BASE & DEPENDENCIES
# ------------------------------------------------------------------------------
FROM nvcr.io/nvidia/tensorrt:22.08-py3 AS base

# Set environment variable to non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Install locale settings
RUN apt-get update && apt-get install -y \
  locales \
  && locale-gen en_US.UTF-8 \
  && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
  && rm -rf /var/lib/apt/lists/*
ENV LANG en_US.UTF-8

# Set the timezone to UTC
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime \
  && apt-get update \
  && apt-get install -y tzdata \
  && dpkg-reconfigure --frontend noninteractive tzdata \
  && rm -rf /var/lib/apt/lists/*

# Install ROS2 environment
ENV ROS_DISTRO=galactic
ENV AMENT_PREFIX_PATH=/opt/ros/galactic
ENV COLCON_PREFIX_PATH=/opt/ros/galactic
ENV LD_LIBRARY_PATH=/opt/ros/galactic/lib
ENV PATH=/opt/ros/galactic/bin:$PATH
ENV PYTHONPATH=/opt/ros/galactic/lib/python3.8/site-packages
ENV ROS_PYTHON_VERSION=3
ENV ROS_VERSION=2
ENV DEBIAN_FRONTEND=

RUN apt-get update && apt-get install -y \
  curl \
  gnupg2 \
  lsb-release \
  sudo \
  && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
  && apt-get update && apt-get install -y \
  ros-${ROS_DISTRO}-ros-base \
  python3-argcomplete \
  && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
#                              Development image
# ------------------------------------------------------------------------------

FROM base AS dev

# Set the environment variable for non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Install development tools and dependencies
RUN apt-get update && apt-get install -y \
  bash-completion \
  build-essential \
  cmake \
  gdb \
  git \
  pylint3 \
  python3-argcomplete \
  python3-colcon-common-extensions \
  python3-pip \
  python3-rosdep \
  python3-vcstool \
  vim \
  wget \
  ros-${ROS_DISTRO}-ament-lint \
  ros-${ROS_DISTRO}-launch-testing \
  ros-${ROS_DISTRO}-launch-testing-ament-cmake \
  ros-${ROS_DISTRO}-launch-testing-ros \
  python3-autopep8 \
  && rm -rf /var/lib/apt/lists/* \
  && rosdep init || echo "rosdep already initialized" \
  && pip install --upgrade pydocstyle

# Define non-root user and group for development
ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create a non-root user and configure sudo access
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  && rm -rf /var/lib/apt/lists/* \
  && echo "source /usr/share/bash-completion/completions/git" >> /home/$USERNAME/.bashrc \
  && echo "if [ -f /opt/ros/${ROS_DISTRO}/setup.bash ]; then source /opt/ros/${ROS_DISTRO}/setup.bash; fi" >> /home/$USERNAME/.bashrc

# Reset environment variables
ENV DEBIAN_FRONTEND=
ENV AMENT_CPPCHECK_ALLOW_SLOW_VERSIONS=1

# ------------------------------------------------------------------------------
#                              Full image
# ------------------------------------------------------------------------------

FROM dev AS full

# Set the environment variable for non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Install the full ROS release
RUN apt-get update && apt-get install -y \
  ros-${ROS_DISTRO}-desktop \
  && rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND=

# Install dependencies for OpenGL and Nvidia driver
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
  libglvnd0 \
  libgl1 \
  libglx0 \
  libegl1 \
  libxext6 \
  libx11-6

# Set Nvidia-related environment variables
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
ENV QT_X11_NO_MITSHM 1

# Install additional dependencies
RUN apt-get update && apt-get install -y \
  libopencv-dev \
  libeigen3-dev \
  libatlas-base-dev \
  libsuitesparse-dev \
  libeigen3-dev \
  libgoogle-glog-dev \
  libspdlog-dev \
  libboost-all-dev \
  libfmt-dev \
  libyaml-cpp-dev\
  swig

# Clone and build G2O
RUN git clone https://github.com/RainerKuemmerle/g2o.git /tmp/g2o
RUN mkdir -p /tmp/g2o/build && cd /tmp/g2o/build && cmake ../ && make -j3 && make install
RUN ldconfig

# Clone and build ONNX C++ library
RUN git clone --recursive https://github.com/onnx/onnx.git /tmp/onnx
RUN mkdir -p /tmp/onnx/build && cd /tmp/onnx/build && cmake .. && make -j3 && make install

# Clone and build Casadi
RUN git clone https://github.com/casadi/casadi.git -b main /tmp/casadi
RUN mkdir -p /tmp/casadi/build && cd /tmp/casadi/build && cmake -DWITH_PYTHON=ON -DWITH_PYTHON3=ON .. && make -j3 && make install
RUN ldconfig


# Cleanup
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install torch \
  && pip install pybind11 \
  && pip install yacs


RUN pip install --upgrade pip
RUN pip install pyquaternion \
  && pip3 install torchvision>=0.3 \
  && pip3 install numpy \
  && pip3 install tqdm>=4.36.0 \
  && pip3 install scipy \
  && pip3 install transformers \
  && pip3 install Pillow


FROM full as build 

# Switch to the ROS2 workspace and install Python packages
WORKDIR /workspace/ros2_ws/src
COPY ${ROOT_DIR} /workspace/ros2_ws/src/ur_mvo
WORKDIR /workspace/ros2_ws/src/ur_mvo
RUN pip install -e .

RUN pip install ruamel.yaml==0.17.32
WORKDIR /workspace/ros2_ws



ENV PYTHONPATH "/workspace/ros2_ws/src/ur_mvo:${PYTHONPATH}"
ENV PYTHONPATH "/workspace/ros2_ws/build/ur_mvo:${PYTHONPATH}"

# Build code in the ROS2 workspace
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash "
RUN /bin/bash -c "colcon build --executor sequential"
RUN /bin/bash -c "source install/setup.bash"

# Copy the entrypoint script
COPY ${ROOT_DIR}/docker/ros_entrypoint.sh /

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["/bin/bash"]
