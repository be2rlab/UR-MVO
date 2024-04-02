# Use the official ROS 2 Galactic base image
FROM ros:galactic

# Set the working directory
WORKDIR /viz_ws

# Install additional dependencies for RViz
RUN apt-get update && apt-get install -y \
    ros-galactic-rviz2 \
    && rm -rf /var/lib/apt/lists/*

# Copy the entrypoint script
COPY ${ROOT_DIR}/docker/ros_entrypoint_visualization.sh /viz_ws/ros_entrypoint_visualization.sh

# Copy rviz2 settings from rviz folder
COPY ${ROOT_DIR}/rviz/. /viz_ws/rviz

# Set the entrypoint script as executable
RUN chmod +x /viz_ws/ros_entrypoint_visualization.sh

# Set the entrypoint command
ENTRYPOINT ["/viz_ws/ros_entrypoint_visualization.sh"]