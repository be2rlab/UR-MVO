#ifndef ROS2_PUBLISHER_H_
#define ROS2_PUBLISHER_H_

#include <Eigen/Core>
#include <map>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "read_configs.h"
#include "thread_publisher.h"
#include "utils.h"

struct FeatureMessage {
  double time;
  cv::Mat image;
  std::vector<bool> inliers;
  std::vector<cv::KeyPoint> keypoints;
};
using FeatureMessagePtr = std::shared_ptr<FeatureMessage>;
using FeatureMessageConstPtr = std::shared_ptr<const FeatureMessage>;

struct DebugMessage {
  double time;
  cv::Mat image;
};
using DebugMessagePtr = std::shared_ptr<DebugMessage>;
using DebugMessageConstPtr = std::shared_ptr<const DebugMessage>;

struct FramePoseMessage {
  double time;
  Eigen::Matrix4d pose;
};
using FramePoseMessagePtr = std::shared_ptr<FramePoseMessage>;
using FramePoseMessageConstPtr = std::shared_ptr<const FramePoseMessage>;

struct KeyframeMessage {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double time;
  std::vector<int> ids;
  std::vector<Eigen::Matrix4d> poses;
  // std::vector<double> stamps;
};
using KeyframeMessagePtr = std::shared_ptr<KeyframeMessage>;
using KeyframeMessageConstPtr = std::shared_ptr<const KeyframeMessage>;

struct MapMessage {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double time;
  bool reset;
  std::vector<int> ids;
  std::vector<Eigen::Vector3d> points;
};
using MapMessagePtr = std::shared_ptr<MapMessage>;
using MapMessageConstPtr = std::shared_ptr<const MapMessage>;

double GetCurrentTime();

class Ros2Publisher : public rclcpp::Node {
public:
  Ros2Publisher(const RosPublisherConfig &ros_publisher_config);

  void PublishFeature(FeatureMessagePtr feature_message);
  void PublishDebug(DebugMessagePtr debug_message);
  void PublishFramePose(FramePoseMessagePtr frame_pose_message);
  void PublishKeyframe(KeyframeMessagePtr keyframe_message);
  void PublishMap(MapMessagePtr map_message);

  void ShutDown();
  void reset() {
    _ros_feature_pub.reset();
    _ros_frame_pose_pub.reset();
    _ros_keyframe_pub.reset();
    _ros_path_pub.reset();
    _ros_map_pub.reset();
  }

private:
  RosPublisherConfig _config;
  // for publishing features
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _ros_feature_pub;
  ThreadPublisher<FeatureMessage> _feature_publisher;

  // for publishing debug message
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _ros_debug_pub;
  ThreadPublisher<DebugMessage> _debug_publisher;

  // for publishing frame
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr
      _ros_frame_pose_pub;
  ThreadPublisher<FramePoseMessage> _frame_pose_publisher;

  // for publishing keyframes
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr _ros_keyframe_pub;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr _ros_path_pub;

  std::map<int, int> _keyframe_id_to_index;
  geometry_msgs::msg::PoseArray _ros_keyframe_array;
  nav_msgs::msg::Path _ros_path;
  ThreadPublisher<KeyframeMessage> _keyframe_publisher;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr _ros_map_pub;
  std::unordered_map<int, int> _mappoint_id_to_index;

  sensor_msgs::msg::PointCloud _ros_mappoints;
  ThreadPublisher<MapMessage> _map_publisher;
};
using Ros2PublisherPtr = std::shared_ptr<Ros2Publisher>;

#endif // ROS2_PUBLISHER_H_
